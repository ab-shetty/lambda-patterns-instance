#!/usr/bin/env python3
"""Boundary-patch refiner (after BPR, Tang et al. CVPR 2021), trained on synthetic sheets only (2026-10-20).

Works map -> map at the evaluator's inference resolution (sheet resized to long side 4096, where `--save-probs` maps live), so
synthetic and real sheets are refined at the same scale and the output feeds scripts/polygonize.py / scripts/editcost.py
unchanged. Per question: 64 x 64 patches centred on the coarse mask's edge (P > 0.35), input = sheet RGB + P + (P > 0.35),
output = the patch's mask logits; overlapping patches are averaged with a Hann window and replace P only within --band px of
the coarse edge; everything else keeps the model's P.

  build  PATCHES from a synthetic pool + the best model's maps on it (edges of P and of the target; plus corrupted-target
         coarse masks: dilate / erode / shift / blur), split 90 / 10 by sheet
  train  a small U-Net on them (BCE + Dice), flips / rot90 / tone jitter
  apply  refine a maps dir (hf14 / val / gemini30 / a pool) into OUT/<split>_4096/probs, same file names

  python3 scripts/bpr/bpr.py build --pool data/synthetic/bpr_s900000 --maps data/bpr/maps_s900000/probs --out data/bpr/patches
  python3 scripts/bpr/bpr.py train --patches data/bpr/patches --out data/bpr/refiner.pth
  python3 scripts/bpr/bpr.py apply --ckpt data/bpr/refiner.pth --split val --out data/bpr/refined_b8 --band 8
Labels: synthetic labels split boundary strokes like the real ones (synth_progress 2026-10-20 "BPR precheck")."""
import argparse, json, random, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")

K = 64          # patch side, inference-resolution px
T = 89          # P > 0.35 on the uint8 maps


def edge(b):
    b8 = b.astype(np.uint8)
    return b & ~cv2.erode(b8, np.ones((3, 3), np.uint8)).astype(bool)


def centres(b, stride, rng=None):
    """One edge pixel per stride x stride cell (random at training, the cell's first at inference)."""
    ys, xs = np.nonzero(edge(b))
    if not len(ys):
        return np.zeros((0, 2), int)
    cell = (ys // stride) * 100_000 + xs // stride
    order = np.arange(len(ys)) if rng is None else rng.permutation(len(ys))
    _, first = np.unique(cell[order], return_index=True)
    sel = order[first]
    return np.stack([ys[sel], xs[sel]], 1)


def crop(a, y, x):
    h = K // 2
    pad = [(h, h), (h, h)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, mode="edge")[y:y + K, x:x + K]


def sheet_inputs(img0, p):
    """The sheet at the map's resolution, resized exactly as the evaluator does (linear)."""
    return cv2.resize(img0, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_LINEAR)


def stack(img, p, ys, xs):
    """[N, 5, K, K] uint8: RGB, P, P > T."""
    big = np.concatenate([img, p[..., None], ((p > T) * 255).astype(np.uint8)[..., None]], 2)
    big = np.pad(big, [(K // 2, K // 2), (K // 2, K // 2), (0, 0)], mode="edge")
    return np.stack([big[y:y + K, x:x + K] for y, x in zip(ys, xs)]).transpose(0, 3, 1, 2)


# ----------------------------------------------------------------------------- build
def _pool_jobs(pool):
    from evaluate_refunet_selection import _prepare_sheet  # noqa: F401  (import check)
    files = sorted((Path(pool) / "annotations").glob("*.json"))
    return [(i, str(f)) for i, f in enumerate(files)]


def _corrupt(t, rng):
    m = t.astype(np.uint8)
    k = int(rng.integers(-3, 4))
    if k:
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1,) * 2)
        m = cv2.dilate(m, el) if k > 0 else cv2.erode(m, el)
    dy, dx = rng.integers(-2, 3, 2)
    m = np.roll(m, (int(dy), int(dx)), (0, 1)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), float(rng.uniform(0.8, 3.0)))
    return np.clip(m * 255 + rng.normal(0, 12, m.shape), 0, 255).astype(np.uint8)


def _build_sheet(args):
    (i, ann_path), maps, per_q, seed = args
    from evaluate_refunet_selection import _prepare_sheet
    ann = json.loads(Path(ann_path).read_text())
    rec = {"image_path": str(Path(ann_path).parent.parent / "images" / ann["image"]["file_name"]),
           "annotations": ann["annotations"], "ref_exclude": ann.get("ref_exclude")}
    sh = _prepare_sheet((i, rec, 4096, 224, False, {}, [], False))
    img0 = cv2.cvtColor(cv2.imread(rec["image_path"]), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(seed * 1_000_003 + i)
    X, Y = [], []
    img = None
    for ref_idx, cat, box, _, _ in sh["questions"]:
        f = Path(maps) / f"{i:03d}_{ref_idx:03d}.npz"
        if not f.exists():
            continue
        p = np.load(f)["prob"]
        if img is None:
            img = sheet_inputs(img0, p)
        bits, _ = sh["targets"][cat]
        t0 = np.unpackbits(bits, count=sh["h0"] * sh["w0"]).reshape(sh["h0"], sh["w0"])
        t = cv2.resize(t0, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        for src, n in ((p, per_q), (None, per_q // 4), ("corrupt", per_q // 3)):
            if src is None:            # edges of the target: where the model missed or overshot
                q, c = p, centres(t, 24, rng)
            elif isinstance(src, str):
                q = _corrupt(t, rng); c = centres(q > T, 24, rng)
            else:
                q, c = p, centres(p > T, 24, rng)
            if not len(c):
                continue
            c = c[rng.permutation(len(c))[:n]]
            X.append(stack(img, q, c[:, 0], c[:, 1]))
            Y.append(np.stack([crop(t, y, x) for y, x in c]).astype(np.uint8))
    if not X:
        return None
    return i, np.concatenate(X), np.concatenate(Y)


def build(a):
    jobs = _pool_jobs(a.pool)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0); ids = [j[0] for j in jobs]; rng.shuffle(ids)
    val_ids = set(ids[: len(ids) // 10])
    acc = {"train": ([], []), "val": ([], [])}
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_build_sheet, [(j, a.maps, a.per_q, 0) for j in jobs], chunksize=4):
            if r is None:
                continue
            i, X, Y = r
            s = "val" if i in val_ids else "train"
            acc[s][0].append(X); acc[s][1].append(Y)
    for s, (X, Y) in acc.items():
        X, Y = np.concatenate(X), np.concatenate(Y)
        np.save(out / f"{s}_x.npy", X); np.save(out / f"{s}_y.npy", Y)
        print(s, X.shape, "positive share", round(float(Y.mean()), 3))


# ----------------------------------------------------------------------------- model
def make_net():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self, c=(32, 64, 128, 256)):
            super().__init__()
            self.d = nn.ModuleList([block(5, c[0])] + [block(c[k], c[k + 1]) for k in range(3)])
            self.u = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, 2) for k in range(3)])
            self.ub = nn.ModuleList([block(2 * c[k], c[k]) for k in range(3)])
            self.head = nn.Conv2d(c[0], 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for k, d in enumerate(self.d):
                x = d(x if k == 0 else self.pool(x)); skips.append(x)
            for k in (2, 1, 0):
                x = self.ub[k](__import__("torch").cat([self.u[k](x), skips[k]], 1))
            return self.head(x)
    return UNet()


def to_input(xb, torch):
    x = xb.float() / 255
    x[:, :3] = (x[:, :3] - 0.5) / 0.25
    x[:, 3:] = (x[:, 3:] - 0.5) / 0.5
    return x


# ----------------------------------------------------------------------------- train
def train(a):
    import torch, torch.nn.functional as F
    torch.manual_seed(a.seed)
    d = Path(a.patches); dev = "cuda"
    # fully in RAM (~13 GB): a tensor over a read-only memory map bus-errors when indexed
    Xtr_t, Ytr_t = torch.from_numpy(np.load(d / "train_x.npy")), torch.from_numpy(np.load(d / "train_y.npy"))
    Xva, Yva = np.load(d / "val_x.npy"), np.load(d / "val_y.npy")
    net = make_net().to(dev)
    steps = a.epochs * (len(Xtr_t) // a.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    print(f"train {len(Xtr_t)} val {len(Xva)} patches, {steps} steps")

    def loss_fn(lo, y):
        bce = F.binary_cross_entropy_with_logits(lo, y)
        p = lo.sigmoid(); i = (p * y).flatten(1).sum(1); s = p.flatten(1).sum(1) + y.flatten(1).sum(1)
        return bce + (1 - (2 * i + 1) / (s + 1)).mean()

    def evaluate():
        net.eval(); tot = [0, 0, 0, 0]
        with torch.no_grad():
            for k in range(0, len(Xva), 2048):
                x = to_input(torch.from_numpy(Xva[k:k + 2048]).to(dev), torch); y = torch.from_numpy(Yva[k:k + 2048]).to(dev).bool()
                pr = net(x)[:, 0] > 0; co = x[:, 4] > 0       # coarse = P > T channel
                tot[0] += (pr & y).sum().item(); tot[1] += (pr | y).sum().item()
                tot[2] += (co & y).sum().item(); tot[3] += (co | y).sum().item()
        net.train()
        return tot[0] / tot[1], tot[2] / tot[3]

    step = 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr_t))
        for k in range(0, len(perm) - a.batch + 1, a.batch):
            idx = perm[k:k + a.batch]
            x = Xtr_t[idx].to(dev, non_blocking=True); y = Ytr_t[idx].to(dev, non_blocking=True).float()[:, None]
            # flips / rot90 (same on input and target), tone jitter on RGB
            r = int(torch.randint(4, ()))
            x, y = torch.rot90(x, r, (2, 3)), torch.rot90(y, r, (2, 3))
            if torch.rand(()) < 0.5:
                x, y = x.flip(3), y.flip(3)
            x = to_input(x, torch)
            g = torch.empty(len(x), 1, 1, 1, device=dev)
            x[:, :3] = x[:, :3] * g.uniform_(0.8, 1.25) + g.clone().uniform_(-0.3, 0.3)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(x)
            loss = loss_fn(lo.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
            if step % 500 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
        ref_iou, coarse_iou = evaluate()
        print(f"epoch {ep}: synthetic val patch IoU refined {ref_iou:.4f} vs coarse {coarse_iou:.4f}", flush=True)
        torch.save({"model": net.state_dict(), "epoch": ep, "args": vars(a)}, a.out)


# ----------------------------------------------------------------------------- apply
_NET = [None]


def refine(net, img, p, band, stride, dev, torch):
    b = p > T
    c = centres(b, stride)
    if not len(c):
        return p
    acc = np.zeros(p.shape, np.float32); wsum = np.zeros(p.shape, np.float32)
    w = np.outer(np.hanning(K + 2)[1:-1], np.hanning(K + 2)[1:-1]).astype(np.float32)
    H, W = p.shape; h = K // 2
    accp = np.pad(acc, h); wp = np.pad(wsum, h)
    with torch.no_grad():
        for k in range(0, len(c), 1024):
            cc = c[k:k + 1024]
            x = to_input(torch.from_numpy(stack(img, p, cc[:, 0], cc[:, 1])).to(dev), torch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(x)[:, 0].float().cpu().numpy()
            for (y, xx), l in zip(cc, lo):
                accp[y:y + K, xx:xx + K] += l * w; wp[y:y + K, xx:xx + K] += w
    acc, wsum = accp[h:h + H, h:h + W], wp[h:h + H, h:h + W]
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5) + cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5)
    use = (wsum > 1e-3) & (d <= band)
    out = p.copy()
    out[use] = np.clip(255 / (1 + np.exp(-acc[use] / wsum[use])), 0, 255).astype(np.uint8)
    return out


def apply(a):
    import torch, common
    dev = "cuda"
    net = make_net().to(dev); net.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"]); net.eval()
    common.MAPS = Path(a.maps)
    out = Path(a.out) / f"{'hf14' if a.split.startswith('hf14') else a.split}_4096" / "probs"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for job in common.jobs(a.split):
        idx, img0, anyann, qs = common.load(job)
        img = None
        for ref, cat, tgt, p in qs:
            if img is None:
                img = sheet_inputs(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), p)
            np.savez_compressed(out / f"{idx:03d}_{ref:03d}.npz", prob=refine(net, img, p, a.band, a.stride, dev, torch)); n += 1
    print(a.split, n, "maps ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--pool"); b.add_argument("--maps"); b.add_argument("--out")
    b.add_argument("--per-q", type=int, default=48); b.add_argument("--workers", type=int, default=32)
    t = sp.add_parser("train"); t.add_argument("--patches"); t.add_argument("--out"); t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=512); t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("apply"); p.add_argument("--ckpt"); p.add_argument("--split"); p.add_argument("--out")
    p.add_argument("--maps", default="data/floz-diag/xblend-kq30-kms30"); p.add_argument("--band", type=float, default=8)
    p.add_argument("--stride", type=int, default=16)
    a = ap.parse_args()
    {"build": build, "train": train, "apply": apply}[a.cmd](a)


if __name__ == "__main__":
    main()
