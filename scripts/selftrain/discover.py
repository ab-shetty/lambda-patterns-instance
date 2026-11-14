#!/usr/bin/env python3
"""Self-training / pre-labelling, stage A (GPU): find the pattern families of an UNLABELLED sheet with the model itself.

Per sheet (the image backbone runs once per scale; only the decoder runs per box):
  1. candidate reference boxes on a grid (side ~5% of the long side, 112-320 px native), skipping blank paper and boxes
     mostly covered by text (scripts/text_mask.py);
  2. each box is asked at 4096; a box is ACCEPTED only if it lies wholly inside its own confident answer
     (>= IN_FRAC of its pixels at P > 0.9) and the answer is a sensible size (A_MIN..A_MAX of the sheet);
  3. accepted boxes are grouped into families by complete linkage on answer IoU >= LINK: every pair in a family must agree.
     Conservative on purpose -- two look-alike families are kept apart rather than merged (a merge is the expensive error
     for both a reviewer and self-training); family pairs whose answers overlap (IoU >= FLAG) are flagged;
  4. one representative box per family (the medoid: highest mean IoU with the other members) is asked at the product
     scales 5120 / 4096 / 3072, and the three maps are saved in the --save-probs layout, for stage B (pseudolabel.py).

    PYTHONPATH=. python3 scripts/selftrain/discover.py --checkpoint data/runs/ck_fgavg78.pth \
        --local-pool data/eval_pools/gemini30_s20261006 --out data/selftrain/disc_gemini30
    ... --real 4,5,6 --out ...            # sheets of the HF real-world-test set (labels used only by stage B's report)
    ... --images DIR --out ...            # a folder of unlabelled images (round 5)
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_refunet_selection import load_refunet, _normalize_on, load_parquet_records  # noqa: E402
from refmask2former.dataset import _normalize_chw  # noqa: E402
from text_mask import text_mask  # noqa: E402

SCALES = (5120, 4096, 3072)
DISC_SCALE = 4096
LOW = 1024              # long side of the low-res answers used for grouping
IN_FRAC = 0.9           # share of the box at P > 0.9 for the box to count as inside its own answer
A_MIN, A_MAX = 0.002, 0.8
PASSES = (1, 2, 4)      # box side divisors: later passes use smaller boxes, only where no accepted answer covers yet
B_MIN = 40
LINK = 0.75             # complete-linkage answer IoU to share a family
FLAG = 0.25             # family pairs overlapping at least this much are flagged


def sheets(a):
    if a.local_pool:
        pool = Path(a.local_pool)
        for f in sorted((pool / "annotations").glob("*.json")):
            ann = json.loads(f.read_text())
            yield f.stem, (pool / "images" / ann["image"]["file_name"]).read_bytes()
    elif a.real:
        recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
        for i in [int(x) for x in a.real.split(",")]:
            yield f"real{i:03d}", recs[i]["image"]
    else:
        for f in sorted(Path(a.images).iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                yield f.stem, f.read_bytes()


def box_plane(box, h0, w0, nh, nw, device):
    x, y, w, h = box
    rows = np.zeros((h0, 1), np.uint8); rows[y:y + h] = 1
    cols = np.zeros((1, w0), np.uint8); cols[0, x:x + w] = 1
    r = torch.from_numpy(cv2.resize(rows, (1, nh), interpolation=cv2.INTER_NEAREST)[:, 0]).to(device).float()
    c = torch.from_numpy(cv2.resize(cols, (nw, 1), interpolation=cv2.INTER_NEAREST)[0]).to(device).float()
    return r[:, None] * c[None, :]


@torch.inference_mode()
def ask(model, img0, boxes, side, device, ref_size=224, q_batch=16):
    """uint8 probability maps (len(boxes), nh, nw) at long side `side`, the evaluator's inference path."""
    h0, w0 = img0.shape[:2]
    s = side / max(h0, w0)
    nh, nw = max(1, round(h0 * s)), max(1, round(w0 * s))
    img = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t = _normalize_on(img, device).unsqueeze(0)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feats = model.features(t)
        for c0 in range(0, len(boxes), q_batch):
            chunk = boxes[c0:c0 + q_batch]
            refs = torch.stack([_normalize_chw(cv2.resize(img0[y:y + h, x:x + w], (ref_size, ref_size),
                                                          interpolation=cv2.INTER_LINEAR)) for x, y, w, h in chunk]).to(device)
            planes = torch.stack([box_plane(b, h0, w0, nh, nw, device) for b in chunk])[:, None] if model.roi_ref else None
            logit = model.forward_cached(feats, t.shape[-2:], refs, planes, image=t)
            out.append((logit.sigmoid().float()[:, 0] * 255).round().to(torch.uint8).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, nh, nw), np.uint8)


def candidates(img0, txt, div=1, covered=None):
    """Grid boxes of side ~5% of the long side / div. `covered` (native bool): skip boxes mostly inside it."""
    h0, w0 = img0.shape[:2]
    b = max(B_MIN, int(np.clip(round(0.05 * max(h0, w0)), 112, 320)) // div)
    gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    out = []
    for y in range(0, h0 - b + 1, b):
        for x in range(0, w0 - b + 1, b):
            if covered is not None and covered[y:y + b, x:x + b].mean() > 0.5:
                continue
            g = gray[y:y + b, x:x + b]
            if g.mean() > 245 and g.std() < 6:          # blank paper
                continue
            if txt[y:y + b, x:x + b].mean() > 0.3:      # mostly text
                continue
            out.append((x, y, b, b))
    return out


def complete_linkage(iou, thr):
    """Clusters in which every pair has IoU >= thr (scipy complete linkage on 1 - IoU)."""
    if len(iou) == 1:
        return [[0]]
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = np.clip(1 - iou, 0, 1)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "complete"), t=1 - thr + 1e-9, criterion="distance")
    return [list(np.nonzero(lab == c)[0]) for c in np.unique(lab)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-pool"); src.add_argument("--real"); src.add_argument("--images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--q-batch", type=int, default=16, help="boxes per decoder call (smaller = less GPU memory)")
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's share of GPU memory (next to training)")
    a = ap.parse_args()
    device = torch.device("cuda")
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    model, _ = load_refunet(a.checkpoint, device)
    model.roi_ref = bool(getattr(model, "roi_ref", False))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for n, (sid, data) in enumerate(sheets(a)):
        if a.limit and n >= a.limit:
            break
        d = out / sid
        if (d / "families.json").exists():
            continue
        img0 = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        h0, w0 = img0.shape[:2]
        txt = text_mask(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)).astype(bool)
        acc, low, n_boxes = [], [], 0
        ls = LOW / max(h0, w0)
        lh, lw = max(1, round(h0 * ls)), max(1, round(w0 * ls))
        for div in PASSES:
            covered = (cv2.resize(np.logical_or.reduce(low).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
                       if low else None)
            boxes = candidates(img0, txt, div, covered)
            n_boxes += len(boxes)
            P = ask(model, img0, boxes, DISC_SCALE, device, q_batch=a.q_batch)
            nh, nw = P.shape[1:] if len(P) else (1, 1)
            for b, p in zip(boxes, P):
                x, y, w, h = b
                s = nw / w0
                inbox = p[int(y * s):int((y + h) * s), int(x * s):int((x + w) * s)]
                frac_in = float((inbox > 229).mean()) if inbox.size else 0.0
                area = float((p > 127).mean())
                if frac_in >= IN_FRAC and A_MIN <= area <= A_MAX:
                    acc.append(b)
                    low.append(cv2.resize(p, (lw, lh), interpolation=cv2.INTER_AREA) > 127)
            del P
        fams = []
        if acc:
            L = np.stack(low).reshape(len(low), -1)
            inter = L.astype(np.float32) @ L.T.astype(np.float32)
            area = L.sum(1).astype(np.float32)
            iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
            for cl in complete_linkage(iou, LINK):
                med = cl[int(np.argmax(iou[np.ix_(cl, cl)].mean(1)))]
                fams.append({"members": [list(acc[i]) for i in cl], "support": len(cl), "box": list(acc[med]),
                             "low": np.logical_or.reduce([low[i] for i in cl])})
            fams.sort(key=lambda f: -f["support"])
        # flag overlapping family pairs (merge / split risk)
        for i, f in enumerate(fams):
            f["overlaps"] = []
            for j, g in enumerate(fams):
                if i != j:
                    u = (f["low"] | g["low"]).sum()
                    o = float((f["low"] & g["low"]).sum() / max(u, 1))
                    if o >= FLAG:
                        f["overlaps"].append([j, round(o, 3)])
        # product-scale maps of each family's representative box
        md = d / "maps"
        for z in SCALES:
            (md / str(z)).mkdir(parents=True, exist_ok=True)
            if fams:
                M = ask(model, img0, [tuple(f["box"]) for f in fams], z, device, q_batch=a.q_batch)
                for k, m in enumerate(M):
                    np.savez_compressed(md / str(z) / f"{k:03d}.npz", prob=m)
        meta = {"sheet": sid, "h": h0, "w": w0, "n_candidates": n_boxes, "n_accepted": len(acc),
                "families": [{k: v for k, v in f.items() if k != "low"} for f in fams],
                "params": dict(PASSES=PASSES, B_MIN=B_MIN, IN_FRAC=IN_FRAC, A_MIN=A_MIN, A_MAX=A_MAX, LINK=LINK, FLAG=FLAG, DISC_SCALE=DISC_SCALE),
                "checkpoint": a.checkpoint}
        (d / "families.json").write_text(json.dumps(meta, indent=1))
        print(f"{sid}: {n_boxes} boxes, {len(acc)} accepted, {len(fams)} families "
              f"(support {[f['support'] for f in fams]}), peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
