#!/usr/bin/env python3
"""Fine-tune SAM 3's mask decoder for box-prompted region segmentation on plans.

Trains the labelling-assist model: a hand-drawn box over a material region ->
the mask of that region. The 454M vision encoder is FROZEN and only the 4.2M
mask decoder is trained, which is the right scale for 767 training instances and
lets the encoder run under no_grad.

Target is the OUTER ring (holes filled); windows are cut afterwards as separate
`remove` polygons, matching the Roboflow pipeline.

Boxes are jittered every epoch because the deployed prompt is hand-drawn, not a
tight ground-truth box.

    PYTHONPATH=. python3 scripts/train_sam3_boxseg.py --epochs 12 --out data/runs/sam3_ft
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask
from scripts.regularize_polygon import regularize
from scripts.sam3_augment import PRESETS, augment

MASK_RES = 288


def load_split(path):
    return [json.loads(l) for l in open(path)]


def jitter(box, W, H, frac, rng):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    j = [x0 + rng.uniform(-frac, frac) * bw, y0 + rng.uniform(-frac, frac) * bh,
         x1 + rng.uniform(-frac, frac) * bw, y1 + rng.uniform(-frac, frac) * bh]
    return [max(0.0, j[0]), max(0.0, j[1]), min(W - 1.0, j[2]), min(H - 1.0, j[3])]


def interior_points(mask, n, rng, keep_frac=0.5):
    """`n` clicks inside `mask`, biased away from the boundary by a distance
    transform. A person clicks somewhere obviously inside a region, not on its
    edge, so uniform sampling over the mask would model the gesture badly."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if d.max() <= 0:
        return []
    ys, xs = np.where(d >= keep_frac * d.max())
    if len(xs) == 0:
        ys, xs = np.where(mask)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return [[int(xs[i]), int(ys[i])] for i in idx]


def negative_points(region, others, n, rng, band=(8, 40)):
    """`n` clicks meaning "not this region".

    Half come from a band just OUTSIDE the boundary and half from other labelled
    regions on the sheet. The band is the informative half: a single positive
    click cannot say where a region ends -- on repeating siding it is equally
    consistent with one course, one panel, or the whole wall -- and a negative
    just past the edge supplies exactly that missing extent.
    """
    if n <= 0:
        return []
    d = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where((d >= band[0]) & (d <= band[1]))
    pts = []
    n_band = n if others is None or not others.any() else (n + 1) // 2
    if len(xs):
        idx = rng.choice(len(xs), size=min(n_band, len(xs)), replace=False)
        pts += [[int(xs[i]), int(ys[i])] for i in idx]
    if others is not None and others.any() and len(pts) < n:
        oy, ox = np.where(others & ~region)
        if len(ox):
            idx = rng.choice(len(ox), size=min(n - len(pts), len(ox)), replace=False)
            pts += [[int(ox[i]), int(oy[i])] for i in idx]
    return pts[:n]


def _hole_union(inst, H, W):
    """Union of an instance's holes. NOT render_instance_mask: that treats the
    first polygon as an outer ring and subtracts the rest, where here every
    polygon is a hole and they are OR-ed together."""
    m = np.zeros((H, W), np.uint8)
    for hp in inst.get("hole_polygons", []):
        pts = np.asarray(hp, np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def targets_for(scene, res=MASK_RES, target="outer"):
    H, W = scene["height"], scene["width"]
    out = []
    for inst in scene["instances"]:
        if target == "holes":
            m = _hole_union(inst, H, W)
        else:
            m = render_instance_mask([inst["outer_polygon"]], H, W).astype(np.uint8)
        out.append(cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.float32)


def dice_bce(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum((-2, -1)) + eps
    den = p.sum((-2, -1)) + target.sum((-2, -1)) + eps
    return bce + (1.0 - num / den).mean()


@torch.no_grad()
def evaluate(model, proc, scenes, dev, jitter_frac, seed=7, multimask=False):
    rng = np.random.default_rng(seed)
    mask_ious, poly_ious, verts = [], [], []
    for s in scenes:
        im = Image.open(s["image"]).convert("RGB")
        W, H = im.size
        boxes = [jitter(i["bbox_xyxy"], W, H, jitter_frac, rng) for i in s["instances"]]
        inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
        out = model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"],
                    multimask_output=multimask)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        for k, inst in enumerate(s["instances"]):
            gt = render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
            mk = masks[k].numpy()
            mk = mk if mk.ndim == 2 else mk[int(np.argmax(
                out.iou_scores[0, k].detach().cpu().numpy().reshape(-1)))]
            mk = mk.astype(bool)
            mask_ious.append(float((mk & gt).sum()) / max(1, int((mk | gt).sum())))
            p = regularize(mk)
            if p is None or len(p) < 3:
                poly_ious.append(0.0); continue
            pm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(pm, [np.round(p).astype(np.int32)], 1)
            pm = pm.astype(bool)
            poly_ious.append(float((pm & gt).sum()) / max(1, int((pm | gt).sum())))
            verts.append(len(p))
    return (np.array(mask_ious), np.array(poly_ious), np.array(verts))


@torch.no_grad()
def evaluate_holes(model, proc, scenes, dev, jitter_frac, seed=7):
    """Mask IoU of the predicted hole union. An instance with no holes has an
    empty target: predicting empty scores 1.0, predicting anything scores 0.0,
    so false holes on solid walls are punished as hard as missed ones."""
    rng = np.random.default_rng(seed)
    all_iou, holed_iou, solid_ok = [], [], []
    for s in scenes:
        im = Image.open(s["image"]).convert("RGB")
        W, H = im.size
        boxes = [jitter(i["bbox_xyxy"], W, H, jitter_frac, rng) for i in s["instances"]]
        inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
        out = model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"],
                    multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        for k, inst in enumerate(s["instances"]):
            gt = _hole_union(inst, H, W).astype(bool)
            mk = masks[k].numpy()
            mk = (mk if mk.ndim == 2 else mk[0]).astype(bool)
            # a hole only counts inside its own region
            mk &= render_instance_mask([inst["outer_polygon"]], H, W).astype(bool)
            if gt.sum() == 0:
                v = 1.0 if mk.sum() == 0 else 0.0
                solid_ok.append(v)
            else:
                v = float((mk & gt).sum()) / max(1, int((mk | gt).sum()))
                holed_iou.append(v)
            all_iou.append(v)
    return (np.array(all_iou), np.array(holed_iou),
            np.array(solid_ok) if solid_ok else np.array([1.0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sam3_ft")
    ap.add_argument("--out", default="data/runs/sam3_ft")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--max-boxes", type=int, default=12, help="cap boxes per forward (memory)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="box", choices=["box", "point", "mixed"],
                    help="gesture the decoder is trained on. 'mixed' alternates "
                         "per sheet so one decoder serves both a drag and a click.")
    ap.add_argument("--max-pos", type=int, default=3,
                    help="positive clicks per instance; drawn 1..max each sheet")
    ap.add_argument("--max-neg", type=int, default=0,
                    help="negative clicks per instance, drawn 0..max each sheet. "
                         "Without these the decoder never sees a negative in "
                         "training but is handed them at inference.")
    ap.add_argument("--target", default="outer", choices=["outer", "holes"],
                    help="'holes' trains the subtractive second model: same box "
                         "prompt, target is the union of that region's holes")
    ap.add_argument("--aug", default="none", choices=sorted(PRESETS),
                    help="sheet augmentation arm (scripts/sam3_augment.py); "
                         "'none' reproduces the 2026-09-08 result")
    args = ap.parse_args()
    aug_cfg = PRESETS[args.aug]

    dev = "cuda"
    tok = os.environ.get("HF_TOKEN")
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3", token=tok)
    model = Sam3TrackerModel.from_pretrained("facebook/sam3", token=tok).to(dev)

    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in train_params)
    print(f"trainable: {n_tr/1e6:.2f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train = load_split(f"{args.data}/train.jsonl")
    val = load_split(f"{args.data}/val.jsonl")
    print(f"train {len(train)} images / {sum(len(s['instances']) for s in train)} instances | "
          f"val {len(val)} images / {sum(len(s['instances']) for s in val)} instances")

    def val_report(ep_label):
        """-> (score used for checkpointing, printable line, history entry)"""
        if args.target == "holes":
            a, h, so = evaluate_holes(model, proc, val, dev, args.jitter, seed=args.seed)
            # weight the two failure modes equally: missing a real hole, and
            # inventing one on a solid wall. A plain mean would be dominated by
            # the 79% of instances that have no holes at all.
            sc = 0.5 * float(h.mean()) + 0.5 * float(so.mean())
            line = (f"{ep_label}: hole IoU {h.mean():.4f} (n={len(h)})  "
                    f"solid-clean {so.mean():.1%}  score {sc:.4f}")
            e = {"hole_iou": float(h.mean()), "solid_clean": float(so.mean()),
                 "score": sc, "poly_median": sc, "mask_mean": float(a.mean()),
                 "ge80": float((h >= .8).mean())}
        else:
            mi, pi, vv = evaluate(model, proc, val, dev, args.jitter, seed=args.seed)
            sc = float(np.median(pi))
            line = (f"{ep_label}: mask {mi.mean():.4f}/{np.median(mi):.4f}  "
                    f"poly {pi.mean():.4f}/{np.median(pi):.4f}  "
                    f">=0.8 {(pi>=.8).mean():.1%}  verts {np.median(vv):.0f}")
            e = {"poly_mean": float(pi.mean()), "poly_median": sc,
                 "mask_mean": float(mi.mean()), "ge80": float((pi >= .8).mean())}
        return sc, line, e

    model.eval()
    os.makedirs(args.out, exist_ok=True)
    best, line, e0 = val_report("\nepoch  -1 (zero-shot)")
    print(line)
    hist = [dict(e0, epoch=-1)]

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    for ep in range(args.epochs):
        model.train(); model.vision_encoder.eval()
        order = rng.permutation(len(train)); tot, nb = 0.0, 0
        t0 = time.time()
        for si in order:
            s = train[si]
            im = Image.open(s["image"]).convert("RGB")
            insts = s["instances"]
            if aug_cfg is not None:
                arr, insts = augment(np.array(im), insts, rng, aug_cfg)
                if not insts:          # crop lost every region; skip the sheet
                    continue
                im = Image.fromarray(arr)
            W, H = im.size
            idx = np.arange(len(insts))
            if len(idx) > args.max_boxes:
                idx = rng.choice(idx, args.max_boxes, replace=False)
            tgt = torch.from_numpy(targets_for(
                {"height": H, "width": W,
                 "instances": [insts[k] for k in idx]}, target=args.target)).to(dev)
            # gesture is chosen per SHEET, not per instance: one processor call
            # takes one prompt type, and padding a ragged mix buys nothing here
            use_pt = (args.prompt == "point" or
                      (args.prompt == "mixed" and rng.random() < 0.5))
            if use_pt:
                # counts are drawn once per sheet: one processor call takes a
                # rectangular points tensor, so every instance in this forward
                # must carry the same number of clicks
                n_pos = int(rng.integers(1, args.max_pos + 1))
                n_neg = int(rng.integers(0, args.max_neg + 1)) if args.max_neg else 0
                all_outer = (np.logical_or.reduce(
                    [render_instance_mask([insts[k]["outer_polygon"]], H, W).astype(bool)
                     for k in idx]) if args.max_neg else None)
                pts, labs = [], []
                for k in idx:
                    region = render_instance_mask(
                        [insts[k]["outer_polygon"]], H, W).astype(bool)
                    solid = render_instance_mask(
                        [insts[k]["outer_polygon"]] + insts[k].get("hole_polygons", []),
                        H, W).astype(bool)
                    p = interior_points(solid if solid.any() else region, n_pos, rng)
                    if not p:                      # degenerate: fall back to centre
                        x0, y0, x1, y1 = insts[k]["bbox_xyxy"]
                        p = [[int((x0 + x1) / 2), int((y0 + y1) / 2)]]
                    while len(p) < n_pos:          # keep the tensor rectangular
                        p.append(p[-1])
                    l = [1] * n_pos
                    if n_neg:
                        neg = negative_points(region, all_outer, n_neg, rng)
                        while len(neg) < n_neg:
                            neg.append(neg[-1] if neg else p[0])
                        l = l + [0] * n_neg
                        p = p + neg
                    pts.append(p); labs.append(l)
                inp = proc(images=im, input_points=[pts], input_labels=[labs],
                           return_tensors="pt").to(dev)
            else:
                boxes = [jitter(insts[k]["bbox_xyxy"], W, H, args.jitter, rng)
                         for k in idx]
                inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
            with torch.no_grad():
                emb = model.get_image_embeddings(inp["pixel_values"])
            out = model(image_embeddings=[e.detach() for e in emb],
                        input_points=inp.get("input_points"),
                        input_labels=inp.get("input_labels"),
                        input_boxes=inp.get("input_boxes"), multimask_output=False)
            logits = out.pred_masks[0, :, 0]                     # [N, 288, 288]
            loss = dice_bce(logits.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        model.eval()
        sc, line, e = val_report(f"epoch {ep:3d}")
        print(f"{line}  loss {tot/max(nb,1):.4f}  {time.time()-t0:.0f}s", flush=True)
        hist.append(dict(e, epoch=ep, loss=tot/max(nb,1)))
        if sc > best:
            best = sc
            torch.save({"mask_decoder": model.mask_decoder.state_dict(),
                        "epoch": ep, "poly_median": best, "target": args.target,
                        "poly_mean": e.get("poly_mean", best), "ge80": e["ge80"],
                        "aug": args.aug, "prompt": args.prompt,
                        "max_pos": args.max_pos, "max_neg": args.max_neg,
                        "seed": args.seed, "epochs": args.epochs,
                        "jitter": args.jitter, "lr": args.lr,
                        "base_model": "facebook/sam3",
                        "n_train_images": len(train),
                        "n_train_instances": sum(len(s["instances"]) for s in train)},
                       f"{args.out}/best.pth")
            print(f"   -> saved best ({args.target} score {best:.4f})")
        json.dump(hist, open(f"{args.out}/history.json", "w"), indent=1)
    print(f"\nbest val polygon IoU (median): {best:.4f}")


if __name__ == "__main__":
    main()
