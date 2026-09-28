#!/usr/bin/env python3
"""Is a failing real question limited by what the user's box SHOWS, or by what
the model can tell apart? Re-ask each failing question with larger boxes drawn
inside the SAME instance the evaluator's box came from, everything else as
`evaluate_refunet_selection.py` (resize, threshold, native-resolution IoU, the
box plane for --roi-ref / anchored models):

  protocol   the evaluator's box (reproduces its IoU)
  big512     the largest square inside the instance, capped at the sampler's
             512 px maximum -- still a box a user could draw
  biggest    the largest square inside the instance, uncapped

If big boxes fix a question, the limit is reference information (fixable at
inference or with UI guidance). If not, the model cannot separate the materials.

    PYTHONPATH=. python3 scripts/box_size_probe.py --checkpoint ck.pth \\
        --metrics data/evaluations/x_val4096.json data/evaluations/x_hf14.json \\
        --max-iou 0.6 --image-max-size 4096 --device cpu --out probe.json
"""
import argparse
import io
import json
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_parquet_records                                          # noqa: E402
from refmask2former.dataset import _normalize_chw, render_instance_mask, sample_reference_box  # noqa: E402
from scripts.evaluate_refunet_selection import load_refunet                             # noqa: E402


def largest_square(mask, cap=None):
    dt = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_C, 3)[1:-1, 1:-1]
    half = int(dt.max()) - 1
    if cap:
        half = min(half, (cap - 1) // 2)
    if half < 0:
        return None
    cy, cx = np.argwhere(dt >= half + 1)[len(np.argwhere(dt >= half + 1)) // 2]
    return int(cx - half), int(cy - half), 2 * half + 1, 2 * half + 1


def elongate(mask, box, max_ratio=4):
    """Grow the square box along one axis (whichever reaches further) while it
    stays inside the instance, up to max_ratio x its side -- the long thin
    rectangle a user draws on a band. Returns the ROI box; the crop stays the
    protocol square."""
    x, y, w, h = box
    best = box
    for axis in (0, 1):
        lo, hi = (x, x + w) if axis == 0 else (y, y + h)
        limit = max_ratio * (w if axis == 0 else h)
        while hi - lo < limit:
            grown = False
            for d in (-1, 1):
                nlo, nhi = (lo - 1, hi) if d < 0 else (lo, hi + 1)
                if nlo < 0 or nhi > (mask.shape[1] if axis == 0 else mask.shape[0]):
                    continue
                strip = (mask[y:y + h, nlo:nlo + 1] if d < 0 else mask[y:y + h, nhi - 1:nhi]) if axis == 0 \
                    else (mask[nlo:nlo + 1, x:x + w] if d < 0 else mask[nhi - 1:nhi, x:x + w])
                if strip.all():
                    lo, hi, grown = nlo, nhi, True
            if not grown:
                break
        cand = (lo, y, hi - lo, h) if axis == 0 else (x, lo, w, hi - lo)
        if cand[2] * cand[3] > best[2] * best[3]:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="evaluator metrics json(s) for this checkpoint; failing questions are read from them")
    ap.add_argument("--max-iou", type=float, default=0.6)
    ap.add_argument("--image-max-size", type=int, default=4096)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rect-only", action="store_true",
                    help="only protocol vs rect4 (protocol crop, ROI box elongated along the region)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.set_num_threads(48)
    model, _ = load_refunet(a.checkpoint, device)
    model.float()
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                config="real-world-test", split="test")
    todo = {}
    for f in a.metrics:
        for r in json.load(open(f))["selections"]:
            if r["iou"] < a.max_iou:
                todo.setdefault(r["image_index"], []).append((r["reference_instance"], r["iou"]))
    rows = []
    with torch.inference_mode():
        for idx, qs in sorted(todo.items()):
            rec = recs[idx]
            image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
            anns = rec["annotations"]
            anns = json.loads(anns) if isinstance(anns, str) else anns
            h0, w0 = image0.shape[:2]
            masks0 = [render_instance_mask(x["segmentation"], h0, w0).astype(bool) for x in anns]
            cats = [x.get("category_name", "pattern") for x in anns]
            scale = a.image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            img_t = _normalize_chw(cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR))[None].to(device)
            feats = model.features(img_t)
            for q, logged in qs:
                rng = random.Random(idx * 1_000_003 + q * 65_537 + 12_345)
                pbox = sample_reference_box(masks0[q], 128, 512, rng=rng)
                boxes = {"protocol": pbox, "rect4": elongate(masks0[q], pbox)} if a.rect_only else {
                         "protocol": pbox,
                         "big512": largest_square(masks0[q], 512),
                         "biggest": largest_square(masks0[q])}
                target = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cats[q]])
                row = {"image": idx, "question": q, "category": cats[q], "logged_iou": logged}
                for name, box in boxes.items():
                    if box is None:
                        continue
                    x, y, w, h = box
                    cx, cy, cw, ch = pbox if name == "rect4" else box   # rect4: square crop, long ROI
                    crop = cv2.resize(image0[cy:cy + ch, cx:cx + cw], (224, 224), interpolation=cv2.INTER_LINEAR)
                    ref = _normalize_chw(crop)[None].to(device)
                    rf = model.features(ref)
                    protos = [f.mean((-2, -1)) for f in rf]
                    if getattr(model, "roi_ref", False):
                        bn = np.zeros((h0, w0), np.uint8)
                        bn[y:y + h, x:x + w] = 1
                        bm = torch.from_numpy(cv2.resize(bn, (nw, nh), interpolation=cv2.INTER_NEAREST)).float()[None, None]
                        protos = model.roi_prototypes(feats, bm.to(device), protos)
                    logits = model.decode(feats, protos, rf)
                    prob = torch.nn.functional.interpolate(logits, size=(nh, nw), mode="bilinear",
                                                           align_corners=False).sigmoid()[0, 0]
                    pred = cv2.resize((prob > a.mask_thresh).numpy().astype(np.uint8), (w0, h0),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                    iou = (pred & target).sum() / max((pred | target).sum(), 1)
                    row[name] = float(iou)
                    row[name + "_side_at_2048"] = round(max(w, h) * 2048 / max(h0, w0), 1)
                    row[name + "_pred_over_target"] = round(float(pred.sum() / max(target.sum(), 1)), 2)
                rows.append(row)
                print(f"img{idx:2d} q{q:2d} {cats[q]:9s} logged {logged:.3f} | " + " | ".join(
                    f"{n} {row.get(n, float('nan')):.3f} ({row.get(n + '_side_at_2048', 0):.0f}px, "
                    f"x{row.get(n + '_pred_over_target', 0)})" for n in boxes), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for n in ("protocol", "big512", "biggest", "rect4"):
        v = [r[n] for r in rows if n in r]
        print(f"{n:9s} mean {np.mean(v):.3f} over {len(v)} failing questions")


if __name__ == "__main__":
    main()
