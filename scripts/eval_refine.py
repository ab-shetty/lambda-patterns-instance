#!/usr/bin/env python3
"""Two-pass inference: re-reference from the model's own confident region.

Motivation (2026-09-24, validation audit of swin_t mixr4): 63% of remaining
validation loss is image 17, and both of its failures come from a TINY user
rectangle -- on a thin stone band the model matches the upscaled blurry crop by
tone (selects the dark roof), and on a dormer gable it under-commits on the main
wall. Pass 1 is the normal prediction; pass 2 re-runs the model with a better
reference built from the confident pass-1 component that contains the user's
rectangle:

  bigbox  the largest square that fits inside that component (<= 512 native)
  mosaic  a 3x3 grid of tiles, each the size of the user's box, sampled across
          the component -- more texture evidence at the same scale, which is
          what a thin band needs (a bigger square cannot fit in it)

`--combine replace` uses pass 2 alone; `avg` averages the two probability
maps. Uses only the model and the user's rectangle -- no labels, no ensemble.
Pass 1 reproduces `evaluate_refunet_selection.py` exactly (same RNG, resize,
threshold).

    PYTHONPATH=. python3 scripts/eval_refine.py --checkpoint ck.pth \\
        --indices 4,5,6,8,9,10,13,15,17,19,20,21,22,26 --mode mosaic \\
        --combine avg --metrics-out val_refine.json
"""
import argparse
import io
import json
import random

import cv2
import numpy as np
import torch
from PIL import Image

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask, sample_reference_box
from scripts.evaluate_refunet_selection import HOLDOUT, _normalize_chw, load_refunet


def predict(model, image_tensor, crop, ref_size, device):
    crop = cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
    ref = _normalize_chw(crop).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(image_tensor, ref).sigmoid().float()[0, 0]


def second_reference(image0, prob_native, box, mode, conf, rng, grid=3):
    """Build the pass-2 reference crop, or None to keep pass 1."""
    x, y, w, h = box
    m = (prob_native > conf).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    ids = lab[y:y + h, x:x + w]
    ids = ids[ids > 0]
    if ids.size == 0:
        return None
    comp = (lab == np.bincount(ids).argmax()).astype(np.uint8)
    dt = cv2.distanceTransform(comp, cv2.DIST_C, 3)      # Chebyshev: half-side of fitting square
    if mode == "bigbox":
        r = int(min(dt.max(), 256))
        if 2 * r + 1 <= max(w, h) * 1.2:
            return None
        cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
        return image0[cy - r:cy + r + 1, cx - r:cx + r + 1]
    # mosaic: tiles the size of the user's box, centred where they fit
    s = max(w, h)
    ys, xs = np.where(dt >= s / 2)
    if len(ys) < grid * grid:
        return None
    tiles = []
    for k in rng.sample(range(len(ys)), grid * grid):
        cy, cx = ys[k], xs[k]
        t = image0[cy - s // 2:cy - s // 2 + s, cx - s // 2:cx - s // 2 + s]
        if t.shape[:2] != (s, s):
            return None
        tiles.append(t)
    rows = [np.hstack(tiles[i * grid:(i + 1) * grid]) for i in range(grid)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--indices", default=HOLDOUT)
    ap.add_argument("--image-max-size", type=int, default=2048)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--mode", choices=["bigbox", "mosaic"], default="mosaic")
    ap.add_argument("--combine", choices=["replace", "avg"], default="avg")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--metrics-out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, _ = load_refunet(a.checkpoint, device)
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    rows = []
    with torch.inference_mode():
        for image_idx in [int(i) for i in a.indices.split(",")]:
            rec = records[image_idx]
            image0 = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
            anns = rec["annotations"]
            anns = json.loads(anns) if isinstance(anns, str) else anns
            h0, w0 = image0.shape[:2]
            masks0 = [render_instance_mask(x["segmentation"], h0, w0).astype(bool) for x in anns]
            cats = [x.get("category_name", "pattern") for x in anns]
            scale = a.image_max_size / max(h0, w0)
            nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
            image = cv2.resize(image0, (nw, nh), interpolation=cv2.INTER_LINEAR)
            image_tensor = _normalize_chw(image).unsqueeze(0).to(device)
            for ref_idx, (ref_mask, cat) in enumerate(zip(masks0, cats)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, w, h = sample_reference_box(ref_mask, 128, 512, rng=rng)
                crop = image0[y:y + h, x:x + w]
                if not crop.size:
                    continue
                p1 = predict(model, image_tensor, crop, a.ref_size, device)
                p1n = cv2.resize(p1.cpu().numpy(), (w0, h0), interpolation=cv2.INTER_LINEAR)
                crop2 = second_reference(image0, p1n, (x, y, w, h), a.mode, a.conf,
                                         random.Random(image_idx * 7 + ref_idx))
                used2 = crop2 is not None and crop2.size > 0
                if used2:
                    p2 = predict(model, image_tensor, crop2, a.ref_size, device)
                    p = p2 if a.combine == "replace" else (p1 + p2) / 2
                else:
                    p = p1
                pred = cv2.resize((p > a.mask_thresh).cpu().numpy().astype(np.uint8), (w0, h0),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                tgt = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
                rows.append({"image_index": image_idx, "reference_instance": ref_idx,
                             "category": cat, "reference_box_native": [x, y, w, h],
                             "used_pass2": bool(used2),
                             "iou": int((pred & tgt).sum()) / max(int((pred | tgt).sum()), 1)})
    out = {"checkpoint": a.checkpoint, "mode": a.mode, "combine": a.combine, "conf": a.conf,
           "mean_iou": float(np.mean([r["iou"] for r in rows])),
           "pass2_used": int(sum(r["used_pass2"] for r in rows)), "selections": rows}
    json.dump(out, open(a.metrics_out, "w"))
    print(f"{a.mode}/{a.combine}: mean {out['mean_iou']:.4f}  pass2 used on "
          f"{out['pass2_used']}/{len(rows)}")


if __name__ == "__main__":
    main()
