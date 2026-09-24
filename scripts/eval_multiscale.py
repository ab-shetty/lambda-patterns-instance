#!/usr/bin/env python3
"""One model, several inference sizes, probability maps averaged at native size.

Inference size is the largest single lever on the real-data swin_t model
(validation 2048 .785 -> 4096 .801, 2026-09-24), and different sheets prefer
different sizes, so averaging the sigmoid maps across sizes may beat any one.
Same reference crop, RNG, threshold and scoring as
`evaluate_refunet_selection.py` (a single size reproduces it exactly).

    PYTHONPATH=. python3 scripts/eval_multiscale.py --checkpoint ck.pth \\
        --sizes 2048,4096 --indices 4,5,6,8,9,10,13,15,17,19,20,21,22,26 \\
        --metrics-out val_ms.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sizes", default="2048,4096")
    ap.add_argument("--weights", default="", help="per-size weights, default equal")
    ap.add_argument("--indices", default=HOLDOUT)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--metrics-out", required=True)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(sizes)
    w = [x / sum(w) for x in w]
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
            tensors = []
            for S in sizes:
                sc = S / max(h0, w0)
                im = cv2.resize(image0, (max(1, round(w0 * sc)), max(1, round(h0 * sc))),
                                interpolation=cv2.INTER_LINEAR)
                tensors.append(_normalize_chw(im).unsqueeze(0).to(device))
            for ref_idx, (ref_mask, cat) in enumerate(zip(masks0, cats)):
                rng = random.Random(image_idx * 1_000_003 + ref_idx * 65_537 + 12_345)
                x, y, bw, bh = sample_reference_box(ref_mask, 128, 512, rng=rng)
                crop = image0[y:y + bh, x:x + bw]
                if not crop.size:
                    continue
                ref = _normalize_chw(cv2.resize(crop, (a.ref_size, a.ref_size),
                                                interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(device)
                acc = np.zeros((h0, w0), np.float32)
                for t, wt in zip(tensors, w):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = model(t, ref).sigmoid().float()[0, 0].cpu().numpy()
                    acc += wt * cv2.resize(p, (w0, h0), interpolation=cv2.INTER_LINEAR)
                pred = acc > a.mask_thresh
                tgt = np.logical_or.reduce([m for m, c in zip(masks0, cats) if c == cat])
                rows.append({"image_index": image_idx, "reference_instance": ref_idx, "category": cat,
                             "iou": int((pred & tgt).sum()) / max(int((pred | tgt).sum()), 1)})
    out = {"checkpoint": a.checkpoint, "sizes": sizes, "weights": w,
           "mean_iou": float(np.mean([r["iou"] for r in rows])), "selections": rows}
    json.dump(out, open(a.metrics_out, "w"))
    print(f"sizes {sizes} weights {[round(x, 2) for x in w]}: mean {out['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
