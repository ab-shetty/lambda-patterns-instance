#!/usr/bin/env python3
"""Score ensembles of RefUNet models offline from saved probability maps.

Each model is first run through `evaluate_refunet_selection.py --save-probs DIR` (same
indices, same --image-max-size). This averages the saved maps with the given weights,
thresholds, and scores the reference-selection IoU exactly as the evaluator does
(prediction resized to native resolution, union of the reference's category).

    python3 scripts/ensemble_selection.py --probs A_dir B_dir --weights 0.5,0.5 \
        --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 --mask-thresh 0.35 [--sweep]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask

VAL = "4,5,6,8,9,10,13,15,17,19,20,21,22,26"
HF14 = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"


def targets_for(rec):
    anns = rec["annotations"]
    anns = json.loads(anns) if isinstance(anns, str) else anns
    h0, w0 = int(rec["height"]), int(rec["width"])
    masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool) for a in anns]
    cats = [a.get("category_name", "pattern") for a in anns]
    return h0, w0, masks, cats


def score(probs, weights, indices, records, thresh):
    rows = []
    for i in indices:
        h0, w0, masks, cats = targets_for(records[i])
        union = {c: np.logical_or.reduce([m for m, cc in zip(masks, cats) if cc == c]) for c in set(cats)}
        for q, c in enumerate(cats):
            maps = [np.load(Path(d) / f"{i:03d}_{q:03d}.npz")["prob"].astype(np.float32) / 255 for d in probs]
            p = sum(w * m for w, m in zip(weights, maps)) / sum(weights)
            pred = cv2.resize((p > thresh).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
            t = union[c]
            rows.append({"image_index": i, "reference_instance": q,
                         "iou": int((pred & t).sum()) / max(int((pred | t).sum()), 1)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--indices", default=HF14)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--sweep", action="store_true", help="also report weights 0..1 in 0.1 steps (2 models)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                   config="real-world-test", split="test")
    idx = [int(v) for v in a.indices.split(",") if v.strip()]
    w = [float(v) for v in a.weights.split(",")] if a.weights else [1.0] * len(a.probs)
    rows = score(a.probs, w, idx, records, a.mask_thresh)
    print(json.dumps({"weights": w, "mask_thresh": a.mask_thresh, "n": len(rows),
                      "mean_iou": float(np.mean([r["iou"] for r in rows]))}))
    if a.sweep and len(a.probs) == 2:
        for k in range(11):
            r = score(a.probs, [1 - k / 10, k / 10], idx, records, a.mask_thresh)
            print(f"  w2={k / 10:.1f}  {np.mean([x['iou'] for x in r]):.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"weights": w, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
