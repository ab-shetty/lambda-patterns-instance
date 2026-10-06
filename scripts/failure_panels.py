#!/usr/bin/env python3
"""Contact sheet of the worst questions for a model, from evaluator outputs.

Needs the metrics JSON (for IoU and the reference box) and the --save-probs directory of
the same run. Each panel: prediction filled red, target outline green, reference box blue,
title "sheet/question IoU pred/target size".

    python3 scripts/failure_panels.py --metrics M.json --probs DIR --indices 12,16,... \
        --worst 16 --out panels.jpg [--other M2.json]   # --other adds that model's IoU
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--worst", type=int, default=16)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--other", default="")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-pool", default="", help="score a local pool (as evaluate_refunet_selection.py "
                    "--local-pool); --indices all = every sheet")
    ap.add_argument("--drop", default="", help="JSON with 'dropped': [[image_index, reference_instance], ...] "
                    "(e.g. unrealistic boxes) to leave out")
    a = ap.parse_args()
    rows = json.load(open(a.metrics))["selections"]
    if a.indices != "all":
        idx = {int(v) for v in a.indices.split(",")}
        rows = [r for r in rows if r["image_index"] in idx]
    if a.drop:
        dropped = {tuple(k) for k in json.load(open(a.drop))["dropped"]}
        rows = [r for r in rows if (r["image_index"], r["reference_instance"]) not in dropped]
    other = {}
    if a.other:
        other = {(r["image_index"], r["reference_instance"]): r["iou"]
                 for r in json.load(open(a.other))["selections"]}
    rows.sort(key=lambda r: r["iou"])
    if a.local_pool:
        pool = Path(a.local_pool)
        records = []
        for f in sorted((pool / "annotations").glob("*.json")):
            ann = json.loads(f.read_text())
            records.append({"image": (pool / "images" / ann["image"]["file_name"]).read_bytes(),
                            "annotations": ann["annotations"]})
    else:
        records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                       config="real-world-test", split="test")
    panels = []
    for r in rows[:a.worst]:
        i, q = r["image_index"], r["reference_instance"]
        rec = records[i]
        img = cv2.imdecode(np.frombuffer(rec["image"], np.uint8), cv2.IMREAD_COLOR)
        h0, w0 = img.shape[:2]
        anns = rec["annotations"]
        anns = json.loads(anns) if isinstance(anns, str) else anns
        cat = anns[q].get("category_name")
        tgt = np.logical_or.reduce([render_instance_mask(x["segmentation"], h0, w0).astype(bool)
                                    for x in anns if x.get("category_name") == cat])
        p = np.load(Path(a.probs) / f"{i:03d}_{q:03d}.npz")["prob"].astype(np.float32) / 255
        pred = cv2.resize((p > a.mask_thresh).astype(np.uint8), (w0, h0),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        s = a.width / w0
        ov = img.copy()
        ov[pred] = (0.45 * ov[pred] + 0.55 * np.array([40, 40, 230])).astype(np.uint8)
        cs, _ = cv2.findContours(tgt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (0, 170, 0), max(2, int(3 / s)))
        x, y, w, h = r["reference_box_native"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 80, 0), max(3, int(4 / s)))
        ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = f"{i}/q{q:02d} IoU {r['iou']:.2f}"
        if other:
            t += f" (other {other.get((i, q), float('nan')):.2f})"
        t += f"  pred/target {r['prediction_pixels'] / max(r['target_pixels'], 1):.1f}x"
        cv2.rectangle(ov, (0, 0), (len(t) * 11 + 10, 26), (255, 255, 255), -1)
        cv2.putText(ov, t, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        panels.append(ov)
    left, right = panels[0::2], panels[1::2]
    col = lambda c: np.vstack([np.vstack([x, np.full((8, a.width, 3), 255, np.uint8)]) for x in c]) \
        if c else np.full((1, a.width, 3), 255, np.uint8)
    L, R = col(left), col(right)
    H = max(L.shape[0], R.shape[0])
    pad = lambda x: np.vstack([x, np.full((H - x.shape[0], a.width, 3), 255, np.uint8)])
    cv2.imwrite(a.out, np.hstack([pad(L), np.full((H, 8, 3), 0, np.uint8), pad(R)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(panels)} panels -> {a.out}")


if __name__ == "__main__":
    main()
