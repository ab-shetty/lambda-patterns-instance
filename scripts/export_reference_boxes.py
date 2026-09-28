#!/usr/bin/env python3
"""Export the real-world-test images with the EXACT reference boxes the
evaluator draws (`evaluate_refunet_selection.py`: every labelled instance is a
question, `sample_reference_box(mask, 128, 512)` with the same per-question
seed), so a person can judge whether they look like boxes a user would draw.

Writes, under --out:
  overview/<split>_img<NN>.jpg   the sheet, every question's box numbered and
                                 coloured by family, family masks tinted
  questions/<split>_img<NN>_q<RR>.jpg
                                 the box in context (a window around it) next to
                                 the 224x224 crop the model is actually given
  boxes.csv                      one row per question: split, image, question,
                                 family, native box, box side at 2048 input, the
                                 instance's size, the image size

    python3 scripts/export_reference_boxes.py --out ~/ref_box_audit
"""
import argparse
import csv
import io
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_parquet_records                                # noqa: E402
from refmask2former.dataset import render_instance_mask, sample_reference_box  # noqa: E402

HF14 = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]
VAL = [4, 5, 6, 8, 9, 10, 13, 15, 17, 19, 20, 21, 22, 26]
COLOURS = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
           (166, 86, 40), (247, 129, 191), (0, 170, 170)]


def label(img, text, org, scale, colour):
    th = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--input-size", type=int, default=2048)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(f"{out}/overview", exist_ok=True)
    os.makedirs(f"{out}/questions", exist_ok=True)
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir=a.cache_dir,
                                config="real-world-test", split="test")
    rows = []
    for idx in range(len(recs)):
        split = "hf14" if idx in HF14 else "val" if idx in VAL else "other"
        rec = recs[idx]
        img = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h0, w0 = img.shape[:2]
        masks = [render_instance_mask(x["segmentation"], h0, w0).astype(bool) for x in anns]
        cats = [x.get("category_name", "pattern") for x in anns]
        fams = sorted(set(cats))
        scale_in = a.input_size / max(h0, w0)
        over = img.copy()
        tint = over.copy()
        for m, c in zip(masks, cats):
            tint[m] = COLOURS[fams.index(c) % len(COLOURS)]
        over = cv2.addWeighted(over, 0.75, tint, 0.25, 0)
        lw = max(2, int(max(h0, w0) / 600))
        fs = max(0.6, max(h0, w0) / 1800)
        boxes = []
        for q, (m, c) in enumerate(zip(masks, cats)):
            rng = random.Random(idx * 1_000_003 + q * 65_537 + 12_345)
            x, y, w, h = sample_reference_box(m, 128, 512, rng=rng)
            boxes.append((x, y, w, h))
            col = COLOURS[fams.index(c) % len(COLOURS)]
            cv2.rectangle(over, (x, y), (x + w, y + h), (0, 0, 0), lw + 2)
            cv2.rectangle(over, (x, y), (x + w, y + h), col, lw)
            label(over, str(q), (x, max(int(20 * fs), y - 4)), fs, col)
            # question panel: context window around the box + the model's crop
            pad = max(3 * max(w, h), 300)
            cx0, cy0 = max(0, x - pad), max(0, y - pad)
            cx1, cy1 = min(w0, x + w + pad), min(h0, y + h + pad)
            ctx = img[cy0:cy1, cx0:cx1].copy()
            cm = np.zeros(ctx.shape[:2], np.uint8)
            cm[m[cy0:cy1, cx0:cx1]] = 1
            edge = cv2.morphologyEx(cm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            ctx[edge] = col
            cv2.rectangle(ctx, (x - cx0, y - cy0), (x - cx0 + w, y - cy0 + h), (255, 0, 255), max(2, lw))
            H = 448
            ctx_r = cv2.resize(ctx, (max(1, int(ctx.shape[1] * H / ctx.shape[0])), H),
                               interpolation=cv2.INTER_AREA)
            crop = cv2.resize(img[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_LINEAR)
            crop = cv2.resize(crop, (H, H), interpolation=cv2.INTER_NEAREST)
            gap = np.full((H, 12, 3), 255, np.uint8)
            panel = np.concatenate([ctx_r, gap, crop], 1)
            head = np.full((56, panel.shape[1], 3), 255, np.uint8)
            side = w * scale_in
            label(head, f"{split} img{idx} q{q} {c}  box {w}px native = {side:.0f}px at {a.input_size} "
                        f"input  (left: context, magenta box; right: model's 224 crop)", (8, 36), 0.7, (0, 0, 0))
            cv2.imwrite(f"{out}/questions/{split}_img{idx:02d}_q{q:02d}.jpg",
                        cv2.cvtColor(np.concatenate([head, panel], 0), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            ys, xs = np.nonzero(m)
            rows.append({"split": split, "image": idx, "question": q, "family": c,
                         "box_x": x, "box_y": y, "box_w": w, "box_h": h,
                         f"box_side_at_{a.input_size}": round(side, 1),
                         "instance_w": int(xs.max() - xs.min() + 1) if len(xs) else 0,
                         "instance_h": int(ys.max() - ys.min() + 1) if len(ys) else 0,
                         "instance_area": int(m.sum()), "image_w": w0, "image_h": h0})
        s = min(1.0, 3000 / max(h0, w0))
        over = cv2.resize(over, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/overview/{split}_img{idx:02d}.jpg", cv2.cvtColor(over, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(f"{out}/boxes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(recs)} images, {len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
