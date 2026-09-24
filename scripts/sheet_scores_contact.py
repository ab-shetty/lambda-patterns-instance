#!/usr/bin/env python3
"""Contact sheet of HF real-world-test sheets with labels overlaid, sorted by
score -- for LOOKING at where a model fails, one panel per sheet.

    python3 scripts/sheet_scores_contact.py --metrics data/evaluations/val_x.json --out sheet.jpg
"""
import argparse, io, json, collections
import numpy as np, cv2
from PIL import Image
from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tile", type=int, default=520)
a = ap.parse_args()
R = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
d = json.load(open(a.metrics)); rows = d.get("selections", d.get("rows"))
sc = collections.defaultdict(list)
for r in rows:
    sc[r["image_index"]].append(r["iou"])
T, cols = a.tile, 5
order = sorted(sc, key=lambda i: np.mean(sc[i]))
S = np.full((((len(order) + cols - 1) // cols) * (T + 40), cols * T, 3), 255, np.uint8)
C = [(255, 80, 80), (80, 200, 80), (80, 80, 255), (230, 180, 0), (200, 0, 200)]
for k, i in enumerate(order):
    rec = R[i]; im = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB")).copy()
    anns = rec["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
    H, W = im.shape[:2]; cats = sorted({x["category_name"] for x in anns}); ov = im.copy()
    for x in anns:
        ov[render_instance_mask(x["segmentation"], H, W).astype(bool)] = C[cats.index(x["category_name"]) % 5]
    im = (0.6 * im + 0.4 * ov).astype(np.uint8)
    f = (T - 6) / max(H, W); im = cv2.resize(im, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    y, x = (k // cols) * (T + 40), (k % cols) * T
    S[y + 40:y + 40 + im.shape[0], x + 3:x + 3 + im.shape[1]] = im
    cv2.putText(S, f"img{i}  {np.mean(sc[i]):.2f} (n={len(sc[i])})", (x + 6, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
Image.fromarray(S).save(a.out, quality=88)
print(a.out)
