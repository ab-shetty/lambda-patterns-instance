#!/usr/bin/env python3
"""Illustrate instance granularity: real GT splits adjacent same-material
regions per unit; synthetic merges them. Each instance drawn a distinct color
with its category_name, so 'same material but separate instances' is visible."""
import io, json, colorsys
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from refmask2former.dataset import render_instance_mask, load_parquet_records

OUT = "/home/ubuntu/lambda-patterns-instance/diagnostics/granularity.png"


def load(rec):
    image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
    anns = rec["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    return image, anns


def distinct_colors(n):
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 1.0))
            for i in range(n)]


def render(image, anns, max_w=2400):
    h, w = image.shape[:2]
    s = min(1.0, max_w / w)
    img = cv2.resize(image, (int(w * s), int(h * s)))
    H, W = img.shape[:2]
    over = img.copy()
    cols = distinct_colors(len(anns))
    for k, a in enumerate(anns):
        m = render_instance_mask(a["segmentation"], h, w)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        col = np.array(cols[k], np.uint8)
        over[m > 0] = (0.5 * over[m > 0] + 0.5 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 0, 0), 2)
        ys, xs = np.where(m > 0)
        if len(ys):
            cx, cy = int(xs.mean()), int(ys.mean())
            name = a.get("category_name", "?")
            cv2.putText(over, f"{k}:{name}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(over, f"{k}:{name}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return over


real = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                            config="real-world-test", split="test")
synth = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data")

rimg, ranns = load(real[17])
# choose a synthetic elevation with several instances for contrast
si, sanns = None, None
for i in range(len(synth)):
    img, anns = load(synth[i])
    if 4 <= len(anns) <= 8:
        si, sanns = img, anns
        si_idx = i
        break

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
axes[0].imshow(render(rimg, ranns))
axes[0].set_title(f"REAL img17: {len(ranns)} GT instances — note same material (e.g. siding) "
                  f"is SPLIT into separate per-unit instances", fontsize=11)
axes[1].imshow(render(si, sanns))
axes[1].set_title(f"SYNTH img{si_idx}: {len(sanns)} instances — same-material adjacent "
                  f"regions are MERGED into one instance", fontsize=11)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT, dpi=90, bbox_inches="tight")
print("saved", OUT)

# Print real category breakdown so we can confirm same-category neighbors.
from collections import Counter
print("\nREAL img17 category_name counts:", dict(Counter(a.get("category_name") for a in ranns)))
print("REAL img17 reference_tile values:", set(a.get("reference_tile", "n/a") for a in ranns) if "reference_tile" in (ranns[0] if ranns else {}) else "no reference_tile field")
