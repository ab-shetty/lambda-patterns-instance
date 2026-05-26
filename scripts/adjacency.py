#!/usr/bin/env python3
"""Do the same-category real instances physically touch, or are they separate?
Measures pairwise gaps between instances of each category in img17, and renders
each category's instances alone so we can see what they are."""
import io, json
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from refmask2former.dataset import render_instance_mask, load_parquet_records

OUT = "/home/ubuntu/lambda-patterns-instance/diagnostics/adjacency.png"


def load(rec):
    image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
    anns = rec["annotations"]
    if isinstance(anns, str):
        anns = json.loads(anns)
    return image, anns


real = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                            config="real-world-test", split="test")
image, anns = load(real[17])
h, w = image.shape[:2]

masks = [render_instance_mask(a["segmentation"], h, w) for a in anns]
cats = [a.get("category_name") for a in anns]

for target in ("pattern1", "pattern2"):
    idxs = [i for i, c in enumerate(cats) if c == target]
    print(f"\n=== {target}: {len(idxs)} instances ===")
    # pairwise minimum gap (px) between instance masks via distance transform
    touch_pairs, gaps = 0, []
    for a_pos, i in enumerate(idxs):
        mi = masks[i] > 0
        # distance from instance i to nearest other-pixel
        inv = (~mi).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        for j in idxs[a_pos + 1:]:
            mj = masks[j] > 0
            min_gap = dist[mj].min() if mj.any() else np.inf
            gaps.append(min_gap)
            if min_gap <= 3:
                touch_pairs += 1
    gaps = np.array(gaps)
    sizes = [(int((masks[i] > 0).sum()),
              int(np.ptp(np.where(masks[i] > 0)[1])),   # bbox w
              int(np.ptp(np.where(masks[i] > 0)[0]))) for i in idxs]  # bbox h
    ws = [s[1] for s in sizes]; hs = [s[2] for s in sizes]
    print(f"  bbox w: median {int(np.median(ws))}  h: median {int(np.median(hs))}")
    print(f"  pairwise gaps (px): min {gaps.min():.0f}  median {np.median(gaps):.0f}  "
          f"that touch (<=3px): {touch_pairs}/{len(gaps)} pairs")
    # how many instances have >=1 same-cat neighbor within 3px
    has_touch = 0
    for a_pos, i in enumerate(idxs):
        mi = masks[i] > 0
        dist = cv2.distanceTransform((~mi).astype(np.uint8), cv2.DIST_L2, 5)
        touching = any((masks[j] > 0).any() and dist[masks[j] > 0].min() <= 3
                       for j in idxs if j != i)
        has_touch += touching
    print(f"  instances touching >=1 same-category neighbor: {has_touch}/{len(idxs)}")

# Render each category alone
s = min(1.0, 2400 / w)
disp = cv2.resize(image, (int(w * s), int(h * s)))
H, W = disp.shape[:2]
fig, axes = plt.subplots(2, 1, figsize=(16, 11))
for ax, target in zip(axes, ("pattern1", "pattern2")):
    o = disp.copy()
    idxs = [i for i, c in enumerate(cats) if c == target]
    for i in idxs:
        m = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
        o[m > 0] = (0.5 * o[m > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnts, -1, (0, 0, 255), 2)
    ax.imshow(o); ax.set_title(f"{target}: {len(idxs)} instances (red fill)", fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT, dpi=90, bbox_inches="tight")
print("\nsaved", OUT)
