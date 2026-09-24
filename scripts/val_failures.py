#!/usr/bin/env python3
"""Score the validation failure targets for one or more metrics files.

Buckets every selection by target thickness at the 2048 input (thin < 74 px,
mid < 147 px, thick; cut points from the 2026-09-23 analysis of 129 real
selections), and reports the sheets the visual audit flagged: 13 (column on a
5:1 strip), 17 (thin foundation band), 10 and 9 (roof vs unlabelled walls).

    PYTHONPATH=. python3 scripts/val_failures.py base.json new.json
"""
import io, json, sys, collections
import numpy as np, cv2
from PIL import Image
from refmask2former import load_parquet_records
from refmask2former.dataset import render_instance_mask

FLAGGED = [13, 17, 10, 9]
R = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data", config="real-world-test", split="test")
thick = {}


def thickness(i, cat):
    if (i, cat) not in thick:
        W, H = Image.open(io.BytesIO(R[i]["image"])).size
        anns = R[i]["annotations"]; anns = json.loads(anns) if isinstance(anns, str) else anns
        t = np.logical_or.reduce([render_instance_mask(a["segmentation"], H, W).astype(bool)
                                  for a in anns if a.get("category_name") == cat])
        dt = cv2.distanceTransform(t.astype(np.uint8), cv2.DIST_L2, 5)
        thick[(i, cat)] = 2 * np.percentile(dt[t], 90) * 2048 / max(W, H)
    return thick[(i, cat)]


print(f"{'file':44s} {'mean':>6s} {'thin':>6s} {'mid':>6s} {'thick':>6s}  " +
      "  ".join(f"img{i:<3d}" for i in FLAGGED))
for path in sys.argv[1:]:
    d = json.load(open(path)); rows = d.get("selections", d.get("rows"))
    b = collections.defaultdict(list); im = collections.defaultdict(list)
    for r in rows:
        t = thickness(r["image_index"], r["category"])
        b["thin" if t < 73.8 else ("mid" if t < 147 else "thick")].append(r["iou"])
        im[r["image_index"]].append(r["iou"])
    print(f"{path.split('/')[-1][:44]:44s} {np.mean([r['iou'] for r in rows]):6.3f} "
          f"{np.mean(b['thin']):6.3f} {np.mean(b['mid']):6.3f} {np.mean(b['thick']):6.3f}  " +
          "  ".join(f"{np.mean(im[i]):6.3f}" for i in FLAGGED))
