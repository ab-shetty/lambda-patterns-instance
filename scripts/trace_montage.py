#!/usr/bin/env python3
"""Original | trace twin | twin labels, one row per sheet (facsimile_trace.py output).

    python3 scripts/trace_montage.py ORIG_DIR TRACE_DIR OUT.jpg [N] [SEED]
"""
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask  # noqa: E402

COLS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 160, 255), (200, 0, 200), (0, 200, 200), (120, 120, 0)]


def main(orig, trace, out, n=8, seed=0):
    anns = sorted(glob.glob(f"{trace}/annotations/trace_*.json"))
    random.Random(seed).shuffle(anns)
    rows = []
    for f in anns[:n]:
        stem = os.path.basename(f)[len("trace_"):-len(".json")]
        A = json.load(open(f))
        img = cv2.imread(f"{orig}/{stem}.png"); tw = cv2.imread(f"{trace}/images/trace_{stem}.png")
        h, w = img.shape[:2]
        lab = tw.copy()
        cats = sorted({a["category_name"] for a in A["annotations"]})
        for a in A["annotations"]:
            m = render_instance_mask(a["segmentation"], h, w).astype(bool)
            lab[m] = (0.45 * lab[m] + 0.55 * np.array(COLS[cats.index(a["category_name"]) % len(COLS)])).astype(np.uint8)
        W = 560
        row = np.hstack([cv2.resize(x, (W, int(h * W / w))) for x in (img, tw, lab)])
        cv2.putText(row, stem[-3:], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        rows.append(row)
    Wm = max(r.shape[1] for r in rows)
    cv2.imwrite(out, np.vstack([cv2.copyMakeBorder(r, 0, 6, 0, Wm - r.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
                                for r in rows]))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 8, int(a[4]) if len(a) > 4 else 0)
