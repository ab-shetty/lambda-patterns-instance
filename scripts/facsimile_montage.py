#!/usr/bin/env python3
"""Side-by-side montage of facsimile pairs (original | twin), named on each row.

    python3 scripts/facsimile_montage.py ORIG_DIR TWIN_POOL OUT.jpg [N] [SEED]
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def fit(f, H=300, W=560):
    im = cv2.imread(str(f)); h, w = im.shape[:2]; s = min(W / w, H / h)
    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(im, 2, H + 2 - im.shape[0], 2, W + 2 - im.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))


def main(orig, pool, out, n=8, seed=3):
    ms = json.load(open(Path(pool, "spec_matches.json")))
    random.Random(seed).shuffle(ms)
    rows = []
    for m in ms:
        t = Path(pool, "images", f"synth6_{m['image_id']}.png")
        if not t.exists():
            continue
        row = np.hstack([fit(Path(orig, m["name"] + ".png")), fit(t)])
        cv2.putText(row, m["name"][-3:], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        rows.append(row)
        if len(rows) == n:
            break
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 8, int(a[4]) if len(a) > 4 else 3)
