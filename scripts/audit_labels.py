"""Label-consistency audit for generated pools (2026-09-29, after the window-hole bug).

Checks every annotation JSON + image in a pool dir:
  overlap   pixels labelled as two different families on one sheet
  bounds    label polygons outside the image
  invalid   self-intersecting / empty / hole-outside-outer polygons
  sliver    labels thinner than 4 px (vanish at training resolution)
  plain     labels whose interior has almost no ink (a blank region called a material)
Usage: python3 scripts/audit_labels.py DIR [DIR ...]
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former.dataset import render_instance_mask  # noqa: E402


def audit(d, show=5):
    C = Counter()
    ex = defaultdict(list)
    fs = sorted(glob.glob(os.path.join(d, "annotations", "*.json")))
    for f in fs:
        a = json.load(open(f))
        W, H = a["image"]["width"], a["image"]["height"]
        mode = a.get("mode", "?")
        img = cv2.imread(os.path.join(d, "images", a["image"]["file_name"]), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 60, 160) > 0
        fam_mask = {}
        for x in a["annotations"]:
            seg = x["segmentation"]
            C["labels"] += 1
            try:
                outer = Polygon(list(zip(seg[0][0::2], seg[0][1::2])))
                holes = [Polygon(list(zip(h[0::2], h[1::2]))) for h in seg[1:]]
                bad = (not outer.is_valid) or any(not h.is_valid for h in holes) or \
                    any(not outer.buffer(1).contains(h) for h in holes)
            except Exception:
                bad = True
            if bad:
                C[f"invalid/{mode}"] += 1
                ex["invalid"].append((os.path.basename(f), x["family"]))
            xs, ys = seg[0][0::2], seg[0][1::2]
            if min(xs) < -1 or min(ys) < -1 or max(xs) > W + 1 or max(ys) > H + 1:
                C[f"bounds/{mode}"] += 1
                ex["bounds"].append((os.path.basename(f), x["family"]))
            m = render_instance_mask(seg, H, W).astype(bool)
            if m.sum() == 0:
                C[f"empty/{mode}"] += 1
                continue
            # thickness: max of distance transform
            dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
            if dt.max() < 2.0:
                C[f"sliver/{mode}"] += 1
                ex["sliver"].append((os.path.basename(f), x["family"], int(m.sum())))
            core = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if core.sum() > 2000:
                ink = edges[core].mean()
                if ink < 0.002:
                    C[f"plain/{mode}/{x['family']}"] += 1
                    ex["plain"].append((os.path.basename(f), x["family"], round(float(ink), 4)))
            fam_mask[x["family"]] = fam_mask.get(x["family"], np.zeros((H, W), bool)) | m
        fams = list(fam_mask)
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                ov = (fam_mask[fams[i]] & fam_mask[fams[j]]).sum()
                small = min(fam_mask[fams[i]].sum(), fam_mask[fams[j]].sum())
                if ov > 50 and ov > 0.002 * small:
                    C[f"overlap/{mode}/{fams[i]}+{fams[j]}"] += 1
                    ex["overlap"].append((os.path.basename(f), fams[i], fams[j], round(ov / small, 3)))
    print(f"== {d}: {len(fs)} sheets, {C['labels']} labels")
    for k, v in sorted(C.items()):
        if k != "labels":
            print(f"   {k}: {v}")
    for k, v in ex.items():
        print(f"   e.g. {k}: {v[:show]}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        audit(d)
