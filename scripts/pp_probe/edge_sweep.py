"""Boundary offset and simplification sweep on production polygons: dilate/erode the mask by d px (native) before
polygonising; approxPolyDP eps; plus a rectilinear snap (minAreaRect-free: snap near-axis edges). IoU and vertices."""
import sys, json
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize
common.WITH_BOX[0] = True
split, maps = sys.argv[1], sys.argv[2]
common.MAPS = Path(maps)
D = (-2, -1, 0, 1, 2); EPS = (1.0, 2.0, 3.0, 5.0)
acc = {}; nv = {}; nt = []
for job in common.jobs(split):
    idx, img, _, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        P = cv2.resize(p4, (w, h), interpolation=cv2.INTER_LINEAR)
        cs, _ = cv2.findContours(tgt.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        nt.append(sum(len(cv2.approxPolyDP(c, 1.0, True)) for c in cs))
        for d in D:
            m = (P > int(0.35 * 255)).astype(np.uint8)
            if d: m = (cv2.dilate if d > 0 else cv2.erode)(m, np.ones((2 * abs(d) + 1,) * 2, np.uint8))
            for e in EPS if d == 0 else (1.0,):
                polys = probs_to_polygons(m * 255, (h, w), eps=e)
                r = rasterize(polys, (h, w)).astype(bool)
                acc.setdefault((d, e), []).append((r & tgt).sum() / max(1, (r | tgt).sum()))
                nv.setdefault((d, e), []).append(sum(len(p["outer"]) + sum(len(x) for x in p["holes"]) for p in polys))
for k in sorted(acc):
    print(split, "d %+d eps %.0f  iou %.4f  vertices/label median %.2f" % (k[0], k[1], np.mean(acc[k]),
          np.median(np.array(nv[k]) / np.maximum(1, nt))))
