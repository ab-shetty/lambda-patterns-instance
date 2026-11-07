"""Polygon simplification + straight-edge snapping (2026-11-07): drawings are mostly axis-aligned with sloped roofs, and a
person draws a wall as 4 corners, not 30. Per polygon ring:
  1. approxPolyDP(eps)
  2. snap: an edge within `ang` degrees of horizontal / vertical gets both endpoints set to its mean y / x (a corner between
     a horizontal and a vertical edge takes y from one and x from the other: an exact right angle)
  3. drop vertices that are now collinear (< 1 px off the line through their neighbours) or duplicated
Rows: IoU of the rasterised result and vertices / label vertices, on the 4096 map. Production (polygonize.py defaults since
2026-11-07): eps 2 + snap 6.
Usage: python3 scripts/pp_probe/snap.py SPLIT MAPS_DIR"""
import sys
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from pathlib import Path
import common
from polygonize import probs_to_polygons, rasterize, snap_ring, clean_ring
common.WITH_BOX[0] = True


def simplify(polys, ang=None):
    out = []
    for poly in polys:
        rings = [poly["outer"]] + poly["holes"]
        if ang is not None:
            rings = [clean_ring(snap_ring(r, ang)) for r in rings]
        rings = [np.round(np.asarray(r)).astype(int).tolist() for r in rings]
        if len(rings[0]) >= 3:
            out.append({"outer": rings[0], "holes": [r for r in rings[1:] if len(r) >= 3]})
    return out


def nverts(polys):
    return sum(len(p["outer"]) + sum(len(h) for h in p["holes"]) for p in polys)


if __name__ == "__main__":
    split, maps = sys.argv[1], sys.argv[2]
    common.MAPS = Path(maps)
    CFG = [(1.0, None), (2.0, None), (3.0, None), (1.0, 6.0), (2.0, 6.0), (3.0, 6.0), (2.0, 10.0)]
    iou, nv, nt = {c: [] for c in CFG}, {c: [] for c in CFG}, []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            cs, _ = cv2.findContours(tgt.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            nt.append(max(1, sum(len(cv2.approxPolyDP(c, 1.0, True)) for c in cs)))
            for eps, ang in CFG:
                polys = simplify(probs_to_polygons(p4, (h, w), eps=eps, snap_deg=None), ang)
                m = rasterize(polys, (h, w)).astype(bool)
                iou[(eps, ang)].append((m & tgt).sum() / max(1, (m | tgt).sum()))
                nv[(eps, ang)].append(nverts(polys))
    for c in CFG:
        r = np.array(nv[c]) / np.array(nt)
        print(split, "eps %.0f snap %-4s iou %.4f  vertices/label median %.2f mean %.2f" %
              (c[0], "-" if c[1] is None else "%.0f" % c[1], np.mean(iou[c]), np.median(r), r.mean()))
