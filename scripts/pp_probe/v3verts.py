"""PRODUCT_V3's vertex count after the 1 px offset (2026-11-11): stack ms3+cc_v3 is hf14fix4 1.24x (cellsnap counted before
the offset: 1.20x). Variants on the final polygons: simplify after the offset (shapely, preserve topology) or a larger eps."""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
from shapely.geometry import Polygon
import common
from stack import colour_cut, iou, CUR, nv
from curve import label_vertices
from polygonize import _fill_holes, fill_gaps, mask_to_polygons, rasterize, offset_polys, snap_to_cells, clean_ring
common.WITH_BOX[0] = True


def resimp(polys, tol):
    out = []
    for p in polys:
        g = Polygon(p["outer"], p["holes"]).simplify(tol, preserve_topology=True)
        for q in (getattr(g, "geoms", None) or [g]):
            if q.is_empty:
                continue
            ring = lambda r: np.round(clean_ring(np.round(np.asarray(r.coords)[:-1]).astype(int))).astype(int).tolist()
            o = ring(q.exterior)
            if len(o) >= 3:
                out.append({"outer": o, "holes": [x for x in (ring(h) for h in q.interiors) if len(x) >= 3]})
    return out


split, maps = sys.argv[1], Path(sys.argv[2]); common.MAPS = maps; ms = common.map_split(split)
V = ["eps6 off", "eps6 off s1", "eps6 off s1.5", "eps7 off", "eps8 off", "eps7 off s1"]
R = []
for job in common.jobs(split):
    idx, img, anyann, qs = common.load(job); h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        CUR[0] = (idx, ref); s = p4.shape[1] / w
        P = cv2.resize(p4, (w, h)).astype(np.float32) / 255 / 3
        for z in (5120, 3072):
            P += cv2.resize(np.load(maps / f"{ms}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"], (w, h)).astype(np.float32) / 255 / 3
        m = snap_to_cells(colour_cut(img, fill_gaps(_fill_holes(P > 0.35, 4096 / s / s), P), box), img, s)
        r = {"lv": label_vertices(job[1], cat)}
        for k in V:
            e = float(k.split()[0][3:]); pl = offset_polys(mask_to_polygons(m, 0.005, e), 1.0)
            if " s" in k:
                pl = resimp(pl, float(k.split(" s")[1]))
            r[k] = (iou(rasterize(pl, (h, w)), tgt), nv(pl))
        R.append(r)
L = [r for r in R if r["lv"] > 0]
for k in V:
    print(f"{split} {k:14s} IoU {np.mean([r[k][0] for r in R]):.4f}  vertices/label {np.median([r[k][1] / r['lv'] for r in L]):.2f}x")
