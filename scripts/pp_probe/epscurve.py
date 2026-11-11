"""Per-question simplification curve on the product v3 mask (2026-11-11): IoU and vertices at Douglas-Peucker eps 4..20 (axis
snap + 1 px offset as in the product), plus label-free features a per-question eps could key on. Rows ->
scripts/pp_probe/rows_epscurve_{split}.json.   Usage: python3 scripts/pp_probe/epscurve.py SPLIT [MAPS_DIR]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common
from vertcount import label_vertices
import product
from polygonize import LOW_VERTEX, PRODUCT_V2, mask_to_polygons, offset_polys, rasterize

EPS = (4, 6, 8, 10, 12, 16, 20)

if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            m = product.product_mask(product.blend(mp, (h, w)), img, box)
            n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            per = float(sum(cv2.arcLength(c, True) for c in cs))
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat), "h": h, "w": w,
                 "area": int(m.sum()), "pieces": int((st[1:, 4] >= LOW_VERTEX["min_frac"] * st[1:, 4].sum()).sum()) if n > 1 else 0,
                 "perimeter": per, "rings": len(cs)}
            for e in EPS:
                polys = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], float(e)), PRODUCT_V2["offset"])
                r[f"iou{e}"] = common.masked_iou(rasterize(polys, (h, w)), tgt, (idx, ref))
                r[f"v{e}"] = product.nverts(polys)
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_epscurve_{split}.json", "w"))
    lab = np.array([r["label_v"] for r in rows], float)
    for e in EPS:
        print(split, f"eps {e:2d}: IoU {np.mean([r[f'iou{e}'] for r in rows]):.4f}  vertices/label median "
              f"{np.median(np.array([r[f'v{e}'] for r in rows]) / lab):.2f}x")
