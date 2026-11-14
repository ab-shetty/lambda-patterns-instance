"""Cell snap deletes confident regions (2026-11-14, human_calib 11_0: a hatched triangle with P ~ 1 removed because its cell
leaks through a gap into a large unlabelled area, so the cell's coverage is <= lo). Variant: snap_to_cells may not remove
pixels whose blended probability is >= PH (it still adds whole cells and joins strokes). Prints IoU vs the product per PH.
Usage: python3 scripts/pp_probe/snapguard.py SPLIT [MAPS]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import T, _fill_holes, fill_gaps, snap_to_cells, mask_to_polygons, offset_polys, rasterize, LOW_VERTEX, PRODUCT_V2
common.WITH_BOX[0] = True
PHS = (0.8, 0.9, 0.95)


def polys_of(m, P):
    total = max(1, int(m.sum()))
    polys = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]), PRODUCT_V2["offset"])
    return product.drop_islands(product.simplify_small_rings(polys, total), P, total)


if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        s = product.REF_SCALE / max(h, w)
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            P = product.blend(mp, (h, w))
            pre = product.colour_cut(img, fill_gaps(_fill_holes(P > T, 4096 / s / s), P), box)
            snap = snap_to_cells(pre, img, s)
            r = {"sheet": idx, "q": ref}
            r["base"] = common.masked_iou(rasterize(polys_of(snap, P), (h, w)), tgt, (idx, ref))
            for ph in PHS:
                guard = pre & ~snap & (P >= ph)
                r[str(ph)] = (common.masked_iou(rasterize(polys_of(snap | guard, P), (h, w)), tgt, (idx, ref))
                              if guard.sum() > 0.001 * max(1, snap.sum()) else r["base"])
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_snapguard_{split}_{maps.name}.json", "w"))
    b = np.array([x["base"] for x in rows])
    print(f"{split} {maps.name}: {len(rows)} q, product {b.mean():.4f}")
    for ph in PHS:
        a = np.array([x[str(ph)] for x in rows])
        print(f"  guard P>={ph}: {a.mean():.4f} ({a.mean() - b.mean():+.4f})  up {(a > b + 1e-3).sum()} / down {(a < b - 1e-3).sum()}"
              f"  biggest up {max(a - b):+.3f}  biggest down {min(a - b):+.3f}")
