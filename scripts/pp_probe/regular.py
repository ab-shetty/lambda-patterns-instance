"""Post-processing undoing regular drawn shapes (2026-11-14, pp_probe/blame.py): HF14 27 q2's door (P 0.05) is filled by
fill_gaps (its mean P sits at pmin 0.05; the swing lines make it look irregular); HF14 14 q4's panel under a window (a clean
rectangle, P 0.6) is dropped by drop_islands. Variants: fill_gaps pmin 0.1 / 0.15 / 0.2; drop_islands spares pieces with
rectangularity >= 0.9 / 0.8; both. Prints IoU vs the product (up / down). Usage: regular.py SPLIT [MAPS]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import T, _fill_holes, fill_gaps, snap_to_cells, mask_to_polygons, offset_polys, rasterize, LOW_VERTEX, PRODUCT_V2
from holes_irregular import rectangularity
common.WITH_BOX[0] = True


def mask(P, img, box, pmin):
    h, w = P.shape; s = product.REF_SCALE / max(h, w)
    pre = product.colour_cut(img, fill_gaps(_fill_holes(P > T, 4096 / s / s), P, pmin=pmin), box)
    snap = snap_to_cells(pre, img, s)
    g = pre & ~snap & (P >= product.SNAP_GUARD)
    return snap | g if g.sum() > 0.001 * max(1, snap.sum()) else snap


def polys(m, P, rect):
    total = max(1, int(m.sum()))
    ps = product.simplify_small_rings(offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]),
                                                   PRODUCT_V2["offset"]), total)
    if not rect:
        return product.drop_islands(ps, P, total)
    # drop_islands with rectangular pieces spared (the same rule otherwise; never every piece)
    keep = []
    for p in ps:
        if len(ps) > 1 and abs(cv2.contourArea(np.asarray(p["outer"], np.float32))) < product.ISLANDS["max_frac"] * total \
                and rectangularity(p["outer"]) < rect:
            m = rasterize([p], P.shape)
            if m.any() and P[m].mean() < product.ISLANDS["p"]:
                continue
        keep.append(p)
    return keep or ps


VARIANTS = {"pmin.1": (0.1, 0), "pmin.15": (0.15, 0), "pmin.2": (0.2, 0), "rect.9": (0.05, 0.9), "rect.8": (0.05, 0.8),
            "pmin.1+rect.9": (0.1, 0.9), "pmin.15+rect.9": (0.15, 0.9)}

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
            P = product.blend(mp, (h, w))
            M = {pm: mask(P, img, box, pm) for pm in {v[0] for v in VARIANTS.values()} | {0.05}}
            r = {"sheet": idx, "q": ref, "base": common.masked_iou(rasterize(polys(M[0.05], P, 0), (h, w)), tgt, (idx, ref))}
            for k, (pm, rect) in VARIANTS.items():
                r[k] = common.masked_iou(rasterize(polys(M[pm], P, rect), (h, w)), tgt, (idx, ref))
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_regular_{split}_{maps.name}.json", "w"))
    b = np.array([x["base"] for x in rows])
    print(f"{split} {maps.name}: {len(rows)} q, product {b.mean():.4f}")
    for k in VARIANTS:
        a = np.array([x[k] for x in rows])
        print(f"  {k:15s} {a.mean():.4f} ({a.mean() - b.mean():+.4f})  up {(a > b + 1e-3).sum()} / down {(a < b - 1e-3).sum()}"
              f"  biggest {max(a - b):+.3f} / {min(a - b):+.3f}")
