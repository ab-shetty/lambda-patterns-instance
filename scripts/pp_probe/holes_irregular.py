"""Irregular holes (2026-11-14, the user on human_calib 0_0: the patio table cut out of the pattern "is the only major issue",
irregular errors are the annoying ones). Real holes in a pattern region are drawn objects -- windows, doors, columns, walls --
and are rectangles; a hole the shape of furniture, a fixture or text is the pattern running on underneath. Variant: a hole
ring under F of the question's mask area whose rectangularity (area / its minimum-area rectangle) is under R is filled
(dropped from the polygon). Prints IoU, change vs the product, holes filled, and up / down per question.
Usage: python3 scripts/pp_probe/holes_irregular.py SPLIT [MAPS]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import rasterize
common.WITH_BOX[0] = True

GRID = [(r, f) for r in (0.6, 0.7, 0.8, 0.9) for f in (0.02, 0.05, 0.2)]


def rectangularity(ring):
    a = np.asarray(ring, np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(a)
    return abs(cv2.contourArea(a)) / max(1e-6, w * h)


def fill_irregular(polys, total, r, f):
    out, n = [], 0
    for p in polys:
        keep = [h for h in p["holes"] if not (abs(cv2.contourArea(np.asarray(h, np.float32))) < f * total
                                              and rectangularity(h) < r)]
        n += len(p["holes"]) - len(keep)
        out.append({"outer": p["outer"], "holes": keep})
    return out, n


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
            polys = product.product_polygons(mp, img, box)
            total = max(1, int(product.product_mask(P, img, box).sum()))
            r = {"sheet": idx, "q": ref, "base": common.masked_iou(rasterize(polys, (h, w)), tgt, (idx, ref))}
            for R, F in GRID:
                q, n = fill_irregular(polys, total, R, F)
                r[f"{R}/{F}"] = (common.masked_iou(rasterize(q, (h, w)), tgt, (idx, ref)) if n else r["base"], n)
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_holes_irregular_{split}_{maps.name}.json", "w"))
    b = np.array([x["base"] for x in rows])
    print(f"{split} {maps.name}: {len(rows)} q, product {b.mean():.4f}")
    for R, F in GRID:
        a = np.array([x[f"{R}/{F}"][0] for x in rows]); n = sum(x[f"{R}/{F}"][1] for x in rows)
        print(f"  R<{R} F<{F:<4}  {a.mean():.4f} ({a.mean() - b.mean():+.4f})  holes filled {n:3d}  up {(a > b + 1e-3).sum()} / "
              f"down {(a < b - 1e-3).sum()}")
