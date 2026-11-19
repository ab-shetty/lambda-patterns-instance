"""Probability-map ensembles of different models through the current product (2026-11-19). Each model's three scales are
blended as in product.py, then the models' blended maps are averaged and run through product_mask + polygons. Different from
blend_ckpts (weight averaging): costs one model run per member at deployment. Usage:
    python3 scripts/pp_probe/ens.py SPLIT NAME=MAPS[,MAPS...] [NAME=...]    (first MAPS of the first entry = the baseline)"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import mask_to_polygons, offset_polys, rasterize, LOW_VERTEX, PRODUCT_V2
common.WITH_BOX[0] = True


def polys_of_P(P, img, box):
    m = product.product_mask(P, img, box); total = max(1, int(m.sum()))
    ps = product.simplify_small_rings(offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]),
                                                   PRODUCT_V2["offset"]), total)
    return product.drop_islands(ps, P, total)


if __name__ == "__main__":
    split = sys.argv[1]
    E = {a.split("=")[0]: [Path(x) for x in a.split("=")[1].split(",")] for a in sys.argv[2:]}
    common.MAPS = next(iter(E.values()))[0]
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, _, box in qs:
            cache = {}
            def Pm(mdir):
                if mdir not in cache:
                    cache[mdir] = product.blend({z: np.load(mdir / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"]
                                                 for z in product.SCALES}, (h, w))
                return cache[mdir]
            r = {"sheet": idx, "q": ref}
            for name, ms in E.items():
                P = sum(Pm(m) for m in ms) / len(ms)
                r[name] = common.masked_iou(rasterize(polys_of_P(P, img, box), (h, w)), tgt, (idx, ref))
            rows.append(r)
    names = list(E)
    json.dump(rows, open(f"scripts/pp_probe/rows_ens_{split}.json", "w"))
    b = np.array([r[names[0]] for r in rows])
    for n in names:
        a = np.array([r[n] for r in rows])
        print(f"{split} {n:22s} {a.mean():.4f} ({a.mean() - b.mean():+.4f})  up {(a > b + 1e-3).sum()} / down {(a < b - 1e-3).sum()}")
