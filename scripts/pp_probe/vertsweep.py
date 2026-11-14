"""Vertex fixes on the product v3 polygons (2026-11-14, after the fgab A/B): label-free variants scored per question.
Variants: eps (6 = v3) x small-ring k / max_frac x island filter (drop a piece -- outer ring with its holes -- under
`isl_frac` of the mask whose mean blended probability is under `isl_p`). Picks are made on valfix4 (IoU) and Gemini30
(vertices); HF14 is a check. Usage: python3 scripts/pp_probe/vertsweep.py SPLIT MAPS [--procs 24] -> rows JSON + table"""
import sys, json, argparse
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from vertcount import label_vertices
from polygonize import LOW_VERTEX, PRODUCT_V2, mask_to_polygons, offset_polys, rasterize
common.WITH_BOX[0] = True

import os
VARIANTS = {   # name: (eps, k, max_frac, isl_frac, isl_p)
    "v3":               (6, 0.1, 0.05, 0, 0),
    "eps7":             (7, 0.1, 0.05, 0, 0),
    "eps8":             (8, 0.1, 0.05, 0, 0),
    "sr.2/.10":         (6, 0.2, 0.10, 0, 0),
    "isl2%<.5":         (6, 0.1, 0.05, 0.02, 0.5),
    "isl2%<.6":         (6, 0.1, 0.05, 0.02, 0.6),
    "isl5%<.6":         (6, 0.1, 0.05, 0.05, 0.6),
    "isl5%<.7":         (6, 0.1, 0.05, 0.05, 0.7),
    "isl2%<.6+sr.2":    (6, 0.2, 0.10, 0.02, 0.6),
    "eps7+isl2%<.6":    (7, 0.1, 0.05, 0.02, 0.6),
    "eps7+isl2%<.6+sr.2": (7, 0.2, 0.10, 0.02, 0.6),
    "eps7+isl5%<.6":    (7, 0.1, 0.05, 0.05, 0.6),
    "eps8+isl5%<.6":    (8, 0.1, 0.05, 0.05, 0.6),
}

if os.environ.get("VARIANTS"):
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in os.environ["VARIANTS"].split(",")}


def drop_islands(polys, P, total, frac, pmin):
    if not frac:
        return polys
    keep = []
    for p in polys:
        a = abs(cv2.contourArea(np.asarray(p["outer"], np.float32)))
        if a < frac * total and len(polys) > 1:
            m = rasterize([p], P.shape)
            if m.any() and P[m].mean() < pmin:
                continue
        keep.append(p)
    return keep or polys


def run_sheet(args):
    split, maps, job = args
    common.MAPS = Path(maps)
    idx, img, _, qs = common.load(job)
    out = []
    for ref, cat, tgt, p4, box in qs:
        mp = {z: (p4 if z == 4096 else np.load(Path(maps) / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
              for z in product.SCALES}
        P = product.blend(mp, img.shape[:2])
        m = product.product_mask(P, img, box)
        total = max(1, int(m.sum()))
        row = {"sheet": idx, "q": ref, "label": label_vertices(job[1], cat)}
        for name, (eps, k, mf, isf, isp) in VARIANTS.items():
            polys = offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], eps), PRODUCT_V2["offset"])
            polys = product.simplify_small_rings(polys, total, k=k, max_frac=mf)
            polys = drop_islands(polys, P, total, isf, isp)
            row[name] = (common.masked_iou(rasterize(polys, img.shape[:2]), tgt, (idx, ref)), product.nverts(polys))
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("maps"); ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    common.MAPS = Path(a.maps)
    jobs = common.jobs(a.split)
    with Pool(min(a.procs, len(jobs))) as pool:
        rows = [r for rs in pool.map(run_sheet, [(a.split, a.maps, j) for j in jobs], chunksize=1) for r in rs]
    Path("data/diag/vertsweep").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(f"data/diag/vertsweep/{Path(a.maps).name}_{a.split}.json", "w"))
    lab = np.array([r["label"] for r in rows]); ok = lab > 0
    for name in VARIANTS:
        iou = np.array([r[name][0] for r in rows]); v = np.array([r[name][1] for r in rows])
        rr = v[ok] / lab[ok]
        print(f"{Path(a.maps).name:14s} {a.split:9s} {name:20s} mIoU {iou.mean():.4f}  median {np.median(rr):.2f}x  "
              f"<=1.2x {100 * np.mean(rr <= 1.2):3.0f}%  total {v[ok].sum() / lab[ok].sum():.2f}x")
