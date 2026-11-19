"""Inference scale vs sheet size (2026-11-19): the small-sheet edge error is symmetric (short = over), so not an offset; small
sheets are upsampled 2-5x to the fixed SCALES. Product (all steps) from each scale alone and from pairs, by size group, from the
saved per-scale maps. Usage: scalesize.py SPLIT [MAPS]"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import common, product
from ens import polys_of_P
from polygonize import rasterize
common.WITH_BOX[0] = True
GROUPS = ((0, 2500), (2500, 4500), (4500, 99999))
SETS = [(5120, 4096, 3072), (3072,), (4096,), (5120,), (4096, 3072), (5120, 4096)]

if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/fgavg78")
    common.MAPS = maps
    rows = []
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            r = {"long": max(h, w)}
            for S in SETS:
                P = product.blend({z: mp[z] for z in S}, (h, w))
                r[S] = common.masked_iou(rasterize(polys_of_P(P, img, box), (h, w)), tgt, (idx, ref))
            rows.append(r)
    for lo, hi in GROUPS:
        g = [r for r in rows if lo <= r["long"] < hi]
        print(f"{split} {lo}-{hi} px ({len(g)} q): " + "  ".join(f"{'+'.join(map(str, S))}: {np.mean([r[S] for r in g]):.4f}" for S in SETS))
