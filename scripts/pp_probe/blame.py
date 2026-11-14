"""Who made each visible error, the model or a post-processing step? (2026-11-14, after the cell-snap guard.) For every visible
miss / extra patch of the product polygons (human.py's definition), the share of its pixels on the "right" side after each
stage: P > T (model), hole/gap fill, colour cut, cell snap (guarded), polygons. A patch is blamed on the first stage after which
it is mostly (> 50%) wrong. Prints patch area (as a share of label area, summed) per stage. Usage: blame.py SPLIT [MAPS]"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product, human
from polygonize import T, _fill_holes, fill_gaps, snap_to_cells, rasterize
common.WITH_BOX[0] = True
STAGES = ("model", "fill", "colour", "cells", "polygons")

if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 else "data/floz-diag/gem2s_x")
    common.MAPS = maps
    blame = {k: {s: 0.0 for s in STAGES} for k in ("miss", "extra")}
    who = {k: {s: [] for s in STAGES} for k in ("miss", "extra")}
    lab = 0
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        s = product.REF_SCALE / max(h, w)
        for ref, cat, tgt, p4, box in qs:
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            P = product.blend(mp, (h, w))
            m0 = P > T
            m1 = fill_gaps(_fill_holes(m0, 4096 / s / s), P)
            m2 = product.colour_cut(img, m1, box)
            m3 = product.product_mask(P, img, box)
            pred = rasterize(product.product_polygons(mp, img, box), (h, w))
            ign = common.IGNORE.get((idx, ref))
            keep = ~ign if ign is not None else np.ones((h, w), bool)
            u8 = tgt.astype(np.uint8); k = np.ones((2 * human.TOL + 1,) * 2, np.uint8)
            keep &= ~(cv2.dilate(u8, k) != cv2.erode(u8, k))
            amin = (human.VIS * max(h, w) / human.SCREEN) ** 2
            T_ = max(1, int(tgt.sum())); lab += 1
            for kind, err, want in (("miss", tgt & ~pred & keep, True), ("extra", pred & ~tgt & keep, False)):
                e = cv2.morphologyEx(err.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                n, cc, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
                for i in range(1, n):
                    if st[i, 4] < amin:
                        continue
                    M = cc == i
                    for stage, m in zip(STAGES, (m0, m1, m2, m3, pred)):
                        if (m[M] != want).mean() > 0.5:
                            blame[kind][stage] += st[i, 4] / T_
                            who[kind][stage].append((idx, ref, round(st[i, 4] / T_, 4)))
                            break
    print(f"{split} {maps.name}: visible patch area per question (share of label, mean) by the stage that made it wrong")
    for kind in blame:
        print(f"  {kind:5s} " + "  ".join(f"{s} {blame[kind][s] / lab:.4f} ({len(who[kind][s])})" for s in STAGES))
        for s in STAGES[1:]:
            if who[kind][s]:
                print(f"        {s}: " + ", ".join(f"{a}q{b} {c}" for a, b, c in sorted(who[kind][s], key=lambda x: -x[2])[:6]))
