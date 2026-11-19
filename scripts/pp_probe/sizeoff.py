"""Edge rule by sheet size (2026-11-19): the real-sheet gap sits on sheets under 2,500 px (fgavg78: val 0.936 vs 0.966 on
larger sheets, almost all of it within 2 px of the label's edge). Product polygons with the outward offset at 0 / 0.5 / 1 (the
product) / 1.5 / 2 native px, and the within-2-px error split into falling short (label not predicted) vs over, by size group.
Usage: sizeoff.py SPLIT [MAPS]"""
import sys, json
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from polygonize import mask_to_polygons, offset_polys, rasterize, LOW_VERTEX, PRODUCT_V2
common.WITH_BOX[0] = True
OFFS = (0.0, 0.5, 1.0, 1.5, 2.0)
GROUPS = ((0, 2500), (2500, 4500), (4500, 99999))

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
            P = product.blend(mp, (h, w)); m = product.product_mask(P, img, box); total = max(1, int(m.sum()))
            base = mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"])
            r = {"sheet": idx, "q": ref, "long": max(h, w)}
            for o in OFFS:
                ps = product.drop_islands(product.simplify_small_rings(offset_polys(base, o) if o else base, total), P, total)
                pred = rasterize(ps, (h, w)); r[str(o)] = common.masked_iou(pred, tgt, (idx, ref))
                if o == 1.0:
                    u8 = tgt.astype(np.uint8); k = np.ones((5, 5), np.uint8)
                    band = cv2.dilate(u8, k).astype(bool) & ~cv2.erode(u8, k).astype(bool)
                    r["short"] = int((tgt & ~pred & band).sum()) / max(1, int(tgt.sum()))
                    r["over"] = int((pred & ~tgt & band).sum()) / max(1, int(tgt.sum()))
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_sizeoff_{split}_{maps.name}.json", "w"))
    for lo, hi in GROUPS:
        g = [r for r in rows if lo <= r["long"] < hi]
        if not g:
            continue
        print(f"{split} {lo}-{hi} px ({len(g)} q): " + "  ".join(f"off {o}: {np.mean([r[str(o)] for r in g]):.4f}" for o in OFFS)
              + f"  | within 2 px @off 1: short {np.mean([r['short'] for r in g]):.4f}, over {np.mean([r['over'] for r in g]):.4f}")
    print(f"{split} all ({len(rows)} q): " + "  ".join(f"off {o}: {np.mean([r[str(o)] for r in rows]):.4f}" for o in OFFS))
