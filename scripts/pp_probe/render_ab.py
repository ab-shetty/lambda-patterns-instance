"""Before / after renders of a post-processing variant for questions the user judges (2026-11-14). Whole sheet at fit-to-screen,
label-only red, prediction-only blue. Usage: render_ab.py SPLIT MAPS VARIANT SHEET:Q [SHEET:Q ...] -> data/diag/ab_<variant>_*.jpg"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product, regular
from polygonize import rasterize
common.WITH_BOX[0] = True
V = {"rect.9": ((0.1, 0), (0.1, 0.9)), "pmin.1": ((0.05, 0), (0.1, 0))}

split, maps, var = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
want = {tuple(int(x) for x in a.split(":")) for a in sys.argv[4:]}
common.MAPS = maps
for job in common.jobs(split):
    if job[0] not in {a for a, _ in want}:
        continue
    idx, img, _, qs = common.load(job)
    h, w = img.shape[:2]
    for ref, cat, tgt, p4, box in qs:
        if (idx, ref) not in want:
            continue
        mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
              for z in product.SCALES}
        P = product.blend(mp, (h, w)); ign = common.IGNORE.get((idx, ref)); tiles = []
        for name, (pm, rect) in zip(("before", "after"), V[var]):
            pred = rasterize(regular.polys(regular.mask(P, img, box, pm), P, rect), (h, w))
            if ign is not None:
                pred &= ~ign
            v = img.copy(); v[pred] = (0.6 * v[pred] + [0, 100, 0]).astype(np.uint8)
            v[tgt & ~pred] = (0, 0, 255); v[pred & ~tgt] = (255, 0, 0)
            x, y, bw, bh = box; cv2.rectangle(v, (x, y), (x + bw, y + bh), (0, 200, 255), max(2, int(max(h, w) / 800)))
            s = 1400 / max(h, w); v = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.putText(v, f"{split} {idx} q{ref} {name}  IoU {common.masked_iou(pred, tgt, (idx, ref)):.3f}", (10, 35), 0, 0.9, (0, 0, 0), 2)
            tiles.append(v)
        cv2.imwrite(f"data/diag/ab_{var}_{split}_{idx}_{ref}.jpg", np.vstack(tiles) if w > h else np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
