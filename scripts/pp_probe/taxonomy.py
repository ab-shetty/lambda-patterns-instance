"""Working backwards: classify every error piece of the default polygon mask and measure what fixing each class would buy.
Error pieces = connected components of FP (mask \\ label) and FN (label \\ mask). Per piece: area, width (2 x max inner distance),
texture density (ink v2 averaged over 15 px, inside the piece) vs the reference box's density.
  FP_blank   FP piece with texture density < 0.4 x the reference's (an unhatched band / blank cell next to the region)
  FP_tex     other FP pieces (textured: wrong family or look-alike)
  FN_clutter FN piece with density > 1.5 x the reference's (hatch under text / symbols / dashes)
  FN_blank   FN piece with density < 0.4 x the reference's
  FN_tex     other FN pieces
  thin       any piece narrower than 4 px (boundary slivers), FP or FN
Gain of a class = IoU with every piece of that class fixed (set to the label) minus the default."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts')
import common
from common import jobs, load, ink_map_v2
from polygonize import probs_to_polygons, rasterize
common.WITH_BOX[0] = True
CL = ["thin", "FP_blank", "FP_tex", "FN_clutter", "FN_blank", "FN_tex"]

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]
    dens = cv2.blur(ink_map_v2(img).astype(np.float32), (15, 15))
    rows = []
    for ref, cat, G, p, box in qs:
        M = rasterize(probs_to_polygons(p, (h0, w0)), (h0, w0))
        x, y, w, h = [int(round(v)) for v in box]
        rd = float(dens[max(y, 0):y + h, max(x, 0):x + w].mean()) + 1e-3
        lab_fix = {c: np.zeros_like(G) for c in CL}
        for kind, E in (("FP", M & ~G), ("FN", G & ~M)):
            n, lab, st, _ = cv2.connectedComponentsWithStats(E.astype(np.uint8), connectivity=8)
            if n <= 1: continue
            dt = cv2.distanceTransform(E.astype(np.uint8), cv2.DIST_L2, 3)
            width = 2 * np.array([0] + [0] * (n - 1), float)
            mx = np.zeros(n); np.maximum.at(mx, lab.ravel(), dt.ravel()); width = 2 * mx
            dm = np.bincount(lab.ravel(), weights=dens.ravel(), minlength=n) / np.maximum(st[:, 4], 1)
            for j in range(1, n):
                if width[j] < 4: c = "thin"
                elif dm[j] < 0.4 * rd: c = kind + "_blank"
                elif kind == "FN" and dm[j] > 1.5 * rd: c = "FN_clutter"
                else: c = kind + "_tex"
                lab_fix[c] |= lab == j
        I, U = (M & G).sum(), (M | G).sum()
        r = {"i": idx, "q": ref, "iou": I / U, "refdens": rd}
        for c in CL:
            F = lab_fix[c]   # fixing: FP pieces removed, FN pieces added
            Mf = (M & ~F) if c.startswith("FP") else (M | F) if c.startswith("FN") else np.where(F, G, M)
            r[c] = float((Mf & G).sum() / (Mf | G).sum()) - r["iou"]
        rows.append(r)
    return rows

if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_taxonomy_{split}.json", "w"))
        print(f"== {split}: n={len(rows)} default {np.mean([r['iou'] for r in rows]):.4f}; gain if every piece of a class were fixed:")
        for c in CL:
            v = np.array([r[c] for r in rows]); print(f"  {c:11s} {v.mean():+.4f}  (questions gaining > 0.01: {(v > .01).sum()})")
