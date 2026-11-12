"""Edges from the sheet's own appearance (2026-11-12): the product mask is right almost everywhere, so learn THIS sheet's
inside / outside colour models from its confident interior and exterior and let a graph cut (cv2.grabCut, which also prefers
cuts along strong image edges -- the drawn lines) re-decide only a band of +-r native px around the predicted edge.
Then the product polygons as usual. Rows: product vs grabcut r = 3 / 6 / 12 (IoU, vertices / label).
Usage: python3 scripts/pp_probe/grabcut.py SPLIT [MAPS_DIR] [--limit N]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2
import common, product
from vertcount import label_vertices
from polygonize import LOW_VERTEX, PRODUCT_V2, mask_to_polygons, offset_polys, rasterize

RS = tuple(int(x) for x in __import__("os").environ.get("GC_RS", "3,6,12").split(","))


def grabcut_band(img, m, r, pad=40, iters=2):
    """Re-decide pixels within r px of m's edge; everything else fixed (definite FG / BG). Runs on m's bounding box + pad."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m
    h, w = m.shape
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    mm = m[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner, outer = cv2.erode(mm, k), cv2.dilate(mm, k)
    gc = np.full(mm.shape, cv2.GC_BGD, np.uint8)
    gc[outer > 0] = cv2.GC_PR_BGD
    gc[mm > 0] = cv2.GC_PR_FGD
    gc[inner > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return m
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(img[y0:y1, x0:x1]), gc, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
    out = m.copy()
    out[y0:y1, x0:x1] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return out


def polys_of(m):
    return product.simplify_small_rings(offset_polys(mask_to_polygons(m, LOW_VERTEX["min_frac"], PRODUCT_V2["eps"]),
                                                     PRODUCT_V2["offset"]), max(1, int(m.sum())))


if __name__ == "__main__":
    split = sys.argv[1]
    maps = Path(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "data/floz-diag/gem2s_x")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9
    common.MAPS = maps
    rows, t0 = [], time.time()
    for job in common.jobs(split):
        idx, img, _, qs = common.load(job)
        h, w = img.shape[:2]
        for ref, cat, tgt, p4, box in qs:
            if len(rows) >= limit:
                break
            mp = {z: (p4 if z == 4096 else np.load(maps / f"{common.map_split(split)}_{z}" / "probs" / f"{idx:03d}_{ref:03d}.npz")["prob"])
                  for z in product.SCALES}
            m = product.product_mask(product.blend(mp, (h, w)), img, box)
            r = {"sheet": idx, "q": ref, "label_v": label_vertices(job[1], cat)}
            for name, mm in [("product", m)] + [(f"gc{k}", grabcut_band(img, m, k)) for k in RS]:
                p = polys_of(mm)
                r[name] = (common.masked_iou(rasterize(p, (h, w)), tgt, (idx, ref)), product.nverts(p))
            rows.append(r)
    json.dump(rows, open(f"scripts/pp_probe/rows_grabcut_{split}.json", "w"))
    lab = np.array([r["label_v"] for r in rows], float)
    b = np.array([r["product"][0] for r in rows])
    print(f"{split}: {len(rows)} q, {time.time() - t0:.0f} s")
    for name in ["product"] + [f"gc{k}" for k in RS]:
        a = np.array([r[name] for r in rows], float)
        print(f"  {name:8s} IoU {a[:, 0].mean():.4f} ({a[:, 0].mean() - b.mean():+.4f}, up {(a[:, 0] > b + 1e-3).sum()} / down "
              f"{(a[:, 0] < b - 1e-3).sum()})  vertices/label median {np.median(a[:, 1] / lab):.2f}x")
