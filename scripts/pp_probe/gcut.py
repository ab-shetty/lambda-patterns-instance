"""GrabCut seeded by the prediction, on top of lin_fill (+poly). Pixels more than b native px inside lin_fill's edge (and
probability > 0.5) are sure foreground, more than b px outside (and probability < 0.2) sure background, the band between
is probable fg/bg by lin_fill; GrabCut (colour GMMs + contrast-sensitive smoothness) relabels only the band.
Run on a crop around the band. Variants gc_b{b}_it{it}."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load
from methods import fill_holes, polygons, iou
T = 0.35

def gcut(img, M, P, b, it):
    din = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
    band = (din <= b) & (dout <= b)
    ys, xs = np.nonzero(band)
    if not len(ys): return M
    pad = 4 * b + 8; h0, w0 = M.shape
    y1, y2, x1, x2 = max(ys.min() - pad, 0), min(ys.max() + pad, h0), max(xs.min() - pad, 0), min(xs.max() + pad, w0)
    m = np.where(M, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
    m[band & M] = cv2.GC_PR_FGD; m[band & ~M] = cv2.GC_PR_BGD
    m[~band & M & (P < 0.5)] = cv2.GC_PR_FGD; m[~band & ~M & (P > 0.2)] = cv2.GC_PR_BGD
    mc = m[y1:y2, x1:x2].copy()
    if not ((mc == cv2.GC_FGD) | (mc == cv2.GC_PR_FGD)).any() or not ((mc == cv2.GC_BGD) | (mc == cv2.GC_PR_BGD)).any(): return M
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try: cv2.grabCut(np.ascontiguousarray(img[y1:y2, x1:x2]), mc, None, bg, fg, it, cv2.GC_INIT_WITH_MASK)
    except cv2.error: return M
    out = M.copy(); out[y1:y2, x1:x2] = (mc == cv2.GC_FGD) | (mc == cv2.GC_PR_FGD)
    return out

FEAT = [False]

def feat_img(img):
    """Texture image for GrabCut on line drawings: grey, hatch density (black-hat averaged over 15 px), local contrast."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    dens = cv2.blur(bh, (15, 15)); gf = g.astype(np.float32)
    sd = np.sqrt(np.maximum(cv2.blur(gf * gf, (15, 15)) - cv2.blur(gf, (15, 15)) ** 2, 0))
    n = lambda a: np.clip(a / max(np.percentile(a, 99.5), 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.dstack([g, n(dens), n(sd)])

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; rows = []
    if FEAT[0]: img = feat_img(img)
    for ref, cat, tgt, p in qs:
        s = p.shape[1] / w0
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, 4096 / s / s); r = {"i": idx, "q": ref, "ref": iou(polygons(lf)[0], tgt)}
        for b in (8, 20, 40):
            g = gcut(img, lf, P, b, 3); r[f"gc_b{b}"] = iou(polygons(fill_holes(g, 4096 / s / s))[0], tgt)
        rows.append(r)
    return rows

if __name__ == "__main__":
    if sys.argv[1] == "--feat": FEAT[0] = True; sys.argv.pop(1)
    split = sys.argv[1]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
    json.dump(rows, open(f"data/diag/rows_gcut{'feat' if FEAT[0] else ''}_{split}.json", "w"))
    b = np.array([r["ref"] for r in rows])
    for k in [k for k in rows[0] if k not in ("i", "q", "ref")]:
        a = np.array([r[k] for r in rows]); d = a - b
        print(f"{k:10s} {a.mean():.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f})", flush=True)
