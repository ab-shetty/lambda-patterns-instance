"""Pre-registered composition of the Gemini30 LOSO picks (morph.py): lin_fill -> gated opening (ellipse 15 inference px, drop only
pixels with probability < 0.7) -> marker watershed over black-hat stroke darkness in a 6 native px band -> polygons.
Rows: ref (lin_fill+poly), base (evaluator mask), open, ws, combo."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
from skimage.segmentation import watershed
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load
from methods import fill_holes, polygons, iou
T = 0.35

def gated_open(M, P, s, k=15, t=0.7):
    kk = max(3, int(round(k / s)) | 1)
    op = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))).astype(bool)
    return M & ~(~op & M & (P < t))

def ws_band(M, bh, b=6):
    din = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
    band = (din <= b) & (dout <= b); ys, xs = np.nonzero(band)
    if not len(ys): return M
    y1, y2, x1, x2 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
    mk = np.where(band, 0, np.where(M, 2, 1))[y1:y2, x1:x2].astype(np.int32)
    out = M.copy(); out[y1:y2, x1:x2] = watershed(bh[y1:y2, x1:x2], mk) == 2
    return out

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; rows = []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    for ref, cat, tgt, p in qs:
        s = p.shape[1] / w0; ha = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, ha)
        r = {"i": idx, "q": ref, "base": iou(cv2.resize((p > T * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool), tgt),
             "ref": iou(polygons(lf)[0], tgt)}
        o = gated_open(lf, P, s); r["open"] = iou(polygons(o)[0], tgt)
        r["ws"] = iou(polygons(fill_holes(ws_band(lf, bh), ha))[0], tgt)
        r["combo"] = iou(polygons(fill_holes(ws_band(o, bh), ha))[0], tgt)
        rows.append(r)
    return rows

if __name__ == "__main__":
    split = sys.argv[1]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
    json.dump(rows, open(f"data/diag/rows_combo_{split}.json", "w"))
    by = {}
    for r in rows: by.setdefault(r["i"], []).append(r)
    for k in ("base", "ref", "open", "ws", "combo"):
        d = np.array([r[k] - r["base"] for r in rows])
        sd = np.array([np.mean([r[k] - r["base"] for r in v]) for v in by.values()])
        print(f"{split} {k:6s} q-mean {np.mean([r[k] for r in rows]):.4f} ({d.mean():+.4f} vs evaluator; per-sheet {sd.mean():+.4f}, "
              f"sheets down >0.005: {(sd < -.005).sum()}, worst sheet {sd.min():+.3f})", flush=True)
