"""Ink split on top of lin_fill (+poly): non-ink pixels keep the prediction; ink pixels within b native px of the mask's edge take
the decision of their nearest non-ink pixel, so a boundary stroke is cut along its centre line (labels split boundary strokes
~50/50; predictions take 58-73%). Variants split{ink}_b{b}; ws_b6 (marker watershed, morph.py) and open_k15_t0.7 composed after."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map, ink_map_v2
from methods import fill_holes, polygons, iou
T = 0.35

def nearest_nonink(ink):
    h0, w0 = ink.shape
    _, lbl = cv2.distanceTransformWithLabels(ink.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~ink); im = np.zeros(len(ys) + 1, np.int64); im[1:] = ys * w0 + xs
    return im[lbl]

def split(M, ink, near, b):
    din = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5); dout = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
    sel = ink & (din <= b) & (dout <= b)
    out = M.copy(); out[sel] = M.ravel()[near[sel]]
    return out

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; rows = []
    inks = {"v1": ink_map(img), "v2": ink_map_v2(img)}; nears = {k: nearest_nonink(v) for k, v in inks.items()}
    for ref, cat, tgt, p in qs:
        s = p.shape[1] / w0
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, 4096 / s / s); r = {"i": idx, "q": ref, "ref": iou(polygons(lf)[0], tgt)}
        for k in inks:
            for b in (2, 4, 8):
                r[f"split{k}_b{b}"] = iou(polygons(fill_holes(split(lf, inks[k], nears[k], b), 4096 / s / s))[0], tgt)
        rows.append(r)
    print(idx, flush=True, file=sys.stderr)
    return rows

if __name__ == "__main__":
    split_ = sys.argv[1]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as pool: rows = [r for rs in pool.imap(sheet, jobs(split_)) for r in rs]
    json.dump(rows, open(f"data/diag/rows_inksplit_{split_}.json", "w"))
    b = np.array([r["ref"] for r in rows])
    for k in [k for k in rows[0] if k not in ("i", "q", "ref")]:
        a = np.array([r[k] for r in rows]); d = a - b
        print(f"{k:12s} {a.mean():.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f})", flush=True)
