"""Mask -> polygon post-processing candidates, scored as the frontend sees them (IoU of the rasterised output).
Per question, from the saved prob map (inference res, uint8):
  base       evaluator: threshold 0.35 at inference res, nearest upsample
  lin        linear upsample of the probability, threshold
  lin_fill   + fill holes up to 4096 inference px
  gf_r_e     guided filter (guide = native grey image) on the linear-upsampled probability, radius r native px, eps e
  cell_t     voting over ink-bounded cells: connected components of non-ink pixels; a cell (smaller than 2% of the sheet) is in
             if its mean probability > t, larger cells keep the per-pixel decision; ink pixels follow their nearest cell pixel
  *+poly     polygon step on a mask: drop pieces < 0.2% of the mask, trace contours with holes, simplify (Douglas-Peucker 1 px),
             rasterise back. Vertex counts reported."""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map
T = 0.35

def fill_holes(b, max_area):
    inv = (~b).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = b.shape; ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        ok[i] = not (x == 0 or y == 0 or x + ww == w or y + hh == h) and a <= max_area
    return b | ok[lab]

def guided(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.boxFilter(I, -1, k), cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)

def polygons(m, eps=1.0, min_frac=0.002):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool); keep[1:] = st[1:, 4] >= min_frac * st[1:, 4].sum(); m = keep[lab]
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8); nv = 0
    if hier is None: return out.astype(bool), 0
    polys = [cv2.approxPolyDP(c, eps, True) for c in cs]
    for j, (c, hh) in enumerate(zip(polys, hier[0])):
        if hh[3] == -1 and len(c) >= 3: cv2.fillPoly(out, [c], 1); nv += len(c)
    for j, (c, hh) in enumerate(zip(polys, hier[0])):
        if hh[3] != -1 and len(c) >= 3: cv2.fillPoly(out, [c], 0); nv += len(c)
    return out.astype(bool), nv

def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)

def sheet(job):
    idx, img, anyann, qs = load(job)
    h0, w0 = img.shape[:2]
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    ink = ink_map(img)
    ncell, cells, cst, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    big = cst[:, 4] > 0.02 * h0 * w0
    _, near = cv2.distanceTransformWithLabels(ink.astype(np.uint8) * 255 ^ 255, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL) \
        if False else (None, None)
    # nearest non-ink pixel for every ink pixel: distanceTransformWithLabels labels zero pixels; zeros = non-ink
    _, lbl = cv2.distanceTransformWithLabels(ink.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~ink); idxmap = np.zeros(len(ys) + 1, np.int64); idxmap[1:] = ys * w0 + xs
    nearest = idxmap[lbl]                      # flat index of nearest non-ink pixel
    rows = []
    for ref, cat, tgt, p in qs:
        r = {"i": idx, "ref": ref}
        s = p.shape[1] / w0
        r["base"] = iou(cv2.resize((p > T * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool), tgt)
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lin = P > T; r["lin"] = iou(lin, tgt)
        lf = fill_holes(lin, 4096 / s / s); r["lin_fill"] = iou(lf, tgt)
        masks = {"lin_fill": lf}
        q = guided(I, P, 4, 1e-2) > T; r["gf4"] = iou(q, tgt)
        mean = np.bincount(cells.ravel(), weights=P.ravel(), minlength=ncell) / np.maximum(cst[:, 4], 1)
        for t in (0.4, 0.5, 0.6):
            sel = mean > t
            cm = np.where(big[cells], lin, sel[cells])
            cm = np.where(ink, cm.ravel()[nearest].reshape(h0, w0), cm)
            r[f"cell{t}"] = iou(cm, tgt)
            if t == 0.5:
                masks["cell0.5"] = cm
                dl = cv2.distanceTransform((~lf).astype(np.uint8), cv2.DIST_L2, 5) + cv2.distanceTransform(lf.astype(np.uint8), cv2.DIST_L2, 5)
                for bw in (4, 8, 16):
                    hy = np.where(dl <= bw, cm, lf); k = f"hyb{bw}"; r[k] = iou(hy, tgt)
                    if bw == 8: masks[k] = hy
                hf = fill_holes(cm, 4096 / s / s); r["cell0.5_fill"] = iou(hf, tgt); masks["cell0.5_fill"] = hf
        for k, m in masks.items():
            pm, nv = polygons(m); r[k + "+poly"] = iou(pm, tgt); r[k + "+poly_nv"] = nv
        rows.append(r)
    return rows

if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: rows = [r for rs in pool.map(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"{sys.argv[0].rsplit('/', 1)[0]}/rows_{split}.json", "w"))
        b = np.array([r["base"] for r in rows])
        def ps(k):
            by = {}
            for r in rows: by.setdefault(r["i"], []).append(r[k])
            return np.mean([np.mean(v) for v in by.values()])
        print(f"== {split}: n={len(rows)} base {b.mean():.4f} (per-sheet {ps('base'):.4f})")
        for k in [k for k in rows[0] if k not in ("i", "ref", "base") and not k.endswith("_nv")]:
            a = np.array([r[k] for r in rows]); d = a - b
            nv = f"  vertices median {np.median([r[k + '_nv'] for r in rows]):.0f}" if k + "_nv" in rows[0] else ""
            print(f"  {k:18s} {a.mean():.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f}) per-sheet {ps(k):.4f}{nv}")
