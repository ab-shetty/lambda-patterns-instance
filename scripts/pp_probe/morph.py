"""Prediction-informed geometry and morphology on top of the default (lin_fill+poly, scripts/polygonize.py). Tune on Gemini30
only (leave-one-sheet-out), read val / HF14 once for the pre-registered picks. All variants start from lf = linear-upsampled
probability > 0.35 with holes filled, and end as rasterised polygons; deltas are against lin_fill+poly ("ref").
  reg_e{eps}_t{deg}        orthogonal regularisation: Douglas-Peucker (eps native px); an edge within deg of an axis becomes the
                           exact axis line at the median of the contour pixels it covers; short diagonals (< 2 eps) between axis
                           edges are dropped so corners come out square; consecutive same-axis edges within 2 px are merged;
                           vertices are the intersections of consecutive lines
  snap_d{d}_c{c}           reg_e2_t10, then each axis edge >= 20 px is moved onto a parallel drawn line within d px when the
                           line covers >= c of the edge's middle 70% (ink v2); the edge lands on the line's centre (labels split
                           boundary strokes 50/50)
  open_k{k}_t{t}           remove what an opening (ellipse, k inference px) removes, but only pixels with probability < t
                           (thin spurs / trim bleed the model is unsure of)
  close_k{k}_t{t}          add what a closing adds, only where probability > t
  rect_r{r}                a component whose area fills >= r of its axis bounding box becomes the box (only if the added pixels
                           have mean probability > 0.15)
  ws_b{b}                  marker watershed: pixels more than b native px from lf's edge are fixed markers (inside / outside),
                           the band in between floods over the drawing's stroke darkness (black-hat), so the edge settles on lines"""
import sys, json, math
from multiprocessing import Pool
import cv2, numpy as np
from skimage.segmentation import watershed
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from common import jobs, load, ink_map_v2
from methods import fill_holes, polygons, iou
T = 0.35


def _lines(c, eps, tol, ink=None, d=0, cmin=1.0, lmin=20):
    """Regularise one closed ring c (Nx2, CHAIN_APPROX_NONE). Returns vertex array or None (keep plain DP)."""
    n = len(c)
    if n < 12: return None
    ap = cv2.approxPolyDP(c.reshape(-1, 1, 2), eps, True)[:, 0, :]
    if len(ap) < 3: return None
    s0 = np.nonzero((c == ap[0]).all(1))[0]
    if not len(s0): return None
    c = np.roll(c, -s0[0], 0); idx = [0]; j = 0
    for v in ap[1:]:
        while j < n and not (c[j] == v).all(): j += 1
        if j == n: return None
        idx.append(j)
    idx.append(n)
    tt = math.tan(math.radians(tol))
    segs = []                                   # [cls, pts]
    for a, b in zip(idx[:-1], idx[1:]):
        pts = c[a:b + 1] if b < n else np.vstack([c[a:], c[:1]])
        dx, dy = pts[-1] - pts[0]
        cls = "h" if abs(dy) <= tt * abs(dx) else "v" if abs(dx) <= tt * abs(dy) else "d"
        segs.append([cls, pts])
    # drop short diagonals between two axis edges (rounded corners)
    k = 0
    while k < len(segs) and len(segs) > 3:
        cls, pts = segs[k]
        if cls == "d" and np.hypot(*(pts[-1] - pts[0])) < 2 * eps and segs[k - 1][0] != "d" and segs[(k + 1) % len(segs)][0] != "d":
            segs.pop(k); continue
        k += 1
    # merge consecutive same-axis edges whose coordinates are within 2 px
    def coord(cls, pts): return float(np.median(pts[:, 1] if cls == "h" else pts[:, 0]))
    k = 0
    while k < len(segs) and len(segs) > 3:
        a, b = segs[k], segs[(k + 1) % len(segs)]
        if a[0] == b[0] != "d" and abs(coord(*a) - coord(*b)) <= 2:
            a[1] = np.vstack([a[1], b[1]]); segs.pop((k + 1) % len(segs)); continue
        k += 1
    if len(segs) < 3: return None
    L = []                                      # (cls, coordinate) or ("d", p0, p1)
    for cls, pts in segs:
        if cls == "d": L.append(("d", pts[0].astype(float), pts[-1].astype(float))); continue
        y = coord(cls, pts)
        if ink is not None and d > 0:
            ax = pts[:, 0] if cls == "h" else pts[:, 1]
            lo, hi = ax.min(), ax.max(); span = hi - lo
            if span >= lmin:
                lo, hi = int(lo + 0.15 * span), int(hi - 0.15 * span) + 1
                yi = int(round(y)); offs = range(-d, d + 1); cov = []
                for o in offs:
                    r = yi + o
                    if cls == "h": cov.append(ink[r, lo:hi].mean() if 0 <= r < ink.shape[0] else 0)
                    else: cov.append(ink[lo:hi, r].mean() if 0 <= r < ink.shape[1] else 0)
                cov = np.array(cov)
                if cov.max() >= cmin:
                    m = int(np.argmax(cov - 1e-3 * np.abs(np.arange(-d, d + 1))))
                    a = m; b = m
                    while a > 0 and cov[a - 1] >= cmin: a -= 1
                    while b < len(cov) - 1 and cov[b + 1] >= cmin: b += 1
                    y = yi + (a + b) / 2 - d
        L.append((cls, y))
    V = []
    for k in range(len(L)):
        A, B = L[k - 1], L[k]
        orig = segs[k][1][0].astype(float)
        if A[0] == "h" and B[0] == "v": V.append((B[1], A[1]))
        elif A[0] == "v" and B[0] == "h": V.append((A[1], B[1]))
        elif A[0] == B[0] == "h": V += [(orig[0], A[1]), (orig[0], B[1])]
        elif A[0] == B[0] == "v": V += [(A[1], orig[1]), (B[1], orig[1])]
        else:
            def nc(X):
                if X[0] == "h": return np.array([0., 1.]), X[1]
                if X[0] == "v": return np.array([1., 0.]), X[1]
                t = X[2] - X[1]; nn = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9); return nn, nn @ X[1]
            (n1, c1), (n2, c2) = nc(A), nc(B)
            M = np.array([n1, n2])
            if abs(np.linalg.det(M)) < 1e-6: V.append(tuple(orig)); continue
            p = np.linalg.solve(M, [c1, c2])
            V.append(tuple(p) if np.hypot(*(p - orig)) <= 4 * eps + 2 else tuple(orig))
    return np.round(np.array(V)).astype(np.int32)


def reg_mask(m, eps, tol, ink=None, d=0, cmin=1.0, min_frac=0.002):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool); keep[1:] = st[1:, 4] >= min_frac * st[1:, 4].sum(); m = keep[lab]
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8); nv = 0
    if hier is None: return out.astype(bool), 0
    rings = []
    for c in cs:
        v = _lines(c[:, 0, :], eps, tol, ink, d, cmin)
        rings.append(v if v is not None and len(v) >= 3 else cv2.approxPolyDP(c, 1.0, True)[:, 0, :])
    for r, hh in zip(rings, hier[0]):
        if hh[3] == -1 and len(r) >= 3: cv2.fillPoly(out, [r], 1); nv += len(r)
    for r, hh in zip(rings, hier[0]):
        if hh[3] != -1 and len(r) >= 3: cv2.fillPoly(out, [r], 0); nv += len(r)
    return out.astype(bool), nv


def sheet(job):
    idx, img, anyann, qs = load(job)
    h0, w0 = img.shape[:2]
    ink = ink_map_v2(img)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    rows = []
    for ref, cat, tgt, p in qs:
        r = {"i": idx, "q": ref}
        s = p.shape[1] / w0
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        lf = fill_holes(P > T, 4096 / s / s)
        pm, nv = polygons(lf); r["ref"] = iou(pm, tgt); r["ref_nv"] = nv
        for e in (1.5, 3):
            for t in (5, 12):
                pm, nv = reg_mask(lf, e, t); r[f"reg_e{e}_t{t}"] = iou(pm, tgt); r[f"reg_e{e}_t{t}_nv"] = nv
        for dd in (2, 4):
            for c in (0.5, 0.7):
                pm, nv = reg_mask(lf, 2, 10, ink, dd, c); r[f"snap_d{dd}_c{c}"] = iou(pm, tgt); r[f"snap_d{dd}_c{c}_nv"] = nv
        for k in (3, 7, 15):
            kk = max(3, int(round(k / s)) | 1); K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            op = cv2.morphologyEx(lf.astype(np.uint8), cv2.MORPH_OPEN, K).astype(bool)
            cl = cv2.morphologyEx(lf.astype(np.uint8), cv2.MORPH_CLOSE, K).astype(bool)
            for t in (0.5, 0.7):
                r[f"open_k{k}_t{t}"] = iou(polygons(lf & ~(~op & lf & (P < t)))[0], tgt)
            for t in (0.1, 0.2):
                r[f"close_k{k}_t{t}"] = iou(polygons(lf | (cl & (P > t)))[0], tgt)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lf.astype(np.uint8), connectivity=8)
        for rr in (0.85, 0.92):
            m = lf.copy()
            for j in range(1, n):
                x, y, ww, hh, a = st[j]
                if a >= rr * ww * hh:
                    add = ~m[y:y + hh, x:x + ww]
                    if add.any() and P[y:y + hh, x:x + ww][add].mean() > 0.15: m[y:y + hh, x:x + ww] = True
            r[f"rect_r{rr}"] = iou(polygons(m)[0], tgt)
        din = cv2.distanceTransform(lf.astype(np.uint8), cv2.DIST_L2, 5)
        dout = cv2.distanceTransform((~lf).astype(np.uint8), cv2.DIST_L2, 5)
        for b in (3, 6):
            band = (din <= b) & (dout <= b)
            ys, xs = np.nonzero(band)
            if not len(ys): r[f"ws_b{b}"] = r["ref"]; continue
            y1, y2, x1, x2 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
            mk = np.where(band, 0, np.where(lf, 2, 1))[y1:y2, x1:x2].astype(np.int32)
            ws = watershed(bh[y1:y2, x1:x2], mk) == 2
            m = lf.copy(); m[y1:y2, x1:x2] = ws
            r[f"ws_b{b}"] = iou(polygons(m)[0], tgt)
        rows.append(r)
    return rows


if __name__ == "__main__":
    workers = 1
    if sys.argv[1].startswith("--workers="): workers = int(sys.argv.pop(1)[10:])
    for split in sys.argv[1:]:
        with Pool(workers) as pool: rows = [r for rs in pool.imap(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_morph_{split}.json", "w"))
        b = np.array([r["ref"] for r in rows])
        def ps(k):
            by = {}
            for r in rows: by.setdefault(r["i"], []).append(r[k])
            return np.mean([np.mean(v) for v in by.values()])
        print(f"== {split}: n={len(rows)} ref (lin_fill+poly) {b.mean():.4f} per-sheet {ps('ref'):.4f}")
        for k in [k for k in rows[0] if k not in ("i", "q", "ref") and not k.endswith("_nv")]:
            a = np.array([r[k] for r in rows]); d = a - b
            nv = f"  vertices median {np.median([r[k + '_nv'] for r in rows]):.0f}" if k + "_nv" in rows[0] else ""
            print(f"  {k:16s} {a.mean():.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f}) per-sheet {ps(k):.4f}{nv}", flush=True)
