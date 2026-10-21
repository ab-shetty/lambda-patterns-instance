"""Shapes of prediction pieces and gaps vs whether they are right (2026-10-21, descriptive; no rule tuned here).
Per question, mask M = lin_fill (linear upsample, threshold 0.35, holes <= 4096 inference px filled), at a working scale
(long side 2048 native-equivalent) for speed. Pieces = connected components of M. Per piece (no labels used):
  area_rel   area / area of the answer's largest piece          width   2 x max inscribed radius (px @2048)
  solidity   area / convex hull area                            rough   perimeter^2 / (4 pi area)
  elong      major / minor axis of the fitted ellipse           pmean   mean probability
  dref       distance from the reference box (px @2048, 0 if it overlaps)
Truth: prec = share of the piece inside the label; "wrong" = prec < 0.5. Gaps = connected components of the label NOT in M
(FN), split into holes (enclosed by M: the outer ring is >= 80% M) and outside; their share of the label area is reported.
Writes rows_shapes_{split}.json (one row per piece)."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0]); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import common
from methods import fill_holes
common.WITH_BOX[0] = True
T = 0.35


def feats(m, P, box, s):
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return [], cc
    amax = st[1:, 4].max()
    x, y, bw, bh = [int(round(v * s)) for v in box]
    ref = np.zeros_like(m, np.uint8); ref[y:y + bh, x:x + bw] = 1
    dist_ref = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 3)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    out = []
    for i in range(1, n):
        sel = cc == i
        a = st[i, 4]
        cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c)) or 1
        per = cv2.arcLength(c, True)
        el = 1.0
        if len(c) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(c)
            el = max(MA, ma) / max(1e-3, min(MA, ma))
        out.append({"i": i, "area": int(a), "area_rel": a / amax, "width": float(2 * dt[sel].max()),
                    "solidity": a / hull, "rough": per * per / (4 * np.pi * a), "elong": el,
                    "pmean": float(P[sel].mean()), "dref": float(dist_ref[sel].min())})
    return out, cc


def sheet(job):
    idx, img, anyann, qs = common.load(job)
    h0, w0 = img.shape[:2]
    s = 2048 / max(h0, w0)
    W, H = round(w0 * s), round(h0 * s)
    rows, gaps = [], []
    for ref, cat, tgt, p, box in qs:
        P = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        t = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sp = p.shape[1] / W
        M = fill_holes(P > T, int(4096 / (sp * sp)))
        fs, cc = feats(M, P, box, s)
        for f in fs:
            sel = cc == f.pop("i")
            f.update(sheet=idx, q=ref, prec=float(t[sel].mean()), tp=int((t & sel).sum()), fp=int((~t & sel).sum()),
                     label=int(t.sum()))
            rows.append(f)
        fn = (t & ~M).astype(np.uint8)
        n, gcc, st, _ = cv2.connectedComponentsWithStats(fn, connectivity=8)
        Md = cv2.dilate(M.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        for i in range(1, n):
            sel = gcc == i
            ring = cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~sel
            gaps.append({"sheet": idx, "q": ref, "area": int(st[i, 4]), "label": int(t.sum()),
                         "enclosed": float(M[ring].mean()) if ring.any() else 0.0,
                         "pmean": float(P[sel].mean())})
    return rows, gaps


if __name__ == "__main__":
    split = sys.argv[1]
    R, Gp = [], []
    for job in common.jobs(split):
        r, g = sheet(job); R += r; Gp += g
    json.dump({"pieces": R, "gaps": Gp}, open(f"scripts/pp_probe/rows_shapes_{split}.json", "w"))
    print(split, len(R), "pieces", len(Gp), "gaps")
