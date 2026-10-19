"""Probability-map post-processing variants (2026-10-19). Tuned on Gemini30 only (leave-one-sheet-out CV); HF14 / val read once.
Every variant: linear upsample of the probability P to native size, a decision rule, then fill holes <= 4096 inference px (the
polygonize.py default). Rows: per question IoU for each variant -> rows_pmap_<split>.json.
  thr_t         P > t
  otsu          Otsu threshold on P over pixels with P > 0.05
  boxrel_a      P > a * median(P inside the reference box)        (the box is a known positive)
  hyst_lo       hysteresis: pixels P > lo connected to a P > 0.5 core
  gs_s          Gaussian-smoothed P (sigma s native px) > 0.35
  locnorm       P > max(0.15, 0.5 * local max of P in a 31 px window), only where that local max > 0.5 (boundary at half height)
  lapl          P > 0.15 and (P > 0.6 or Laplacian of smoothed P <= 0): boundary where the profile's curvature changes sign
  prune_m       thr 0.35, drop components whose peak P < m"""
import sys, json
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import common
from common import jobs, load
common.WITH_BOX[0] = True

def fill_holes(b, max_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats((~b).astype(np.uint8), connectivity=4)
    h, w = b.shape; ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        ok[i] = not (x == 0 or y == 0 or x + ww == w or y + hh == h) and a <= max_area
    return b | ok[lab]

def hyst(P, lo, hi):
    b = P > lo; n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[P > hi])] = True; keep[0] = False
    return keep[lab]

def prune(b, P, m):
    n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    mx = np.zeros(n); np.maximum.at(mx, lab.ravel(), P.ravel())
    keep = mx >= m; keep[0] = False
    return keep[lab]

def iou(a, b): return float((a & b).sum() / max((a | b).sum(), 1))

def sheet(job):
    idx, img, anyann, qs = load(job); h0, w0 = img.shape[:2]; rows = []
    for ref, cat, tgt, p, box in qs:
        s = p.shape[1] / w0; H = 4096 / s / s
        P = cv2.resize(p.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
        x, y, bw, bh = box; bm = float(np.median(P[y:y + bh, x:x + bw]))
        V = {}
        for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6): V[f"thr{t}"] = P > t
        v = P[P > 0.05]
        if v.size > 100:
            ot, _ = cv2.threshold((v * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); V["otsu"] = P > ot / 255
        else: V["otsu"] = P > 0.35
        for a in (0.3, 0.4, 0.5, 0.6): V[f"boxrel{a}"] = P > a * bm
        for lo in (0.1, 0.15, 0.2, 0.25): V[f"hyst{lo}"] = hyst(P, lo, 0.5)
        for sg in (1.5, 3, 6): V[f"gs{sg}"] = cv2.GaussianBlur(P, (0, 0), sg) > 0.35
        lm = cv2.dilate(P, np.ones((31, 31), np.uint8)); V["locnorm"] = (P > np.maximum(0.15, 0.5 * lm)) & (lm > 0.5)
        Ps = cv2.GaussianBlur(P, (0, 0), 2); L = cv2.Laplacian(Ps, cv2.CV_32F, ksize=3)
        V["lapl"] = (P > 0.15) & ((P > 0.6) | (L <= 0))
        for m in (0.5, 0.6, 0.7): V[f"prune{m}"] = prune(P > 0.35, P, m)
        r = {"i": idx, "ref": ref, "box_med": bm}
        for k, b in V.items(): r[k] = iou(fill_holes(b, H), tgt)
        rows.append(r)
    return rows

if __name__ == "__main__":
    for split in sys.argv[1:]:
        with Pool(2) as pool: rows = [r for rs in pool.map(sheet, jobs(split)) for r in rs]
        json.dump(rows, open(f"data/diag/rows_pmap_{split}.json", "w"))
        b = np.array([r["thr0.35"] for r in rows])
        print(f"== {split}: n={len(rows)} lin+fill @0.35 {b.mean():.4f}")
        for k in [k for k in rows[0] if k not in ("i", "ref", "box_med")]:
            a = np.array([r[k] for r in rows]); d = a - b
            print(f"  {k:12s} {a.mean():.4f} ({d.mean():+.4f}; up {(d > .005).sum()} / down {(d < -.005).sum()}; worst {d.min():+.3f})")
