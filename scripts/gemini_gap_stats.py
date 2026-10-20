#!/usr/bin/env python3
"""Interpretable style statistics, generator vs Gemini, each scored alone by how well it separates the two sources
(univariate AUC; 0.5 = same distribution). The whole-image domain classifier (gemini_match.py) separates them at
AUC ~1.0 (2026-10-20), which says nothing about what to fix; these say which property gives the generator away.
All measured at the training scale (long side 2048). Elevation sheets only by default (--modes), both sides.

    python3 scripts/gemini_gap_stats.py --gemini data/image_generation/gemini_r5 \
        --gemini-modes data/image_generation/gemini_r5/specs_described.jsonl --gen data/synthetic/pd_probe
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def stats(path):
    im = cv2.imread(str(path)); h, w = im.shape[:2]
    s = 2048 / max(h, w); im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    paper = float(np.percentile(g, 95))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    stroke = bh > 40
    dt = cv2.distanceTransform(stroke.astype(np.uint8), cv2.DIST_L2, 3)
    # stroke width: 2 x the distance transform on the stroke's medial pixels (local maxima)
    mx = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & stroke
    sw = 2 * dt[mx] if mx.any() else np.array([0.0])
    dark = g[stroke] if stroke.any() else np.array([255])
    sat = hsv[..., 1] / 255.0
    nonwhite = g < paper - 12
    n, lab, st, _ = cv2.connectedComponentsWithStats(stroke.astype(np.uint8), connectivity=8)
    small = ((st[1:, 2] < 18) & (st[1:, 3] < 18) & (st[1:, 2] > 3) & (st[1:, 3] > 5)).sum() if n > 1 else 0
    ys, xs = np.nonzero(nonwhite)
    extent = ((ys.max() - ys.min()) * (xs.max() - xs.min()) / (h * w)) if len(ys) else 0.0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy); ang = (np.degrees(np.arctan2(gy, gx)) % 180)
    sel = mag > 40
    hist = np.bincount(np.minimum((ang[sel] / 15).astype(int), 11), minlength=12).astype(float)
    hist /= max(hist.sum(), 1)
    axis = hist[0] + hist[11] + hist[5] + hist[6]                       # near 0 / 90 degrees
    edge_soft = float(((mag > 10) & (mag < 60)).sum() / max((mag >= 60).sum(), 1))   # anti-aliasing / blur
    col = cv2.resize(im, (64, max(1, round(64 * h / w)))).reshape(-1, 3)
    return {
        "aspect": w / h, "paper": paper, "ink_frac": float(nonwhite.mean()),
        "stroke_px_median": float(np.median(sw)), "stroke_px_p90": float(np.percentile(sw, 90)),
        "stroke_dark_median": float(np.median(dark)), "colour_frac": float((sat > 0.15).mean()),
        "sat_nonwhite": float(sat[nonwhite].mean()) if nonwhite.any() else 0.0,
        "glyphs_per_mpx": small / (h * w / 1e6), "drawing_extent": float(extent),
        "axis_aligned_edges": float(axis), "edge_softness": edge_soft,
        "grey_levels": float(len(np.unique(g[nonwhite] // 8))) if nonwhite.any() else 0.0,
        "palette_size": float(len(np.unique((col // 32).astype(np.int32) @ np.array([64, 8, 1])))),
    }


def auc(a, b):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", required=True)
    ap.add_argument("--gemini-modes")
    ap.add_argument("--gen", required=True, nargs="+", help="one or more generator pools (compared one by one)")
    ap.add_argument("--modes", default="elevation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keep = set(a.modes.split(","))
    gm = {r["name"] + ".png": r.get("mode") for r in map(json.loads, open(a.gemini_modes)) if "error" not in r} \
        if a.gemini_modes else {}
    gp = [p for p in sorted(Path(a.gemini).glob("*.png")) if not gm or gm.get(p.name) in keep]
    with Pool(4) as pool:
        G = pool.map(stats, gp)
        res = {}
        for pool_dir in a.gen:
            modes = {json.load(open(f))["image"]["file_name"]: json.load(open(f)).get("mode")
                     for f in Path(pool_dir, "annotations").glob("*.json")}
            sp = [p for p in sorted(Path(pool_dir, "images").glob("*.png")) if modes.get(p.name) in keep]
            S = pool.map(stats, sp)
            print(f"\n== {pool_dir}: gemini {len(G)} vs generator {len(S)} ({a.modes})")
            print(f"{'statistic':22s} {'gemini med':>11s} {'gen med':>9s} {'AUC':>6s}")
            rows = []
            for k in G[0]:
                ga, sa = np.array([x[k] for x in G]), np.array([x[k] for x in S])
                rows.append((abs(auc(ga, sa) - 0.5), k, np.median(ga), np.median(sa), auc(ga, sa)))
            for _, k, gmed, smed, au in sorted(rows, reverse=True):
                print(f"{k:22s} {gmed:11.3f} {smed:9.3f} {au:6.3f}")
            res[pool_dir] = {k: {"gemini": gmed, "gen": smed, "auc": au} for _, k, gmed, smed, au in rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
