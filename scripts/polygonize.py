#!/usr/bin/env python3
"""Probability map -> polygons for the frontend (2026-10-19, synth_progress.md "mask -> polygon post-processing").

probs_to_polygons(prob_u8, native_hw): linear upsample of the probability to the sheet's native size, threshold 0.35, fill holes
up to 4096 px at inference resolution, drop pieces under 0.2% of the mask, trace outer contours and holes, simplify
(Douglas-Peucker, 2 native px), snap edges within 6 degrees of horizontal / vertical to exact (right-angle corners) and drop
the vertices that leaves collinear. Returns [{"outer": [[x, y], ...], "holes": [[[x, y], ...], ...]}, ...] in native pixels.
Scored as rasterised polygons on the best model's maps: val +0.006, HF14 +0.002 (fix v2 +0.002), Gemini30 +0.008 IoU over
the evaluator's nearest-upsampled mask; median 180-440 vertices per question.
2026-11-07 (scripts/pp_probe/snap.py, gem2s-x maps): eps 1 -> 2 + snap halves the vertices (median 8.7x -> 4.1x the hand
label's on Gemini30, 4.7x -> 2.7x on val, 6.4x -> 2.9x on HF14) for -0.0007 / -0.0003 / -0.0008 IoU. eps=1, snap_deg=None is
the previous output.
rasterize(polys, hw) turns the output back into a mask (for scoring)."""
import cv2
import numpy as np

T = 0.35


def _fill_holes(b, max_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats((~b).astype(np.uint8), connectivity=4)
    h, w = b.shape
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        ok[i] = not (x == 0 or y == 0 or x + ww == w or y + hh == h) and a <= max_area
    return b | ok[lab]


def snap_ring(pts, ang=6.0):
    """An edge within `ang` degrees of horizontal / vertical gets both endpoints set to its mean y / x; a corner between a
    horizontal and a vertical edge takes y from one and x from the other, so it becomes an exact right angle."""
    p = np.asarray(pts, np.float64).copy()
    n = len(p)
    if n < 3:
        return p
    t = np.tan(np.radians(ang))
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if dy <= t * dx:
            p[i, 1] = p[(i + 1) % n, 1] = (a[1] + b[1]) / 2
        elif dx <= t * dy:
            p[i, 0] = p[(i + 1) % n, 0] = (a[0] + b[0]) / 2
    return p


def clean_ring(p, tol=1.0):
    """Drop vertices within `tol` px of the line through their neighbours (collinear after snapping, or duplicated)."""
    changed = True
    while changed and len(p) > 3:
        changed = False
        for i in range(len(p)):
            a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
            ab, ac = b - a, c - a
            L = np.hypot(*ac)
            if (abs(ab[0] * ac[1] - ab[1] * ac[0]) / L if L > 0 else np.hypot(*ab)) < tol:
                p = np.delete(p, i, 0); changed = True
                break
    return p


def probs_to_polygons(prob_u8, native_hw, thresh=T, hole_px=4096, min_frac=0.002, eps=2.0, snap_deg=6.0):
    h0, w0 = native_hw
    s = prob_u8.shape[1] / w0
    m = cv2.resize(prob_u8.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR) > thresh
    m = _fill_holes(m, hole_px / s / s)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, 4] >= min_frac * st[1:, 4].sum()
        m = keep[lab]
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    polys = [cv2.approxPolyDP(c, eps, True)[:, 0, :] for c in cs]
    if snap_deg is not None:
        polys = [np.round(clean_ring(snap_ring(q, snap_deg))).astype(int) for q in polys]
    out, index = [], {}
    for j, hh in enumerate(hier[0]):
        if hh[3] == -1 and len(polys[j]) >= 3:
            index[j] = len(out)
            out.append({"outer": polys[j].tolist(), "holes": []})
    for j, hh in enumerate(hier[0]):
        if hh[3] != -1 and hh[3] in index and len(polys[j]) >= 3:
            out[index[hh[3]]]["holes"].append(polys[j].tolist())
    return out


def rasterize(polys, hw):
    m = np.zeros(hw, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p["outer"], np.int32)], 1)
        for hole in p["holes"]:
            cv2.fillPoly(m, [np.array(hole, np.int32)], 0)
    return m.astype(bool)
