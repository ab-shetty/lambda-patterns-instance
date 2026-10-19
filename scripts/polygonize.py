#!/usr/bin/env python3
"""Probability map -> polygons for the frontend (2026-10-19, synth_progress.md "mask -> polygon post-processing").

probs_to_polygons(prob_u8, native_hw): linear upsample of the probability to the sheet's native size, threshold 0.35, fill holes
up to 4096 px at inference resolution, drop pieces under 0.2% of the mask, trace outer contours and holes, simplify
(Douglas-Peucker, 1 native px). Returns [{"outer": [[x, y], ...], "holes": [[[x, y], ...], ...]}, ...] in native pixels.
Scored as rasterised polygons on the best model's maps: val +0.006, HF14 +0.002 (fix v2 +0.002), Gemini30 +0.008 IoU over
the evaluator's nearest-upsampled mask; median 180-440 vertices per question.
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


def probs_to_polygons(prob_u8, native_hw, thresh=T, hole_px=4096, min_frac=0.002, eps=1.0):
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
