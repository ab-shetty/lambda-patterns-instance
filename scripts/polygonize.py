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
2026-11-09 (scripts/pp_probe/curve.py, synth_progress 2026-11-09): low_vertex=True (off by default; the user picks the trade)
fills irregular holes and inlets (convex-hull gaps) whose mean probability is >= 0.05, then simplifies at 12 px and drops pieces
under 0.5%: ~1.2x the labeller's vertices (median) on Gemini30 / val / HF14 / gemdev for -0.006..-0.009 IoU. LOW_VERTEX holds
its settings; fill_gaps(m, P) is the fill on its own (stack.py runs it before the colour cut).
2026-11-11 (synth_progress 2026-11-10/11, scripts/pp_probe/lvdecomp.py / tune3s.py): PRODUCT_V2 = low_vertex at eps 6 (12 cost
0.003-0.007 on the real sheets; 6 stays <= 1.2x there) + every polygon offset 1 native px outward (the real-sheet labels put the
edge on the outer side of the boundary stroke; Gemini labels do not, -0.007 there). With the 5120+4096+3072 map blend:
valfix3 0.9495 / 1.13x, hf14fix2 0.9407 / 1.12x vertices / label. offset_polys(polys, d) is the offset on its own.
2026-11-11 (scripts/pp_probe/cellsnap.py rounds 1-3): PRODUCT_V3 = PRODUCT_V2 after snapping the mask to the drawing's cells
(non-ink regions, ink = darker than the 9 px local max by > 40): a cell >= 80% selected is taken whole, one <= 20% selected is
dropped, cells under 1000 px or over 15% of the sheet are left alone, then holes refilled -- snap_to_cells(m, img, s). On
gem2s-x 3-scale maps vs PRODUCT_V2: valfix4 +0.0016 (0.9511 / 1.06x), hf14fix4 +0.0016 (0.9473 / 1.20x), Gemini30 +0.0002,
gemdev (xblend, unseen) +0.0019 (487 up / 161 down). Lower take thresholds gain more on the real sheets but lose on Gemini.
rasterize(polys, hw) turns the output back into a mask (for scoring)."""
import cv2
import numpy as np

T = 0.35
LOW_VERTEX = dict(eps=12.0, min_frac=0.005)
PRODUCT_V2 = dict(low_vertex=True, eps=6.0, offset=1.0)
CELL_SNAP = dict(ink_t=40, hi=0.8, lo=0.2, min_px=1000, max_frac=0.15)


def _fill_holes(b, max_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats((~b).astype(np.uint8), connectivity=4)
    h, w = b.shape
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        ok[i] = not (x == 0 or y == 0 or x + ww == w or y + hh == h) and a <= max_area
    return b | ok[lab]


def _rect_like(comp, rect_min):
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return True
    rw, rh = cv2.minAreaRect(max(cs, key=cv2.contourArea))[1]
    return int(comp.sum()) / max(1.0, rw * rh) >= rect_min


def fill_gaps(m, P, rect_min=0.8, max_frac=0.05, pmin=0.05):
    """Fill confidence dropouts inside a correct region. Holes (background components off the border) and inlets (components
    of a piece's convex hull minus the piece) up to max_frac of the mask / piece are filled when irregular -- area / min-area
    rotated rectangle < rect_min, so skylights, insets and an L-shape's notch stay -- and the model's mean probability over
    them is >= pmin. P is the probability at m's resolution."""
    m = m.copy()
    h, w = m.shape
    tot = max(1, int(m.sum()))
    n, lab, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if x == 0 or y == 0 or x + ww == w or y + hh == h or a > max_frac * tot:
            continue
        sel = lab[y:y + hh, x:x + ww] == i
        if not _rect_like(sel.astype(np.uint8), rect_min) and P[y:y + hh, x:x + ww][sel].mean() >= pmin:
            m[y:y + hh, x:x + ww] |= sel
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if a < 64:
            continue
        piece = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(piece); cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cs))], 1)
        k, gl, gs, _ = cv2.connectedComponentsWithStats((hull & (1 - piece)).astype(np.uint8), connectivity=4)
        sub, Ps = m[y:y + hh, x:x + ww], P[y:y + hh, x:x + ww]
        for j in range(1, k):
            if gs[j, 4] > max_frac * a or gs[j, 4] < 4:
                continue
            gx, gy, gw, gh = gs[j, :4]
            if _rect_like((gl[gy:gy + gh, gx:gx + gw] == j).astype(np.uint8), rect_min):
                continue
            sel = gl == j
            if Ps[sel].mean() >= pmin:
                sub[sel] = True
    return m


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


def probs_to_polygons(prob_u8, native_hw, thresh=T, hole_px=4096, min_frac=0.002, eps=2.0, snap_deg=6.0, low_vertex=False,
                      offset=0.0):
    h0, w0 = native_hw
    s = prob_u8.shape[1] / w0
    P = cv2.resize(prob_u8.astype(np.float32) / 255, (w0, h0), interpolation=cv2.INTER_LINEAR)
    m = _fill_holes(P > thresh, hole_px / s / s)
    if low_vertex:
        m = fill_gaps(m, P)
        # low_vertex keeps its own eps unless the caller asks for another (PRODUCT_V2: 6)
        eps = LOW_VERTEX["eps"] if eps == 2.0 else eps
        min_frac = LOW_VERTEX["min_frac"]
    polys = mask_to_polygons(m, min_frac, eps, snap_deg)
    return offset_polys(polys, offset) if offset else polys


def snap_to_cells(m, img, s=1.0, ink_t=CELL_SNAP["ink_t"], hi=CELL_SNAP["hi"], lo=CELL_SNAP["lo"],
                  min_px=CELL_SNAP["min_px"], max_frac=CELL_SNAP["max_frac"]):
    """Snap a native-resolution mask to the drawing's cells (4-connected non-ink regions; ink = darker than the 9 px local
    maximum by > ink_t). Cells with coverage >= hi are taken whole, <= lo dropped; ink pixels whose 3x3 neighbourhood is
    mostly selected join (strokes between taken cells); holes up to 4096 px at inference scale s are refilled."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = (cv2.dilate(g, np.ones((9, 9), np.uint8)).astype(np.int16) - g) > ink_t
    n, lab = cv2.connectedComponents((~ink).astype(np.uint8), connectivity=4)
    tot = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    c = np.bincount(lab.ravel(), weights=m.ravel().astype(np.float64), minlength=n) / np.maximum(tot, 1)
    ok = (tot >= min_px) & (tot <= max_frac * lab.size)
    ok[0] = False
    out = m.copy()
    out[(ok & (c >= hi))[lab]] = True
    out[(ok & (c <= lo) & (c > 0))[lab]] = False
    out[ink] = m[ink]
    out |= ink & (cv2.blur(out.astype(np.float32), (3, 3)) >= 0.5)
    return _fill_holes(out, 4096 / s / s)


def offset_polys(polys, d):
    """Grow every polygon by d native px (holes shrink by d). Mitre limit 10 so sharp corners stay one vertex (limit 2 bevels
    them: +4-12% vertices, scripts/pp_probe/offverts.py), then the usual collinear clean-up. Pieces that touch after the offset
    stay separate polygons -- the frontend draws them as before."""
    from shapely.geometry import Polygon
    out = []
    for p in polys:
        g = Polygon(p["outer"], p["holes"])
        if not g.is_valid:
            g = g.buffer(0)
        g = g.buffer(d, join_style=2, mitre_limit=10.0)
        for q in (getattr(g, "geoms", None) or [g]):
            if q.is_empty or q.area <= 0:
                continue
            ring = lambda r: np.round(clean_ring(np.round(np.asarray(r.coords)[:-1]).astype(int))).astype(int).tolist()
            o = ring(q.exterior)
            if len(o) >= 3:
                out.append({"outer": o, "holes": [x for x in (ring(h) for h in q.interiors) if len(x) >= 3]})
    return out


def mask_to_polygons(m, min_frac=0.002, eps=2.0, snap_deg=6.0):
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
