#!/usr/bin/env python3
"""Turn a raster mask into a small, architecturally-plausible polygon.

SAM-family masks give a stair-stepped contour (median 284 vertices here) where a
human draws a 5-vertex quadrilateral, which is why raw mask->contour proposals
cost a labeller more time than they save. This does:

    contour -> Douglas-Peucker  (-> optional orientation snap, measured NEGATIVE)

Measured on 321 instances (40 real + 25 gemini plans), SAM3 masks vs human polygons:

    method             med verts   mean IoU   med IoU   >=0.8
    raw contour              284     0.7748    0.8483
    DP eps=0.010               7     0.7571    0.8306   58.6%
    DP + snap  8 deg           7     0.7462    0.8141   55.1%
    DP + snap 12 deg           7     0.7434    0.8117   53.3%
    DP + snap 20 deg           7     0.7347    0.8007   50.5%

So DP alone takes 284 vertices to 7 for 1.7 points of IoU -- the whole usability
problem -- and **orientation snapping is negative and monotone in the tolerance**.
The rectilinear assumption does not hold: these regions are bounded by rakes,
gables and eaves at many angles, so forcing edges onto a lattice mod one dominant
angle costs more than the wobble it removes. `snap_tol_deg=0` disables it and is
the default; the knob is kept only so the result stays reproducible.
"""
import numpy as np, cv2


def _dominant_angle(pts, weights):
    """Length-weighted circular mean of edge angles, taken mod 90 degrees."""
    a = (pts * 4.0) % (2 * np.pi)          # mod 90 deg -> full circle for averaging
    s = float((weights * np.sin(a)).sum())
    c = float((weights * np.cos(a)).sum())
    if s == 0.0 and c == 0.0:
        return 0.0
    return float(np.arctan2(s, c) / 4.0)


def _intersect(p0, d0, p1, d1):
    """Intersection of lines p0+t*d0 and p1+u*d1; None when near-parallel."""
    den = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(den) < 1e-9:
        return None
    t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den
    return p0 + t * d0


def regularize(mask, eps_frac=0.010, snap_tol_deg=0.0, min_edge_px=3.0):
    """mask (HxW bool) -> (N,2) float polygon, or None if no contour."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps_frac * per, True).reshape(-1, 2).astype(np.float64)
    n = len(poly)
    if n < 3:
        return None

    seg = np.roll(poly, -1, axis=0) - poly
    length = np.hypot(seg[:, 0], seg[:, 1])
    keep = length >= min_edge_px
    if keep.sum() >= 3:
        poly = poly[keep]
        seg = np.roll(poly, -1, axis=0) - poly
        length = np.hypot(seg[:, 0], seg[:, 1])
        n = len(poly)
    if n < 3:
        return None

    if snap_tol_deg <= 0.0:      # default: DP only -- snapping measured negative
        return poly

    ang = np.arctan2(seg[:, 1], seg[:, 0])
    theta = _dominant_angle(ang, length)

    # snap each edge to the nearest member of {theta + k*90deg} when close enough
    tol = np.deg2rad(snap_tol_deg)
    dirs = np.empty((n, 2))
    for i in range(n):
        k = np.round((ang[i] - theta) / (np.pi / 2))
        target = theta + k * (np.pi / 2)
        delta = (ang[i] - target + np.pi) % (2 * np.pi) - np.pi
        use = target if abs(delta) <= tol else ang[i]
        dirs[i] = (np.cos(use), np.sin(use))

    # each snapped edge keeps its midpoint; vertices are consecutive intersections
    mid = poly + seg / 2.0
    out = []
    for i in range(n):
        j = (i - 1) % n
        p = _intersect(mid[j], dirs[j], mid[i], dirs[i])
        out.append(poly[i] if p is None else p)
    out = np.asarray(out, dtype=np.float64)

    if not np.all(np.isfinite(out)):
        return poly
    # a snap that moves a vertex absurdly far means the assumption broke; fall back
    if np.max(np.hypot(*(out - poly).T)) > 0.25 * per:
        return poly
    return out
