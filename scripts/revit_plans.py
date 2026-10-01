"""Revit-style roof plans and floor plans (--revit-plans in generate_synthetic_fc.py).

Added 2026-09-28. The Revit elevation pool beat v6d on every elevation type of
the real eval sheets but LOST on plans (roof plans 0.92 -> 0.81, floor plan
HF14 12 0.44 -> 0.36): its plan sheets were still v6-drawn. This module draws
them the way the real Revit plan sheets in the eval set look (HF14 0/1/7/11/12/
15/16), and labels what those sheets label:

  roof plan  -- every pitched roof surface is one family (all facets, across
                separate buildings); flat roofs usually unlabelled. Facets come
                from the FreeCAD model seen from above (true valleys / hips /
                intersections). ALWAYS textured (2026-09-29, from the 14
                Gemini + 4 real roof plans): running-bond shingle courses
                turning with each facet's eave, global shingle bond,
                eave-parallel lines, down-slope seams, diamond crosshatch or
                dot grids, on white / grey / tinted / dark bases, optionally
                sun-shaded per facet; 35% a wing in a second labelled roof
                material, flat roofs gravel / membrane dots / seams (labelled
                60%). Skylights and chimneys are ALWAYS excluded: holes in the roof
                label, never labelled themselves. Grid lines + bubbles, section heads, slope
                arrows ("3\" / 1'-0\""), ridge/valley text, spot elevations,
                walls-below dashed, gutters, trellis / louver slats (unlabelled
                parallel-line look-alikes), copies of the building.
  floor plan -- rewritten 2026-09-29 from the ~40 Gemini + 3 real floor plans.
                Rooms in EVERY block (wings too) inside a shaped footprint
                (notches, chamfers, 45-degree bays, U courtyards), corridors,
                merged L-shaped open-plan rooms, closets carved from corners;
                doors / wide cased openings
                on a spanning tree, windows on the exterior. Three sheet types:
                finish plans (55%, Gemini): one material per room group (wet
                tile / mosaic, living planks / tile / basketweave, bedroom
                carpet / planks, garage concrete / dots), a label runs through
                the openings between rooms of the same material, identical
                materials are ONE family, every patterned finish is labelled;
                rendered plans (22%, HF14 0 / 12): textured grey interiors,
                labelled exterior hardscape; under-floor / foundation plans
                (23%, HF14 7 + Gemini): crawlspace hatch / joists / dots and
                slab labelled, piers cut out of the label.
                Plan symbols (FreeCAD BIM window/door presets, ArchStairs):
                swing / double / pocket / bifold / barn doors, exterior
                patio sliders, garage overhead doors; 3-line / 2-line / sill
                windows (one style per sheet); straight or U stairs with
                treads, UP/DN arrow, break line, rail -- always holes in the
                floor label.

Labels and the annotation JSON follow v6's format, so v6's _job post-steps
(tight crop, res-degrade) apply unchanged.
"""
import math
import random

import cv2
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

G = None      # generate_synthetic_v6
FC = None     # generate_synthetic_fc


def install(g, fc):
    global G, FC
    G, FC = g, fc


def _house(image_id, seed, mode_weights):
    """Replay compose()'s rng prefix: the same house FreeCAD built."""
    rng = random.Random(seed * 1_000_003 + image_id)
    rng.choices(list(mode_weights.keys()), weights=list(mode_weights.values()))
    G.make_appearance(rng)
    return G.make_house(rng)


# ----------------------------------------------------------------------------
# geometry: roof faces seen from above
# ----------------------------------------------------------------------------
def top_faces(faces, chim):
    cand = []
    for f in faces:
        n = f["n"]
        if n[2] <= 0.05 or not f["loops"]:
            continue
        try:
            poly = Polygon([(p[0], p[1]) for p in f["loops"][0]],
                           [[(p[0], p[1]) for p in lp] for lp in f["loops"][1:]]).buffer(0)
        except Exception:
            continue
        if poly.is_empty or poly.area < 1e-3:
            continue
        P = np.array([p for lp in f["loops"] for p in lp])
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        plane = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        cx, cy, cz = f["c"]
        if chim and chim["x0"] - 0.05 <= cx <= chim["x0"] + chim["w"] + 0.05 and \
                chim["y0"] - 0.05 <= cy <= chim["y0"] + chim["d"] + 0.05 and cz > chim["z0"]:
            kind = "chimney"
        else:
            kind = "flat" if n[2] > 0.995 else "pitched"
        cand.append({"poly": poly, "plane": plane, "n": n, "kind": kind, "zmax": float(P[:, 2].max()),
                     "pts3": P})

    def z(F, pt):
        return F["plane"][0] * pt.x + F["plane"][1] * pt.y + F["plane"][2]

    out = []
    for i, F in enumerate(cand):
        occ = []
        for j, H in enumerate(cand):
            if i == j or not F["poly"].intersects(H["poly"]):
                continue
            ov = F["poly"].intersection(H["poly"])
            if ov.area < 1e-4:
                continue
            # planes can cross inside the overlap (a wing's slab running into the
            # main roof): H hides F only where z_H - z_F > 0, a half-plane
            d = H["plane"] - F["plane"]
            hp = _halfplane(ov.bounds, d)
            if hp is None:
                continue
            part = ov.intersection(hp)
            if not part.is_empty and part.area > 1e-5:
                occ.append(part)
        vis = F["poly"].difference(unary_union([o.buffer(0) for o in occ]).buffer(0)) if occ else F["poly"]
        # the half-plane split leaves near-degenerate slivers; keep a valid polygon set
        vis = shapely.set_precision(vis.buffer(0), 1e-4)
        vis = unary_union([q for q in G.polys_of(vis) if q.area > 1e-3]) if not vis.is_empty else vis
        if vis.area < 0.05:
            continue
        out.append(dict(F, vis=vis))
    return out


def _halfplane(bounds, d, eps=1e-4):
    """Polygon of {(x, y) in bounds : d0*x + d1*y + d2 > eps}, or None if empty."""
    x0, y0, x1, y1 = bounds
    x0, y0, x1, y1 = x0 - 1, y0 - 1, x1 + 1, y1 + 1
    f = lambda p: d[0] * p[0] + d[1] * p[1] + d[2] - eps
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    out = []
    for k in range(4):
        a, b = ring[k], ring[(k + 1) % 4]
        fa, fb = f(a), f(b)
        if fa > 0:
            out.append(a)
        if (fa > 0) != (fb > 0):
            t = fa / (fa - fb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return Polygon(out) if len(out) >= 3 else None


def ridge_segments(F):
    """Horizontal edges at the facet's top (ridges) and sloping edges (hips / valleys)."""
    P = F["pts3"]
    ring = list(F["poly"].exterior.coords) if isinstance(F["poly"], Polygon) else []
    segs = []
    for a, b in zip(ring[:-1], ring[1:]):
        za = F["plane"][0] * a[0] + F["plane"][1] * a[1] + F["plane"][2]
        zb = F["plane"][0] * b[0] + F["plane"][1] * b[1] + F["plane"][2]
        if math.dist(a, b) < 3:
            continue
        top = F["zmax"]
        if abs(za - top) < 0.05 and abs(zb - top) < 0.05:
            segs.append(("RIDGE", a, b))
        elif abs(za - zb) > 0.5:
            segs.append(("SLOPE", a, b))
    return segs


# ----------------------------------------------------------------------------
# transforms (building copies)
# ----------------------------------------------------------------------------
def _xf(g, rot, dx, dy):
    return affinity.translate(affinity.rotate(g, rot, origin=(0, 0)), dx, dy)


def _xv(v, rot):
    t = math.radians(rot)
    return (v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t))


def _xp(p, rot, dx, dy):
    q = _xv(p, rot)
    return (q[0] + dx, q[1] + dy)


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def _dash_line(canvas, a, b, colour, w, pattern):
    """pattern: list of (on_px, off_px) repeated."""
    L = math.dist(a, b)
    if L < 1:
        return
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    s, k = 0.0, 0
    while s < L:
        on, off = pattern[k % len(pattern)]
        e = min(L, s + on)
        G.cv_line(canvas, (a[0] + ux * s, a[1] + uy * s), (a[0] + ux * e, a[1] + uy * e), colour, w)
        s = e + off
        k += 1


def _dash_poly(canvas, poly_px, colour, w, pattern):
    for p in G.polys_of(poly_px):
        c = list(p.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            _dash_line(canvas, a, b, colour, w, pattern)


def _bubble(canvas, tq, c, rad, text, ink, lw, size):
    cv2.circle(canvas, G._pt(c), int(rad * G.SCALE), G.bgr((255, 255, 255)), -1, cv2.LINE_AA, G.SHIFT)
    cv2.circle(canvas, G._pt(c), int(rad * G.SCALE), G.bgr(ink), max(1, int(round(lw))), cv2.LINE_AA, G.SHIFT)
    tq.add(c, text, size, ink, anchor="mm")


def _section_head(canvas, tq, c, rad, direction, num, sheet, ink, lw, size):
    """Revit section / callout head: circle split by a rule, filled wedge pointing along `direction`."""
    dx, dy = direction
    tip = (c[0] + dx * rad * 1.7, c[1] + dy * rad * 1.7)
    px, py = -dy, dx
    wedge = [tip, (c[0] + px * rad, c[1] + py * rad), (c[0] - px * rad, c[1] - py * rad)]
    arr = np.array([G._pt(p) for p in wedge], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [arr], G.bgr(ink), cv2.LINE_AA, G.SHIFT)
    cv2.circle(canvas, G._pt(c), int(rad * G.SCALE), G.bgr((255, 255, 255)), -1, cv2.LINE_AA, G.SHIFT)
    cv2.circle(canvas, G._pt(c), int(rad * G.SCALE), G.bgr(ink), max(1, int(round(lw))), cv2.LINE_AA, G.SHIFT)
    G.cv_line(canvas, (c[0] - rad, c[1]), (c[0] + rad, c[1]), ink, lw)
    tq.add((c[0], c[1] - rad * 0.12), num, size, ink, anchor="ms", bold=True)
    tq.add((c[0], c[1] + rad * 0.12), sheet, size * 0.8, ink, anchor="ma")


def _slope_arrow(canvas, tq, a, b, text, ink, lw, size, half=True):
    G.cv_line(canvas, a, b, ink, lw)
    L = math.dist(a, b)
    if L < 2:
        return
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    h = min(L * 0.3, size * 0.8)
    G.cv_line(canvas, b, (b[0] - ux * h - uy * h * 0.35, b[1] - uy * h + ux * h * 0.35), ink, lw)
    if not half:
        G.cv_line(canvas, b, (b[0] - ux * h + uy * h * 0.35, b[1] - uy * h - ux * h * 0.35), ink, lw)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ang = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
    if abs(ang) > 95:
        ang = ang - 180 if ang > 0 else ang + 180
    if abs(ang) < 5:
        tq.add((mid[0], mid[1] - size * 0.35), text, size, ink, anchor="ms")
    else:
        tq.add((mid[0] - size * 0.6 * math.sin(math.radians(ang)) - size * 1.5, mid[1] - size * 0.4), text, size, ink,
               angle=ang)


def _dim_chain(canvas, tq, pts, axis, off, V, ink, lw, size, S):
    """Revit dimension chain: witness ticks as 45-degree slashes, text centred above.
    axis "x": points are x positions at y=off; "y": y positions at x=off."""
    pts = sorted(set(round(p, 2) for p in pts))
    if len(pts) < 2:
        return
    P = [V.px(p, off) if axis == "x" else V.px(off, p) for p in pts]
    G.cv_line(canvas, P[0], P[-1], ink, lw)
    t = 0.18 * S
    for q in P:
        G.cv_line(canvas, (q[0] - t, q[1] + t), (q[0] + t, q[1] - t), ink, lw * 1.6)
        if axis == "x":
            G.cv_line(canvas, (q[0], q[1] - t * 1.4), (q[0], q[1] + t * 1.4), ink, lw)
        else:
            G.cv_line(canvas, (q[0] - t * 1.4, q[1]), (q[0] + t * 1.4, q[1]), ink, lw)
    for (p0, p1), (a, b) in zip(zip(pts[:-1], pts[1:]), zip(P[:-1], P[1:])):
        if p1 - p0 < 1.2:
            continue
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if axis == "x":
            tq.add((m[0], m[1] - size * 0.25), G.ft_label(p1 - p0), size, ink, anchor="ms")
        else:
            tq.add((m[0] - size * 1.35, m[1] + size * 1.6), G.ft_label(p1 - p0), size, ink, angle=90)


def _grid(canvas, tq, V, xs, ys, ext, ink, lw, size, S, r):
    """Grid lines (dash-dot) with bubbles; letters across x, numbers down y."""
    x0, y0, x1, y1 = ext
    rad = size * r.uniform(0.95, 1.25)
    both = r.random() < 0.5
    col = G.mix(ink, (255, 255, 255), r.uniform(0.1, 0.45))
    pat = [(1.6 * S, 0.3 * S), (0.25 * S, 0.3 * S)]
    for i, x in enumerate(xs):
        a, b = V.px(x, y0), V.px(x, y1)
        _dash_line(canvas, a, b, col, lw, pat)
        _bubble(canvas, tq, (a[0], a[1] - rad), rad, "ABCDEFGHJKLMNP"[i % 14], ink, lw, size * 1.05)
        if both:
            _bubble(canvas, tq, (b[0], b[1] + rad), rad, "ABCDEFGHJKLMNP"[i % 14], ink, lw, size * 1.05)
    for j, y in enumerate(ys):
        a, b = V.px(x0, y), V.px(x1, y)
        _dash_line(canvas, a, b, col, lw, pat)
        _bubble(canvas, tq, (a[0] - rad, a[1]), rad, str(j + 1), ink, lw, size * 1.05)
        if both:
            _bubble(canvas, tq, (b[0] + rad, b[1]), rad, str(j + 1), ink, lw, size * 1.05)


def _merge_close(vals, d):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > d:
            out.append(v)
    return out


def _title(canvas, tq, V, x_ft, y_ft, x_end_ft, name, r, ink, lw, size):
    tx, ty = V.px(x_ft, y_ft)
    xe = V.px(x_end_ft, y_ft)[0]
    scale = r.choice(['1/4" = 1\'-0"', '1/8" = 1\'-0"', '3/16" = 1\'-0"'])
    if r.random() < 0.6:
        big = size * 1.9
        tq.add((tx, ty), str(r.randint(1, 4)), big, ink, bold=True, anchor="lm")
        tq.add((tx + big * 1.2, ty - big * 0.12), name, size * 1.15, ink, bold=True, anchor="ls")
        G.cv_line(canvas, (tx + big * 1.2, ty + big * 0.05), (max(xe, tx + big * 8), ty + big * 0.05), ink, lw)
        tq.add((tx + big * 1.2, ty + big * 0.25), scale, size * 0.75, ink, anchor="la")
    else:
        # bracket + north glyph style (HF14 16)
        h = size * 2.4
        G.cv_line(canvas, (tx, ty - h / 2), (tx, ty + h / 2), ink, lw * 2)
        G.cv_line(canvas, (tx + size * 1.2, ty - h / 2), (tx + size * 1.2, ty + h / 2), ink, lw * 2)
        G.cv_line(canvas, (tx + size * 1.2, ty - h / 2), (tx + size * 0.4, ty), ink, lw)
        G.cv_line(canvas, (tx + size * 0.4, ty), (tx + size * 1.2, ty + h / 2), ink, lw)
        tq.add((tx + size * 1.6, ty - size * 0.1), name, size * 1.2, ink, bold=True, anchor="ls")
        G.cv_line(canvas, (tx + size * 1.6, ty + size * 0.05), (max(xe, tx + size * 14), ty + size * 0.05), ink, lw)
        tq.add((tx + size * 1.6, ty + size * 0.3), scale, size * 0.8, ink, bold=True, anchor="la")


def _north(canvas, tq, c, rad, ink, lw, size):
    cv2.circle(canvas, G._pt(c), int(rad * G.SCALE), G.bgr(ink), max(1, int(round(lw))), cv2.LINE_AA, G.SHIFT)
    pts = [(c[0], c[1] - rad), (c[0] - rad * 0.35, c[1] + rad * 0.3), (c[0], c[1] + rad * 0.05)]
    arr = np.array([G._pt(p) for p in pts], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [arr], G.bgr(ink), cv2.LINE_AA, G.SHIFT)
    G.cv_polyline(canvas, [(c[0], c[1] - rad), (c[0] + rad * 0.35, c[1] + rad * 0.3), (c[0], c[1] + rad * 0.05)],
                  ink, lw, closed=True)
    tq.add((c[0], c[1] - rad - size * 0.2), "N", size * 1.1, ink, anchor="ms", bold=True)


def _notes(tq, x, y, title, lines, ink, size):
    tq.add((x, y), title, size * 1.05, ink, bold=True)
    for i, n in enumerate(lines):
        tq.add((x, y + size * 1.5 * (i + 1)), n, size * 0.85, ink)


def _mfill(canvas, geom_px, colour, W, H):
    """Fill respecting holes (G.cv_fill paints holes white: wrong for a wall ring around patterned rooms)."""
    if geom_px.is_empty:
        return
    minx, miny, maxx, maxy = geom_px.bounds
    x0, y0 = max(0, int(minx) - 2), max(0, int(miny) - 2)
    x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    m = G.poly_mask(affinity.translate(geom_px, -x0, -y0), x1 - x0, y1 - y0)
    layer = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
    layer[:] = G.bgr(colour)
    G.blend_mask(canvas[y0:y1, x0:x1], layer, m)


# ---- FreeCAD TechDraw hatch library (scripts/hatch_tiles/*.png, seamless 256 px tiles)
import os
HATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hatch_tiles")
# kind -> (tile file, look class, effective line/unit spacing as a fraction of the tile,
#          tile size range in feet)
TILES = {"t_hbone": ("hbone", "hbone", 1 / 3, (2.5, 4.5)), "t_earth": ("earth", "basket", 1 / 2, (2.0, 3.5)),
         "t_brick": ("brick01", "bond", 1 / 4, (2.5, 4.0)), "t_aggregate": ("concrete", "aggregate", 1 / 3, (2.5, 5.0)),
         "t_woodgrain": ("woodgrain", "woodgrain", 1 / 5, (2.5, 5.0)), "t_plus": ("plus", "plus", 1 / 2, (2.0, 4.0)),
         "t_cross": ("cross", "diag", 1 / 5, (2.5, 4.5)), "t_zinc": ("zinc", "diag", 1 / 8, (2.0, 3.5)),
         "t_steel": ("steel", "lines", 1 / 8, (2.5, 4.5)), "t_aluminium": ("aluminium", "lines", 1 / 8, (3.0, 5.0)),
         "t_cuprous": ("cuprous", "lines", 1 / 10, (2.5, 4.5)), "t_glass": ("glass", "dashes", 1 / 10, (2.5, 4.5)),
         "t_plastic": ("plastic", "plastic", 1 / 6, (2.5, 4.0))}
_TILE_CACHE = {}


def _tile_alpha(name):
    if name not in _TILE_CACHE:
        g = cv2.imread(os.path.join(HATCH_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
        _TILE_CACHE[name] = 1.0 - g.astype(np.float32) / 255.0
    return _TILE_CACHE[name]


def hatch_layer(layer, ox, oy, S, kind, tile_ft, colour, ang=0.0):
    """Composite TechDraw tile `kind` (tile_ft feet per repeat, rotated by ang degrees)
    in `colour` over `layer`, which covers sheet pixels (ox, oy)+(w, h). Phase is
    anchored to the sheet so one family's pieces line up."""
    h, w = layer.shape[:2]
    T = max(8, int(round(tile_ft * S)))
    a = _tile_alpha(TILES[kind][0])
    t = cv2.resize(a, (T, T), interpolation=cv2.INTER_AREA if T < a.shape[0] else cv2.INTER_LINEAR)
    t = np.clip(t * max(1.0, (a.shape[0] / T) ** 0.6), 0, 1)       # keep thin lines visible when shrunk
    if ang % 180:
        D = int(math.hypot(w, h)) + 2 * T
        big = np.tile(t, (D // T + 2, D // T + 2))[:D, :D]
        M = cv2.getRotationMatrix2D((D / 2, D / 2), -ang, 1.0)
        big = cv2.warpAffine(big, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        a2 = big[(D - h) // 2:(D - h) // 2 + h, (D - w) // 2:(D - w) // 2 + w]
    else:
        py, px = int(oy) % T, int(ox) % T
        big = np.tile(t, (h // T + 2, w // T + 2))
        a2 = big[py:py + h, px:px + w]
    a2 = a2[..., None]
    col = np.array(G.bgr(colour), np.float32)
    layer[:] = (layer.astype(np.float32) * (1 - a2) + col * a2).astype(np.uint8)


def _shade(canvas, geom_px, k, W, H):
    m = np.zeros((H, W), np.uint8)
    for q in G.polys_of(geom_px):
        m = np.maximum(m, G.poly_mask(q, W, H))
    a = (m.astype(np.float32) / 255.0 * (1.0 - k))[..., None]
    canvas[:] = (canvas.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def _shade_soft(canvas, geom_px, k, W, H, blur_px):
    """_shade with a blurred edge, worked on the shape's bounding box only."""
    if geom_px.is_empty:
        return
    bx0, by0, bx1, by1 = geom_px.bounds
    x0, y0 = max(0, int(bx0) - 3 * blur_px - 2), max(0, int(by0) - 3 * blur_px - 2)
    x1, y1 = min(W, int(bx1) + 3 * blur_px + 2), min(H, int(by1) + 3 * blur_px + 2)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for q in G.polys_of(affinity.translate(geom_px, -x0, -y0)):
        m = np.maximum(m, G.poly_mask(q, x1 - x0, y1 - y0))
    ks = 2 * blur_px + 1
    a = (cv2.GaussianBlur(m, (ks, ks), 0).astype(np.float32) / 255.0 * (1.0 - k))[..., None]
    sub = canvas[y0:y1, x0:x1]
    sub[:] = (sub.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def _annotations(labelled, S, W, H, image_id, mode, colour):
    fam_ids, anns = {}, []
    min_area = max(400, (0.02 * S) ** 2 * 100)
    for fam, p in labelled:
        for q in G.polys_of(p.buffer(0)):
            if q.area < min_area:
                continue
            q = q.simplify(0.5)
            if q.is_empty or not isinstance(q, Polygon):
                continue
            fam_ids.setdefault(fam, len(fam_ids) + 1)
            outer = [round(v, 2) for c in q.exterior.coords[:-1] for v in c]
            holes = [[round(v, 2) for c in ring.coords[:-1] for v in c] for ring in q.interiors]
            holes = [hh for hh in holes if len(hh) >= 6]
            bx0, by0, bx1, by1 = q.bounds
            anns.append({"id": len(anns) + 1, "category_id": fam_ids[fam], "category_name": f"pattern{fam_ids[fam]}",
                         "family": fam, "segmentation": [outer] + holes, "num_holes": len(holes),
                         "bbox": [round(bx0, 1), round(by0, 1), round(bx1 - bx0, 1), round(by1 - by0, 1)],
                         "area": round(q.area, 1)})
    return {"image": {"file_name": f"synth6_{image_id:06d}.png", "width": W, "height": H},
            "mode": mode, "appearance": "colour" if colour else "mono_normal",
            "px_per_ft": round(S, 2), "annotations": anns, "render": "revit"}


def _finish(canvas, tq, r):
    canvas = tq.flush(canvas)
    if r.random() < G.JPEG_PROB:
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, r.randint(60, 92)])
        if ok:
            canvas = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return canvas


ROOF_NOTES = ["COMPOSITE SHINGLE ROOFING, TYP.", "GUTTER, TYP.", "(N) CLASS A ROOF", "STANDING SEAM METAL ROOF",
              "ASPHALT SHINGLES TO MATCH (E)", "NEW PITCHED ROOF TO MATCH (E)", "ROOF DRAIN INSIDE WALL BELOW",
              "LINE OF WALL BELOW", "SKYLIGHT, SEE SPEC", "OVERFLOW SCUPPER", "CRICKET"]
DARK_ROOF = [(122, 34, 34), (64, 64, 68), (92, 72, 58), (44, 44, 48), (104, 44, 38), (60, 72, 84)]
TINT_ROOF = [(196, 160, 140), (150, 160, 172), (172, 152, 124), (140, 150, 136), (205, 190, 170), (170, 120, 110)]


# ----------------------------------------------------------------------------
# roof materials
# ----------------------------------------------------------------------------
PITCHED_KINDS = ["bond_facet", "bond_global", "eave_lines", "seam", "cross", "dots", "t_zinc", "t_brick"]
PITCHED_W = [30, 9, 17, 15, 8, 5, 8, 8]


def roof_material(r, avoid, seed):
    """One roof family's look. `avoid`: another family's material this one must differ from."""
    kinds, wts = PITCHED_KINDS, PITCHED_W
    if avoid is not None:
        kw = [(k, w) for k, w in zip(kinds, wts) if k != avoid["kind"]]
        kinds, wts = [k for k, _ in kw], [w for _, w in kw]
    kind = r.choices(kinds, weights=wts)[0]
    style = r.choices(["white", "grey", "tint", "dark"], weights=[50, 15, 17, 18])[0]
    if avoid is not None and style == avoid["style"] and r.random() < 0.6:
        style = r.choice([x for x in ["white", "grey", "tint", "dark"] if x != avoid["style"]])
    base = {"white": (255, 255, 255), "grey": (r.randint(222, 246),) * 3,
            "tint": r.choice(TINT_ROOF), "dark": r.choice(DARK_ROOF)}[style]
    lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    line = G.mix(base, (255, 255, 255), r.uniform(0.2, 0.4)) if lum < 110 else \
        ((r.randint(40, 150),) * 3 if style in ("white", "grey") else G.mix(base, (0, 0, 0), r.uniform(0.35, 0.55)))
    return {"kind": kind, "style": style, "base": base, "line": line, "seed": seed,
            "row": r.uniform(0.35, 0.8), "unit": r.uniform(0.9, 1.8), "sp": r.uniform(0.45, 1.4),
            "seam_sp": r.uniform(1.0, 2.0), "cross_sp": r.uniform(1.2, 3.0), "dot_sp": r.uniform(0.6, 1.3),
            "dot_fill": r.random() < 0.5, "lwk": r.uniform(0.55, 0.9),
            "tile": r.uniform(*TILES[kind][3]) if kind in TILES else 0.0}


ROOF_LOOK = {"bond_facet": "bond", "bond_global": "bond", "eave_lines": "lines", "seam": "lines",
             "flat_seams": "lines", "cross": "diag", "dots": "dots", "membrane": "dots", "gravel": "stipple",
             "t_zinc": "diag", "t_brick": "bond", "t_aggregate": "aggregate", "t_glass": "dashes"}
ROOF_SP = {"bond_facet": "row", "bond_global": "row", "eave_lines": "sp", "seam": "seam_sp", "flat_seams": "seam_sp",
           "cross": "cross_sp", "dots": "dot_sp", "membrane": "dot_sp", "gravel": "density",
           "t_zinc": "tile", "t_brick": "tile", "t_aggregate": "tile", "t_glass": "tile"}


def _roof_eff(m):
    v = m[ROOF_SP[m["kind"]]]
    return v * TILES[m["kind"]][2] if m["kind"] in TILES else v


def roof_separate(m, others):
    """Same rule as the floors: a roof family of the same pattern type as another (direction
    ignored: eave lines vs seams, per-facet vs global bond) must differ in spacing or tone."""
    key = ROOF_SP[m["kind"]]
    for _ in range(4):
        clash = None
        for o in others:
            if ROOF_LOOK[o["kind"]] != ROOF_LOOK[m["kind"]]:
                continue
            if abs(_lum(o["base"]) - _lum(m["base"])) >= 30 or math.dist(o["base"], m["base"]) >= 60:
                continue
            a, b = _roof_eff(m), _roof_eff(o)
            if max(a, b) / min(a, b) < 1.4:
                clash = b
                break
        if clash is None:
            return
        f = TILES[m["kind"]][2] if m["kind"] in TILES else 1.0
        m[key] = (clash * 1.6 if _roof_eff(m) >= clash else clash / 1.6) / f


def flat_material(r, seed):
    kind = r.choices(["gravel", "membrane", "flat_seams", "t_aggregate", "t_glass"], weights=[32, 28, 20, 12, 8])[0]
    base = (r.randint(236, 252),) * 3 if r.random() < 0.7 else (255, 255, 255)
    return {"kind": kind, "style": "grey", "base": base, "line": (r.randint(90, 160),) * 3, "seed": seed,
            "density": r.uniform(1.0, 3.0), "dot_sp": r.uniform(0.9, 1.8), "dot_fill": r.random() < 0.4,
            "seam_sp": r.uniform(2.5, 4.5), "seam_ang": r.choice([0, 90]), "lwk": r.uniform(0.5, 0.8),
            "tile": r.uniform(*TILES[kind][3]) if kind in TILES else 0.0}


def _dotgrid(layer, ox, oy, S, sp, rpx, colour, filled, lw, jit=0.0, seed=0):
    """Staggered dot grid; jit > 0 (plan v3) moves each dot up to jit*spacing and varies its size,
    hashed on its grid cell so neighbouring regions with the same material line up."""
    h, w = layer.shape[:2]
    step = sp * S
    if step < 3:
        return
    j0, j1 = int(math.floor(oy / step)) - 1, int(math.ceil((oy + h) / step)) + 1
    for j in range(j0, j1):
        off = (j % 2) * step / 2
        i0, i1 = int(math.floor((ox - off) / step)) - 1, int(math.ceil((ox + w - off) / step)) + 1
        for i in range(i0, i1):
            c = (i * step + off - ox, j * step - oy)
            rr = rpx
            if jit > 0:
                hr = random.Random(hash((i, j, seed)) & 0xFFFFFFFF)
                c = (c[0] + hr.uniform(-jit, jit) * step, c[1] + hr.uniform(-jit, jit) * step)
                rr = rpx * hr.uniform(0.6, 1.4)
            cv2.circle(layer, G._pt(c), int(rr * G.SCALE), G.bgr(colour), -1 if filled else max(1, int(lw)),
                       cv2.LINE_AA, G.SHIFT)


def draw_material(canvas, poly_px, m, ang, tone, S, W, H, lw_thin):
    """Paint material `m` into poly_px; `ang` = eave direction (deg, y-down) of the facet."""
    if poly_px.is_empty:
        return
    minx, miny, maxx, maxy = poly_px.bounds
    x0, y0 = max(0, int(math.floor(minx)) - 2), max(0, int(math.floor(miny)) - 2)
    x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    base = tuple(int(v * tone) for v in m["base"])
    line = tuple(int(v * tone) for v in m["line"])
    lw = max(0.8, lw_thin * m["lwk"])
    k = m["kind"]
    if k == "bond_facet":
        # running-bond courses parallel to this facet's eave: draw horizontal, rotate
        D = int(math.hypot(w, h)) + 8
        big = np.empty((D, D, 3), np.uint8)
        big[:] = G.bgr(base)
        G._courses(big, 0, 0, S, m["row"], m["unit"], line, lw, m["seed"], stagger=0.5)
        M = cv2.getRotationMatrix2D((D / 2, D / 2), -ang, 1.0)
        big = cv2.warpAffine(big, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        layer = np.ascontiguousarray(big[(D - h) // 2:(D - h) // 2 + h, (D - w) // 2:(D - w) // 2 + w])
    else:
        layer = np.empty((h, w, 3), np.uint8)
        layer[:] = G.bgr(base)
        if k == "bond_global":
            G._courses(layer, x0, y0, S, m["row"], m["unit"], line, lw, m["seed"], stagger=0.5)
        elif k == "eave_lines":
            G._dlines(layer, x0, y0, S, m["sp"], ang, line, lw)
        elif k == "seam":
            G._dlines(layer, x0, y0, S, m["seam_sp"], ang + 90, line, lw)
        elif k == "cross":
            G._dlines(layer, x0, y0, S, m["cross_sp"], 45, line, lw)
            G._dlines(layer, x0, y0, S, m["cross_sp"], -45, line, lw)
        elif k == "dots":
            _dotgrid(layer, x0, y0, S, m["dot_sp"], max(1.2, 0.09 * S), line, m["dot_fill"], lw)
        elif k == "gravel":
            G._stipple(layer, x0, y0, S, m["density"], line, m["seed"], r_px=1)
        elif k == "membrane":
            _dotgrid(layer, x0, y0, S, m["dot_sp"], max(1.2, 0.07 * S), line, m["dot_fill"], lw)
        elif k == "flat_seams":
            G._dlines(layer, x0, y0, S, m["seam_sp"], m["seam_ang"], line, lw)
        elif k in TILES:
            # brick-bond tiles turn with the facet's eave, like the drawn shingle bond
            hatch_layer(layer, x0, y0, S, k, m["tile"], line, ang if k == "t_brick" else 0.0)
    mask = G.poly_mask(affinity.translate(poly_px, -x0, -y0), w, h)
    G.blend_mask(canvas[y0:y1, x0:x1], layer, mask)



# ----------------------------------------------------------------------------
# roof plan
# ----------------------------------------------------------------------------
def _bpoly(b):
    """A spec block's footprint: its polygon (--shaped) or its rectangle."""
    return Polygon(b["poly"]) if b.get("poly") else box(b["x0"], b["y0"], b["x0"] + b["w"], b["y0"] + b["d"])


def compose_roof(image_id, seed, mode_weights):
    house = _house(image_id, seed, mode_weights)
    r = random.Random(seed * 1_000_003 + image_id * 181 + 53)
    spec, faces = FC._FACES[FC._CUR[0]]
    chim = spec.get("chimney")
    tops = top_faces(faces, chim)
    # a flat face is a flat ROOF only over a flat-roofed block; elsewhere it is a
    # sliver where pitched roofs meet and is drawn / labelled with the roof
    flat_fp = unary_union([_bpoly(b).buffer(b["ov"] + 0.1, join_style=2) for b in spec["blocks"] if b["roof"] == "flat"])
    for t in tops:
        if t["kind"] == "flat" and (flat_fp.is_empty or not flat_fp.contains(t["vis"].representative_point())):
            t["kind"] = "pitched"
    if not any(t["kind"] in ("pitched", "flat") for t in tops):
        raise RuntimeError("no roof faces")
    footprint = unary_union([_bpoly(b) for b in spec["blocks"]])
    ink = (r.randint(0, 30),) * 3

    # ---- roof materials: every labelled roof carries a texture (Gemini + real roof plans,
    # 2026-09-29): running-bond shingle courses turning with each facet's eave is the
    # commonest; then eave-parallel lines, down-slope standing seams, global shingle
    # bond, diamond crosshatch, dot grids; flat roofs gravel stipple / membrane dots /
    # seam lines. Gemini sheets often carry 2-4 roof families (a wing in another
    # material, flat roofs, skylights), real Revit sheets usually one.
    fam_seed = r.randrange(1 << 30)
    sun3 = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0.8, 1.6)])
    sun3 /= np.linalg.norm(sun3)
    shade_facets = r.random() < 0.2                       # Revit sun-shaded tone ON TOP of the pattern
    main = roof_material(r, None, fam_seed)
    pitched_blocks = [i for i, b in enumerate(spec["blocks"]) if b["roof"] != "flat"]
    alt_block = None
    if len(pitched_blocks) >= 2 and r.random() < 0.35:
        alt_block = r.choice(pitched_blocks[1:])
        alt = roof_material(r, main, fam_seed + 11)
        roof_separate(alt, [main])
    fmat = flat_material(r, fam_seed + 7)
    roof_separate(fmat, [main] + ([alt] if alt_block is not None else []))
    flat_label = r.random() < 0.6
    if not any(t["kind"] == "pitched" for t in tops):          # all-flat house: the flat roof is the question
        flat_label = True
    skylights_on = r.random() < 0.35
    sky_look = r.choice(["double", "x", "glass"])
    colourful = main["base"] != (255, 255, 255) and max(main["base"]) - min(main["base"]) > 12
    block_of = {}
    for t in tops:
        c = t["poly"].representative_point()
        for i, bl in enumerate(spec["blocks"]):
            if _bpoly(bl).buffer(bl["ov"] + 0.05, join_style=2).contains(c):
                block_of[id(t)] = i
                if i != 0:
                    break

    # ---- building copies (HF14 15/16: several buildings on one roof plan)
    ncopy = r.choices([1, 2, 3], weights=[70, 20, 10])[0]
    insts = [(0, 0.0, 0.0)]
    bx0, by0, bx1, by1 = unary_union([t["vis"] for t in tops] + [footprint]).bounds
    cur = (bx0, by0, bx1, by1)
    for k in range(1, ncopy):
        rot = r.choice([0, 90, 180, 270])
        g = _xf(box(bx0, by0, bx1, by1), rot, 0, 0)
        gx0, gy0, gx1, gy1 = g.bounds
        gap = r.uniform(6, 18)
        if r.random() < 0.6:
            dx, dy = cur[2] + gap - gx0, cur[1] + r.uniform(-0.3, 0.3) * (by1 - by0) - gy0
        else:
            dx, dy = cur[0] + r.uniform(-0.2, 0.4) * (bx1 - bx0) - gx0, cur[3] + gap - gy0
        insts.append((rot, dx, dy))
        cur = unary_union([box(*cur), _xf(box(bx0, by0, bx1, by1), rot, dx, dy)]).bounds

    # ---- per-instance geometry (feet)
    geo = []
    for (rot, dx, dy) in insts:
        items = []
        for t in tops:
            v = _xf(t["vis"], rot, dx, dy)
            n2 = _xv((t["n"][0], t["n"][1]), rot)
            items.append((t, v, n2))
        geo.append({"items": items, "fp": _xf(footprint, rot, dx, dy), "rot": rot, "dx": dx, "dy": dy})
    allg = unary_union([it[1] for g in geo for it in g["items"]] + [g["fp"] for g in geo])
    ex0, ey0, ex1, ey1 = allg.bounds

    # trellis / louver slats next to the house: unlabelled parallel lines (HF14 16)
    extras = []
    for g in geo:
        if r.random() < 0.3:
            fx0, fy0, fx1, fy1 = g["fp"].bounds
            tw, td = r.uniform(8, 18), r.uniform(6, 12)
            side = r.choice(["N", "S", "E", "W"])
            if side in "NS":
                x = r.uniform(fx0, max(fx0, fx1 - tw))
                rr = box(x, fy0 - td - 1, x + tw, fy0 - 1) if side == "N" else box(x, fy1 + 1, x + tw, fy1 + td + 1)
            else:
                y = r.uniform(fy0, max(fy0, fy1 - td))
                rr = box(fx0 - tw - 1, y, fx0 - 1, y + td) if side == "W" else box(fx1 + 1, y, fx1 + tw + 1, y + td)
            if not rr.intersects(allg):
                extras.append(("trellis", rr, r.choice([0, 90]), r.uniform(0.4, 0.9)))
    if extras:
        allg = unary_union([allg] + [e[1] for e in extras])
        ex0, ey0, ex1, ey1 = allg.bounds

    # ---- layout
    grid_on = r.random() < 0.65
    pad = r.uniform(7, 11) if grid_on else r.uniform(3, 6)
    bot = r.uniform(7, 10)
    total_w, total_h = (ex1 - ex0) + 2 * pad, (ey1 - ey0) + pad + bot + 2
    S = min(130.0, r.uniform(*G.LONG_SIDE_PX) / max(total_w, total_h))
    W, H = int(total_w * S), int(total_h * S)
    V = G.View(S, (pad - ex0) * S, (pad - ey0) * S)
    canvas = np.full((H, W, 3), 255, np.uint8)
    tq = G.TextQueue()
    size = max(12, max(W, H) * r.uniform(0.0045, 0.0075))
    lw_thin = max(1.0, S * r.uniform(0.009, 0.015))
    lw_heavy = max(1.6, S * r.uniform(0.022, 0.036))
    app = {"ink": ink, "text_px": size, "outline_lw": lw_thin}

    # ---- roof surfaces
    labelled = []
    all_skylights = []
    for g in geo:
        pitched_polys, alt_polys = [], []
        for (t, v, n2) in g["items"]:
            if t["kind"] != "pitched":
                continue
            nn = math.hypot(*n2)
            ang = math.degrees(math.atan2(n2[0], -n2[1])) if nn > 1e-6 else 0.0
            is_alt = alt_block is not None and block_of.get(id(t)) == alt_block
            mat = alt if is_alt else main
            tone = 1.0
            if shade_facets:
                nv = np.array(t["n"], float)
                nv[:2] = n2
                nv /= np.linalg.norm(nv)
                tone = 0.72 + 0.28 * max(0.0, float(nv @ sun3))
            draw_material(canvas, V.geom(v), mat, ang, tone, S, W, H, lw_thin)
            (alt_polys if is_alt else pitched_polys).append(v)
        for (t, v, n2) in g["items"]:
            if t["kind"] == "flat":
                draw_material(canvas, V.geom(v), fmat, 0.0, 1.0, S, W, H, lw_thin)
                # parapet line + crickets to a drain
                inner = v.buffer(-r.uniform(0.5, 1.0), join_style=2)
                if not inner.is_empty:
                    G.cv_outline(canvas, V.geom(inner), ink, lw_thin)
                    if r.random() < 0.7:
                        c = inner.representative_point()
                        for q in G.polys_of(inner):
                            for corner in list(q.exterior.coords)[:-1]:
                                G.cv_line(canvas, V.px(*corner), V.px(c.x, c.y), G.mix(ink, (255, 255, 255), 0.35), lw_thin * 0.8)
                        cv2.circle(canvas, G._pt(V.px(c.x, c.y)), int(0.35 * S * G.SCALE), G.bgr(ink),
                                   max(1, int(lw_thin)), cv2.LINE_AA, G.SHIFT)
                if flat_label:
                    labelled.append(("flat", V.geom(v)))
            elif t["kind"] == "chimney":
                G.cv_fill(canvas, V.geom(v), (255, 255, 255))
                G.cv_outline(canvas, V.geom(v), ink, lw_thin * 1.3)
                fl = v.buffer(-0.5, join_style=2)
                if not fl.is_empty:
                    G.cv_outline(canvas, V.geom(fl), ink, lw_thin)
        if alt_polys:
            labelled.append(("roof2", V.geom(unary_union(alt_polys))))
        # skylights on big facets
        sky = []
        if skylights_on:
            for p in pitched_polys:
                if p.area > 120 and r.random() < 0.35:
                    c = p.representative_point()
                    sw, sd = r.uniform(2, 3.5), r.uniform(3, 5)
                    s = box(c.x - sw / 2, c.y - sd / 2, c.x + sw / 2, c.y + sd / 2)
                    if p.buffer(-0.3).contains(s):
                        sky.append(s)
        for s in sky:
            G.cv_fill(canvas, V.geom(s), (255, 255, 255))
            G.cv_outline(canvas, V.geom(s), ink, lw_thin * 1.3)
            q = s.buffer(-0.35, join_style=2)
            G.cv_outline(canvas, V.geom(q), ink, lw_thin)
            x0s, y0s, x1s, y1s = q.bounds
            if sky_look == "x":
                G.cv_line(canvas, V.px(x0s, y0s), V.px(x1s, y1s), ink, lw_thin * 0.8)
                G.cv_line(canvas, V.px(x0s, y1s), V.px(x1s, y0s), ink, lw_thin * 0.8)
            elif sky_look == "glass":
                for f in (0.35, 0.5):
                    G.cv_line(canvas, V.px(x0s + (x1s - x0s) * f, y0s + (y1s - y0s) * (1 - f) * 0.9),
                              V.px(x0s + (x1s - x0s) * (f + 0.3), y0s + (y1s - y0s) * (0.6 - f) * 0.9), ink, lw_thin * 0.7)
        all_skylights += sky
        if pitched_polys:
            lab = unary_union(pitched_polys)
            if sky:
                lab = lab.difference(unary_union(sky))
            labelled.append(("roof", V.geom(lab)))
        # edges: every facet (ridges / hips / valleys), heavier roof outline
        for (t, v, n2) in g["items"]:
            G.cv_outline(canvas, V.geom(v), ink, lw_thin)
        outline = unary_union([it[1] for it in g["items"]])
        for p in G.polys_of(outline):
            G.cv_polyline(canvas, list(V.geom(p).exterior.coords), ink, lw_heavy, closed=True)
        if r.random() < 0.35:                                  # gutter line
            for p in G.polys_of(outline.buffer(0.35, join_style=2)):
                G.cv_polyline(canvas, list(V.geom(p).exterior.coords), ink, lw_thin * 0.8, closed=True)
        if r.random() < 0.7:                                   # walls below
            wcol = G.mix(ink, (255, 255, 255), r.uniform(0.15, 0.5))
            _dash_poly(canvas, V.geom(g["fp"].intersection(outline)), wcol, lw_thin,
                       [(0.6 * S, 0.3 * S)])

    # grid lines over the roof (HF14 1)
    if grid_on:
        g0 = geo[0]
        xs = _merge_close([v for b in spec["blocks"] for v in (b["x0"], b["x0"] + b["w"])], 3.0)
        ys = _merge_close([v for b in spec["blocks"] for v in (b["y0"], b["y0"] + b["d"])], 3.0)
        pts = [_xp((x, 0), g0["rot"], g0["dx"], g0["dy"]) for x in xs]
        xs2 = _merge_close([p[0] for p in pts], 3.0)
        pts = [_xp((0, y), g0["rot"], g0["dx"], g0["dy"]) for y in ys]
        ys2 = _merge_close([p[1] for p in pts], 3.0)
        if g0["rot"] in (90, 270):
            xs2 = _merge_close([_xp((0, y), g0["rot"], g0["dx"], g0["dy"])[0] for y in ys], 3.0)
            ys2 = _merge_close([_xp((x, 0), g0["rot"], g0["dx"], g0["dy"])[1] for x in xs], 3.0)
        e = (ex0 - pad + 2.2, ey0 - pad + 2.2, ex1 + pad - 2.2, ey1 + pad * 0.6)
        _grid(canvas, tq, V, xs2, ys2, e, ink, lw_thin * 0.9, size, S, r)

    for (kind, rr, ang, spc) in extras:
        G.draw_fill(canvas, V.geom(rr), G.Style("hatch", (255, 255, 255), (110, 110, 110), lw_thin * 0.8,
                                                {"sp": spc, "angle": ang}, fam_seed + 9, "trellis"), S, W, H)
        G.cv_outline(canvas, V.geom(rr), ink, lw_thin * 1.3)
        for c in list(rr.exterior.coords)[:4]:
            G.cv_outline(canvas, V.geom(box(c[0] - 0.4, c[1] - 0.4, c[0] + 0.4, c[1] + 0.4)), ink, lw_thin)
        if r.random() < 0.6:
            q = rr.representative_point()
            G.draw_leader(canvas, tq, V, (q.x, q.y), r.choice(["WOOD SLAT TRELLIS", "LOUVER SYSTEM",
                                                               "(N) PERGOLA ABOVE", "ALUM. LOUVERS"]), app, S, 1)

    # ---- annotation on the roof
    slope_txt = r.choice(["rev", "ratio", "word"])
    for g in geo:
        for (t, v, n2) in g["items"]:
            if t["kind"] != "pitched" or v.area < 30 or r.random() < 0.3:
                continue
            nn = math.hypot(*n2)
            if nn < 1e-6:
                continue
            d = (n2[0] / nn, n2[1] / nn)
            c = v.representative_point()
            L = r.uniform(2.0, 3.5)
            a, b = V.px(c.x - d[0] * L / 2, c.y - d[1] * L / 2), V.px(c.x + d[0] * L / 2, c.y + d[1] * L / 2)
            pitch12 = max(1, round(12 * nn / max(t["n"][2], 1e-3)))
            txt = {"rev": f'{pitch12}" / 1\'-0"', "ratio": f"{pitch12}:12", "word": f"SLOPE {pitch12}:12"}[slope_txt]
            _slope_arrow(canvas, tq, a, b, txt, ink, lw_thin, size * 0.8)
    if r.random() < 0.3:
        for (t, v, n2) in geo[0]["items"]:
            if t["kind"] != "pitched":
                continue
            for (name, a, b) in ridge_segments(t)[:2]:
                a2, b2 = _xp(a, geo[0]["rot"], geo[0]["dx"], geo[0]["dy"]), _xp(b, geo[0]["rot"], geo[0]["dx"], geo[0]["dy"])
                m = V.px((a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2)
                ang = math.degrees(math.atan2(-(b2[1] - a2[1]), b2[0] - a2[0]))
                if abs(ang) > 90:
                    ang = ang - 180 if ang > 0 else ang + 180
                word = name if name == "RIDGE" else r.choice(["HIP", "VALLEY"])
                if abs(ang) < 3:
                    tq.add((m[0], m[1] - size * 0.3), word, size * 0.9, ink, anchor="ms")
                elif r.random() < 0.5:
                    tq.add((m[0], m[1]), word, size * 0.9, ink, angle=ang)
    if r.random() < 0.25:
        for _ in range(r.randint(2, 7)):
            g = r.choice(geo)
            t, v, _ = r.choice(g["items"])
            c = v.representative_point()
            p = V.px(c.x + r.uniform(-2, 2), c.y + r.uniform(-2, 2))
            txt = f"{r.uniform(5, 30):.1f}" if r.random() < 0.6 else f"{r.randint(100, 240)}"
            cv2.ellipse(canvas, G._pt(p), (int(size * 1.1 * G.SCALE), int(size * 0.55 * G.SCALE)), 0, 0, 360,
                        G.bgr((255, 255, 255)), -1, cv2.LINE_AA, G.SHIFT)
            cv2.ellipse(canvas, G._pt(p), (int(size * 1.1 * G.SCALE), int(size * 0.55 * G.SCALE)), 0, 0, 360,
                        G.bgr(ink), max(1, int(lw_thin * 0.8)), cv2.LINE_AA, G.SHIFT)
            tq.add(p, txt, size * 0.65, ink, anchor="mm")
    if r.random() < 0.5:
        for _ in range(r.randint(1, 4)):
            rad = size * r.uniform(1.0, 1.3)
            side = r.choice(["N", "S", "E", "W"])
            if side in "NS":
                x = r.uniform(ex0, ex1)
                c = V.px(x, ey0 - pad * 0.45 if side == "N" else ey1 + pad * 0.45)
                d = (0, 1 if side == "N" else -1)
            else:
                y = r.uniform(ey0, ey1)
                c = V.px(ex0 - pad * 0.45 if side == "W" else ex1 + pad * 0.45, y)
                d = (1 if side == "W" else -1, 0)
            _section_head(canvas, tq, c, rad, d, str(r.randint(1, 6)), r.choice(["A5.1", "A5.2", "A2.4", "A4-1", "A3.1"]),
                          ink, lw_thin, size * 0.75)
    for _ in range(r.choices([0, 1, 2, 3], weights=[35, 30, 25, 10])[0]):
        g = r.choice(geo)
        t, v, _ = r.choice([it for it in g["items"] if it[0]["kind"] == "pitched"] or g["items"])
        q = v.representative_point()
        G.draw_leader(canvas, tq, V, (q.x, q.y), r.choice(ROOF_NOTES), app, S, 1 if q.x > (ex0 + ex1) / 2 else -1)
    if r.random() < 0.3:
        _dim_chain(canvas, tq, [ex0, ex1], "x", ey0 - r.uniform(1.5, 3), V, ink, lw_thin * 0.8, size * 0.8, S)
    if r.random() < 0.5:
        _north(canvas, tq, V.px(ex1 + pad * 0.5, ey1 + bot * 0.35), size * 1.3, ink, lw_thin, size)
    if r.random() < 0.3:
        _notes(tq, W * r.uniform(0.72, 0.8), H * r.uniform(0.75, 0.85), "SHEET NOTES",
               r.sample(["ALL AREAS OF FLAT ROOF TO SLOPE TO DRAIN AT MIN. 1/4\" / 1'",
                         "VERIFY ALL DIMENSIONS IN FIELD.", "PROVIDE FLASHING AT ALL ROOF PENETRATIONS.",
                         "GUTTERS AND DOWNSPOUTS PER CIVIL."], 2), ink, size * 0.8)
    _title(canvas, tq, V, ex0, ey1 + bot * 0.55, ex0 + (ex1 - ex0) * r.uniform(0.3, 0.6),
           r.choice(["ROOF PLAN", "ROOF PLAN", "PROPOSED ROOF PLAN", "(N) ROOF PLAN"]), r, ink, lw_thin, size)
    canvas = _finish(canvas, tq, r)
    return canvas, _annotations(labelled, S, W, H, image_id, "roof_plan", colourful)


# ----------------------------------------------------------------------------
# floor plan
# ----------------------------------------------------------------------------
HARD_NOTES = {"ashlar": ["FLAGSTONE PATIO", "STONE PAVERS", "RANDOM ASHLAR PAVERS"],
              "stone": ["STONE PATIO", "FLAGSTONE"], "grid": ["CONC. PAVERS", "PAVER WALK", "24x24 PAVERS"],
              "plank": ["(N) WOOD DECK", "COMPOSITE DECKING", "DECK"], "concrete": ["CONC. WALK", "CONCRETE PATIO"],
              "basket": ["BRICK PAVERS", "PAVERS, BASKETWEAVE"]}
MEP_NOTES = ["wall supply", "refer to specs for exact outlet location", "(n) clg. return air grille",
             "route supply duct btw (n) roof rafters", "mech soffit as needed", "skylight above",
             "verify location with architect", "wood soffit, typ."]


def _hard_style(kind, r, lw, seed, colour):
    base = (255, 255, 255)
    if colour:
        base = r.choice([(214, 206, 196), (200, 200, 196), (226, 218, 204), (190, 176, 160)])
    line = (r.randint(60, 150),) * 3 if not colour else G.mix(base, (0, 0, 0), 0.45)
    if kind == "ashlar":
        return G.Style("ashlar", base, line, lw, {"unit": r.uniform(0.9, 1.6)}, seed, "hard")
    if kind == "stone":
        return G.Style("rubble", base, line, lw, {}, seed, "hard")
    if kind == "grid":
        return G.Style("grid", base, line, lw, {"sp": r.uniform(1.0, 2.0)}, seed, "hard")
    if kind == "plank":
        return G.Style("hatch", base, line, lw, {"sp": r.uniform(0.35, 0.6), "angle": 0}, seed, "hard")
    if kind == "basket":
        return G.Style("basket", base, line, lw, {"sp": r.uniform(0.8, 1.4)}, seed, "hard")
    if kind in TILES:
        return G.Style(kind, base, line, lw, {"sp": r.uniform(*TILES[kind][3]), "angle": r.choice([0, 90])}, seed, "hard")
    return G.Style("concrete", base, line, lw, {"density": r.uniform(0.8, 1.8)}, seed, "hard")


def _fill(canvas, poly_px, st, S, W, H):
    if st.kind in TILES:
        minx, miny, maxx, maxy = poly_px.bounds
        x0, y0 = max(0, int(minx) - 2), max(0, int(miny) - 2)
        x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        layer = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
        layer[:] = G.bgr(st.base)
        hatch_layer(layer, x0, y0, S, st.kind, st.params["sp"], st.line,
                    st.params.get("angle", 0) if st.kind in ("t_hbone", "t_brick") else 0)
        m = G.poly_mask(affinity.translate(poly_px, -x0, -y0), x1 - x0, y1 - y0)
        G.blend_mask(canvas[y0:y1, x0:x1], layer, m)
        return
    if st.kind == "basket":
        layer_st = G.Style("grid", st.base, st.line, st.lw, {"sp": st.params["sp"]}, st.seed, st.label_name)
        minx, miny, maxx, maxy = poly_px.bounds
        x0, y0 = max(0, int(minx) - 2), max(0, int(miny) - 2)
        x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        layer = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
        layer[:] = G.bgr(st.base)
        G._basketweave(layer, x0, y0, S, st.params["sp"], st.line, st.lw)
        m = G.poly_mask(affinity.translate(poly_px, -x0, -y0), x1 - x0, y1 - y0)
        G.blend_mask(canvas[y0:y1, x0:x1], layer, m)
        return layer_st
    G.draw_fill(canvas, poly_px, st, S, W, H)


# ---- floor layout: rooms in EVERY block (v6's build_floor_plan only splits the main one)
WET = ("BATH", "LAUNDRY", "POWDER", "MUD", "UTIL")
LIVING = ("LIVING", "DINING", "KITCHEN", "HALL", "ENTRY", "FOYER", "FAMILY", "GREAT", "PANTRY")
BED = ("BEDROOM", "OFFICE", "STUDY", "BONUS", "W.I.C.", "CLOSET")


def _zone(name):
    if name.startswith("GARAGE"):
        return "garage"
    if any(k in name for k in WET):
        return "wet"
    if any(k in name for k in LIVING):
        return "living"
    return "bed"


def _convex_corners(poly):
    """Convex 90-degree corners of a polygon's exterior: (corner, dir_a, dir_b) with unit
    directions along the two edges leaving the corner."""
    from shapely.geometry.polygon import orient
    pts = list(orient(poly, 1.0).exterior.coords)[:-1]
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        a = (p0[0] - p1[0], p0[1] - p1[1])
        b = (p2[0] - p1[0], p2[1] - p1[1])
        la, lb = math.hypot(*a), math.hypot(*b)
        if la < 1e-6 or lb < 1e-6:
            continue
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross > 0 and abs(a[0] * b[0] + a[1] * b[1]) < 1e-6 * la * lb:
            out.append((p1, (a[0] / la, a[1] / la), (b[0] / lb, b[1] / lb), la, lb))
    return out


def shaped_footprint(blocks, r):
    """Union of the blocks, then notches, chamfered corners, 45-degree bays and a U
    courtyard, so plans are not always unions of rectangles."""
    foot = unary_union([G.rect(b.x0, b.y0, b.x1, b.y1) for b in blocks])

    def ok(g):
        return isinstance(g, Polygon) and g.is_valid and g.area > 0.6 * foot.area and \
            g.buffer(-4.0, join_style=2).geom_type == "Polygon"
    if r.random() < 0.35:                                         # notch a convex corner
        cs = [c for c in _convex_corners(foot) if c[3] > 14 and c[4] > 14]
        if cs:
            p, da, db, la, lb = r.choice(cs)
            u, v = r.uniform(0.2, 0.4) * la, r.uniform(0.2, 0.4) * lb
            q = Polygon([p, (p[0] + da[0] * u, p[1] + da[1] * u),
                         (p[0] + da[0] * u + db[0] * v, p[1] + da[1] * u + db[1] * v),
                         (p[0] + db[0] * v, p[1] + db[1] * v)])
            g = foot.difference(q)
            foot = g if ok(g) else foot
    for _ in range(r.choice([0, 0, 1, 2])):                       # chamfer corners
        cs = [c for c in _convex_corners(foot) if c[3] > 10 and c[4] > 10]
        if cs:
            p, da, db, la, lb = r.choice(cs)
            d = r.uniform(3, 6)
            g = foot.difference(Polygon([p, (p[0] + da[0] * d, p[1] + da[1] * d), (p[0] + db[0] * d, p[1] + db[1] * d)]))
            foot = g if ok(g) else foot
    if r.random() < 0.3:                                          # 45-degree bay on a long wall
        from shapely.geometry.polygon import orient
        pts = list(orient(foot, 1.0).exterior.coords)
        segs = [(a, b) for a, b in zip(pts[:-1], pts[1:]) if math.dist(a, b) >= 14 and
                (abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6)]
        if segs:
            a, b = r.choice(segs)
            L = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            nx, ny = uy, -ux                                     # outward for a CCW ring
            w, dpt = r.uniform(7, min(12, L - 4)), r.uniform(2.5, 4)
            t0 = r.uniform(2, L - 2 - w)
            p0 = (a[0] + ux * t0, a[1] + uy * t0)
            p3 = (p0[0] + ux * w, p0[1] + uy * w)
            p1 = (p0[0] + ux * dpt + nx * dpt, p0[1] + uy * dpt + ny * dpt)
            p2 = (p3[0] - ux * dpt + nx * dpt, p3[1] - uy * dpt + ny * dpt)
            try:
                bay = Polygon([p0, p1, p2, p3])
                g = foot.union(bay) if bay.is_valid else foot
                if not ok(g) or g.area <= foot.area:
                    bay = Polygon([p0, (p0[0] + ux * dpt - nx * dpt, p0[1] + uy * dpt - ny * dpt),
                                   (p3[0] - ux * dpt - nx * dpt, p3[1] - uy * dpt - ny * dpt), p3])
                    g = foot.union(bay) if bay.is_valid else foot
                foot = g if ok(g) and g.area > foot.area else foot
            except Exception:                                  # degenerate bay: keep the footprint
                pass
    mb = blocks[0]
    if r.random() < 0.12 and min(mb.w, mb.d) > 34:                # U: courtyard into a long side
        if mb.w >= mb.d:
            cw, cd = r.uniform(0.3, 0.5) * mb.w, r.uniform(0.3, 0.45) * mb.d
            cx = r.uniform(mb.x0 + 10, mb.x1 - 10 - cw)
            q = G.rect(cx, mb.y1 - cd, cx + cw, mb.y1 + 1)
        else:
            cw, cd = r.uniform(0.3, 0.5) * mb.d, r.uniform(0.3, 0.45) * mb.w
            cy = r.uniform(mb.y0 + 10, mb.y1 - 10 - cw)
            q = G.rect(mb.x1 - cd, cy, mb.x1 + 1, cy + cw)
        g = foot.difference(q)
        foot = g if ok(g) else foot
    return foot


def _shared_seg(a, b):
    """Longest axis-aligned piece of the boundary two cells share, as bounds, or None."""
    sh = a.boundary.intersection(b.boundary)
    best = None
    for seg in G.lines_of(sh):
        c = list(seg.coords)
        for p, q in zip(c[:-1], c[1:]):
            if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
                continue
            L = math.dist(p, q)
            if best is None or L > best[1]:
                best = ((min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])), L)
    return best


def plan_layout(house, r):
    blocks = house["blocks"]
    t_ext, t_int = r.uniform(0.55, 0.85), r.uniform(0.35, 0.5)
    foot = shaped_footprint(blocks, r)
    inner_foot = foot.buffer(-t_ext, join_style=2)
    cells, taken = [], None
    for b in blocks:
        R = G.rect(b.x0, b.y0, b.x1, b.y1)
        reg = (R if taken is None else R.difference(taken)).intersection(foot)
        taken = R if taken is None else taken.union(R)
        for q in G.polys_of(reg):
            if q.area < 20:
                continue
            if b.kind == "garage":
                cells.append((q, "garage"))
                continue
            x0, y0, x1, y1 = q.bounds
            parts = [(x0, y0, x1, y1, "room")]
            # a corridor through the block (Gemini plans: halls connect the rooms)
            if min(x1 - x0, y1 - y0) >= 24 and r.random() < 0.45:
                hw = r.uniform(3.5, 4.5)
                if x1 - x0 >= y1 - y0:
                    yc = r.uniform(y0 + 0.35 * (y1 - y0), y0 + 0.65 * (y1 - y0))
                    parts = [(x0, y0, x1, yc - hw / 2, "room"), (x0, yc - hw / 2, x1, yc + hw / 2, "hall"),
                             (x0, yc + hw / 2, x1, y1, "room")]
                else:
                    xc = r.uniform(x0 + 0.35 * (x1 - x0), x0 + 0.65 * (x1 - x0))
                    parts = [(x0, y0, xc - hw / 2, y1, "room"), (xc - hw / 2, y0, xc + hw / 2, y1, "hall"),
                             (xc + hw / 2, y0, x1, y1, "room")]
            for (a0, b0, a1, b1, kind) in parts:
                subs = [(a0, b0, a1, b1)] if kind == "hall" else G.split_rooms(r, a0, b0, a1, b1)
                for (s0, u0, s1, u1) in subs:
                    for piece in G.polys_of(G.rect(s0, u0, s1, u1).intersection(q)):
                        if piece.area > 1:
                            cells.append((piece, kind))
    # anything of the footprint outside the blocks (bays) joins the cell it touches most
    covered = unary_union([c for c, _ in cells])
    for extra in G.polys_of(foot.difference(covered)):
        if extra.area < 0.5:
            continue
        k = max(range(len(cells)), key=lambda i: cells[i][0].boundary.intersection(extra.boundary).length)
        cells[k] = (cells[k][0].union(extra), cells[k][1])
    # slivers join their longest-shared neighbour
    changed = True
    while changed:
        changed = False
        for i, (c, kind) in enumerate(cells):
            if c.area < 25 and kind != "garage" and len(cells) > 1:
                j = max((j for j in range(len(cells)) if j != i and cells[j][1] != "garage"),
                        key=lambda j: cells[j][0].boundary.intersection(c.boundary).length, default=None)
                if j is not None and cells[j][0].boundary.intersection(c.boundary).length > 1:
                    cells[j] = (cells[j][0].union(c), cells[j][1])
                    cells.pop(i)
                    changed = True
                    break
    # open plan: merge neighbours into L-shaped / larger rooms
    for _ in range(r.choice([0, 1, 1, 2])):
        pairs = [(i, j) for i in range(len(cells)) for j in range(i + 1, len(cells))
                 if cells[i][1] == cells[j][1] == "room" and (_shared_seg(cells[i][0], cells[j][0]) or (0, 0))[1] > 5]
        if not pairs:
            break
        i, j = r.choice(pairs)
        m = cells[i][0].union(cells[j][0])
        if isinstance(m, Polygon):
            cells[i] = (m, "room")
            cells.pop(j)
    # closets / baths carved out of big rooms' corners -> L-shaped rooms
    for i in range(len(cells)):
        c, kind = cells[i]
        if kind != "room" or c.area < 150 or r.random() > 0.35:
            continue
        cs = [k for k in _convex_corners(c) if k[3] > 10 and k[4] > 10]
        if not cs:
            continue
        p, da, db, la, lb = r.choice(cs)
        u, v = r.uniform(4, 7), r.uniform(4.5, 8)
        q = Polygon([p, (p[0] + da[0] * u, p[1] + da[1] * u),
                     (p[0] + da[0] * u + db[0] * v, p[1] + da[1] * u + db[1] * v), (p[0] + db[0] * v, p[1] + db[1] * v)])
        rest = c.difference(q)
        if isinstance(rest, Polygon) and q.within(c.buffer(1e-6)):
            cells[i] = (rest, "room")
            cells.append((q, "carve"))
    rooms = []
    for (c, kind) in cells:
        inner = c.buffer(-t_int / 2, join_style=2).intersection(inner_foot)
        if inner.is_empty or inner.area < 8:
            continue
        inner = max(G.polys_of(inner), key=lambda q: q.area)
        rooms.append({"cell": c, "inner": inner, "kind": kind})
    # names by size / shape
    free = sorted([k for k, rm in enumerate(rooms) if rm["kind"] == "room"], key=lambda k: -rooms[k]["inner"].area)
    names = {}
    for k in [k for k, rm in enumerate(rooms) if rm["kind"] == "garage"]:
        names[k] = r.choice(["GARAGE", "GARAGE", "2-CAR GARAGE"])
    for k in [k for k, rm in enumerate(rooms) if rm["kind"] == "hall"]:
        names[k] = r.choice(["HALL", "HALLWAY", "CORRIDOR"])
    for k in [k for k, rm in enumerate(rooms) if rm["kind"] == "carve"]:
        names[k] = r.choice(["CLOSET", "W.I.C.", "BATH", "PANTRY", "LAUNDRY", "POWDER"])
    nb = 1
    for rank, k in enumerate(free):
        q = rooms[k]["inner"]
        x0, y0, x1, y1 = q.bounds
        w, h = x1 - x0, y1 - y0
        a = q.area
        if min(w, h) < 6.5 and max(w, h) / max(min(w, h), 1) > 2.3:
            nm = "HALL"
        elif rank == 0:
            nm = r.choice(["LIVING ROOM", "GREAT ROOM", "LIVING / DINING"])
        elif rank == 1:
            nm = "KITCHEN"
        elif a < 45:
            nm = r.choice(["CLOSET", "W.I.C.", "PANTRY", "LINEN"])
        elif a < 95:
            nm = r.choice(["BATH", "BATH", "LAUNDRY", "POWDER", "MUDROOM", "UTILITY"])
        elif rank == 2 and r.random() < 0.6:
            nm = r.choice(["DINING", "FAMILY ROOM", "ENTRY"])
        else:
            nm = r.choice(["BEDROOM", "BEDROOM", "BEDROOM", "OFFICE", "PRIMARY BEDROOM"])
            if nm == "BEDROOM":
                nb += 1
                nm = f"BEDROOM {nb}"
        names[k] = nm
    for k, rm in enumerate(rooms):
        rm["name"] = names[k]
        rm["zone"] = _zone(rm["name"])
    # openings between rooms: a spanning tree first (every room reachable), then extras
    pairs = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            sh = _shared_seg(rooms[i]["cell"], rooms[j]["cell"])
            if sh is not None and sh[1] > 3.6:
                pairs.append((i, j, sh[0], sh[1]))
    r.shuffle(pairs)
    parent = list(range(len(rooms)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    chosen = []
    for pr in pairs:
        a, b = find(pr[0]), find(pr[1])
        if a != b:
            parent[a] = b
            chosen.append(pr)
        elif r.random() < 0.25:
            chosen.append(pr)
    doors, openings = [], []
    for (i, j, (x0, y0, x1, y1), L) in chosen:
        vert = (y1 - y0) > (x1 - x0)
        both_open = rooms[i]["zone"] == rooms[j]["zone"] == "living"
        wide = both_open and r.random() < 0.6
        closet = any(k in rooms[i]["name"] + rooms[j]["name"] for k in ("CLOSET", "W.I.C.", "PANTRY", "LINEN", "LAUNDRY"))
        if wide:
            dtype = "open"
        elif closet:
            dtype = r.choices(["bifold", "barn", "pocket", "swing", "double"], weights=[35, 15, 15, 25, 10])[0]
        else:
            dtype = r.choices(["swing", "pocket", "double", "barn"], weights=[72, 12, 8, 8])[0]
        w = min(L - 1.0, r.uniform(4, 9)) if wide else \
            min(L - 1.0, r.uniform(4.5, 6) if dtype in ("double", "bifold") else r.choice([2.5, 2.67, 3.0]))
        if vert:
            c = r.uniform(y0 + 0.5 + w / 2, y1 - 0.5 - w / 2) if y1 - y0 > w + 1 else (y0 + y1) / 2
            x = (x0 + x1) / 2
            op = G.rect(x - t_int, c - w / 2, x + t_int, c + w / 2)
            doors.append(("v", x, c, w, dtype))
        else:
            c = r.uniform(x0 + 0.5 + w / 2, x1 - 0.5 - w / 2) if x1 - x0 > w + 1 else (x0 + x1) / 2
            y = (y0 + y1) / 2
            op = G.rect(c - w / 2, y - t_int, c + w / 2, y + t_int)
            doors.append(("h", y, c, w, dtype))
        openings.append((i, j, op))
    # front door + windows on the exterior
    mb = blocks[0]
    fx = mb.x0 + house["front_door_u"] * mb.w
    # front door on the front-most horizontal exterior wall (the footprint may be notched)
    fsegs = [(a, b) for a, b in zip(list(foot.exterior.coords)[:-1], list(foot.exterior.coords)[1:])
             if abs(a[1] - b[1]) < 1e-6 and abs(a[0] - b[0]) > 6]
    if fsegs:
        a, b = min(fsegs, key=lambda sg: (sg[0][1], -abs(sg[0][0] - sg[1][0])))
        lo, hi = min(a[0], b[0]) + 2, max(a[0], b[0]) - 2
        fx = min(max(fx, lo), hi)
        doors.append(("h", a[1], fx, 5.5, "double") if r.random() < 0.3 else ("h", a[1], fx, 3.0, "swing"))
        front_y = a[1]
    else:
        front_y = None
    edge = foot.boundary

    def ext_segs(cell, min_len):
        out = []
        for a, b in zip(list(cell.exterior.coords)[:-1], list(cell.exterior.coords)[1:]):
            if (abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6) or math.dist(a, b) < min_len:
                continue
            if edge.distance(LineString([a, b]).interpolate(0.5, normalized=True)) > 0.05:
                continue
            out.append((a, b))
        return out

    def add_ext(a, b, w, dtype):
        L = math.dist(a, b)
        t0 = r.uniform(1 + w / 2, L - 1 - w / 2) if L > w + 2 else L / 2
        if abs(a[1] - b[1]) < 1e-6:
            doors.append(("h", a[1], min(a[0], b[0]) + t0, w, dtype))
        else:
            doors.append(("v", a[0], min(a[1], b[1]) + t0, w, dtype))
    for rm in rooms:
        if rm["kind"] == "garage":
            sg = ext_segs(rm["cell"], 11)
            if sg:
                a, b = max(sg, key=lambda ab: math.dist(*ab))
                L = math.dist(a, b)
                add_ext(a, b, min(16.0, L - 2) if L >= 19 else min(9.0, L - 2), "garage")
        elif rm["zone"] == "living" and r.random() < 0.35:
            sg = ext_segs(rm["cell"], 9)
            if sg:
                a, b = r.choice(sg)
                add_ext(a, b, r.uniform(6, min(8, math.dist(a, b) - 2)), "slider")
    windows = []
    for rm in rooms:
        if rm["kind"] == "garage":
            continue
        for a, b in zip(list(rm["cell"].exterior.coords)[:-1], list(rm["cell"].exterior.coords)[1:]):
            seg = LineString([a, b])
            if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:      # no windows in diagonal walls
                continue
            if seg.length < 5 or edge.distance(seg.interpolate(0.5, normalized=True)) > 0.05:
                continue
            n = max(0, int(seg.length / r.uniform(7, 13)))
            for k in range(n):
                t = (k + 0.5) / n
                p = seg.interpolate(t, normalized=True)
                ww = r.choice([2.5, 3.0, 4.0, 5.0])
                o = "h" if abs(a[1] - b[1]) < 1e-6 else "v"
                pos, cc = (a[1], p.x) if o == "h" else (a[0], p.y)
                if any(d[0] == o and abs(d[1] - pos) < 1e-6 and abs(d[2] - cc) < (d[3] + ww) / 2 + 1 for d in doors):
                    continue                                        # no window over a door / slider / garage door
                windows.append((o, pos, cc, ww))
    cut = []
    for (o, pos, c, w, *_) in doors + windows:
        cut.append(G.rect(pos - 1.0, c - w / 2, pos + 1.0, c + w / 2) if o == "v" else
                   G.rect(c - w / 2, pos - 1.0, c + w / 2, pos + 1.0))
    walls_full = foot.difference(unary_union([rm["inner"] for rm in rooms]))
    walls = walls_full.difference(unary_union(cut))
    return {"rooms": rooms, "doors": doors, "windows": windows, "openings": openings, "walls": walls,
            "walls_full": walls_full, "foot": foot, "t_ext": t_ext, "t_int": t_int}


# ---- plan symbols: windows, door types, stairs (FreeCAD BIM window/door presets, ArchStairs)
def _seg(canvas, V, o, pos, a, b, off, ink, lw):
    """Line along the wall (o, pos) from a to b, offset `off` across the wall."""
    if o == "h":
        G.cv_line(canvas, V.px(a, pos + off), V.px(b, pos + off), ink, lw)
    else:
        G.cv_line(canvas, V.px(pos + off, a), V.px(pos + off, b), ink, lw)


def _pt_(o, pos, along, across):
    return (along, pos + across) if o == "h" else (pos + across, along)


def _draw_window(canvas, V, o, pos, c, w, t, style, ink, lw):
    a, b = c - w / 2, c + w / 2
    if style == "3line":
        for k in (-0.4, 0, 0.4):
            _seg(canvas, V, o, pos, a, b, k * t, ink, lw)
    elif style == "2line":
        for k in (-0.12, 0.12):
            _seg(canvas, V, o, pos, a, b, k * t, ink, lw)
    else:                                                     # glass line + projecting sill
        _seg(canvas, V, o, pos, a, b, 0, ink, lw)
        for k in (-0.5, 0.5):
            _seg(canvas, V, o, pos, a - 0.2, b + 0.2, k * t * 1.35, ink, lw)
        for x in (a - 0.2, b + 0.2):
            p, q = _pt_(o, pos, x, -0.5 * t * 1.35), _pt_(o, pos, x, 0.5 * t * 1.35)
            G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw)
    for x in (a, b):                                          # jambs
        p, q = _pt_(o, pos, x, -t / 2), _pt_(o, pos, x, t / 2)
        G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw)


def _arc(canvas, V, hinge, rad, a0, a1, ink, lw, S):
    cv2.ellipse(canvas, G._pt(V.px(*hinge)), (int(rad * S * G.SCALE), int(rad * S * G.SCALE)), 0, a0, a1,
                G.bgr(ink), max(1, int(lw * 0.8)), cv2.LINE_AA, G.SHIFT)


def _swing(canvas, V, o, pos, hx, leaf, sgn, toward, ink, lw, S):
    """One leaf hinged at along-wall position hx, swinging to side sgn, opening toward +1/-1 along the wall."""
    hinge = _pt_(o, pos, hx, 0)
    end = _pt_(o, pos, hx, sgn * leaf)
    G.cv_line(canvas, V.px(*hinge), V.px(*end), ink, lw * 1.3)
    # arc from the leaf tip back to the closed position (along +toward)
    ang_leaf = math.degrees(math.atan2(end[1] - hinge[1], end[0] - hinge[0]))
    closed = _pt_(o, pos, hx + toward * leaf, 0)
    ang_closed = math.degrees(math.atan2(closed[1] - hinge[1], closed[0] - hinge[0]))
    a0, a1 = sorted([ang_leaf, ang_closed])
    if a1 - a0 > 180:
        a0, a1 = a1, a0 + 360
    _arc(canvas, V, hinge, leaf, a0, a1, ink, lw, S)


def _draw_door(canvas, V, d, t_ext, t_int, ink, lw, S, r):
    o, pos, c, w, dtype = d
    a, b = c - w / 2, c + w / 2
    sgn = r.choice([-1, 1])
    if dtype == "open":
        return
    if dtype == "swing":
        if r.random() < 0.5:
            _swing(canvas, V, o, pos, a, w, sgn, 1, ink, lw, S)
        else:
            _swing(canvas, V, o, pos, b, w, sgn, -1, ink, lw, S)
    elif dtype == "double":
        _swing(canvas, V, o, pos, a, w / 2, sgn, 1, ink, lw, S)
        _swing(canvas, V, o, pos, b, w / 2, sgn, -1, ink, lw, S)
    elif dtype == "pocket":                                   # leaf slides into the wall (dashed inside it)
        _seg(canvas, V, o, pos, a, b, 0, ink, lw * 1.3)
        p0, p1 = _pt_(o, pos, b, 0), _pt_(o, pos, b + w * 0.9, 0)
        _dash_line(canvas, V.px(*p0), V.px(*p1), ink, lw, [(0.3 * S, 0.2 * S)])
    elif dtype == "bifold":                                   # zig-zag panels from both jambs
        for side, x0 in ((1, a), (-1, b)):
            n = 2
            pw = (w / 2) / n
            pts = [_pt_(o, pos, x0 + side * k * pw, (sgn * pw * 0.45) * (k % 2)) for k in range(n + 1)]
            G.cv_polyline(canvas, [V.px(*p) for p in pts], ink, lw * 1.2)
    elif dtype == "barn":                                     # surface-mounted slab outside the opening
        off = sgn * (t_int / 2 + 0.25)
        _seg(canvas, V, o, pos, a - 0.3, b + 0.3, off, ink, lw * 1.4)
        _seg(canvas, V, o, pos, b + 0.3, b + w * 0.9, off, ink, lw)
    elif dtype == "slider":                                   # two glazed panels, overlapping, in the wall
        for k, (u, v) in enumerate(((a, c + 0.3), (c - 0.3, b))):
            off = (k - 0.5) * t_ext * 0.35
            _seg(canvas, V, o, pos, u, v, off - 0.05, ink, lw)
            _seg(canvas, V, o, pos, u, v, off + 0.05, ink, lw)
        for x in (a, b):
            p, q = _pt_(o, pos, x, -t_ext / 2), _pt_(o, pos, x, t_ext / 2)
            G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw)
    elif dtype == "garage":                                   # overhead door: panel line + dashed track inside
        _seg(canvas, V, o, pos, a, b, 0, ink, lw * 1.2)
        for x in (a, b):
            p, q = _pt_(o, pos, x, -t_ext / 2), _pt_(o, pos, x, t_ext / 2)
            G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw)


def plan_stairs(rooms, r):
    """0-1 stair runs (straight or U) inside a hall / living room, as plan polygons with
    tread geometry. Stairs are always holes in the floor label (like skylights, piers)."""
    cand = []
    for rm in rooms:
        q = rm["inner"]
        x0, y0, x1, y1 = q.bounds
        if rm["zone"] in ("living",) and q.area > 0.9 * (x1 - x0) * (y1 - y0) and max(x1 - x0, y1 - y0) >= 12 \
                and min(x1 - x0, y1 - y0) >= 4.2:
            cand.append(rm)
    if not cand:
        return []
    rm = r.choice(cand)
    x0, y0, x1, y1 = rm["inner"].bounds
    horiz = (x1 - x0) >= (y1 - y0)
    along0, along1 = (x0, x1) if horiz else (y0, y1)
    across0, across1 = (y0, y1) if horiz else (x0, x1)
    sw = r.uniform(3.0, 3.8)
    L = min(along1 - along0 - 1.0, r.uniform(10, 13))
    u_type = (across1 - across0) >= 2 * sw + 0.6 and r.random() < 0.4
    s0 = r.uniform(along0 + 0.2, along1 - 0.2 - L)
    side = r.choice([0, 1])                                   # against one of the long walls
    runs = []
    if u_type:
        L = min(L, 8.5)
        ca = across0 + 0.1 if side == 0 else across1 - 0.1 - (2 * sw + 0.3)
        runs = [(s0, s0 + L, ca, ca + sw, 1), (s0, s0 + L, ca + sw + 0.3, ca + 2 * sw + 0.3, -1)]
        land = (s0 + L, s0 + L + sw, ca, ca + 2 * sw + 0.3) if s0 + L + sw < along1 - 0.1 else None
        if land is None:
            return []
    else:
        ca = across0 + 0.1 if side == 0 else across1 - 0.1 - sw
        runs = [(s0, s0 + L, ca, ca + sw, r.choice([1, -1]))]
        land = None

    def box_(a0, a1, c0, c1):
        return G.rect(a0, c0, a1, c1) if horiz else G.rect(c0, a0, c1, a1)
    polys = [box_(*run[:4]) for run in runs] + ([box_(*land)] if land else [])
    return [{"poly": unary_union(polys), "runs": runs, "land": land, "horiz": horiz, "box": box_,
             "tread": r.uniform(0.85, 1.0), "rail": r.random() < 0.5, "brk": r.random() < 0.5,
             "label": r.choice(["UP", "UP", "DN"])}]


def draw_stairs(canvas, tq, V, st, ink, lw, size, S):
    horiz, box_ = st["horiz"], st["box"]
    _mfill(canvas, V.geom(st["poly"]), (255, 255, 255), canvas.shape[1], canvas.shape[0])
    G.cv_outline(canvas, V.geom(st["poly"]), ink, lw * 1.2)
    for (a0, a1, c0, c1, direc) in st["runs"]:
        n = int((a1 - a0) / st["tread"])
        brk = a0 + (a1 - a0) * 0.62 if st["brk"] else None
        for k in range(1, n + 1):
            a = a0 + k * (a1 - a0) / (n + 1)
            p, q = ((a, c0), (a, c1)) if horiz else ((c0, a), (c1, a))
            if brk is not None and a > brk:
                _dash_line(canvas, V.px(*p), V.px(*q), ink, lw * 0.8, [(0.25 * S, 0.18 * S)])
            else:
                G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw * 0.8)
        if brk is not None:                                  # diagonal cut line with a zig-zag
            p, q = ((brk - 0.8, c0), (brk + 0.8, c1)) if horiz else ((c0, brk - 0.8), (c1, brk + 0.8))
            G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw * 1.2)
        if st["rail"]:
            off = 0.25
            p, q = ((a0, c1 - off), (a1, c1 - off)) if horiz else ((c1 - off, a0), (c1 - off, a1))
            G.cv_line(canvas, V.px(*p), V.px(*q), ink, lw * 0.8)
        cm = (c0 + c1) / 2
        s_, e_ = (a0 + 0.4, a1 - 0.4) if direc > 0 else (a1 - 0.4, a0 + 0.4)
        p, q = ((s_, cm), (e_, cm)) if horiz else ((cm, s_), (cm, e_))
        _slope_arrow(canvas, tq, V.px(*p), V.px(*q), "", ink, lw, size, half=False)
        tq.add(V.px(*p), st["label"], size * 0.8, ink, bold=True, anchor="mm")
    if st["land"]:
        G.cv_outline(canvas, V.geom(box_(*st["land"])), ink, lw)


# ---- floor finishes (Gemini floor plans: finish per room group, labelled through doorways)
def floor_material(kind, r, seed, colour, ang=None):
    base = (255, 255, 255)
    if colour and r.random() < 0.6:
        base = r.choice([(232, 226, 214), (226, 214, 196), (214, 214, 210), (236, 232, 226), (220, 228, 236)])
    line = (r.randint(80, 175),) * 3 if not colour else G.mix(base, (0, 0, 0), r.uniform(0.3, 0.5))
    m = {"kind": kind, "base": base, "line": line, "seed": seed,
         "ang": ang if ang is not None else r.choice([0, 90])}
    if kind == "plank_lines":
        m["sp"] = r.uniform(0.35, 0.75)
    elif kind == "plank":
        m["w"] = r.uniform(0.45, 0.8)
    elif kind in ("tile", "diag_tile"):
        m["sp"] = r.uniform(0.8, 2.2)
    elif kind == "mosaic":
        m["sp"] = r.uniform(0.3, 0.55)
    elif kind in ("carpet", "concrete"):
        m["density"] = r.uniform(1.2, 3.5)
    elif kind == "dots":
        m["sp"] = r.uniform(0.7, 1.6)
        m["fill"] = r.random() < 0.6
    elif kind == "basket":
        m["sp"] = r.uniform(0.8, 1.4)
    elif kind == "hatch45":
        m["sp"] = r.uniform(0.5, 1.2)
        m["ang"] = r.choice([45, -45])
    elif kind == "joists":
        m["sp"] = r.choice([1.333, 1.333, 2.0])
    elif kind == "ashlar":
        m["unit"] = r.uniform(0.9, 1.6)
    elif kind in TILES:
        m["sp"] = r.uniform(*TILES[kind][3])
    return m


def _mkey(m):
    return (m["kind"],) + tuple(round(m[k], 2) if isinstance(m[k], float) else m[k]
                                for k in sorted(m) if k not in ("kind", "seed"))


# Two families of the same pattern TYPE on one sheet must differ in spacing (>= 1.4x) or
# tone. Direction never counts: a pattern that turns is still the same family (roof
# facets, deck pieces), so two families told apart only by direction would contradict it.
LOOK = {"hatch": "lines", "plank_lines": "lines", "plank": "lines", "joists": "lines", "hatch45": "lines",
        "grid": "grid", "tile": "grid", "mosaic": "grid", "diag_tile": "diag", "cross": "diag",
        "concrete": "stipple", "carpet": "stipple", "stipple": "stipple", "dots": "dots",
        "basket": "basket", "ashlar": "ashlar", "rubble": "stone"}


for _k, (_f, _look, _eff, _rng) in TILES.items():
    LOOK[_k] = _look


def _spacing(p, kind=None):
    """(param key, effective spacing in ft). Tiles: tile size times the tile's line fraction."""
    for k in ("sp", "w", "unit", "density"):
        if p.get(k):
            return k, p[k] * (TILES[kind][2] if kind in TILES else 1.0)
    return None, None


def _lum(b):
    return 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]


def _distinct(k1, p1, b1, k2, p2, b2):
    if LOOK.get(k1, k1) != LOOK.get(k2, k2):
        return True
    if abs(_lum(b1) - _lum(b2)) >= 30 or math.dist(b1, b2) >= 60:
        return True
    s1, s2 = _spacing(p1, k1)[1], _spacing(p2, k2)[1]
    return bool(s1 and s2) and max(s1, s2) / min(s1, s2) >= 1.4


def _separate(kind, params, base, others):
    """Rescale this family's spacing until it differs from every same-type family in `others`."""
    key = _spacing(params, kind)[0]
    if key is None:
        return
    f = TILES[kind][2] if kind in TILES else 1.0
    for _ in range(4):
        clash = [(k, p, b) for (k, p, b) in others if not _distinct(kind, params, base, k, p, b)]
        if not clash:
            return
        s2 = _spacing(clash[0][1], clash[0][0])[1]
        mine = params[key] * f
        params[key] = (s2 * 1.6 if mine >= s2 else s2 / 1.6) / f


def _separate_all(mats):
    """Floor materials (dicts) in order; identical objects are one family and skipped."""
    done, seen = [], set()
    for m in mats:
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        _separate(m["kind"], m, m["base"], done)
        done.append((m["kind"], m, m["base"]))
    return done


def draw_floor(canvas, poly_px, m, S, W, H, lw):
    if poly_px.is_empty:
        return
    minx, miny, maxx, maxy = poly_px.bounds
    x0, y0 = max(0, int(math.floor(minx)) - 2), max(0, int(math.floor(miny)) - 2)
    x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
    layer[:] = G.bgr(m["base"])
    k, c = m["kind"], m["line"]
    lx0, ly0 = x0, y0
    x0, y0 = x0 + m.get("ox", 0.0), y0 + m.get("oy", 0.0)     # v3: pattern origin per room
    if k in ("plank_lines", "joists", "hatch45"):
        G._dlines(layer, x0, y0, S, m["sp"], m["ang"], c, lw)
    elif k == "plank":
        G._planks(layer, x0, y0, S, m["w"], c, lw, m["seed"], angle=m["ang"])
    elif k in ("tile", "mosaic"):
        G._hlines(layer, x0, y0, S, m["sp"], c, lw)
        G._vlines(layer, x0, y0, S, m["sp"], c, lw)
    elif k == "diag_tile":
        G._dlines(layer, x0, y0, S, m["sp"], 45, c, lw)
        G._dlines(layer, x0, y0, S, m["sp"], -45, c, lw)
    elif k == "carpet":
        G._stipple(layer, x0, y0, S, m["density"], c, m["seed"], r_px=1)
    elif k == "concrete":
        G._stipple(layer, x0, y0, S, m["density"], c, m["seed"], r_px=1)
        G._stipple(layer, x0, y0, S, m["density"] * 0.15, c, m["seed"] + 1, r_px=2)
        if m.get("tri"):                    # AR-CONC aggregate: small open triangles, cell-seeded
            cell = 2.0 * S
            hh, ww = layer.shape[:2]
            for cy in range(math.floor(y0 / cell), math.ceil((y0 + hh) / cell) + 1):
                for cx in range(math.floor(x0 / cell), math.ceil((x0 + ww) / cell) + 1):
                    rr = random.Random(m["seed"] * 7919 + cx * 104729 + cy * 1299709)
                    for _ in range(rr.randint(0, 3)):
                        px, py = cx * cell + rr.random() * cell - x0, cy * cell + rr.random() * cell - y0
                        sz = rr.uniform(0.12, 0.3) * S
                        a0 = rr.uniform(0, 2 * math.pi)
                        pts = np.array([[px + sz * math.cos(a0 + k2 * 2.094), py + sz * math.sin(a0 + k2 * 2.094)]
                                        for k2 in range(3)], np.int32)
                        cv2.polylines(layer, [pts], True, G.bgr(c), max(1, int(lw * 0.8)), cv2.LINE_AA)
    elif k == "dots":
        _dotgrid(layer, x0, y0, S, m["sp"], max(1.2, 0.07 * S), c, m["fill"], lw, m.get("jit", 0.0), m["seed"])
    elif k == "basket":
        G._basketweave(layer, x0, y0, S, m["sp"], c, lw)
    elif k == "ashlar":
        G._ashlar(layer, x0, y0, S, m["unit"], c, lw, m["seed"])
    elif k in TILES:
        hatch_layer(layer, x0, y0, S, k, m["sp"], c, m.get("ang", 0) if k in ("t_hbone", "t_earth", "t_woodgrain") else 0)
    x0, y0 = lx0, ly0
    mask = G.poly_mask(affinity.translate(poly_px, -x0, -y0), x1 - x0, y1 - y0)
    G.blend_mask(canvas[y0:y1, x0:x1], layer, mask)


LIVING_FIN = (["plank_lines", "plank", "tile", "diag_tile", "dots", "basket", "t_hbone", "t_earth", "t_woodgrain",
               "t_cross"], [32, 14, 14, 8, 5, 5, 10, 6, 3, 3])
WET_FIN = (["tile", "mosaic", "diag_tile", "dots", "t_zinc", "t_plus", "t_cross"], [42, 20, 12, 8, 8, 5, 5])
BED_FIN = (["carpet", "plank_lines", "plank", "dots", "t_hbone", "t_woodgrain"], [40, 26, 12, 8, 9, 5])
GARAGE_FIN = (["concrete", "dots", "carpet", "t_aggregate"], [42, 18, 10, 30])


def _footprint_notch(foot, pr):
    """A notch of the footprint's bounding box that the walls enclose on two or more sides
    (the covered patio between wings on HF14 0, the porch on 12), or None."""
    x0, y0, x1, y1 = foot.bounds
    best = None
    for q in G.polys_of(box(x0, y0, x1, y1).difference(foot.buffer(0.05, join_style=2))):
        if not 60 <= q.area <= 900:
            continue
        shared = q.exterior.intersection(foot.buffer(0.2, join_style=2)).length
        if shared >= 0.4 * q.exterior.length and (best is None or q.area > best.area):
            best = q
    if best is None:
        return None
    return best.buffer(-0.15, join_style=2) if pr.random() < 0.5 else best


def _per_face(band, foot):
    """Split a band outside `foot` into one piece per footprint edge, mitred along the corner
    bisectors; yields (piece, edge angle in degrees)."""
    coords = list(foot.exterior.coords)[:-1]
    n = len(coords)
    out_n = []
    for k in range(n):
        (ax, ay), (bx, by) = coords[k], coords[(k + 1) % n]
        L = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = (by - ay) / L, -(bx - ax) / L
        mx, my = (ax + bx) / 2, (ay + by) / 2
        if foot.contains(Point(mx + 0.05 * nx, my + 0.05 * ny)):
            nx, ny = -nx, -ny
        out_n.append((nx, ny))
    R = 6.0
    done = None
    for k in range(n):
        a, b = coords[k], coords[(k + 1) % n]
        n0, n1, n2 = out_n[k - 1], out_n[k], out_n[(k + 1) % n]

        def miter(p, q):
            mx, my = p[0] + q[0], p[1] + q[1]
            d = 1.0 + p[0] * q[0] + p[1] * q[1]
            return (mx / d, my / d) if d > 0.2 else (q[0], q[1])
        ma, mb = miter(n0, n1), miter(n1, n2)
        quad = Polygon([a, b, (b[0] + R * mb[0], b[1] + R * mb[1]), (a[0] + R * ma[0], a[1] + R * ma[1])]).buffer(0)
        piece = band.intersection(quad)
        if done is not None:
            piece = piece.difference(done)
        if piece.is_empty or piece.area < 0.5:
            continue
        done = piece if done is None else unary_union([done, piece])
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
        for q in G.polys_of(piece):
            yield q, ang
    rest = band.difference(done) if done is not None else band
    for q in G.polys_of(rest):
        if q.area > 0.5:
            bx = q.bounds
            yield q, 0 if bx[2] - bx[0] >= bx[3] - bx[1] else 90


def _soffit_band(foot, pr):
    """The eave overhang drawn on a ceiling plan: a strip 1-2.5 ft outside the walls, the
    whole ring or the stretches along some sides (an L on HF14 12)."""
    ov = pr.uniform(1.0, 2.5)
    ring = foot.buffer(ov, join_style=2).difference(foot)
    if pr.random() < 0.35:
        return ring
    coords = list(foot.exterior.coords)
    keep = []
    for a, b in zip(coords, coords[1:]):
        if math.dist(a, b) >= 6 and pr.random() < 0.55:
            keep.append(LineString([a, b]).buffer(ov + 0.3, cap_style=2))
    if not keep:
        return ring
    return ring.intersection(unary_union(keep))


# ----------------------------------------------------------------------------
# plan v3 (--plan-v2 2): furniture, framing
# ----------------------------------------------------------------------------
def _ell(cx, cy, rx, ry):
    return affinity.scale(Point(cx, cy).buffer(1.0, 24), rx, ry)


def _room_furniture(name, x0, y0, x1, y1, pr, clip=None):
    """Furniture of one room in plan, as (polygon in ft, 'rug' | 'item' | 'line') pieces, in a frame
    flipped at random so pieces sit against different walls; laid along whichever axis fits. With
    `clip` (the room, less stairs), the best of a few random layouts by how many items fit."""
    if clip is None:
        return _room_furniture0(name, x0, y0, x1, y1, pr)
    best, best_n = [], -1
    for _ in range(5):
        out = _room_furniture0(name, x0, y0, x1, y1, pr)
        n = sum(1 for q, k in out if k == "item" and q.within(clip))
        if n > best_n:
            best, best_n = out, n
        if out and n == sum(1 for q, k in out if k == "item"):
            break
    return best


def _room_furniture0(name, x0, y0, x1, y1, pr):
    w, h = x1 - x0, y1 - y0
    if pr.random() < 0.5:
        out = _room_furniture1(name, x0, y0, x1, y1, pr)
        if out:
            return out
    out = _room_furniture1(name, 0, 0, h, w, pr)          # transposed frame, mapped back
    return [(affinity.translate(affinity.affine_transform(q, [0, 1, 1, 0, 0, 0]), x0, y0), k) for q, k in out] or \
        _room_furniture1(name, x0, y0, x1, y1, pr)


def _room_furniture1(name, x0, y0, x1, y1, pr):
    w, h = x1 - x0, y1 - y0
    fx, fy = pr.random() < 0.5, pr.random() < 0.5
    out = []

    def tr(a, b):                                    # room-local feet -> sheet feet, flips applied
        return (x0 + (w - a if fx else a), y0 + (h - b if fy else b))

    def P(pts, kind="item"):
        out.append((Polygon([tr(a, b) for a, b in pts]), kind))

    def R(a, b, c, d, kind="item", rad=0.0):
        (ax, ay), (cx_, cy_) = tr(a, b), tr(c, d)
        q = box(min(ax, cx_), min(ay, cy_), max(ax, cx_), max(ay, cy_))
        if rad > 0 and min(c - a, d - b) > 2.2 * rad:
            q = q.buffer(-rad).buffer(rad, 8)
        out.append((q, kind))

    def E(cx, cy, rx, ry, kind="item"):
        out.append((_ell(*tr(cx, cy), rx, ry), kind))

    def toilet(cx, wall_y, d=1):                     # tank against the wall, bowl + rim out from it
        R(cx - 0.85, wall_y, cx + 0.85, wall_y + d * 0.75) if d > 0 else R(cx - 0.85, wall_y - 0.75, cx + 0.85, wall_y)
        E(cx, wall_y + d * 1.55, 0.62, 0.85)
        E(cx, wall_y + d * 1.6, 0.45, 0.62, "line")

    def basin(cx, cy, rx=0.7, ry=0.5):
        E(cx, cy, rx, ry, "line"); E(cx, cy, rx * 0.72, ry * 0.68, "line"); E(cx, cy, 0.07, 0.07, "line")

    def chair(cx, cy, back_dy):
        R(cx - 0.75, cy - 0.7, cx + 0.75, cy + 0.7, rad=0.15)
        R(cx - 0.75, cy + back_dy * 0.7 - 0.12, cx + 0.75, cy + back_dy * 0.7 + 0.12, "line")

    if "BED" in name and w > 8 and h > 7.1:
        bw = 6.3 if "PRIMARY" in name else pr.choice([4.8, 5.0, 5.0, 6.3])
        bl = 6.8 if bw > 6 else 6.5
        cx = w / 2 + pr.uniform(-0.15, 0.15) * w
        if pr.random() < 0.6:
            R(cx - bw / 2 - 1.5, 1.2, cx + bw / 2 + 1.5, bl + 2.5, "rug")
        R(cx - bw / 2, 0.3, cx + bw / 2, 0.3 + bl, rad=0.25)                         # mattress
        R(cx - bw / 2 - 0.1, 0.15, cx + bw / 2 + 0.1, 0.45, "line")                   # headboard
        npil = 3 if bw > 6 and pr.random() < 0.4 else 2                               # pillows
        pw_ = (bw - 0.6) / npil
        for k in range(npil):
            R(cx - bw / 2 + 0.3 + k * pw_ + 0.08, 0.7, cx - bw / 2 + 0.3 + (k + 1) * pw_ - 0.08,
              pr.uniform(1.6, 2.0), "line", rad=pr.choice([0.15, 0.3]))
        fold = 0.3 + bl * pr.uniform(0.32, 0.42)
        if pr.random() < 0.8:
            P([(cx - bw / 2, fold), (cx + bw / 2, fold), (cx + bw / 2, fold + 0.6), (cx - bw / 2, fold + 0.6)], "line")
        cs = pr.random()
        if cs < 0.45:
            P([(cx + bw / 2 - 1.6, 0.3 + bl), (cx + bw / 2, 0.3 + bl - 1.6), (cx + bw / 2, 0.3 + bl)], "line")   # turned corner
        elif cs < 0.7:                                                                  # quilt stitching
            for k in range(1, 4):
                R(cx - bw / 2, fold + 0.6 + k * (bl - fold) / 4, cx + bw / 2, fold + 0.6 + k * (bl - fold) / 4 + 0.01, "line")
        elif cs < 0.85:                                                                 # throw across the foot
            R(cx - bw / 2 - 0.1, bl - 1.2, cx + bw / 2 + 0.1, bl - 0.3, "line")
        ns = pr.uniform(1.3, 2.0)
        nst = pr.random()
        for sx in (-1, 1):
            nx = cx + sx * (bw / 2 + 0.15 + ns / 2)
            R(nx - ns / 2, 0.3, nx + ns / 2, 0.3 + ns * pr.uniform(0.8, 1.0))          # nightstand
            if nst < 0.5:
                E(nx, 0.3 + ns / 2, 0.3, 0.3, "line")                                    # lamp
            elif nst < 0.75:
                P([(nx - 0.3, 0.3 + ns / 2 - 0.3), (nx + 0.3, 0.3 + ns / 2 + 0.3), (nx, 0.3 + ns / 2),
                   (nx - 0.3, 0.3 + ns / 2 + 0.3), (nx + 0.3, 0.3 + ns / 2 - 0.3), (nx, 0.3 + ns / 2)], "line")
        if h > 12 and pr.random() < 0.7:
            R(w / 2 - 2.5, h - 2.0, w / 2 + 2.5, h - 0.3)
            for k in range(1, 3):
                R(w / 2 - 2.5 + k * 5 / 3, h - 2.0, w / 2 - 2.5 + k * 5 / 3 + 0.01, h - 0.3, "line")
    elif any(k in name for k in ("LIVING", "GREAT", "FAMILY", "DEN", "BONUS")) and w > 9 and h > 8:
        cx, cy = w * pr.uniform(0.4, 0.6), h / 2 + pr.uniform(-0.1, 0.1) * max(0, h - 9.5)
        if h > 10:
            R(cx - 5, cy - 4, cx + 5, cy + 4, "rug")
        R(cx - 3.9, cy + 1.6, cx + 3.9, cy + 4.6, rad=0.2)                             # sofa
        R(cx - 3.9, cy + 3.8, cx + 3.9, cy + 4.6, "line")                              # back
        R(cx - 3.9, cy + 1.6, cx - 3.2, cy + 3.8, "line"); R(cx + 3.2, cy + 1.6, cx + 3.9, cy + 3.8, "line")   # arms
        for k in range(3):
            R(cx - 3.2 + k * 6.4 / 3 + 0.05, cy + 1.75, cx - 3.2 + (k + 1) * 6.4 / 3 - 0.05, cy + 3.75, "line", rad=0.15)
        if pr.random() < 0.5:
            R(cx + 3.9, cy - 2.0, cx + 6.7, cy + 4.6, rad=0.2)                         # L return
            R(cx + 5.9, cy - 2.0, cx + 6.7, cy + 4.6, "line")
        for sx in (-1, 1):
            ax = cx + sx * 2.4
            R(ax - 1.45, cy - 4.2, ax + 1.45, cy - 1.5, rad=0.2)                       # armchairs
            R(ax - 1.45, cy - 4.2, ax + 1.45, cy - 3.5, "line")
            R(ax - 1.45, cy - 3.5, ax - 0.95, cy - 1.5, "line"); R(ax + 0.95, cy - 3.5, ax + 1.45, cy - 1.5, "line")
        if pr.random() < 0.6:
            R(cx - 2.0, cy - 1.0, cx + 2.0, cy + 1.0, rad=0.1)
        else:
            E(cx, cy, 1.2, 1.2)
        if "DINING" in name or "GREAT" in name:
            dx = w * 0.18 if cx > w / 2 else w * 0.82
            if 3.5 < dx < w - 3.5:
                out.extend(_dining(*tr(dx, h * 0.5), pr))
    elif "DINING" in name and w > 8 and h > 8:
        out.extend(_dining(x0 + w / 2, y0 + h / 2, pr))
    elif "KITCHEN" in name and w > 8 and h > 8:
        R(0, 0, w, 2.1); R(0, 2.1, 2.1, h * pr.uniform(0.5, 0.9))                      # L counter
        R(0, 0, w, 2.1, "line"); R(0.05, 1.95, w - 0.05, 2.1, "line")                  # nosing
        sx = w * pr.uniform(0.3, 0.6)
        R(sx, 0.35, sx + 1.25, 1.65, "line", rad=0.15); R(sx + 1.35, 0.35, sx + 2.6, 1.65, "line", rad=0.15)
        E(sx + 1.3, 0.22, 0.08, 0.08, "line")
        rx = min(w - 3.2, sx + 4.5)
        if rx > sx + 3:
            R(rx + 0.1, 0.15, rx + 2.4, 1.95, "line")
            for dx, dy in [(0.7, 0.6), (1.7, 0.6), (0.7, 1.5), (1.7, 1.5)]:
                E(rx + dx, dy, 0.36, 0.36, "line"); E(rx + dx, dy, 0.2, 0.2, "line")
        R(w - 3.2, 0, w - 0.2, 2.6)                                                      # fridge
        out.append((LineString([tr(0, 1.15), tr(w, 1.15)]).buffer(0.02), "line"))      # uppers (projected)
        R(w - 3.2, 2.35, w - 0.2, 2.6, "line")                                          # fridge door
        if w > 12 and h > 11:
            ix, iy = w * 0.55, h * 0.6
            iw = pr.uniform(2.8, 4.2)
            R(ix - iw, iy - 1.5, ix + iw, iy + 1.5)
            R(ix - iw, iy + 0.4, ix + iw, iy + 0.41, "line")                               # overhang
            if pr.random() < 0.5:
                R(ix - 1.0, iy - 1.2, ix + 0.3, iy - 0.2, "line", rad=0.15)                 # prep sink
            for k in range(3):
                E(ix - 2.2 + 2.2 * k, iy + 2.3, 0.6, 0.6); E(ix - 2.2 + 2.2 * k, iy + 2.3, 0.35, 0.35, "line")
    elif any(k in name for k in ("BATH", "POWDER")) and w > 4.5 and h > 4.5:
        vw = min(w - 0.6, pr.choice([3.0, 5.0, 6.0]))
        R(0.2, 0.2, 0.2 + vw, 2.0)                                                       # vanity
        nb = 1 if vw < 4.5 else 2
        for k in range(nb):
            basin(0.2 + vw * (k + 0.5) / nb, 1.05)
        toilet(w - 1.3, h - 0.2, -1)
        if "POWDER" not in name and h > 7:
            if pr.random() < 0.55 and w > 6:
                R(0.2, h - 2.7, 5.2, h - 0.2)                                               # tub
                R(0.5, h - 2.4, 4.9, h - 0.5, "line", rad=0.6)
                E(4.4, h - 1.45, 0.1, 0.1, "line")
            else:
                R(0.2, h - 3.4, 3.4, h - 0.2)                                               # shower
                P([(0.2, h - 3.4), (3.4, h - 0.2), (0.2, h - 0.2)], "line")
                E(1.8, h - 1.8, 0.12, 0.12, "line")
    elif any(k in name for k in ("OFFICE", "STUDY")) and w > 7 and h > 7:
        R(w * 0.3, 0.2, w * 0.3 + 5, 2.6); chair(w * 0.3 + 2.5, 3.5, 1)
        R(0.2, h * 0.3, 1.4, h * 0.3 + 5)
        for k in range(1, 4):
            R(0.2, h * 0.3 + k * 1.25, 1.4, h * 0.3 + k * 1.25 + 0.01, "line")
    elif "GARAGE" in name and w > 10 and h > 16:
        for k in range(1 if w < 20 else 2):
            cx = w * (k + 0.5) / (1 if w < 20 else 2) + pr.uniform(-0.8, 0.8)
            if k == 1 and pr.random() < 0.3:
                continue                                                                    # one bay empty
            cl, cw = pr.uniform(14.0, 16.5), pr.uniform(2.8, 3.2)
            R(cx - cw, 1.5, cx + cw, 1.5 + cl, rad=pr.uniform(0.7, 1.2))
            P([(cx - 2.6, 5.0), (cx + 2.6, 5.0), (cx + 2.2, 6.6), (cx - 2.2, 6.6)], "line")
            R(cx - 2.2, 6.6, cx + 2.2, 11.6, "line")
            P([(cx - 2.2, 11.6), (cx + 2.2, 11.6), (cx + 2.5, 12.8), (cx - 2.5, 12.8)], "line")
            for sx in (-1, 1):
                R(cx + sx * 3.0 - 0.35, 4.4, cx + sx * 3.0 + 0.35, 5.0, "line")
    elif any(k in name for k in ("MUD", "ENTRY", "FOYER")) and w > 5 and h > 4:
        R(0.2, 0.2, min(w - 0.2, 5.5), 1.6)
        for k in range(int(min(w - 0.4, 5.3) // 1.3)):
            R(0.2 + 1.3 * (k + 1), 0.2, 0.2 + 1.3 * (k + 1) + 0.01, 1.6, "line")
    elif "LAUNDRY" in name and w > 5 and h > 4:
        for k in range(2):
            R(0.3 + 2.6 * k, 0.2, 2.6 + 2.6 * k, 2.6, rad=0.15)
            E(1.45 + 2.6 * k, 1.5, 0.85, 0.85, "line"); E(1.45 + 2.6 * k, 1.5, 0.55, 0.55, "line")
    elif any(k in name for k in ("CLOSET", "W.I.C.")) and min(w, h) > 3:
        R(0, 0, w, 1.8, "line"); R(0, 0.9, w, 0.95, "line")
        for k in range(int(w / 0.6)):
            R(0.3 + 0.6 * k, 0.3, 0.3 + 0.6 * k + 0.01, 1.55, "line")                   # hangers
    return out


def _dining(cx, cy, pr):
    out = []
    if pr.random() < 0.3:
        out.append((_ell(cx, cy, 2.2, 2.2), "item"))
        for a in range(0, 360, 60):
            ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
            ch = box(-0.75, -0.75, 0.75, 0.75).buffer(-0.15).buffer(0.15, 6)
            bk = box(0.5, -0.75, 0.75, 0.75)                                         # back, away from the table
            for g_, k_ in ((ch, "item"), (bk, "line")):
                out.append((affinity.translate(affinity.rotate(g_, a, origin=(0, 0)), cx + 3.0 * ux, cy + 3.0 * uy), k_))
        return out
    tw, th = pr.choice([(6.0, 3.3), (7.5, 3.5), (5.0, 3.0)])
    out.append((box(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2), "item"))
    n = int(tw // 2)
    for k in range(n):
        x = cx - tw / 2 + (k + 0.5) * tw / n
        for sy in (-1, 1):
            y = cy + sy * (th / 2 + 0.85)
            out.append((box(x - 0.75, y - 0.7, x + 0.75, y + 0.7).buffer(-0.15).buffer(0.15, 6), "item"))
            out.append((box(x - 0.75, y + sy * 0.7 - 0.12, x + 0.75, y + sy * 0.7 + 0.12), "line"))   # back
    return out


def _draw_furniture(canvas, V, pieces, ink, lw, solid, W, H, pr, clip=None):
    """solid: furniture blocks filled white with a soft drop shadow (rendered sheets, HF14 0);
    otherwise outlines only, so the floor pattern shows through."""
    skip = False
    placed = []
    for q, kind in pieces:
        if kind == "item":              # an item's detail lines follow it: drop them with it
            skip = clip is not None and not q.within(clip) or \
                any(q.intersection(o).area > 0.02 * min(q.area, o.area) for o in placed)
            if not skip:
                placed.append(q)
        if skip and kind == "line" or clip is not None and not q.within(clip) or kind == "item" and skip:
            continue
        g = V.geom(q)
        if kind == "rug":           # a rug covers the floor pattern: light fill, border band
            if solid:
                G.cv_fill(canvas, g, (pr.randint(236, 250),) * 3)
                G.cv_outline(canvas, V.geom(q.buffer(-0.35, join_style=2)), G.mix(ink, (255, 255, 255), 0.55), lw * 0.6)
            G.cv_outline(canvas, g, G.mix(ink, (255, 255, 255), 0.35), lw * 0.8)
        elif kind == "line":
            G.cv_outline(canvas, g, ink, lw * 0.8)
        else:
            if solid:
                if solid == "shadow":
                    d = 0.2 * V.S
                    _shade_soft(canvas, affinity.translate(g, d, d).difference(g), 0.85, W, H, max(1, int(0.12 * V.S)))
                G.cv_fill(canvas, g, (pr.randint(246, 255),) * 3)
            G.cv_outline(canvas, g, ink, lw)


def _frame_crop(canvas, labelled, focus_px, pr):
    """Crop to the drawing as the real sheets do (they fill ~75% of the image, our full sheets
    ~47%), sometimes cutting through its edges (HF14 11, 12). Returns canvas, labelled or None."""
    H, W = canvas.shape[:2]
    bx0, by0, bx1, by1 = focus_px.bounds
    sw, sh = bx1 - bx0, by1 - by0
    x0 = int(max(0, bx0 - pr.uniform(-0.05, 0.06) * sw)); x1 = int(min(W, bx1 + pr.uniform(-0.05, 0.06) * sw))
    y0 = int(max(0, by0 - pr.uniform(-0.05, 0.06) * sh)); y1 = int(min(H, by1 + pr.uniform(-0.05, 0.06) * sh))
    if x1 - x0 < 0.3 * W or y1 - y0 < 0.3 * H:
        return None
    clip = box(x0, y0, x1, y1)
    lab = []
    for fam, g in labelled:
        q = g.buffer(0).intersection(clip)
        if not q.is_empty and q.area > 400:
            lab.append((fam, affinity.translate(q, -x0, -y0)))
    if not lab:
        return None
    return np.ascontiguousarray(canvas[y0:y1, x0:x1]), lab


def compose_floor(image_id, seed, mode_weights):
    house = _house(image_id, seed, mode_weights)
    r = random.Random(seed * 1_000_003 + image_id * 191 + 61)
    lay = plan_layout(house, random.Random(seed * 1_000_003 + image_id * 193 + 67))
    rooms, foot = lay["rooms"], lay["foot"]
    blocks = house["blocks"]
    mb = blocks[0]
    ink = (r.randint(0, 30),) * 3
    colour = r.random() < 0.3
    fam_seed = r.randrange(1 << 30)
    # three sheet types, in the proportions the Gemini + real floor plans show
    sheet = r.choices(["finish", "rendered", "underfloor"], weights=[55, 22, 23])[0]
    x0f, y0f, x1f, y1f = foot.bounds
    # --plan-v2 (G.PLAN_V2, 2026-10-01): own stream, so 0 leaves every pool byte-identical
    v2 = bool(getattr(G, "PLAN_V2", 0))
    pr = random.Random(seed * 1_000_003 + image_id * 199 + 73)
    v3 = getattr(G, "PLAN_V2", 0) >= 2         # plan v3: furniture, framing (own stream p3)
    p3 = random.Random(seed * 1_000_003 + image_id * 211 + 89)
    if v3:      # one MUDROOM / PANTRY / ... per house; repeats get numbered (BATH 2) or a sibling name
        seen = set()
        alts = {"MUDROOM": ["ENTRY", "STORAGE"], "LAUNDRY": ["UTILITY", "STORAGE"], "PANTRY": ["STORAGE", "LINEN"],
                "UTILITY": ["STORAGE", "MECH."], "CLOSET": ["LINEN", "COATS", "STORAGE"], "W.I.C.": ["CLOSET", "STORAGE"],
                "POWDER": ["BATH 2", "1/2 BATH"], "HALL": ["CORRIDOR", "HALLWAY"], "ENTRY": ["FOYER"],
                "OFFICE": ["STUDY", "DEN"], "DINING": ["NOOK"], "FAMILY ROOM": ["DEN", "BONUS ROOM"]}
        for rm in rooms:
            nm = rm["name"]
            if nm in seen:
                opts = [a for a in alts.get(nm, []) if a not in seen]
                k = 2
                while f"{nm} {k}" in seen:
                    k += 1
                nm = p3.choice(opts) if opts and nm != "BATH" else f"{nm} {k}"
            seen.add(nm)
            rm["name"] = nm
            if "GARAGE" in nm:          # "2-CAR GARAGE" was zoned a bedroom and took the bedroom finish
                rm["zone"] = "garage"
    if v2 and pr.random() < 0.15:
        sheet = "ceiling"                       # reflected ceiling / electrical plan (HF14 12)

    # ---- exterior hardscape (rendered sheets always, finish sheets sometimes)
    hard_polys = []
    if sheet == "rendered" or (sheet == "finish" and r.random() < 0.25):
        pieces = []
        if r.random() < 0.8:
            pw = r.uniform(0.35, 1.0) * (x1f - x0f)
            px = r.uniform(x0f, x1f - pw)
            pieces.append(box(px, y1f, px + pw, y1f + r.uniform(7, 16)))
        if r.random() < 0.6:
            fx = mb.x0 + house["front_door_u"] * mb.w
            ww = r.uniform(3, 5.5)
            pieces.append(box(fx - ww / 2, y0f - r.uniform(6, 16), fx + ww / 2, y0f))
            if v3:      # run the walk on to the wall where the footprint is notched (no floating strip)
                bridge = box(fx - ww / 2, y0f - 0.1, fx + ww / 2, y1f).difference(foot)
                top = [q for q in G.polys_of(bridge) if q.bounds[1] <= y0f]
                if top:
                    pieces.append(top[0])
        if r.random() < 0.35:
            dd, dl = r.uniform(5, 10), r.uniform(10, 0.8 * (y1f - y0f) + 10)
            yy = r.uniform(y0f, max(y0f, y1f - dl + 6))
            pieces.append(box(x1f, yy, x1f + dd, yy + dl) if r.random() < 0.5 else box(x0f - dd, yy, x0f, yy + dl))
        if not pieces:
            pieces.append(box(x0f + 2, y1f, min(x1f, x0f + 22), y1f + r.uniform(8, 14)))
        hard = unary_union(pieces).difference(foot.buffer(0.3, join_style=2))
        hard_polys = [q for q in G.polys_of(hard) if q.area > 12]
        if not hard_polys and sheet == "rendered":              # the rendered sheet's question is the hardscape
            pw = min(22.0, x1f - x0f)
            hard_polys = [box(x0f, y1f + 0.5, x0f + pw, y1f + r.uniform(8, 14))]
        # plan v2: a covered patio in a notch of the footprint, walled on two sides and
        # butting the interior floors (HF14 0), half the time
        if v2 and pr.random() < 0.5:
            notch = _footprint_notch(foot, pr)
            if notch is not None:
                hard_polys = [q for q in G.polys_of(unary_union(hard_polys + [notch])) if q.area > 12]
    soffit = None
    if sheet == "ceiling":
        soffit = _soffit_band(foot, pr)
        notch = _footprint_notch(foot, pr)
        if notch is not None and pr.random() < 0.7:          # porch ceiling: the same soffit boards
            soffit = unary_union([soffit, notch])

    # ---- finishes: one material per zone; identical materials are ONE family
    zone_mat = {}
    if sheet == "finish":
        ang = r.choice([0, 90])
        zone_mat["living"] = floor_material(r.choices(*LIVING_FIN)[0], r, fam_seed + 1, colour, ang)
        zone_mat["wet"] = floor_material(r.choices(*WET_FIN)[0], r, fam_seed + 2, colour)
        zone_mat["bed"] = zone_mat["living"] if r.random() < 0.35 else \
            floor_material(r.choices(*BED_FIN)[0], r, fam_seed + 3, colour, ang)
        zone_mat["garage"] = floor_material(r.choices(*GARAGE_FIN)[0], r, fam_seed + 4, colour)
        for z in ("bed", "living", "wet", "garage"):          # some zones left blank, as Gemini does
            if r.random() < {"bed": 0.3, "living": 0.2, "wet": 0.15, "garage": 0.25}[z]:
                zone_mat[z] = None
        present = [rm["zone"] for rm in rooms]
        if all(zone_mat.get(z) is None for z in present):       # something on the sheet must be patterned
            z0 = max(set(present), key=present.count)
            fin = {"living": LIVING_FIN, "bed": BED_FIN, "wet": WET_FIN, "garage": GARAGE_FIN}[z0]
            zone_mat[z0] = floor_material(r.choices(*fin)[0], r, fam_seed + 1, colour, ang)
    elif sheet == "rendered":
        fl = floor_material(r.choice(["plank", "plank_lines", "tile"]), r, fam_seed + 1, False, r.choice([0, 90]))
        fl["base"] = (r.randint(222, 242),) * 3
        fl["line"] = G.mix(fl["base"], (0, 0, 0), r.uniform(0.12, 0.25))
        if v2:      # real rendered interiors show their planks / tile clearly (HF14 0)
            fl["line"] = G.mix(fl["base"], (0, 0, 0), pr.uniform(0.18, 0.4))
        for z in ("living", "bed"):
            zone_mat[z] = fl
        zone_mat["wet"] = floor_material("tile", r, fam_seed + 2, False)
        zone_mat["wet"]["base"], zone_mat["wet"]["line"] = fl["base"], fl["line"]
        zone_mat["garage"] = floor_material("concrete", r, fam_seed + 4, False) if r.random() < 0.6 else None
    floor_fams = _separate_all([zone_mat.get(z) for z in ("living", "bed", "wet", "garage")])
    if v3:
        # herringbone / basket / wood-grain tiles at material scale (planks ~0.3-0.5 ft, not 2 ft), and
        # floor linework a lighter pen than the walls; one factor for all, so families stay apart
        k_t, k_l = p3.uniform(0.3, 0.5), p3.uniform(0.3, 0.65)     # real floors are soft (HF14 0, 7, 12)
        mats = list({id(m): m for m in zone_mat.values() if m is not None}.values())
        if not any(m["kind"] == "plank" for m in mats):          # staggered planks, not bare parallel lines
            for m in mats:
                if m["kind"] == "plank_lines" and p3.random() < 0.65:
                    m["kind"], m["w"] = "plank", max(0.35, m["sp"])
                    break
        if p3.random() < 0.4:                                   # a light tone under every floor on the sheet
            tone = p3.randint(226, 244)
            for m in mats:
                if m["base"] == (255, 255, 255):
                    m["base"] = (tone,) * 3
        for m in mats:
            if m["kind"] in TILES and m["kind"] not in ("t_earth", "t_aggregate"):
                m["sp"] *= k_t
            elif m["kind"] == "basket":
                m["sp"] *= p3.uniform(0.45, 0.65)
            elif m["kind"] == "dots" and p3.random() < 0.7:
                m["jit"] = p3.uniform(0.12, 0.35)
            elif m["kind"] == "concrete":
                m["tri"] = p3.random() < 0.7
            m["line"] = G.mix(m["line"], m["base"], k_l)

    # ---- layout
    allg = unary_union([foot] + hard_polys + ([soffit] if soffit is not None else []))
    ex0, ey0, ex1, ey1 = allg.bounds
    grid_on = r.random() < (0.55 if sheet != "finish" else 0.35)
    dims_on = r.random() < 0.75
    pad = r.uniform(9, 13) if (grid_on or dims_on) else r.uniform(3, 6)
    bot = r.uniform(7, 10)
    total_w, total_h = (ex1 - ex0) + 2 * pad, (ey1 - ey0) + pad + bot + 2
    S = min(130.0, r.uniform(*G.LONG_SIDE_PX) / max(total_w, total_h))
    W, H = int(total_w * S), int(total_h * S)
    V = G.View(S, (pad - ex0) * S, (pad - ey0) * S)
    canvas = np.full((H, W, 3), 255, np.uint8)
    tq = G.TextQueue()
    size = max(12, max(W, H) * r.uniform(0.004, 0.0065))
    lw_thin = max(1.0, S * r.uniform(0.009, 0.015))
    lw_heavy = max(1.6, S * r.uniform(0.02, 0.032))
    lw_pat = max(0.8, lw_thin * r.uniform(0.55, 0.8))
    app = {"ink": ink, "text_px": size, "outline_lw": lw_thin}
    labelled = []

    # ---- hardscape
    if hard_polys:
        hk = r.choices(["ashlar", "stone", "grid", "plank", "concrete", "basket", "t_brick", "t_hbone", "t_aggregate"],
                       weights=[22, 12, 13, 22, 5, 7, 8, 6, 5])[0]
        if v3 and hk == "grid" and p3.random() < 0.65:          # real patios: irregular stone (HF14 0)
            hk = p3.choice(["ashlar", "stone"])
        hs = _hard_style(hk, r, lw_pat, fam_seed + 11, colour)
        _separate(hs.kind, hs.params, hs.base, floor_fams)
        if v2 and sheet == "rendered" and pr.random() < 0.5:
            # paving in the interior floors' grey, a little darker (HF14 0: grey pavers
            # beside grey planks), instead of a tan that sets it apart
            gv = max(150, zone_mat["living"]["base"][0] - pr.randint(5, 30))
            hs.base, hs.line = (gv,) * 3, G.mix((gv,) * 3, (0, 0, 0), pr.uniform(0.25, 0.45))
        for p in hard_polys:
            st = hs
            if st.kind == "hatch":                               # deck boards run the long way of each piece
                bx = p.bounds
                st = G.Style("hatch", st.base, st.line, st.lw,
                             {"sp": st.params["sp"], "angle": 0 if bx[2] - bx[0] >= bx[3] - bx[1] else 90},
                             st.seed, "hardscape")
            _fill(canvas, V.geom(p), st, S, W, H)
            G.cv_outline(canvas, V.geom(p), ink, lw_thin)
            if v3 and st.kind == "hatch" and p3.random() < 0.3:          # rim / edge board
                G.cv_outline(canvas, V.geom(p.buffer(-0.6, join_style=2)), ink, lw_thin * 0.8)
            labelled.append(("hardscape", V.geom(p)))
        if v3:      # patio furniture on the paving (HF14 0), inside the label as annotators draw it
            for p in hard_polys:
                bx = p.bounds
                if p.area > 90 and min(bx[2] - bx[0], bx[3] - bx[1]) > 7 and p3.random() < 0.55:
                    c = p.representative_point()
                    _draw_furniture(canvas, V, _dining(c.x, c.y, p3), ink, lw_thin, p3.random() < 0.5, W, H, p3,
                                    clip=p.buffer(-0.3))

    # ---- soffit band (ceiling sheets): soffit boards running along each stretch
    if soffit is not None:
        sp = pr.uniform(0.25, 0.45)
        sline = (pr.randint(90, 160),) * 3
        if v3 and hard_polys and hs.kind == "hatch" and abs(hs.params.get("sp", 0) - sp) < 0.15:
            sp = hs.params["sp"] + p3.choice([-1, 1]) * p3.uniform(0.15, 0.3)    # soffit boards != deck boards
            sp = max(0.2, sp)
        perp = p3.random() < 0.65              # v3: boards across the band rather than along it
        if v3:
            sp = p3.uniform(0.15, 0.3)          # T&G boards: dense (HF14 12)
        for q in G.polys_of(soffit):
            bx = q.bounds
            st = G.Style("hatch", (255, 255, 255), sline, lw_pat,
                         {"sp": sp, "angle": 0 if bx[2] - bx[0] >= bx[3] - bx[1] else 90}, fam_seed + 31, "soffit")
            if v3:      # one board direction per face of the house, mitred at the corners
                for piece, ang in _per_face(q, foot):
                    a2 = (ang + 90) % 180 if perp else ang
                    _fill(canvas, V.geom(piece), G.Style("hatch", (255, 255, 255), sline, lw_pat, {"sp": sp, "angle": a2},
                                                         fam_seed + 31, "soffit"), S, W, H)
            else:
                _fill(canvas, V.geom(q), st, S, W, H)
            G.cv_outline(canvas, V.geom(q), ink, lw_thin)
            if v3 and p3.random() < 0.6:                  # fascia / trim line along the edge
                G.cv_outline(canvas, V.geom(q.buffer(-p3.uniform(0.25, 0.5), join_style=2)), sline, lw_thin * 0.8)
            labelled.append(("soffit", V.geom(q)))

    stairs = plan_stairs(rooms, r) if sheet in ("finish", "rendered") and r.random() < 0.45 else []
    stair_u = unary_union([st["poly"] for st in stairs]) if stairs else None
    # ---- floors: regions = rooms of a material + the openings between two such rooms
    if sheet in ("finish", "rendered"):
        fams = {}
        for k, rm in enumerate(rooms):
            m = zone_mat.get(rm["zone"])
            if m is None:
                continue
            fams.setdefault(_mkey(m), (m, []))[1].append(rm["inner"])
        for (i, j, op) in lay["openings"]:
            mi, mj = zone_mat.get(rooms[i]["zone"]), zone_mat.get(rooms[j]["zone"])
            if mi is not None and mj is not None and _mkey(mi) == _mkey(mj):
                fams[_mkey(mi)][1].append(op)
            elif v3 and (mi is not None or mj is not None):    # the finish runs on through the doorway
                m_ = mi if mi is not None else mj
                fams[_mkey(m_)][1].append(op)
        for n, (key, (m, polys)) in enumerate(sorted(fams.items(), key=lambda kv: str(kv[0]))):
            reg = unary_union(polys)
            if stair_u is not None:
                reg = reg.difference(stair_u)
            if v3 and p3.random() < 0.7:                 # each room starts its own tile / plank layout
                for q in G.polys_of(reg):
                    mq = dict(m, ox=p3.uniform(0, 3) * S, oy=p3.uniform(0, 3) * S)
                    draw_floor(canvas, V.geom(q), mq, S, W, H, lw_pat)
            else:
                draw_floor(canvas, V.geom(reg), m, S, W, H, lw_pat)
            if sheet == "finish":                        # patterned finish => always labelled
                labelled.append((f"floor{n}", V.geom(reg)))
        if v3 and p3.random() < 0.7:                     # threshold lines across the openings
            for (i, j, op) in lay["openings"]:
                G.cv_outline(canvas, V.geom(op), ink, lw_thin * 0.7)
    elif sheet == "underfloor":
        # crawlspace / slab / framing plan (real HF14 7, Gemini under-floor sheets)
        slab_rooms = [rm["inner"] for rm in rooms if rm["zone"] == "garage"]
        crawl = foot.buffer(-lay["t_ext"], join_style=2)
        if slab_rooms:
            crawl = crawl.difference(unary_union([rm["cell"] for rm in rooms if rm["zone"] == "garage"]).buffer(lay["t_ext"]))
        ck = r.choices(["hatch45", "joists", "dots", "basket", "carpet", "blank", "t_earth", "t_steel", "t_cuprous",
                        "t_aluminium"], weights=[18, 20, 9, 8, 8, 12, 10, 5, 5, 5])[0]
        if ck == "blank" and not slab_rooms:
            ck = "hatch45"
        fin_rooms = []
        if v3 and len(rooms) >= 3 and p3.random() < 0.45:
            # part of the level is finished rooms, the rest crawlspace (HF14 7: rec room beside the under-floor)
            ys_ = sorted(rm["cell"].centroid.y for rm in rooms)
            cut = ys_[len(ys_) // 2]
            side = p3.random() < 0.5
            fin_rooms = [rm for rm in rooms if (rm["cell"].centroid.y >= cut) == side and rm["zone"] != "garage"]
            if fin_rooms and len(fin_rooms) < len(rooms):
                crawl = foot.buffer(-lay["t_ext"], join_style=2)          # fill runs to the wall centre lines
                if slab_rooms:
                    crawl = crawl.difference(unary_union([rm["cell"] for rm in rooms if rm["zone"] == "garage"])
                                             .buffer(lay["t_int"] / 2, join_style=2))
                crawl = crawl.difference(unary_union([rm["cell"] for rm in fin_rooms]).buffer(lay["t_int"] / 2,
                                                                                              join_style=2))
                crawl = unary_union([q for q in G.polys_of(crawl) if q.area > 40])
            else:
                fin_rooms = []
        # piers on a grid (holes in the crawlspace label, like skylights in a roof)
        px0, py0, px1, py1 = crawl.bounds
        sp_p = r.uniform(5, 8)
        pier_rows, piers = [], []
        if v3:      # girder spacing and pier spacing differ, spans uneven, footing + post (round or square)
            spx, foot_r = p3.uniform(4.5, 8.5), p3.uniform(0.6, 1.1)
            rows_y, yy = [], py0 + p3.uniform(0.6, 1.0) * sp_p
            while yy < py1 - 1:
                rows_y.append(yy)
                yy += sp_p * p3.uniform(0.75, 1.3)
            round_p = p3.random() < 0.3
            for yy in rows_y:
                pier_rows.append(yy)
                sx_row = spx * p3.uniform(0.85, 1.2)
                xx = px0 + p3.uniform(0.4, 1.0) * sx_row
                while xx < px1 - 1:
                    fr = foot_r
                    q = Point(xx, yy).buffer(fr, 20) if round_p else box(xx - fr, yy - fr, xx + fr, yy + fr)
                    if crawl.buffer(-0.3).contains(q):
                        piers.append(q)
                    xx += sx_row * p3.uniform(0.9, 1.1)
        else:
            yy = py0 + sp_p
            while yy < py1 - 1:
                pier_rows.append(yy)
                xx = px0 + sp_p
                while xx < px1 - 1:
                    q = box(xx - 0.6, yy - 0.6, xx + 0.6, yy + 0.6)
                    if crawl.buffer(-0.3).contains(q):
                        piers.append(q)
                    xx += sp_p
                yy += sp_p
        if ck != "blank":
            cm = floor_material(ck, r, fam_seed + 21, colour, r.choice([0, 90]))
            if v3 and ck == "dots" and p3.random() < 0.7:
                cm["jit"] = p3.uniform(0.12, 0.35)
            if v3 and ck == "concrete":
                cm["tri"] = p3.random() < 0.7
            if v3:
                cm["line"] = G.mix(cm["line"], cm["base"], p3.uniform(0.25, 0.6))
            draw_floor(canvas, V.geom(crawl), cm, S, W, H, lw_pat)
            lab = crawl.difference(unary_union(piers)) if piers else crawl
            labelled.append(("crawl", V.geom(lab)))
        if slab_rooms:
            sm = floor_material(r.choice(["concrete", "concrete", "dots", "t_aggregate", "t_aggregate"]), r,
                                fam_seed + 22, colour)
            if v3 and ck != "blank" and sm["kind"] == cm["kind"]:     # never the crawlspace's pattern in a new tint
                sm = floor_material(p3.choice([k for k in ("concrete", "t_aggregate", "dots") if k != cm["kind"]]), p3,
                                    fam_seed + 23, colour)
            if ck != "blank":
                _separate(sm["kind"], sm, sm["base"], [(cm["kind"], cm, cm["base"])])
            reg = unary_union(slab_rooms)
            if v3 and sm["kind"] == "concrete":
                sm["tri"] = p3.random() < 0.7
            draw_floor(canvas, V.geom(reg), sm, S, W, H, lw_pat)
            if v3:
                G.cv_outline(canvas, V.geom(reg), ink, lw_thin)                # slab edge
            if ck == "blank" or r.random() < 0.85:
                labelled.append(("slab", V.geom(reg)))
        # girders dashed through the pier rows (clipped to the crawlspace), then the piers
        gcol = G.mix(ink, (255, 255, 255), 0.3)
        dbl = v3 and p3.random() < 0.45
        for yy in pier_rows:
            for seg in G.lines_of(LineString([(px0, yy), (px1, yy)]).intersection(crawl)):
                c = list(seg.coords)
                if dbl:         # girder drawn as its width, hidden
                    for dy in (-0.25, 0.25):
                        _dash_line(canvas, V.px(c[0][0], c[0][1] + dy), V.px(c[-1][0], c[-1][1] + dy), gcol, lw_thin,
                                   [(0.8 * S, 0.3 * S)])
                else:
                    _dash_line(canvas, V.px(*c[0]), V.px(*c[-1]), gcol, lw_thin, [(0.8 * S, 0.3 * S)])
        for q in piers:
            G.cv_fill(canvas, V.geom(q), (255, 255, 255))
            G.cv_outline(canvas, V.geom(q), ink, lw_thin)
            if v3:                                                            # post on the footing
                c = q.centroid
                G.cv_outline(canvas, V.geom(box(c.x - 0.25, c.y - 0.25, c.x + 0.25, c.y + 0.25)), ink, lw_thin)
        if v3 and not crawl.is_empty:
            # joist span arrows across the girders
            for _ in range(p3.randint(1, 3)):
                c = crawl.buffer(-2).representative_point() if not crawl.buffer(-2).is_empty else crawl.representative_point()
                L = p3.uniform(4, 9)
                a, b = V.px(c.x, c.y - L / 2), V.px(c.x, c.y + L / 2)
                G.cv_line(canvas, a, b, ink, lw_thin)
                for (pt, d) in ((a, 1), (b, -1)):
                    G.cv_line(canvas, pt, (pt[0] - 0.35 * S, pt[1] + d * 0.7 * S), ink, lw_thin)
                    G.cv_line(canvas, pt, (pt[0] + 0.35 * S, pt[1] + d * 0.7 * S), ink, lw_thin)
                G.cv_line(canvas, (a[0] - 0.8 * S, (a[1] + b[1]) / 2), (a[0] + 0.8 * S, (a[1] + b[1]) / 2), ink, lw_thin)
    if not labelled and sheet != "underfloor":
        # every sheet must ask something: label the largest patterned region
        pass

    for st in stairs:
        draw_stairs(canvas, tq, V, st, ink, lw_thin, size, S)
    # ---- walls
    if sheet == "underfloor":
        wg = V.geom(lay["walls_full"].difference(unary_union([rm["inner"] for rm in rooms]).buffer(0)))
        ring = foot.difference(foot.buffer(-lay["t_ext"], join_style=2))
        if v3 and fin_rooms:            # the finished rooms' walls and doors, as on a floor plan
            fz = unary_union([rm["cell"] for rm in fin_rooms]).buffer(0.3, join_style=2)
            fw = unary_union([q for q in G.polys_of(lay["walls"].intersection(fz)) if q.area > 6 * lay["t_int"] ** 2 + 2])
            for d in lay["doors"]:      # doors of the finished rooms
                dc = Point(d[1], d[2]) if d[0] == "v" else Point(d[2], d[1])
                if fz.contains(dc):
                    _draw_door(canvas, V, d, lay["t_ext"], lay["t_int"], ink, lw_thin, S, p3)
            fw_look = p3.random() < 0.5
            _mfill(canvas, V.geom(fw), (p3.randint(0, 120),) * 3 if fw_look else (255, 255, 255), W, H)
            G.cv_outline(canvas, V.geom(fw), ink, lw_thin if fw_look else lw_heavy * 0.8)
        poche = v3 and p3.random() < 0.6
        _mfill(canvas, V.geom(ring), (255, 255, 255) if not poche else (p3.randint(150, 215),) * 3, W, H)
        G.cv_outline(canvas, V.geom(ring), ink, lw_heavy)
        _dash_poly(canvas, V.geom(foot.buffer(0.9, join_style=2)), G.mix(ink, (255, 255, 255), 0.4), lw_thin,
                   [(0.6 * S, 0.3 * S)])                          # footing below
    else:
        wall_look = r.choices(["grey", "black", "double"], weights=[50, 30, 20])[0]
        if v3 and wall_look == "double" and p3.random() < 0.75:     # cut walls read heavier than everything else
            wall_look = p3.choice(["grey", "black", "hatch"])
        wall_sh = sheet == "rendered" and (r.random() < 0.45 or (v3 and p3.random() < 0.65)) or \
            (v3 and sheet == "finish" and p3.random() < 0.3)
        if wall_sh:
            off = (r.uniform(0.35, 0.8), r.uniform(0.35, 0.8))
            sh = affinity.translate(lay["walls_full"], *off).difference(lay["walls_full"])
            k_sh = r.uniform(0.7, 0.85)
            if v3:
                if p3.random() < 0.65:                       # cast outside the building only
                    sh = affinity.translate(foot, *off).difference(foot)
                _shade_soft(canvas, V.geom(sh), k_sh, W, H, max(1, int(p3.uniform(0.15, 0.4) * S)))
            else:
                _shade(canvas, V.geom(sh), k_sh, W, H)
        wg = V.geom(lay["walls"])
        if v3 and p3.random() < 0.6:        # exterior walls read heavier than partitions
            wg = V.geom(unary_union([lay["walls"], foot.buffer(p3.uniform(0.15, 0.35), join_style=2).difference(foot)
                                     .difference(unary_union(hard_polys) if hard_polys else Point(0, 0).buffer(0))]))
        if wall_look == "grey":
            _mfill(canvas, wg, (r.randint(80, 150),) * 3, W, H)
            G.cv_outline(canvas, wg, ink, lw_thin * p3.uniform(1.3, 2.2) if v3 else lw_thin)
        elif wall_look == "black":
            _mfill(canvas, wg, (r.randint(0, 35),) * 3, W, H)
        elif wall_look == "hatch":                  # v3: diagonal poché hatch, heavy outline
            _mfill(canvas, wg, (255, 255, 255), W, H)
            hl = np.full((H, W, 3), 255, np.uint8)
            sp_px = max(3, int(p3.uniform(0.25, 0.45) * S))
            for c0 in range(-H, W, sp_px):
                cv2.line(hl, (c0, H), (c0 + H, 0), (60, 60, 60), max(1, int(lw_thin * 0.7)), cv2.LINE_AA)
            m_ = np.zeros((H, W), np.uint8)
            for q in G.polys_of(wg):
                m_ = np.maximum(m_, G.poly_mask(q, W, H))
            canvas[m_ > 0] = hl[m_ > 0]
            G.cv_outline(canvas, wg, ink, lw_heavy)
        else:
            _mfill(canvas, wg, (255, 255, 255), W, H)
            G.cv_outline(canvas, wg, ink, lw_heavy * 0.8 if v3 else lw_thin * 1.2)
        if v3 and len(rooms) >= 3 and p3.random() < 0.4:
            # existing vs new construction: some rooms' walls in a second rendering (HF14 12 mixes them)
            # by wall class, so a run never switches style part way: the interior partitions or the shell
            shell = foot.difference(foot.buffer(-lay["t_ext"] - 0.05, join_style=2))
            alt = lay["walls"].difference(shell) if p3.random() < 0.6 else lay["walls"].intersection(shell)
            ag = V.geom(alt)
            if wall_look in ("grey", "black", "hatch"):
                _mfill(canvas, ag, (255, 255, 255), W, H)
                G.cv_outline(canvas, ag, ink, lw_thin * 1.1)
            else:
                _mfill(canvas, ag, (p3.randint(90, 160),) * 3, W, H)
                G.cv_outline(canvas, ag, ink, lw_thin)
        t = lay["t_ext"]
        win_style = r.choices(["3line", "2line", "sill"], weights=[45, 30, 25])[0]
        for (o, pos, c, w) in lay["windows"]:
            _draw_window(canvas, V, o, pos, c, w, t, win_style, ink, lw_thin)
        for d in lay["doors"]:
            _draw_door(canvas, V, d, t, lay["t_int"], ink, lw_thin, S, r)
    if grid_on:
        xs = _merge_close([v for b in blocks for v in (b.x0, b.x1)], 4.0)
        ys = _merge_close([v for b in blocks for v in (b.y0, b.y1)], 4.0)
        _grid(canvas, tq, V, xs, ys, (ex0 - pad + 2.2, ey0 - pad + 2.2, ex1 + pad - 2.2, ey1 + pad * 0.5),
              ink, lw_thin * 0.9, size, S, r)

    # ---- fixtures, tags, text
    sf = r.random() < 0.5
    for rm in rooms:
        x0, y0, x1, y1 = rm["inner"].bounds
        if sheet != "underfloor" and v3:
            clip = rm["inner"].buffer(0.05)
            if stair_u is not None:
                clip = clip.difference(stair_u.buffer(0.3))
            pieces = _room_furniture(rm["name"], x0, y0, x1, y1, p3, clip)
            if sheet == "ceiling":          # the floor plan ghosted under the RCP (HF14 12)
                _draw_furniture(canvas, V, [q for q in pieces if q[1] != "rug"],
                                G.mix(ink, (255, 255, 255), p3.uniform(0.5, 0.7)), lw_thin * 0.8, False, W, H, p3, clip=clip)
            elif p3.random() < 0.9:
                _draw_furniture(canvas, V, pieces, ink, lw_thin * p3.uniform(0.6, 0.9),
                                "shadow" if wall_sh else True,
                                W, H, p3, clip=clip)
        if sheet != "underfloor":
            if not v3 and rm["inner"].area > 0.9 * (x1 - x0) * (y1 - y0) and \
                    not (stair_u is not None and rm["inner"].intersects(stair_u)):   # rectangles, no stair
                G.draw_fixtures(canvas, V, rm["name"], (x0, y0, x1, y1), ink, lw_thin, S, r)
            rp_ = rm["inner"].representative_point() if rm["inner"].area < 0.9 * (x1 - x0) * (y1 - y0) else \
                Point((x0 + x1) / 2, (y0 + y1) / 2)
            c = V.px(rp_.x, rp_.y)
            if rm["inner"].area > 30:
                tq.add(c, rm["name"], size * 0.95, ink, bold=True, anchor="mm")
                if r.random() < 0.7:
                    sub = f"{int(rm['inner'].area)} SF" if sf else f"{G.ft_label(x1 - x0)} x {G.ft_label(y1 - y0)}"
                    tq.add((c[0], c[1] + size * 1.15), sub, size * 0.75, ink, anchor="mm")
    if sheet == "underfloor":
        cxy = foot.representative_point()
        tq.add(V.px(cxy.x, cxy.y), r.choice(["CRAWLSPACE", "CRAWL SPACE", "UNDER FLOOR", "(E) CRAWLSPACE"]), size * 1.2,
               ink, bold=True, anchor="mm")
        for rm in rooms:
            if rm["zone"] == "garage":
                c = rm["inner"].representative_point()
                tq.add(V.px(c.x, c.y), r.choice(['4" CONC. SLAB', "GARAGE SLAB ON GRADE", "CONCRETE SLAB"]), size, ink,
                       bold=True, anchor="mm")
        q = foot.buffer(-2).representative_point()
        G.draw_leader(canvas, tq, V, (q.x, q.y), r.choice(['2x10 FLOOR JOISTS @ 16" O.C.', "4x6 GIRDER ON PIERS",
                                                           'PIER BLOCK, TYP.', "6 MIL VAPOR BARRIER"]), app, S, 1)
        if r.random() < 0.6:
            _notes(tq, W * r.uniform(0.7, 0.78), H * r.uniform(0.05, 0.12), "NOTES:",
                   r.sample(["1. ALL STRUCTURAL CONCRETE TO BE 2500 PSI MIN.", "2. PROVIDE CROSS VENTILATION.",
                             "3. VERIFY ALL DIMENSIONS IN FIELD.", "4. SEE STRUCTURAL FOR FRAMING."], 3), ink, size * 0.8)
    for p in hard_polys[:2]:
        if p.area > 80 and r.random() < 0.5:
            c = p.representative_point()
            tq.add(V.px(c.x, c.y), r.choice(["COVERED PATIO", "PATIO", "DECK", "TERRACE", "(N) DECK"]), size, ink,
                   anchor="mm")
    if sheet == "ceiling":
        # dense MEP clutter, as on HF14 12: switch legs (dashed arcs), fixtures, duct runs, notes
        mcol = (pr.randint(20, 90),) * 3
        for _ in range(pr.randint(14, 32)):
            rm = pr.choice(rooms)
            x0, y0, x1, y1 = rm["inner"].bounds
            cx, cy = pr.uniform(x0, x1), pr.uniform(y0, y1)
            kind = pr.random()
            if kind < 0.45:
                rad = pr.uniform(1.5, 5.0) * S
                for a in range(0, 360, 20):
                    cv2.ellipse(canvas, G._pt(V.px(cx, cy)), (int(rad * G.SCALE), int(rad * pr.uniform(0.5, 0.9) * G.SCALE)),
                                0, a, a + 11, G.bgr(mcol), max(1, int(lw_thin)), cv2.LINE_AA, G.SHIFT)
            elif kind < 0.8:
                rr_ = 0.45 * S
                cv2.circle(canvas, G._pt(V.px(cx, cy)), int(rr_ * G.SCALE), G.bgr(mcol), max(1, int(lw_thin)),
                           cv2.LINE_AA, G.SHIFT)
                for dx, dy in ((1, 1), (1, -1)):
                    a, b = V.px(cx - 0.32 * dx, cy - 0.32 * dy), V.px(cx + 0.32 * dx, cy + 0.32 * dy)
                    G.cv_line(canvas, a, b, mcol, lw_thin)
            else:
                ex, ey = pr.uniform(x0, x1), pr.uniform(y0, y1)
                _dash_line(canvas, V.px(cx, cy), V.px(ex, ey), mcol, lw_thin, [(0.7 * S, 0.35 * S)])
            tq.add(V.px(cx + 0.7, cy), pr.choice(["$", "B1", "A1", "SD", "F", "$D", "EF", "GFI"]), size * 0.75, mcol,
                   anchor="lm")
        for _ in range(pr.randint(3, 7)):
            rm = pr.choice(rooms)
            q = rm["inner"].representative_point()
            G.draw_leader(canvas, tq, V, (q.x, q.y), pr.choice(MEP_NOTES), app, S, pr.choice([1, -1]))
        if soffit is not None and pr.random() < 0.8:
            q = G.polys_of(soffit)[0].representative_point()
            G.draw_leader(canvas, tq, V, (q.x, q.y), pr.choice(["wood soffit, typ.", "WOOD SOFFIT", "(N) T&G SOFFIT"]),
                          app, S, pr.choice([1, -1]))
    if sheet not in ("underfloor", "ceiling") and r.random() < 0.15 and not v3:           # MEP overlay (HF14 12)
        mcol = (r.randint(20, 90),) * 3
        for _ in range(r.randint(4, 10)):
            rm = r.choice(rooms)
            x0, y0, x1, y1 = rm["inner"].bounds
            cx, cy = r.uniform(x0, x1), r.uniform(y0, y1)
            rad = r.uniform(1.5, 4.5) * S
            for a in range(0, 360, 24):
                cv2.ellipse(canvas, G._pt(V.px(cx, cy)), (int(rad * G.SCALE), int(rad * 0.7 * G.SCALE)), 0, a, a + 13,
                            G.bgr(mcol), max(1, int(lw_thin)), cv2.LINE_AA, G.SHIFT)
            tq.add(V.px(cx, cy), r.choice(["$", "B1", "A1", "SD", "F", "$D"]), size * 0.8, mcol, anchor="mm")
    if dims_on:
        xs = [v for rm in rooms for v in (rm["cell"].bounds[0], rm["cell"].bounds[2])]
        _dim_chain(canvas, tq, xs, "x", ey0 - r.uniform(2.5, 4), V, ink, lw_thin * 0.8, size * 0.8, S)
        if r.random() < 0.6:
            _dim_chain(canvas, tq, [x0f, x1f], "x", ey0 - r.uniform(5, 6.5), V, ink, lw_thin * 0.8, size * 0.8, S)
        ys = [v for rm in rooms for v in (rm["cell"].bounds[1], rm["cell"].bounds[3])]
        _dim_chain(canvas, tq, ys, "y", ex0 - r.uniform(2.5, 4), V, ink, lw_thin * 0.8, size * 0.8, S)
    if r.random() < 0.35:
        for _ in range(r.randint(1, 3)):
            rad = size * r.uniform(1.0, 1.3)
            side = r.choice(["S", "E", "W"])
            if side == "S":
                c, d = V.px(r.uniform(ex0, ex1), ey1 + pad * 0.4), (0, -1)
            else:
                c = V.px(ex0 - pad * 0.5 if side == "W" else ex1 + pad * 0.5, r.uniform(ey0, ey1))
                d = (1 if side == "W" else -1, 0)
            _section_head(canvas, tq, c, rad, d, str(r.randint(1, 4)), r.choice(["A3-1", "A4-1", "A4-2", "A5.1"]),
                          ink, lw_thin, size * 0.75)
    if r.random() < 0.4:
        _north(canvas, tq, V.px(ex0 - pad * 0.3, ey0 - pad * 0.3), size * 1.3, ink, lw_thin, size)
    title = {"finish": ["FLOOR PLAN", "FIRST FLOOR PLAN", "PROPOSED FLOOR PLAN", "FLOOR FINISH PLAN", "MAIN LEVEL PLAN"],
             "rendered": ["FLOOR PLAN", "SITE / FLOOR PLAN", "PROPOSED FLOOR PLAN"],
             "underfloor": ["UNDER-FLOOR FRAMING PLAN", "FOUNDATION PLAN", "GARAGE UNDER-FLOOR PLAN", "CRAWLSPACE PLAN"],
             "ceiling": ["REFLECTED CEILING PLAN", "ELECTRICAL PLAN", "LIGHTING / RCP PLAN", "MECHANICAL PLAN"]}[sheet]
    _title(canvas, tq, V, ex0, ey1 + bot * 0.55, ex0 + (ex1 - ex0) * r.uniform(0.3, 0.6), r.choice(title), r, ink,
           lw_thin, size)
    if not labelled:
        raise RuntimeError("no labelled region")
    canvas = _finish(canvas, tq, r)
    if v3 and p3.random() < 0.8:
        ring = pad * (p3.uniform(0.35, 1.0) if (grid_on or dims_on) else p3.uniform(0, 0.5))
        fc = _frame_crop(canvas, labelled, V.geom(allg.buffer(ring, join_style=2)), p3)
        if fc is not None:
            canvas, labelled = fc
            H, W = canvas.shape[:2]
    return canvas, _annotations(labelled, S, W, H, image_id, "freeform", colour)
