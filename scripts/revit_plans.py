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
                Rooms in EVERY block (wings too), doors / wide cased openings
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


def _shade(canvas, geom_px, k, W, H):
    m = np.zeros((H, W), np.uint8)
    for q in G.polys_of(geom_px):
        m = np.maximum(m, G.poly_mask(q, W, H))
    a = (m.astype(np.float32) / 255.0 * (1.0 - k))[..., None]
    canvas[:] = (canvas.astype(np.float32) * (1.0 - a)).astype(np.uint8)


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
PITCHED_KINDS = ["bond_facet", "bond_global", "eave_lines", "seam", "cross", "dots"]
PITCHED_W = [32, 10, 18, 16, 9, 5]


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
            "dot_fill": r.random() < 0.5, "lwk": r.uniform(0.55, 0.9)}


ROOF_LOOK = {"bond_facet": "bond", "bond_global": "bond", "eave_lines": "lines", "seam": "lines",
             "flat_seams": "lines", "cross": "diag", "dots": "dots", "membrane": "dots", "gravel": "stipple"}
ROOF_SP = {"bond_facet": "row", "bond_global": "row", "eave_lines": "sp", "seam": "seam_sp", "flat_seams": "seam_sp",
           "cross": "cross_sp", "dots": "dot_sp", "membrane": "dot_sp", "gravel": "density"}


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
            a, b = m[key], o[ROOF_SP[o["kind"]]]
            if max(a, b) / min(a, b) < 1.4:
                clash = b
                break
        if clash is None:
            return
        m[key] = clash * 1.6 if m[key] >= clash else clash / 1.6


def flat_material(r, seed):
    kind = r.choices(["gravel", "membrane", "flat_seams"], weights=[40, 35, 25])[0]
    base = (r.randint(236, 252),) * 3 if r.random() < 0.7 else (255, 255, 255)
    return {"kind": kind, "style": "grey", "base": base, "line": (r.randint(90, 160),) * 3, "seed": seed,
            "density": r.uniform(1.0, 3.0), "dot_sp": r.uniform(0.9, 1.8), "dot_fill": r.random() < 0.4,
            "seam_sp": r.uniform(2.5, 4.5), "seam_ang": r.choice([0, 90]), "lwk": r.uniform(0.5, 0.8)}


def _dotgrid(layer, ox, oy, S, sp, rpx, colour, filled, lw):
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
            cv2.circle(layer, G._pt(c), int(rpx * G.SCALE), G.bgr(colour), -1 if filled else max(1, int(lw)),
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
    mask = G.poly_mask(affinity.translate(poly_px, -x0, -y0), w, h)
    G.blend_mask(canvas[y0:y1, x0:x1], layer, mask)



# ----------------------------------------------------------------------------
# roof plan
# ----------------------------------------------------------------------------
def compose_roof(image_id, seed, mode_weights):
    house = _house(image_id, seed, mode_weights)
    r = random.Random(seed * 1_000_003 + image_id * 181 + 53)
    spec, faces = FC._FACES[FC._CUR[0]]
    chim = spec.get("chimney")
    tops = top_faces(faces, chim)
    # a flat face is a flat ROOF only over a flat-roofed block; elsewhere it is a
    # sliver where pitched roofs meet and is drawn / labelled with the roof
    flat_fp = unary_union([box(b["x0"] - b["ov"] - 0.1, b["y0"] - b["ov"] - 0.1, b["x0"] + b["w"] + b["ov"] + 0.1,
                               b["y0"] + b["d"] + b["ov"] + 0.1) for b in spec["blocks"] if b["roof"] == "flat"])
    for t in tops:
        if t["kind"] == "flat" and (flat_fp.is_empty or not flat_fp.contains(t["vis"].representative_point())):
            t["kind"] = "pitched"
    if not any(t["kind"] in ("pitched", "flat") for t in tops):
        raise RuntimeError("no roof faces")
    footprint = unary_union([box(b["x0"], b["y0"], b["x0"] + b["w"], b["y0"] + b["d"]) for b in spec["blocks"]])
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
            o = bl["ov"] + 0.05
            if bl["x0"] - o <= c.x <= bl["x0"] + bl["w"] + o and bl["y0"] - o <= c.y <= bl["y0"] + bl["d"] + o:
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
    return G.Style("concrete", base, line, lw, {"density": r.uniform(0.8, 1.8)}, seed, "hard")


def _fill(canvas, poly_px, st, S, W, H):
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


def plan_layout(house, r):
    blocks = house["blocks"]
    t_ext, t_int = r.uniform(0.55, 0.85), r.uniform(0.35, 0.5)
    foot = unary_union([G.rect(b.x0, b.y0, b.x1, b.y1) for b in blocks])
    inner_foot = foot.buffer(-t_ext, join_style=2)
    cells, taken = [], None
    for b in blocks:
        R = G.rect(b.x0, b.y0, b.x1, b.y1)
        reg = R if taken is None else R.difference(taken)
        taken = R if taken is None else taken.union(R)
        for q in G.polys_of(reg):
            if q.area < 20:
                continue
            if b.kind == "garage":
                cells.append((q, "garage"))
                continue
            bx = q.bounds
            if q.area > 0.97 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
                for (x0, y0, x1, y1) in G.split_rooms(r, *bx):
                    cells.append((G.rect(x0, y0, x1, y1), "room"))
            else:
                cells.append((q, "room"))
    rooms = []
    for (c, kind) in cells:
        inner = c.buffer(-t_int / 2, join_style=2).intersection(inner_foot)
        if inner.is_empty or inner.area < 8:
            continue
        inner = max(G.polys_of(inner), key=lambda q: q.area)
        rooms.append({"cell": c, "inner": inner, "kind": kind})
    # names by size / shape
    free = sorted([k for k, rm in enumerate(rooms) if rm["kind"] != "garage"], key=lambda k: -rooms[k]["inner"].area)
    names = {}
    for k in [k for k, rm in enumerate(rooms) if rm["kind"] == "garage"]:
        names[k] = r.choice(["GARAGE", "GARAGE", "2-CAR GARAGE"])
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
            sh = rooms[i]["cell"].buffer(0.05).intersection(rooms[j]["cell"].buffer(0.05))
            if sh.is_empty:
                continue
            x0, y0, x1, y1 = sh.bounds
            L = max(x1 - x0, y1 - y0)
            if L > 3.6:
                pairs.append((i, j, (x0, y0, x1, y1), L))
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
        w = min(L - 1.0, r.uniform(4, 9)) if wide else min(L - 1.0, r.choice([2.5, 2.67, 3.0]))
        if vert:
            c = r.uniform(y0 + 0.5 + w / 2, y1 - 0.5 - w / 2) if y1 - y0 > w + 1 else (y0 + y1) / 2
            x = (x0 + x1) / 2
            op = G.rect(x - t_int, c - w / 2, x + t_int, c + w / 2)
            doors.append(("v", x, c, w, not wide))
        else:
            c = r.uniform(x0 + 0.5 + w / 2, x1 - 0.5 - w / 2) if x1 - x0 > w + 1 else (x0 + x1) / 2
            y = (y0 + y1) / 2
            op = G.rect(c - w / 2, y - t_int, c + w / 2, y + t_int)
            doors.append(("h", y, c, w, not wide))
        openings.append((i, j, op))
    # front door + windows on the exterior
    mb = blocks[0]
    fx = mb.x0 + house["front_door_u"] * mb.w
    doors.append(("h", mb.y0, fx, 3.0, True))
    windows = []
    edge = foot.boundary
    for rm in rooms:
        if rm["kind"] == "garage":
            continue
        for a, b in zip(list(rm["cell"].exterior.coords)[:-1], list(rm["cell"].exterior.coords)[1:]):
            seg = LineString([a, b])
            if seg.length < 5 or edge.distance(seg.interpolate(0.5, normalized=True)) > 0.05:
                continue
            n = max(0, int(seg.length / r.uniform(7, 13)))
            for k in range(n):
                t = (k + 0.5) / n
                p = seg.interpolate(t, normalized=True)
                ww = r.choice([2.5, 3.0, 4.0, 5.0])
                if abs(a[1] - b[1]) < 1e-6:
                    if abs(a[1] - mb.y0) < 1e-6 and abs(p.x - fx) < 4:
                        continue
                    windows.append(("h", a[1], p.x, ww))
                else:
                    windows.append(("v", a[0], p.y, ww))
    cut = []
    for (o, pos, c, w, *_) in doors + windows:
        cut.append(G.rect(pos - 1.0, c - w / 2, pos + 1.0, c + w / 2) if o == "v" else
                   G.rect(c - w / 2, pos - 1.0, c + w / 2, pos + 1.0))
    walls_full = foot.difference(unary_union([rm["inner"] for rm in rooms]))
    walls = walls_full.difference(unary_union(cut))
    return {"rooms": rooms, "doors": doors, "windows": windows, "openings": openings, "walls": walls,
            "walls_full": walls_full, "foot": foot, "t_ext": t_ext, "t_int": t_int}


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


def _spacing(p):
    for k in ("sp", "w", "unit", "density"):
        if p.get(k):
            return k, p[k]
    return None, None


def _lum(b):
    return 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]


def _distinct(k1, p1, b1, k2, p2, b2):
    if LOOK.get(k1, k1) != LOOK.get(k2, k2):
        return True
    if abs(_lum(b1) - _lum(b2)) >= 30 or math.dist(b1, b2) >= 60:
        return True
    s1, s2 = _spacing(p1)[1], _spacing(p2)[1]
    return bool(s1 and s2) and max(s1, s2) / min(s1, s2) >= 1.4


def _separate(kind, params, base, others):
    """Rescale this family's spacing until it differs from every same-type family in `others`."""
    key = _spacing(params)[0]
    if key is None:
        return
    for _ in range(4):
        clash = [(k, p, b) for (k, p, b) in others if not _distinct(kind, params, base, k, p, b)]
        if not clash:
            return
        s2 = _spacing(clash[0][1])[1]
        params[key] = s2 * 1.6 if params[key] >= s2 else s2 / 1.6


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
    elif k == "dots":
        _dotgrid(layer, x0, y0, S, m["sp"], max(1.2, 0.07 * S), c, m["fill"], lw)
    elif k == "basket":
        G._basketweave(layer, x0, y0, S, m["sp"], c, lw)
    elif k == "ashlar":
        G._ashlar(layer, x0, y0, S, m["unit"], c, lw, m["seed"])
    mask = G.poly_mask(affinity.translate(poly_px, -x0, -y0), x1 - x0, y1 - y0)
    G.blend_mask(canvas[y0:y1, x0:x1], layer, mask)


LIVING_FIN = (["plank_lines", "plank", "tile", "diag_tile", "dots", "basket"], [40, 18, 18, 10, 7, 7])
WET_FIN = (["tile", "mosaic", "diag_tile", "dots"], [50, 25, 15, 10])
BED_FIN = (["carpet", "plank_lines", "plank", "dots"], [45, 30, 15, 10])
GARAGE_FIN = (["concrete", "dots", "carpet"], [60, 25, 15])


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
        if r.random() < 0.35:
            dd, dl = r.uniform(5, 10), r.uniform(10, 0.8 * (y1f - y0f) + 10)
            yy = r.uniform(y0f, max(y0f, y1f - dl + 6))
            pieces.append(box(x1f, yy, x1f + dd, yy + dl) if r.random() < 0.5 else box(x0f - dd, yy, x0f, yy + dl))
        if not pieces:
            pieces.append(box(x0f + 2, y1f, min(x1f, x0f + 22), y1f + r.uniform(8, 14)))
        hard = unary_union(pieces).difference(foot.buffer(0.3, join_style=2))
        hard_polys = [q for q in G.polys_of(hard) if q.area > 12]

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
        if all(v is None for v in zone_mat.values()):
            zone_mat["living"] = floor_material(r.choices(*LIVING_FIN)[0], r, fam_seed + 1, colour, ang)
    elif sheet == "rendered":
        fl = floor_material(r.choice(["plank", "plank_lines", "tile"]), r, fam_seed + 1, False, r.choice([0, 90]))
        fl["base"] = (r.randint(222, 242),) * 3
        fl["line"] = G.mix(fl["base"], (0, 0, 0), r.uniform(0.12, 0.25))
        for z in ("living", "bed"):
            zone_mat[z] = fl
        zone_mat["wet"] = floor_material("tile", r, fam_seed + 2, False)
        zone_mat["wet"]["base"], zone_mat["wet"]["line"] = fl["base"], fl["line"]
        zone_mat["garage"] = floor_material("concrete", r, fam_seed + 4, False) if r.random() < 0.6 else None
    floor_fams = _separate_all([zone_mat.get(z) for z in ("living", "bed", "wet", "garage")])

    # ---- layout
    allg = unary_union([foot] + hard_polys)
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
        hk = r.choices(["ashlar", "stone", "grid", "plank", "concrete", "basket"], weights=[26, 14, 16, 26, 8, 10])[0]
        hs = _hard_style(hk, r, lw_pat, fam_seed + 11, colour)
        _separate(hs.kind, hs.params, hs.base, floor_fams)
        for p in hard_polys:
            st = hs
            if st.kind == "hatch":                               # deck boards run the long way of each piece
                bx = p.bounds
                st = G.Style("hatch", st.base, st.line, st.lw,
                             {"sp": st.params["sp"], "angle": 0 if bx[2] - bx[0] >= bx[3] - bx[1] else 90},
                             st.seed, "hardscape")
            _fill(canvas, V.geom(p), st, S, W, H)
            G.cv_outline(canvas, V.geom(p), ink, lw_thin)
            labelled.append(("hardscape", V.geom(p)))

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
        for n, (key, (m, polys)) in enumerate(sorted(fams.items(), key=lambda kv: str(kv[0]))):
            reg = unary_union(polys)
            draw_floor(canvas, V.geom(reg), m, S, W, H, lw_pat)
            if sheet == "finish":                        # patterned finish => always labelled
                labelled.append((f"floor{n}", V.geom(reg)))
    elif sheet == "underfloor":
        # crawlspace / slab / framing plan (real HF14 7, Gemini under-floor sheets)
        slab_rooms = [rm["inner"] for rm in rooms if rm["zone"] == "garage"]
        crawl = foot.buffer(-lay["t_ext"], join_style=2)
        if slab_rooms:
            crawl = crawl.difference(unary_union([rm["cell"] for rm in rooms if rm["zone"] == "garage"]).buffer(lay["t_ext"]))
        ck = r.choices(["hatch45", "joists", "dots", "basket", "carpet", "blank"], weights=[25, 25, 12, 12, 11, 15])[0]
        if ck == "blank" and not slab_rooms:
            ck = "hatch45"
        # piers on a grid (holes in the crawlspace label, like skylights in a roof)
        px0, py0, px1, py1 = crawl.bounds
        sp_p = r.uniform(5, 8)
        pier_rows, piers = [], []
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
            draw_floor(canvas, V.geom(crawl), cm, S, W, H, lw_pat)
            lab = crawl.difference(unary_union(piers)) if piers else crawl
            labelled.append(("crawl", V.geom(lab)))
        if slab_rooms:
            sm = floor_material(r.choice(["concrete", "concrete", "dots"]), r, fam_seed + 22, colour)
            if ck != "blank":
                _separate(sm["kind"], sm, sm["base"], [(cm["kind"], cm, cm["base"])])
            reg = unary_union(slab_rooms)
            draw_floor(canvas, V.geom(reg), sm, S, W, H, lw_pat)
            if ck == "blank" or r.random() < 0.85:
                labelled.append(("slab", V.geom(reg)))
        # girders dashed through the pier rows (clipped to the crawlspace), then the piers
        gcol = G.mix(ink, (255, 255, 255), 0.3)
        for yy in pier_rows:
            for seg in G.lines_of(LineString([(px0, yy), (px1, yy)]).intersection(crawl)):
                c = list(seg.coords)
                _dash_line(canvas, V.px(*c[0]), V.px(*c[-1]), gcol, lw_thin, [(0.8 * S, 0.3 * S)])
        for q in piers:
            G.cv_fill(canvas, V.geom(q), (255, 255, 255))
            G.cv_outline(canvas, V.geom(q), ink, lw_thin)
    if not labelled and sheet != "underfloor":
        # every sheet must ask something: label the largest patterned region
        pass

    # ---- walls
    if sheet == "underfloor":
        wg = V.geom(lay["walls_full"].difference(unary_union([rm["inner"] for rm in rooms]).buffer(0)))
        ring = foot.difference(foot.buffer(-lay["t_ext"], join_style=2))
        _mfill(canvas, V.geom(ring), (255, 255, 255), W, H)
        G.cv_outline(canvas, V.geom(ring), ink, lw_heavy)
        _dash_poly(canvas, V.geom(foot.buffer(0.9, join_style=2)), G.mix(ink, (255, 255, 255), 0.4), lw_thin,
                   [(0.6 * S, 0.3 * S)])                          # footing below
    else:
        wall_look = r.choices(["grey", "black", "double"], weights=[50, 30, 20])[0]
        if sheet == "rendered" and r.random() < 0.45:
            off = (r.uniform(0.35, 0.8), r.uniform(0.35, 0.8))
            sh = affinity.translate(lay["walls_full"], *off).difference(lay["walls_full"])
            _shade(canvas, V.geom(sh), r.uniform(0.7, 0.85), W, H)
        wg = V.geom(lay["walls"])
        if wall_look == "grey":
            _mfill(canvas, wg, (r.randint(80, 150),) * 3, W, H)
            G.cv_outline(canvas, wg, ink, lw_thin)
        elif wall_look == "black":
            _mfill(canvas, wg, (r.randint(0, 35),) * 3, W, H)
        else:
            _mfill(canvas, wg, (255, 255, 255), W, H)
            G.cv_outline(canvas, wg, ink, lw_thin * 1.2)
        t = lay["t_ext"]
        for (o, pos, c, w) in lay["windows"]:
            for kk in (-0.5, 0, 0.5):
                if o == "h":
                    G.cv_line(canvas, V.px(c - w / 2, pos + kk * t * 0.8), V.px(c + w / 2, pos + kk * t * 0.8), ink, lw_thin)
                else:
                    G.cv_line(canvas, V.px(pos + kk * t * 0.8, c - w / 2), V.px(pos + kk * t * 0.8, c + w / 2), ink, lw_thin)
        for (o, pos, c, w, leaf) in lay["doors"]:
            if not leaf:
                continue
            sgn = r.choice([-1, 1])
            if o == "h":
                hinge = (c - w / 2, pos)
                G.cv_line(canvas, V.px(*hinge), V.px(hinge[0], pos + sgn * w), ink, lw_thin * 1.3)
                a0, a1 = (0, 90) if sgn > 0 else (270, 360)
            else:
                hinge = (pos, c - w / 2)
                G.cv_line(canvas, V.px(*hinge), V.px(pos + sgn * w, hinge[1]), ink, lw_thin * 1.3)
                a0, a1 = (0, 90) if sgn > 0 else (90, 180)
            cv2.ellipse(canvas, G._pt(V.px(*hinge)), (int(w * S * G.SCALE), int(w * S * G.SCALE)), 0, a0, a1,
                        G.bgr(ink), max(1, int(lw_thin * 0.8)), cv2.LINE_AA, G.SHIFT)
    if grid_on:
        xs = _merge_close([v for b in blocks for v in (b.x0, b.x1)], 4.0)
        ys = _merge_close([v for b in blocks for v in (b.y0, b.y1)], 4.0)
        _grid(canvas, tq, V, xs, ys, (ex0 - pad + 2.2, ey0 - pad + 2.2, ex1 + pad - 2.2, ey1 + pad * 0.5),
              ink, lw_thin * 0.9, size, S, r)

    # ---- fixtures, tags, text
    sf = r.random() < 0.5
    for rm in rooms:
        x0, y0, x1, y1 = rm["inner"].bounds
        if sheet != "underfloor":
            G.draw_fixtures(canvas, V, rm["name"], (x0, y0, x1, y1), ink, lw_thin, S, r)
            c = V.px((x0 + x1) / 2, (y0 + y1) / 2)
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
    if sheet != "underfloor" and r.random() < 0.15:           # MEP overlay (HF14 12)
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
             "underfloor": ["UNDER-FLOOR FRAMING PLAN", "FOUNDATION PLAN", "GARAGE UNDER-FLOOR PLAN", "CRAWLSPACE PLAN"]}[sheet]
    _title(canvas, tq, V, ex0, ey1 + bot * 0.55, ex0 + (ex1 - ex0) * r.uniform(0.3, 0.6), r.choice(title), r, ink,
           lw_thin, size)
    if not labelled:
        raise RuntimeError("no labelled region")
    canvas = _finish(canvas, tq, r)
    return canvas, _annotations(labelled, S, W, H, image_id, "freeform", colour)
