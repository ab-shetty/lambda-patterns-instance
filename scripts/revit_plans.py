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
                60%), skylights labelled 30% of the time. Grid lines + bubbles, section heads, slope
                arrows ("3\" / 1'-0\""), ridge/valley text, spot elevations,
                walls-below dashed, gutters, trellis / louver slats (unlabelled
                parallel-line look-alikes), copies of the building.
  floor plan -- exterior hardscape is the labelled family (flagstone / ashlar
                patio and front walk, deck boards turning per piece, pavers),
                sometimes an interior finish region too. Poche walls, Revit
                wall drop shadows, textured interior floors, rugs, fixtures,
                room tags, tick-mark dimension chains, grid bubbles, MEP
                overlays.

Labels and the annotation JSON follow v6's format, so v6's _job post-steps
(tight crop, res-degrade) apply unchanged.
"""
import math
import random

import cv2
import numpy as np
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
            pt = ov.representative_point()
            if z(H, pt) > z(F, pt) + 1e-4:
                occ.append(H["poly"])
        vis = F["poly"].difference(unary_union(occ)) if occ else F["poly"]
        if vis.area < 0.05:
            continue
        out.append(dict(F, vis=vis))
    return out


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
    fmat = flat_material(r, fam_seed + 7)
    flat_label = r.random() < 0.6
    if not any(t["kind"] == "pitched" for t in tops):          # all-flat house: the flat roof is the question
        flat_label = True
    skylights_on = r.random() < 0.35
    sky_label = skylights_on and r.random() < 0.3
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
            if sky_label:
                labelled.append(("skylight", V.geom(s)))
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


def compose_floor(image_id, seed, mode_weights):
    house = _house(image_id, seed, mode_weights)
    r = random.Random(seed * 1_000_003 + image_id * 191 + 61)
    fp = G.build_floor_plan(house, random.Random(seed * 1_000_003 + image_id * 193 + 67))
    blocks = house["blocks"]
    mb = blocks[0]
    foot = unary_union([G.rect(b.x0, b.y0, b.x1, b.y1) for b in blocks])
    ink = (r.randint(0, 30),) * 3
    colour = r.random() < 0.3
    fam_seed = r.randrange(1 << 30)

    # ---- hardscape (the labelled family on real Revit floor plans: HF14 0 / 12)
    hard_kind = r.choices(["ashlar", "stone", "grid", "plank", "concrete", "basket"], weights=[26, 14, 16, 26, 8, 10])[0]
    pieces = []
    x0f, y0f, x1f, y1f = foot.bounds
    if r.random() < 0.8:                                          # rear patio / deck
        pw = r.uniform(0.35, 1.0) * (x1f - x0f)
        px = r.uniform(x0f, x1f - pw)
        pieces.append(box(px, y1f, px + pw, y1f + r.uniform(7, 16)))
    if r.random() < 0.6:                                          # front walk
        fx = mb.x0 + house["front_door_u"] * mb.w
        ww = r.uniform(3, 5.5)
        pieces.append(box(fx - ww / 2, y0f - r.uniform(6, 16), fx + ww / 2, y0f))
    if r.random() < 0.35:                                         # side deck / terrace
        dd, dl = r.uniform(5, 10), r.uniform(10, 0.8 * (y1f - y0f) + 10)
        yy = r.uniform(y0f, max(y0f, y1f - dl + 6))
        pieces.append(box(x1f, yy, x1f + dd, yy + dl) if r.random() < 0.5 else box(x0f - dd, yy, x0f, yy + dl))
    if not pieces:
        pieces.append(box(x0f + 2, y1f, min(x1f, x0f + 22), y1f + r.uniform(8, 14)))
    hard = unary_union(pieces).difference(foot.buffer(0.3, join_style=2))
    hard_polys = [p for p in G.polys_of(hard) if p.area > 12]
    two_hard = r.random() < 0.2 and len(hard_polys) >= 2
    hard_kind2 = r.choice([k for k in ["ashlar", "grid", "plank", "concrete"] if k != hard_kind]) if two_hard else None

    # ---- interior
    wall_look = r.choices(["black", "grey", "hatched", "double"], weights=[45, 25, 10, 20])[0]
    wall_fill = {"black": (r.randint(0, 35),) * 3, "grey": (r.randint(95, 175),) * 3,
                 "hatched": (255, 255, 255), "double": (255, 255, 255)}[wall_look]
    floor_look = r.choices(["plank", "none", "tile", "carpet"], weights=[40, 35, 13, 12])[0]
    fl_base = (r.randint(214, 240),) * 3 if not colour else r.choice([(226, 214, 196), (214, 206, 200), (232, 226, 214)])
    fl_line = G.mix(fl_base, (0, 0, 0), r.uniform(0.12, 0.3))
    finish_lab = r.random() < 0.3
    fin_kind = r.choice(["grid", "basket", "cross", "ashlar"]) if finish_lab else None
    rugs = r.random() < 0.35
    shadow_on = r.random() < 0.4 and wall_look in ("black", "grey")
    mep = r.random() < 0.2
    dims_on = r.random() < 0.75
    grid_on = r.random() < 0.55

    # ---- layout
    allg = unary_union([foot] + hard_polys + ([fp["patio"]] if fp["patio"] is not None else []))
    ex0, ey0, ex1, ey1 = allg.bounds
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
    app = {"ink": ink, "text_px": size, "outline_lw": lw_thin}
    labelled = []

    # hardscape
    hs = _hard_style(hard_kind, r, lw_thin * 0.8, fam_seed + 1, colour)
    hs2 = _hard_style(hard_kind2, r, lw_thin * 0.8, fam_seed + 2, colour) if two_hard else None
    for i, p in enumerate(hard_polys):
        st = hs2 if (two_hard and i == len(hard_polys) - 1) else hs
        fam = "hardscape2" if st is hs2 else "hardscape"
        if st.kind == "hatch":                                   # deck boards run the long way of each piece
            bx = p.bounds
            st = G.Style("hatch", st.base, st.line, st.lw,
                         {"sp": st.params["sp"], "angle": 0 if bx[2] - bx[0] >= bx[3] - bx[1] else 90}, st.seed, fam)
        _fill(canvas, V.geom(p), st, S, W, H)
        G.cv_outline(canvas, V.geom(p), ink, lw_thin)
        labelled.append((fam, V.geom(p)))
    if fp["patio"] is not None and not fp["patio"].intersects(hard):
        G.cv_outline(canvas, V.geom(fp["patio"]), ink, lw_thin)

    # interior floors
    for i, (rm, name, rp) in enumerate(zip(fp["rooms"], fp["names"], fp["room_polys"])):
        wet = any(k in name for k in ("BATH", "LAUNDRY", "MUD", "ENTRY"))
        if floor_look == "none":
            continue
        kind = "grid" if (wet or floor_look == "tile") else ("stipple" if floor_look == "carpet" else "plank")
        params = {"sp": r.uniform(1.0, 2.0)} if kind == "grid" else \
            {"density": r.uniform(0.8, 2.0)} if kind == "stipple" else {"w": r.uniform(0.4, 0.8), "angle": 0}
        G.draw_fill(canvas, V.geom(rp), G.Style(kind, fl_base, fl_line, max(0.8, lw_thin * 0.6), params,
                                                fam_seed + 3 + (kind == "grid"), "floor"), S, W, H)
    if fp["garage"] is not None and r.random() < 0.6:
        G.draw_fill(canvas, V.geom(fp["garage"]), G.Style("concrete", (250, 250, 250), (150, 150, 150), 1,
                                                          {"density": r.uniform(1.0, 2.5)}, fam_seed + 5, "garage"), S, W, H)
    if finish_lab:
        cand = [rp for rp in fp["room_polys"] if rp.area > 60]
        if cand:
            k = r.randint(1, min(3, len(cand)))
            chosen = r.sample(cand, k)
            st = _hard_style({"grid": "grid", "basket": "basket", "cross": "grid", "ashlar": "ashlar"}[fin_kind], r,
                             lw_thin * 0.7, fam_seed + 6, colour)
            if fin_kind == "cross":
                st = G.Style("cross", st.base, st.line, st.lw, {"sp": r.uniform(1.0, 2.0)}, st.seed, "finish")
            for rp in chosen:
                _fill(canvas, V.geom(rp), st, S, W, H)
                labelled.append(("finish", V.geom(rp)))
    if rugs:
        for (rm, name, rp) in zip(fp["rooms"], fp["names"], fp["room_polys"]):
            if ("LIVING" in name or "BED" in name or "FAMILY" in name) and rp.area > 120 and r.random() < 0.6:
                c = rp.representative_point()
                rw, rd = r.uniform(5, 9), r.uniform(4, 7)
                rug = box(c.x - rw / 2, c.y - rd / 2, c.x + rw / 2, c.y + rd / 2).intersection(rp.buffer(-1))
                if not rug.is_empty:
                    rk = r.choice(["stipple", "cross", "grid"])
                    G.draw_fill(canvas, V.geom(rug), G.Style(rk, (248, 248, 246), (170, 170, 170), 1,
                                                             {"density": 3.0, "sp": r.uniform(0.3, 0.7)},
                                                             fam_seed + 8, "rug"), S, W, H)
                    G.cv_outline(canvas, V.geom(rug), ink, lw_thin * 0.8)

    # wall drop shadow (rendered Revit plan, HF14 0)
    if shadow_on:
        off = (r.uniform(0.35, 0.8), r.uniform(0.35, 0.8))
        sh = affinity.translate(fp["walls_full"].union(foot.boundary.buffer(0.2)), *off).difference(fp["walls_full"])
        _shade(canvas, V.geom(sh), r.uniform(0.7, 0.85), W, H)

    # walls
    wg = V.geom(fp["walls"])
    if wall_look in ("black", "grey"):
        G.cv_fill(canvas, wg, wall_fill)
        G.cv_outline(canvas, wg, ink, lw_thin * (1.0 if wall_look == "black" else 1.3))
    elif wall_look == "hatched":
        G.draw_fill(canvas, wg, G.Style("hatch", (255, 255, 255), (90, 90, 90), max(0.8, lw_thin * 0.6),
                                        {"sp": 0.25, "angle": 45}, fam_seed + 4, "walls"), S, W, H)
        G.cv_outline(canvas, wg, ink, lw_heavy)
    else:
        G.cv_fill(canvas, wg, (255, 255, 255))
        G.cv_outline(canvas, wg, ink, lw_thin * 1.2)
    if grid_on:
        xs = _merge_close([v for b in blocks for v in (b.x0, b.x1)] + [rr[0] for rr in fp["rooms"]], 4.0)
        ys = _merge_close([v for b in blocks for v in (b.y0, b.y1)], 4.0)
        _grid(canvas, tq, V, xs, ys, (ex0 - pad + 2.2, ey0 - pad + 2.2, ex1 + pad - 2.2, ey1 + pad * 0.5),
              ink, lw_thin * 0.9, size, S, r)

    # windows / doors (v6 conventions)
    t = fp["t_ext"]
    for (o, pos, c, w) in fp["windows"]:
        for k in (-0.5, 0, 0.5):
            if o == "h":
                G.cv_line(canvas, V.px(c - w / 2, pos + k * t * 0.8), V.px(c + w / 2, pos + k * t * 0.8), ink, lw_thin)
            else:
                G.cv_line(canvas, V.px(pos + k * t * 0.8, c - w / 2), V.px(pos + k * t * 0.8, c + w / 2), ink, lw_thin)
    for (o, pos, c, w) in fp["doors"]:
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
    # fixtures, room tags
    for (rm, name, rp) in zip(fp["rooms"], fp["names"], fp["room_polys"]):
        G.draw_fixtures(canvas, V, name, rm, ink, lw_thin, S, r)
        c = V.px((rm[0] + rm[2]) / 2, (rm[1] + rm[3]) / 2)
        tq.add(c, name, size * 0.95, ink, bold=True, anchor="mm")
        if r.random() < 0.7:
            tq.add((c[0], c[1] + size * 1.15), f"{G.ft_label(rm[2] - rm[0])} x {G.ft_label(rm[3] - rm[1])}",
                   size * 0.75, ink, anchor="mm")
    if fp["garage"] is not None:
        c = fp["garage"].representative_point()
        tq.add(V.px(c.x, c.y), r.choice(["GARAGE", "(E) GARAGE", "2-CAR GARAGE"]), size, ink, bold=True, anchor="mm")
    for p in hard_polys[:2]:
        if p.area > 80 and r.random() < 0.5:
            c = p.representative_point()
            tq.add(V.px(c.x, c.y), r.choice(["COVERED PATIO", "PATIO", "DECK", "TERRACE", "(N) DECK"]), size, ink,
                   anchor="mm")
    if hard_polys and r.random() < 0.6:
        q = hard_polys[0].representative_point()
        G.draw_leader(canvas, tq, V, (q.x, q.y), r.choice(HARD_NOTES[hard_kind]), app, S,
                      1 if q.x > (ex0 + ex1) / 2 else -1)
    # MEP overlay (HF14 12)
    if mep:
        mcol = (r.randint(20, 90),) * 3
        for _ in range(r.randint(4, 10)):
            rm = r.choice(fp["rooms"])
            cx, cy = r.uniform(rm[0] + 1, rm[2] - 1), r.uniform(rm[1] + 1, rm[3] - 1)
            rad = r.uniform(1.5, 4.5) * S
            for a in range(0, 360, 24):
                cv2.ellipse(canvas, G._pt(V.px(cx, cy)), (int(rad * G.SCALE), int(rad * 0.7 * G.SCALE)), 0, a, a + 13,
                            G.bgr(mcol), max(1, int(lw_thin)), cv2.LINE_AA, G.SHIFT)
            for _ in range(r.randint(1, 4)):
                p = V.px(cx + r.uniform(-2, 2), cy + r.uniform(-2, 2))
                tq.add(p, r.choice(["$", "B1", "A1", "SD", "F", "$D", "B1(g)"]), size * 0.8, mcol, anchor="mm")
        for _ in range(r.randint(2, 5)):
            rm = r.choice(fp["rooms"])
            G.draw_leader(canvas, tq, V, (r.uniform(rm[0] + 1, rm[2] - 1), r.uniform(rm[1] + 1, rm[3] - 1)),
                          r.choice(MEP_NOTES), app, S, r.choice([-1, 1]))
    # dims: exterior chains
    if dims_on:
        xs = [v for rm in fp["rooms"] for v in (rm[0], rm[2])] + [x0f, x1f]
        _dim_chain(canvas, tq, xs, "x", ey0 - r.uniform(2.5, 4), V, ink, lw_thin * 0.8, size * 0.8, S)
        if r.random() < 0.6:
            _dim_chain(canvas, tq, [x0f, x1f], "x", ey0 - r.uniform(5, 6.5), V, ink, lw_thin * 0.8, size * 0.8, S)
        ys = [v for rm in fp["rooms"] for v in (rm[1], rm[3])]
        _dim_chain(canvas, tq, ys, "y", ex0 - r.uniform(2.5, 4), V, ink, lw_thin * 0.8, size * 0.8, S)
    if r.random() < 0.4:
        for _ in range(r.randint(1, 3)):
            side = r.choice(["S", "E", "W"])
            rad = size * r.uniform(1.0, 1.3)
            if side == "S":
                c, d = V.px(r.uniform(ex0, ex1), ey1 + pad * 0.4), (0, -1)
            else:
                c = V.px(ex0 - pad * 0.5 if side == "W" else ex1 + pad * 0.5, r.uniform(ey0, ey1))
                d = (1 if side == "W" else -1, 0)
            _section_head(canvas, tq, c, rad, d, str(r.randint(1, 4)), r.choice(["A3-1", "A4-1", "A4-2", "A5.1"]),
                          ink, lw_thin, size * 0.75)
    if r.random() < 0.45:
        _north(canvas, tq, V.px(ex0 - pad * 0.3, ey0 - pad * 0.3), size * 1.3, ink, lw_thin, size)
    _title(canvas, tq, V, ex0, ey1 + bot * 0.55, ex0 + (ex1 - ex0) * r.uniform(0.3, 0.6),
           r.choice(["FLOOR PLAN", "FIRST FLOOR PLAN", "PROPOSED FLOOR PLAN", "MAIN LEVEL PLAN", "SITE / FLOOR PLAN"]),
           r, ink, lw_thin, size)
    if not labelled:
        raise RuntimeError("no labelled region")
    canvas = _finish(canvas, tq, r)
    return canvas, _annotations(labelled, S, W, H, image_id, "freeform", colour)
