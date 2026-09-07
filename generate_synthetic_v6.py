#!/usr/bin/env python3
"""synthetic v6 -- architectural drawings drawn the way architects draw them.

Why a new regime. Eighty hand-labelled Gemini plans are worth more than 1,600
v5 synthetic plans, and looking at the three pools side by side says why. v5
quilts 224px raster tiles into a smeared texture, stacks two or three
full-width colour bands per elevation, floats windows at random positions, and
gives every family its own saturated colour -- so colour is the boundary cue.
Measured with `pool_style_stats.py`, v5 is twice as inky as the eval set (0.224
vs 0.111), 68% colourised against 57%, square (1.66 vs 2.59) and *over*-ruled
(48,749 against 11,826 on the FFT-regularity measure: a quilted tile of pure
parallel lines with no windows, trim or geometry interrupting it). The
evaluation set and the Gemini plans are the opposite: a material is a perfectly
ruled line field at physical scale (lap siding every 6", brick courses every
2-2/3", shingle rows every 5"), masked by real building geometry, so its
boundary is an eave, a rake, a belt course, a corner board or a change of fill
-- rarely a colour change, and in the monochrome half of the set never one.

What v6 does differently, each item traceable to the eval set:

  1. Fills are drawn, not quilted. Every material is a procedure that draws
     ruled lines in feet, converted through one px/ft scale, as a continuous
     field anchored to sheet coordinates and masked by the region -- so a fill
     interrupted by a window resumes on the same grid, and two instances of a
     family share one grid. This is the mechanism the Gemini v4 prompt asks for.
  2. Geometry is a house, not bands. A footprint of 1-3 blocks with real roofs
     (gable / hip / shed / flat, pitch, overhang) is projected into elevations,
     a roof plan and a floor plan. Boundaries fall where they do in the eval set:
     eave lines, gable triangles, belt courses, wainscot caps, chimneys,
     dormers, porch roofs and the silhouettes of nearer blocks.
  3. One house, one material schedule. The main siding recurs on every wall of
     every view, the roof material on every plane, an accent on gable ends or a
     wainscot -- so families recur on disconnected surfaces because the building
     makes them, not because a probability was set. 1-3 labelled families per
     sheet, matching the eval set's 1.8.
  4. Appearance matches the eval halves. ~55% colourised BIM-export style (flat
     muted base, the material's own lines in a darker tone, and two families
     may share almost the same colour), ~45% monochrome on white paper at three
     contrast levels. Windows have trim and mullions, doors and garage doors
     have panels, and windows are label holes as the annotators cut them.
  5. Sheets look like excerpts. 1 / 2 / 4 views of the same house, titles,
     dimension strings, level marks, material callouts with leaders that cross
     the fills, an occasional graph-paper ground, digital highlighter and dot
     markup, and excerpt crops that leave a region touching the page edge.

Output is the repository's local-data layout (images/ + annotations/), with
`segmentation = [outer, hole, ...]` and image-local `patternN` names, so
`merge_local_datasets.py`, `select_toparea_local.py` and `train_refunet.py`
consume it unchanged. `mode` is one of elevation / roof_plan / freeform
(floor plans keep the v5 name so existing quota tooling works).

    python3 generate_synthetic_v6.py --n 16 --out data/synthetic/v6_smoke --seed 1
    python3 generate_synthetic_v6.py --n 4000 --out data/synthetic/v6_4000 --workers 48
"""
import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPolygon, Polygon, box)
from shapely.ops import unary_union

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------------
# global knobs (all in feet unless named _px)
# ----------------------------------------------------------------------------
MODE_WEIGHTS = {"elevation": 66, "roof_plan": 16, "freeform": 18}
LONG_SIDE_PX = (2800, 5200)            # eval median ~4000, Gemini 3168
COLOUR_PROB = 0.55                     # 16/28 eval images carry colour
VIEW_COUNT_WEIGHTS = {1: 55, 2: 38, 4: 7}
WINDOW_HOLE_PROB = 0.8                 # per image: windows/doors cut as holes
EXCERPT_CROP_PROB = 0.30
GRAPH_PAPER_PROB = 0.12
TOWNHOUSE_PROB = 0.18                  # eval 17-19: rows of identical units
FOUND_ROOF_COLOUR_PROB = 0.0           # v6e tried 0.3: negative
FOUND_LABEL_PROB = 0.4                 # v6e tried 0.7: negative
SAME_COLOUR_PAIR_PROB = 0.2            # accent shares the main colour, fill differs.
                                       # 0.4 (with a roof-coloured, 70%-labelled base
                                       # band) was the v6e round: -0.08 synth-only at
                                       # one seed. Colour-matched look-alike families
                                       # hurt, as v5's confusable tiles did.
HIGHLIGHT_PROB = 0.07
DOT_MARKUP_PROB = 0.05
CLOUD_PROB = 0.04
SOFT_RASTER_PROB = 0.25
JPEG_PROB = 0.25
FLOOR_H = (8.5, 10.0)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)
        except Exception:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------
# Muted BIM-export material colours seen in the eval set (RGB).
BASE_PALETTE = [
    (236, 196, 200), (242, 206, 206), (198, 214, 236), (180, 200, 228),
    (206, 200, 232), (245, 238, 224), (240, 232, 214), (150, 148, 110),
    (170, 165, 120), (74, 74, 80), (96, 96, 104), (120, 140, 160),
    (110, 150, 100), (150, 176, 120), (170, 92, 82), (196, 120, 100),
    (210, 190, 160), (188, 170, 140), (226, 226, 226), (200, 204, 208),
    (150, 110, 90), (128, 128, 128),
    (70, 110, 200), (90, 130, 210), (60, 140, 70), (200, 170, 60), (40, 150, 160),
    (120, 90, 160), (170, 60, 60), (150, 140, 60),
]
TRIM_PALETTE = [(255, 255, 255), (240, 240, 240), (220, 40, 40), (40, 160, 60), (230, 200, 40),
                (40, 180, 200), (200, 60, 200), (60, 60, 60), (255, 255, 255)]
ROOF_PALETTE = [
    (74, 74, 80), (96, 96, 104), (120, 140, 160), (198, 214, 236),
    (236, 196, 200), (170, 92, 82), (110, 150, 100), (150, 148, 110),
    (170, 165, 120), (128, 128, 128), (60, 60, 64), (180, 200, 228),
]
HIGHLIGHT_COLOURS = [(120, 220, 90), (255, 235, 60), (90, 220, 230), (255, 160, 60)]


def darken(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def polys_of(geom):
    """Flatten any shapely geometry to a list of Polygons with area."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 1e-6 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(polys_of(g))
        return out
    return []


def lines_of(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(lines_of(g))
        return out
    return []


def rect(x0, y0, x1, y1):
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class View:
    """Feet -> pixels for one drawing placed on the sheet."""

    def __init__(self, S, ox_px, oy_px):
        self.S, self.ox, self.oy = S, ox_px, oy_px

    def px(self, x, y):
        return (self.ox + x * self.S, self.oy + y * self.S)

    def geom(self, g):
        return affinity.affine_transform(g, [self.S, 0, 0, self.S, self.ox, self.oy])


# ----------------------------------------------------------------------------
# raster helpers (cv2, anti-aliased, sub-pixel)
# ----------------------------------------------------------------------------
SHIFT = 4
SCALE = 1 << SHIFT


def _pt(p):
    return (int(round(p[0] * SCALE)), int(round(p[1] * SCALE)))


def cv_line(img, a, b, colour, w=1):
    cv2.line(img, _pt(a), _pt(b), bgr(colour), max(1, int(round(w))),
             lineType=cv2.LINE_AA, shift=SHIFT)


def cv_polyline(img, pts, colour, w=1, closed=False):
    arr = np.array([_pt(p) for p in pts], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], closed, bgr(colour), max(1, int(round(w))),
                  lineType=cv2.LINE_AA, shift=SHIFT)


def cv_fill(img, poly, colour):
    """Fill a shapely polygon (with holes) in colour."""
    for p in polys_of(poly):
        ext = np.array([_pt(c) for c in p.exterior.coords], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [ext], bgr(colour), lineType=cv2.LINE_AA, shift=SHIFT)
        for ring in p.interiors:
            hole = np.array([_pt(c) for c in ring.coords], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(img, [hole], bgr((255, 255, 255)), lineType=cv2.LINE_AA, shift=SHIFT)


def cv_outline(img, poly, colour, w=1):
    for p in polys_of(poly):
        cv_polyline(img, list(p.exterior.coords), colour, w, closed=True)
        for ring in p.interiors:
            cv_polyline(img, list(ring.coords), colour, w, closed=True)


def poly_mask(poly, W, H):
    m = np.zeros((H, W), dtype=np.uint8)
    for p in polys_of(poly):
        ext = np.array([_pt(c) for c in p.exterior.coords], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [ext], 255, lineType=cv2.LINE_AA, shift=SHIFT)
        for ring in p.interiors:
            hole = np.array([_pt(c) for c in ring.coords], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [hole], 0, lineType=cv2.LINE_AA, shift=SHIFT)
    return m


def blend_mask(dst, layer, mask):
    """Composite `layer` onto `dst` where mask (0-255) says so, anti-aliased."""
    a = (mask.astype(np.float32) / 255.0)[..., None]
    dst[:] = (dst.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a).astype(np.uint8)


# ----------------------------------------------------------------------------
# material fills. Each draws a continuous ruled field in ABSOLUTE sheet pixel
# coordinates into `layer`, which covers sheet rect (ox, oy, ox+w, oy+h).
# `S` is px/ft. `seed` makes the per-row randomness of a family reproducible
# across all its instances, so shingle joints line up across a window.
# ----------------------------------------------------------------------------
class Style:
    def __init__(self, kind, base, line, lw, params, seed, label_name):
        self.kind, self.base, self.line, self.lw = kind, base, line, lw
        self.params, self.seed, self.label_name = params, seed, label_name


def _hlines(layer, ox, oy, S, spacing_ft, colour, lw, phase_ft=0.0):
    h, w = layer.shape[:2]
    sp = spacing_ft * S
    k0 = math.floor((oy - phase_ft * S) / sp) - 1
    k1 = math.ceil((oy + h - phase_ft * S) / sp) + 1
    for k in range(k0, k1 + 1):
        y = k * sp + phase_ft * S - oy
        if -2 <= y <= h + 2:
            cv_line(layer, (-2, y), (w + 2, y), colour, lw)


def _vlines(layer, ox, oy, S, spacing_ft, colour, lw, phase_ft=0.0):
    h, w = layer.shape[:2]
    sp = spacing_ft * S
    k0 = math.floor((ox - phase_ft * S) / sp) - 1
    k1 = math.ceil((ox + w - phase_ft * S) / sp) + 1
    for k in range(k0, k1 + 1):
        x = k * sp + phase_ft * S - ox
        if -2 <= x <= w + 2:
            cv_line(layer, (x, -2), (x, h + 2), colour, lw)


def _dlines(layer, ox, oy, S, spacing_ft, angle_deg, colour, lw):
    """Parallel lines at angle (deg from horizontal), absolute phase."""
    h, w = layer.shape[:2]
    th = math.radians(angle_deg)
    nx, ny = -math.sin(th), math.cos(th)          # normal
    dx, dy = math.cos(th), math.sin(th)           # direction
    sp = spacing_ft * S
    corners = [(ox, oy), (ox + w, oy), (ox, oy + h), (ox + w, oy + h)]
    ds = [cx * nx + cy * ny for cx, cy in corners]
    k0, k1 = math.floor(min(ds) / sp) - 1, math.ceil(max(ds) / sp) + 1
    L = math.hypot(w, h) + 4
    cx0, cy0 = ox + w / 2, oy + h / 2
    for k in range(k0, k1 + 1):
        d = k * sp
        # point on line closest to centre
        t = cx0 * nx + cy0 * ny - d
        px, py = cx0 - t * nx, cy0 - t * ny
        a = (px - dx * L - ox, py - dy * L - oy)
        b = (px + dx * L - ox, py + dy * L - oy)
        cv_line(layer, a, b, colour, lw)


def _courses(layer, ox, oy, S, row_ft, unit_ft, colour, lw, seed,
             stagger=0.5, random_width=None, joint_lw=None, shade=None):
    """Rows of units (brick, shingle, asphalt): course lines + head joints.
    random_width=(lo,hi) gives shingle-like random unit widths per row."""
    h, w = layer.shape[:2]
    rp = row_ft * S
    r0, r1 = math.floor(oy / rp) - 1, math.ceil((oy + h) / rp) + 1
    jl = joint_lw or lw
    for r in range(r0, r1 + 1):
        y_top = r * rp - oy
        y_bot = y_top + rp
        cv_line(layer, (-2, y_top), (w + 2, y_top), colour, lw)
        if random_width is None:
            up = unit_ft * S
            off = (r % 2) * stagger * up
            k0 = math.floor((ox - off) / up) - 1
            k1 = math.ceil((ox + w - off) / up) + 1
            for k in range(k0, k1 + 1):
                x = k * up + off - ox
                if -2 <= x <= w + 2:
                    cv_line(layer, (x, y_top), (x, y_bot), colour, jl)
                    if shade is not None and (k + r) % 3 == 0:
                        x2 = min(w + 2, x + up)
                        cv2.rectangle(layer, _pt((x + 1, y_top + 1)), _pt((x2 - 1, y_bot - 1)),
                                      bgr(shade), -1, lineType=cv2.LINE_AA, shift=SHIFT)
        else:
            rr = random.Random(seed * 7919 + r)
            lo, hi = random_width
            # walk from a fixed origin so the joints are stable across instances
            x_ft = -400.0 + rr.uniform(0, hi)
            xs = []
            while x_ft * S < ox + w + 5:
                xs.append(x_ft)
                x_ft += rr.uniform(lo, hi)
            for xf in xs:
                x = xf * S - ox
                if -2 <= x <= w + 2:
                    cv_line(layer, (x, y_top), (x, y_bot), colour, jl)


def _stones(layer, ox, oy, S, colour, lw, seed, coursed=True, shade=None):
    h, w = layer.shape[:2]
    rr = random.Random(seed * 104729)
    y_ft = -300.0
    rows = []
    while y_ft * S < oy + h + 5:
        rh = rr.uniform(0.55, 1.4)
        rows.append((y_ft, rh))
        y_ft += rh
    for ri, (yf, rh) in enumerate(rows):
        y0 = yf * S - oy
        y1 = (yf + rh) * S - oy
        if y1 < -2 or y0 > h + 2:
            continue
        rr2 = random.Random(seed * 7919 + ri)
        x_ft = -400.0 + rr2.uniform(0, 2.5)
        cv_line(layer, (-2, y0), (w + 2, y0), colour, lw)
        while x_ft * S < ox + w + 5:
            sw = rr2.uniform(0.8, 3.0)
            x0 = x_ft * S - ox
            x1 = (x_ft + sw) * S - ox
            if x1 > -2 and x0 < w + 2:
                if coursed:
                    cv_line(layer, (x0, y0), (x0, y1), colour, lw)
                else:
                    # rubble: jitter the corners
                    j = 0.12 * S
                    pts = [(x0 + rr2.uniform(-j, j), y0 + rr2.uniform(-j, j)),
                           (x1 + rr2.uniform(-j, j), y0 + rr2.uniform(-j, j)),
                           (x1 + rr2.uniform(-j, j), y1 + rr2.uniform(-j, j)),
                           (x0 + rr2.uniform(-j, j), y1 + rr2.uniform(-j, j))]
                    cv_polyline(layer, pts, colour, lw, closed=True)
                if shade is not None and rr2.random() < 0.35:
                    cv2.rectangle(layer, _pt((x0 + 1, y0 + 1)), _pt((x1 - 1, y1 - 1)),
                                  bgr(shade), -1, lineType=cv2.LINE_AA, shift=SHIFT)
            x_ft += sw


def _stipple(layer, ox, oy, S, density_per_sqft, colour, seed, r_px=1):
    """Stucco / carpet dots, cell-seeded so the field is continuous."""
    h, w = layer.shape[:2]
    cell = 2.0 * S
    c0x, c1x = math.floor(ox / cell), math.ceil((ox + w) / cell)
    c0y, c1y = math.floor(oy / cell), math.ceil((oy + h) / cell)
    n = max(1, int(density_per_sqft * 4))
    for cy in range(c0y, c1y + 1):
        for cx in range(c0x, c1x + 1):
            rr = random.Random(seed * 1000003 + cx * 7919 + cy * 104729)
            for _ in range(n):
                x = cx * cell + rr.random() * cell - ox
                y = cy * cell + rr.random() * cell - oy
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(layer, _pt((x, y)), int(r_px * SCALE), bgr(colour), -1,
                               lineType=cv2.LINE_AA, shift=SHIFT)


def _tiles_roof(layer, ox, oy, S, row_ft, colour, lw):
    """S-tile / barrel tile roof: rows of small arcs."""
    h, w = layer.shape[:2]
    rp = row_ft * S
    up = row_ft * 0.9 * S
    r0, r1 = math.floor(oy / rp) - 1, math.ceil((oy + h) / rp) + 1
    for r in range(r0, r1 + 1):
        y = r * rp - oy
        cv_line(layer, (-2, y), (w + 2, y), colour, lw)
        off = (r % 2) * up / 2
        k0, k1 = math.floor((ox - off) / up) - 1, math.ceil((ox + w - off) / up) + 1
        for k in range(k0, k1 + 1):
            x = k * up + off - ox
            if -up <= x <= w + up:
                cv2.ellipse(layer, _pt((x, y)), (int(up / 2 * SCALE), int(rp * 0.55 * SCALE)),
                            0, 180, 360, bgr(colour), max(1, int(lw)), lineType=cv2.LINE_AA,
                            shift=SHIFT)


def _planks(layer, ox, oy, S, plank_w_ft, colour, lw, seed, angle=0):
    """Wood flooring: parallel plank lines with staggered end joints."""
    h, w = layer.shape[:2]
    if angle == 90:
        layer_t = np.ascontiguousarray(np.transpose(layer, (1, 0, 2)))
        _planks(layer_t, oy, ox, S, plank_w_ft, colour, lw, seed, 0)
        layer[:] = np.transpose(layer_t, (1, 0, 2))
        return
    pp = plank_w_ft * S
    r0, r1 = math.floor(oy / pp) - 1, math.ceil((oy + h) / pp) + 1
    for r in range(r0, r1 + 1):
        y0 = r * pp - oy
        cv_line(layer, (-2, y0), (w + 2, y0), colour, lw)
        rr = random.Random(seed * 7919 + r)
        x_ft = -400.0 + rr.uniform(0, 6)
        while x_ft * S < ox + w + 5:
            x = x_ft * S - ox
            if -2 <= x <= w + 2:
                cv_line(layer, (x, y0), (x, y0 + pp), colour, lw)
            x_ft += rr.uniform(3, 8)


def draw_fill(canvas, poly_px, style, S, W, H):
    """Paint `style` into the polygon (sheet pixel coords) on canvas."""
    if poly_px.is_empty:
        return
    minx, miny, maxx, maxy = poly_px.bounds
    x0, y0 = max(0, int(math.floor(minx)) - 2), max(0, int(math.floor(miny)) - 2)
    x1, y1 = min(W, int(math.ceil(maxx)) + 2), min(H, int(math.ceil(maxy)) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    layer = np.empty((h, w, 3), dtype=np.uint8)
    layer[:] = bgr(style.base)
    k, p, c, lw, sd = style.kind, style.params, style.line, style.lw, style.seed
    if k == "lap":
        _hlines(layer, x0, y0, S, p["sp"], c, lw)
        if p.get("shadow"):
            _hlines(layer, x0, y0, S, p["sp"], mix(style.base, c, 0.35), 1, phase_ft=0.06)
    elif k == "vertical":
        _vlines(layer, x0, y0, S, p["sp"], c, lw)
    elif k == "bb":
        _vlines(layer, x0, y0, S, p["sp"], c, lw)
        _vlines(layer, x0, y0, S, p["sp"], c, lw, phase_ft=p["batten"])
    elif k == "shingle":
        _courses(layer, x0, y0, S, p["row"], 0, c, lw, sd, random_width=(0.35, 0.95))
    elif k == "brick":
        _courses(layer, x0, y0, S, p["row"], p["unit"], c, lw, sd, stagger=0.5,
                 shade=p.get("shade"))
    elif k == "block":
        _courses(layer, x0, y0, S, 0.667, 1.333, c, lw, sd, stagger=0.5)
    elif k == "stone":
        _stones(layer, x0, y0, S, c, lw, sd, coursed=True, shade=p.get("shade"))
    elif k == "rubble":
        _stones(layer, x0, y0, S, c, lw, sd, coursed=False, shade=p.get("shade"))
    elif k == "stipple":
        _stipple(layer, x0, y0, S, p["density"], c, sd, r_px=p.get("r", 1))
    elif k == "flat":
        pass
    elif k == "seam":
        _vlines(layer, x0, y0, S, p["sp"], c, lw)
        _vlines(layer, x0, y0, S, p["sp"], mix(style.base, c, 0.4), 1, phase_ft=0.08)
    elif k == "asphalt":
        _courses(layer, x0, y0, S, p["row"], p["unit"], c, lw, sd, stagger=0.5, joint_lw=lw)
    elif k == "tile_roof":
        _tiles_roof(layer, x0, y0, S, p["row"], c, lw)
    elif k == "plywood":
        _hlines(layer, x0, y0, S, 4.0, c, lw)
        _vlines(layer, x0, y0, S, 8.0, c, lw)
    elif k == "hatch":
        _dlines(layer, x0, y0, S, p["sp"], p["angle"], c, lw)
    elif k == "cross":
        _dlines(layer, x0, y0, S, p["sp"], 45, c, lw)
        _dlines(layer, x0, y0, S, p["sp"], -45, c, lw)
    elif k == "grid":
        _hlines(layer, x0, y0, S, p["sp"], c, lw)
        _vlines(layer, x0, y0, S, p["sp"], c, lw)
    elif k == "plank":
        _planks(layer, x0, y0, S, p["w"], c, lw, sd, angle=p.get("angle", 0))
    elif k == "concrete":
        _stipple(layer, x0, y0, S, p["density"], c, sd, r_px=1)
        _stipple(layer, x0, y0, S, p["density"] * 0.15, c, sd + 1, r_px=2)
    elif k == "solid":
        pass
    elif k == "roof_rows":
        # roof plan facets: shingle rows parallel to the eave (angle given)
        _dlines(layer, x0, y0, S, p["row"], p["angle"], c, lw)
        if p.get("joints"):
            _dlines(layer, x0, y0, S, p["row"] * 2.2, p["angle"] + 90, mix(style.base, c, 0.5), 1)
    m = poly_mask(affinity.translate(poly_px, -x0, -y0), w, h)
    blend_mask(canvas[y0:y1, x0:x1], layer, m)


# ----------------------------------------------------------------------------
# material schedule
# ----------------------------------------------------------------------------
WALL_KINDS = ["lap", "lap", "lap", "bb", "bb", "shingle", "brick", "stucco", "stucco",
              "vertical", "stone", "block", "seam"]
ROOF_KINDS = ["asphalt", "asphalt", "seam", "tile_roof", "shingle", "flat_roof"]
CALLOUT = {
    "lap": ["HORIZ. LAP SIDING", "HORIZONTAL LAP SIDING", "FIBER CEMENT LAP SIDING",
            "HORIZ. VINYL SIDING", "6\" EXPOSURE LAP SIDING"],
    "bb": ["BOARD AND BATTEN SIDING", "B&B SIDING", "BOARD & BATTEN"],
    "shingle": ["CEDAR SHINGLES", "SHINGLE SIDING", "SHAKE SIDING"],
    "brick": ["BRICK VENEER", "BRICK"],
    "stucco": ["STUCCO FINISH", "SMOOTH STUCCO", "NEW STUCCO"],
    "vertical": ["VERTICAL SIDING", "T1-11 SIDING", "VERTICAL WOOD SIDING"],
    "stone": ["STONE VENEER", "CULTURED STONE"],
    "rubble": ["STONE VENEER", "RUBBLE STONE"],
    "block": ["CMU", "CONC. BLOCK"],
    "seam": ["STANDING SEAM METAL", "METAL ROOF", "STANDING SEAM METAL ROOF"],
    "asphalt": ["ASPHALT SHINGLES", "COMP. SHINGLES", "SHINGLES (TYP)"],
    "tile_roof": ["CLAY TILE ROOF", "CONCRETE TILE ROOF"],
    "flat_roof": ["MEMBRANE ROOF", "TPO ROOF"],
    "concrete": ["CONC. FOUNDATION", "CONCRETE SLAB"],
    "grid": ["TILE FLOOR", "CERAMIC TILE", "12x12 TILE"],
    "plank": ["WOOD FLOOR", "HARDWOOD", "LVP FLOORING"],
    "stipple": ["CARPET", "STUCCO"],
    "cross": ["HERRINGBONE", "TILE"],
    "solid": ["WALL"],
}


def make_style(rng, kind, appearance, seed, S, label_name, roof=False, base_override=None):
    """Instantiate one family's look: base colour, line colour, weights, params."""
    lw = max(1.0, S * rng.uniform(0.012, 0.022))
    if appearance["colour"]:
        if base_override is not None:
            base = base_override
        else:
            base = rng.choice(ROOF_PALETTE if roof else BASE_PALETTE)
        lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
        if lum < 110:
            line = mix(base, (255, 255, 255), rng.uniform(0.25, 0.5))
        else:
            line = darken(base, rng.uniform(0.55, 0.8))
        if rng.random() < 0.15:
            line = appearance["ink"]
    else:
        base = appearance["paper"]
        line = appearance["ink_fill"]
    params = {}
    k = kind
    if kind == "lap":
        params = {"sp": rng.choice([0.33, 0.42, 0.5, 0.58, 0.67]), "shadow": rng.random() < 0.4}
    elif kind == "vertical":
        params = {"sp": rng.choice([0.5, 0.67, 1.0])}
    elif kind == "bb":
        params = {"sp": rng.choice([1.0, 1.33, 1.5, 2.0]), "batten": rng.uniform(0.15, 0.25)}
    elif kind == "shingle":
        params = {"row": rng.choice([0.42, 0.5, 0.58]) if not roof else rng.choice([0.4, 0.5])}
    elif kind == "brick":
        params = {"row": 0.222 if S >= 45 else 0.333, "unit": 0.667,
                  "shade": mix(base, line, 0.35) if rng.random() < 0.3 else None}
    elif kind == "stucco":
        if rng.random() < 0.55:
            k = "stipple"
            params = {"density": rng.uniform(1.5, 6.0), "r": 1}
        else:
            k = "flat"
    elif kind == "stone":
        params = {"shade": mix(base, line, 0.3) if rng.random() < 0.5 else None}
        if rng.random() < 0.3:
            k = "rubble"
    elif kind == "seam":
        params = {"sp": rng.choice([1.0, 1.33, 1.5])}
    elif kind == "asphalt":
        params = {"row": rng.choice([0.42, 0.5]), "unit": 1.0}
    elif kind == "tile_roof":
        params = {"row": rng.choice([0.9, 1.1])}
    elif kind == "flat_roof":
        k = "flat" if rng.random() < 0.6 else "stipple"
        params = {"density": 1.0, "r": 1}
    elif kind == "grid":
        params = {"sp": rng.choice([1.0, 1.5, 2.0, 0.667])}
    elif kind == "plank":
        params = {"w": rng.choice([0.25, 0.33, 0.42]), "angle": rng.choice([0, 90])}
    elif kind == "cross":
        params = {"sp": rng.choice([0.5, 0.75, 1.0])}
    elif kind == "hatch":
        params = {"sp": rng.choice([0.4, 0.6, 0.8]), "angle": rng.choice([45, -45, 30])}
    elif kind == "concrete":
        params = {"density": rng.uniform(0.6, 2.5)}
    elif kind == "stipple":
        params = {"density": rng.uniform(1.5, 5.0), "r": 1}
    elif kind == "solid":
        base = appearance["ink"] if not appearance["colour"] else darken(appearance["ink"], 1.0)
        line = base
    st = Style(k, base, line, lw, params, seed, label_name)
    st.callout = rng.choice(CALLOUT.get(kind, CALLOUT.get(k, ["FINISH"])))
    return st


def make_appearance(rng):
    colour = rng.random() < COLOUR_PROB
    level = rng.choices(["normal", "light", "faint"], weights=[50, 28, 22])[0]
    ink = {"normal": rng.randint(15, 60), "light": rng.randint(70, 120),
           "faint": rng.randint(120, 175)}[level]
    paper_v = 255 if rng.random() < 0.8 else rng.randint(240, 252)
    app = {"colour": colour, "level": level, "ink": (ink, ink, ink),
           "paper": (paper_v, paper_v, paper_v)}
    # fill lines in mono mode are usually lighter than the outline ink
    fv = min(235, ink + rng.randint(0, 60))
    app["ink_fill"] = (fv, fv, fv)
    app["trim"] = (255, 255, 255) if rng.random() < 0.7 else (rng.randint(225, 245),) * 3
    app["outline"] = (ink, ink, ink)
    app["outline_lw"] = 1.0
    return app


# ----------------------------------------------------------------------------
# the house
# ----------------------------------------------------------------------------
class Block:
    def __init__(self, x0, y0, w, d, stories, roof, ridge, pitch, ov, kind):
        self.x0, self.y0, self.w, self.d = x0, y0, w, d
        self.stories, self.roof, self.ridge = stories, roof, ridge
        self.pitch, self.ov, self.kind = pitch, ov, kind

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.d


def make_house(rng):
    fh = rng.uniform(*FLOOR_H)
    stories = rng.choice([1, 1, 2, 2, 2])
    w = rng.uniform(26, 56)
    d = rng.uniform(20, 36)
    roof = rng.choices(["gable", "hip", "flat", "shed"], weights=[55, 30, 8, 7])[0]
    # townhouse row: n identical units side by side (eval images 17-19)
    units = None
    if rng.random() < TOWNHOUSE_PROB:
        units = rng.choice([2, 3, 3, 4, 4, 5])
        uw = rng.uniform(16, 24)
        w, d, stories = units * uw, rng.uniform(24, 36), rng.choice([2, 2, 2, 3])
        roof = rng.choice(["gable", "gable", "hip"])
    ridge = "x" if w >= d or rng.random() < 0.25 else "y"
    if units:
        ridge = "x"
    pitch = {"gable": rng.uniform(0.3, 0.6), "hip": rng.uniform(0.25, 0.5),
             "shed": rng.uniform(0.15, 0.35), "flat": 0.0}[roof]   # rise/run
    ov = rng.uniform(0.8, 2.2)
    blocks = [Block(0, 0, w, d, stories, roof, ridge, pitch, ov, "main")]
    # attached garage / wing
    if not units and rng.random() < 0.55:
        gw, gd = rng.uniform(12, 26), rng.uniform(0.55, 1.0) * d
        side = rng.choice(["left", "right"])
        gx = -gw if side == "left" else w
        gy = rng.uniform(-0.3 * d, 0.35 * d)
        groof = rng.choice(["gable", "gable", "hip", "shed"])
        gridge = rng.choice(["x", "y"])
        blocks.append(Block(gx, gy, gw, gd, 1, groof, gridge, pitch * rng.uniform(0.8, 1.1),
                            ov, "garage" if rng.random() < 0.75 else "wing"))
    # rear ell
    if not units and rng.random() < 0.35:
        ew, ed = rng.uniform(12, 0.7 * w), rng.uniform(10, 22)
        ex = rng.uniform(0, w - ew)
        blocks.append(Block(ex, d, ew, ed, rng.choice([1, stories]),
                            rng.choice(["gable", "hip"]), rng.choice(["x", "y"]),
                            pitch, ov, "ell"))
    house = {"fh": fh, "plate": rng.uniform(0.4, 1.0), "blocks": blocks,
             "foundation": rng.uniform(0.6, 2.6) if rng.random() < 0.65 else 0.0,
             "belt": rng.random() < 0.5, "belt_h": rng.uniform(0.5, 1.0),
             "corner_boards": rng.random() < 0.5,
             "wainscot_h": rng.uniform(2.5, 4.0),
             "chimney": rng.random() < 0.3, "porch": rng.random() < 0.35,
             "dormers": rng.random() < 0.35,
             "window_style": rng.choice(["2x2", "1x2", "1x1", "3x2", "6lite"]),
             "front_door_u": rng.uniform(0.2, 0.8), "units": units,
             "shadows": rng.random() < 0.5, "shadow_h": rng.uniform(0.6, 1.8), "shadow_k": rng.uniform(0.12, 0.3)}
    if units:
        # one facade template, repeated per unit: slots packed left to right
        uw = w / units
        slots = [("door", 3.0)]
        if rng.random() < 0.5:
            slots.append(("garage", rng.uniform(8.0, 9.5)))
        slots.append(("win", rng.choice([3.0, 4.0, 5.0])))
        if uw > 20 and rng.random() < 0.5:
            slots.append(("win", rng.choice([3.0, 4.0])))
        rng.shuffle(slots)
        total = sum(sw for _, sw in slots)
        gap = max(0.8, (uw - total) / (len(slots) + 1))
        x = gap
        tmpl = {"ground": [], "upper": []}
        for kind, sw in slots:
            tmpl["ground"].append((kind, x, sw))
            x += sw + gap
        nup = rng.choice([2, 2, 3]) if uw > 18 else 2
        upw = rng.choice([3.0, 3.5, 4.0])
        for i in range(nup):
            tmpl["upper"].append(("win", uw * (i + 0.5) / nup - upw / 2, upw))
        house["unit_template"] = tmpl
        house["dormers"] = rng.random() < 0.6
        house["belt"] = rng.random() < 0.7
        house["label_trim"] = rng.random() < 0.6
        house["porch"] = False
        house["chimney"] = False
        house["force_accent"] = True
    # accent placement
    house["accent"] = rng.choices([None, "gable", "wainscot", "upper", "block"],
                                  weights=[15, 30, 25, 15, 15])[0]
    if house.get("force_accent") and house["accent"] in (None, "block"):
        house["accent"] = rng.choice(["wainscot", "gable", "upper"])
    # a second accent on a different zone (stone wainscot under shingle gables, etc.)
    house["accent2"] = None
    if house["accent"] in ("gable", "upper", "block") and rng.random() < 0.35:
        house["accent2"] = "wainscot"
    elif house["accent"] == "wainscot" and rng.random() < 0.35:
        house["accent2"] = "gable"
    # trim (belt courses, fascia, corner boards, window casings) as its own family,
    # the way eval images 17-19 and 25 label it
    if not units:
        house["label_trim"] = rng.random() < 0.35
    if house["label_trim"]:
        house["corner_boards"] = True
        house["belt"] = True
    return house


def view_map(view):
    """Plan (x,y) -> (u, t): screen-x and depth (smaller t = nearer viewer)."""
    if view == "front":
        return lambda x, y: (x, y)
    if view == "rear":
        return lambda x, y: (-x, -y)
    if view == "left":
        return lambda x, y: (-y, x)
    return lambda x, y: (y, -x)          # right


def block_uv(b, view):
    m = view_map(view)
    us, ts = zip(*[m(x, y) for x, y in [(b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1)]])
    u0, u1, t0, t1 = min(us), max(us), min(ts), max(ts)
    ridge_along_u = (b.ridge == "x") == (view in ("front", "rear"))
    return u0, u1, t0, t1, ridge_along_u


# ----------------------------------------------------------------------------
# elevation
# ----------------------------------------------------------------------------
def build_elevation(house, view, rng):
    """Return dict with surfaces (family -> polygons), trims, openings, notes
    in view feet coords: u right, y down, grade at y=0."""
    fh, plate = house["fh"], house["plate"]
    blocks = house["blocks"]
    order = sorted(blocks, key=lambda b: block_uv(b, view)[2])       # near first
    surfaces = []        # (family, poly, block_index, kind)
    trims = []           # polys drawn white/light with outline (unlabelled)
    lines = []           # (a, b, lw) drawn in ink
    openings = []        # window/door polys (holes), with 'type'
    silhouettes = []
    zorder = 0
    for bi, b in enumerate(order):
        u0, u1, t0, t1, ridge_u = block_uv(b, view)
        depth_t = t1 - t0
        width_u = u1 - u0
        H = b.stories * fh + plate
        y_top = -H
        wall = rect(u0, 0, u1, y_top)
        roof_polys, roof_trim, rake_lines = [], [], []
        ov = b.ov
        if b.roof == "flat":
            par = rng.uniform(1.0, 2.5)
            wall = rect(u0, 0, u1, y_top - par)
            trims.append(rect(u0 - 0.3, y_top - par, u1 + 0.3, y_top - par + 0.5))
        elif b.roof in ("gable", "shed"):
            if ridge_u:
                rise = b.pitch * (depth_t / 2 if b.roof == "gable" else depth_t)
                roof_polys.append(rect(u0 - ov, y_top, u1 + ov, y_top - rise))
                roof_trim.append(rect(u0 - ov, y_top, u1 + ov, y_top + rng.uniform(0.4, 0.8)))
            else:
                rise = b.pitch * (width_u / 2 if b.roof == "gable" else width_u)
                if b.roof == "gable":
                    apex = ((u0 + u1) / 2, y_top - rise)
                    wall = Polygon([(u0, 0), (u1, 0), (u1, y_top), apex, (u0, y_top)])
                    # rake boards
                    rk = rng.uniform(0.4, 0.8)
                    for (a, c) in [((u0 - ov, y_top + ov * b.pitch), apex),
                                   ((u1 + ov, y_top + ov * b.pitch), apex)]:
                        seg = LineString([a, (c[0], c[1] - rk * 0.2)])
                        roof_trim.append(seg.buffer(rk / 2, cap_style=2))
                else:
                    wall = Polygon([(u0, 0), (u1, 0), (u1, y_top - rise), (u0, y_top)])
                    roof_trim.append(LineString([(u0 - ov, y_top + ov * b.pitch),
                                                 (u1 + ov, y_top - rise - ov * b.pitch)])
                                     .buffer(0.3, cap_style=2))
        elif b.roof == "hip":
            short = min(width_u, depth_t)
            rise = b.pitch * short / 2
            if ridge_u and width_u >= depth_t:
                top0, top1 = u0 + depth_t / 2, u1 - depth_t / 2
            elif (not ridge_u) and depth_t >= width_u:
                top0 = top1 = (u0 + u1) / 2
            elif ridge_u:
                top0 = top1 = (u0 + u1) / 2
            else:
                top0, top1 = u0 + depth_t / 2, u1 - depth_t / 2
                top0, top1 = min(top0, top1), max(top0, top1)
            roof_polys.append(Polygon([(u0 - ov, y_top), (u1 + ov, y_top),
                                       (top1, y_top - rise), (top0, y_top - rise)]))
            roof_trim.append(rect(u0 - ov, y_top, u1 + ov, y_top + rng.uniform(0.4, 0.8)))
        # dormers on a roof band seen face-on
        dormers = []
        if house["dormers"] and roof_polys and b.roof == "gable" and ridge_u and b.kind == "main":
            rise = b.pitch * depth_t / 2
            if rise > 5 or house.get("units"):
                n = house["units"] if house.get("units") else rng.choice([1, 2, 2, 3])
                dw = rng.uniform(4, 6.5)
                for i in range(n):
                    cu = u0 + (i + 0.5) * width_u / n + (0 if house.get("units") else rng.uniform(-1, 1))
                    base_y = y_top - rng.uniform(1.0, 2.0)
                    dh = rng.uniform(4.5, 6.0)
                    drise = b.pitch * dw / 2
                    dwall = Polygon([(cu - dw / 2, base_y), (cu + dw / 2, base_y),
                                     (cu + dw / 2, base_y - dh), (cu, base_y - dh - drise),
                                     (cu - dw / 2, base_y - dh)])
                    dormers.append((dwall, cu, base_y, dh, dw))
        # chimney
        chimney = None
        if house["chimney"] and b.kind == "main" and rng.random() < 0.7:
            cw = rng.uniform(2.0, 3.5)
            cu = rng.choice([u0 + rng.uniform(2, 6), u1 - rng.uniform(2, 6)])
            top = y_top - (b.pitch * depth_t / 2 if roof_polys else 0) - rng.uniform(1.5, 3.5)
            chimney = rect(cu - cw / 2, -2, cu + cw / 2, top)
        # silhouette of this block
        sil = unary_union([wall] + roof_polys + roof_trim + [d[0] for d in dormers] +
                          ([chimney] if chimney is not None else []))
        occluder = unary_union(silhouettes) if silhouettes else None

        def vis(g):
            return g if occluder is None else g.difference(occluder)

        # material zones on the wall
        zones = {}
        acc = house["accent"]
        wall_v = vis(wall)
        belt_polys = []
        if house["belt"] and b.stories >= 2:
            for s in range(1, b.stories):
                yb = -s * fh
                belt_polys.append(rect(u0 - 0.2, yb - house["belt_h"] / 2, u1 + 0.2, yb + house["belt_h"] / 2))
        def zone_for(placement):
            if placement == "gable" and b.roof == "gable" and not ridge_u:
                belt_polys.append(rect(u0 - 0.2, y_top - 0.1, u1 + 0.2, y_top + rng.uniform(0.4, 0.8)))
                return rect(u0 - 1, y_top, u1 + 1, y_top - 100)
            if placement == "wainscot":
                belt_polys.append(rect(u0 - 0.2, -house["wainscot_h"] - 0.3, u1 + 0.2, -house["wainscot_h"]))
                return rect(u0 - 1, 0.5, u1 + 1, -house["wainscot_h"])
            if placement == "upper" and b.stories >= 2:
                if not house["belt"]:
                    belt_polys.append(rect(u0 - 0.2, -fh - 0.3, u1 + 0.2, -fh + 0.3))
                return rect(u0 - 1, -fh, u1 + 1, -200)
            if placement == "block" and b.kind != "main":
                return rect(u0 - 1, 1, u1 + 1, -200)
            return None
        accent_zone = zone_for(acc) if acc else None
        accent2_zone = zone_for(house["accent2"]) if house.get("accent2") else None
        found = None
        if house["foundation"] > 0:
            found = rect(u0 - 0.1, 0, u1 + 0.1, -house["foundation"])
        belts = unary_union(belt_polys) if belt_polys else None
        main_zone = wall_v
        if found is not None:
            main_zone = main_zone.difference(found)
        if belts is not None:
            main_zone = main_zone.difference(belts)
        if accent_zone is not None:
            accent_v = main_zone.intersection(accent_zone)
            main_zone = main_zone.difference(accent_zone)
            for p in polys_of(accent_v):
                surfaces.append(("accent", p, bi, "wall"))
        if accent2_zone is not None:
            accent_v = main_zone.intersection(accent2_zone)
            main_zone = main_zone.difference(accent2_zone)
            for p in polys_of(accent_v):
                surfaces.append(("accent2", p, bi, "wall"))
        for p in polys_of(main_zone):
            surfaces.append(("main", p, bi, "wall"))
        if found is not None:
            for p in polys_of(vis(found)):
                surfaces.append(("foundation", p, bi, "found"))
        for rp in roof_polys:
            for p in polys_of(vis(rp)):
                surfaces.append(("roof", p, bi, "roof"))
        for tp in roof_trim + ([belts] if belts is not None else []):
            for p in polys_of(vis(tp)):
                trims.append(p)
        for (dwall, cu, base_y, dh, dw) in dormers:
            fam = "accent" if (acc in ("gable", "upper") and rng.random() < 0.7) else "main"
            for p in polys_of(vis(dwall)):
                surfaces.append((fam, p, bi, "dormer"))
            # dormer rake trim
            drise = b.pitch * dw / 2
            for (a, c) in [((cu - dw / 2 - 0.6, base_y - dh + 0.3), (cu, base_y - dh - drise - 0.2)),
                           ((cu + dw / 2 + 0.6, base_y - dh + 0.3), (cu, base_y - dh - drise - 0.2))]:
                trims.append(LineString([a, c]).buffer(0.25, cap_style=2))
            ww = min(3.0, dw - 1.6)
            openings.append({"poly": rect(cu - ww / 2, base_y - 0.8, cu + ww / 2, base_y - 0.8 - min(3.5, dh - 1.2)),
                             "type": "window"})
        if chimney is not None:
            fam = "chimney"
            for p in polys_of(vis(chimney)):
                surfaces.append((fam, p, bi, "chimney"))
            trims.append(vis(rect(chimney.bounds[0] - 0.3, chimney.bounds[1], chimney.bounds[2] + 0.3,
                                  chimney.bounds[1] + 0.5)))
        # openings on this wall face (holes cut later where they lie on visible wall)
        near_face = t0
        is_front_face = view == "front" and b.kind == "main"
        garage_face = (b.kind == "garage" and view == "front")
        door_u = None
        if house.get("units") and b.kind == "main" and view in ("front", "rear"):
            n = house["units"]
            uw = width_u / n
            tmpl = house["unit_template"]
            for i in range(n):
                base = u0 + i * uw
                if i > 0:
                    trims.append(vis(rect(base - 0.15, 0.2, base + 0.15, y_top + 0.2)))   # party wall line
                for kind, off, sw in tmpl["ground"]:
                    if kind == "door" and view == "front":
                        openings.append({"poly": rect(base + off, 0, base + off + sw, -6.8), "type": "door", "block": bi})
                    elif kind == "garage" and view == "front":
                        openings.append({"poly": rect(base + off, 0, base + off + sw, -7.0), "type": "garage", "block": bi})
                    elif kind == "win" or view == "rear":
                        ww = sw if kind == "win" else 3.0
                        wh = 4.0
                        head = -rng.uniform(6.8, 7.2) if i == 0 else head0
                        if i == 0:
                            head0 = head
                        if house["accent"] == "wainscot" and head + wh > -house["wainscot_h"] + 0.3:
                            wh = -house["wainscot_h"] - head - 0.3
                        if wh > 2.0:
                            openings.append({"poly": rect(base + off, head, base + off + ww, head + wh), "type": "window", "block": bi})
                for st in range(1, b.stories):
                    for kind, off, sw in tmpl["upper"]:
                        head = -st * fh - 7.0
                        openings.append({"poly": rect(base + off, head, base + off + sw, head + 4.0), "type": "window", "block": bi})
            is_front_face = False
            garage_face = False
        if is_front_face:
            dw = 3.0 if rng.random() < 0.7 else 6.0
            door_u = u0 + house["front_door_u"] * width_u
            door_u = min(max(door_u, u0 + 2.5), u1 - 2.5 - dw)
            openings.append({"poly": rect(door_u, 0, door_u + dw, -6.8), "type": "door",
                             "block": bi})
        if garage_face and width_u > 11:
            gdw = 8.0 if width_u < 20 or rng.random() < 0.4 else 16.0
            gdw = min(gdw, width_u - 3)
            gu = u0 + (width_u - gdw) / 2 + rng.uniform(-1, 1)
            openings.append({"poly": rect(gu, 0, gu + gdw, -7.0), "type": "garage", "block": bi})
        for s in range(b.stories):
            floor_y = -s * fh
            if house.get("units") and b.kind == "main" and view in ("front", "rear"):
                break
            if b.kind == "garage" and s == 0 and garage_face:
                continue
            n = max(0, int(width_u / rng.uniform(6, 11)))
            if b.kind == "garage":
                n = min(n, 2)
            if n == 0:
                continue
            slots = [u0 + (i + 0.5) * width_u / n for i in range(n)]
            for cu in slots:
                cu += rng.uniform(-1.0, 1.0)
                ww = rng.choice([2.5, 3.0, 3.0, 4.0, 5.0, 6.0])
                wh = rng.choice([3.0, 3.5, 4.0, 4.5, 5.0])
                head = floor_y - rng.uniform(6.6, 7.4)
                if s == 0 and house["accent"] == "wainscot" and head + wh > -house["wainscot_h"] + 0.3:
                    wh = min(wh, -house["wainscot_h"] - head - 0.3)
                    if wh < 2.0:
                        continue
                wp = rect(cu - ww / 2, head, cu + ww / 2, head + wh)
                if door_u is not None and s == 0 and wp.intersects(rect(door_u - 0.8, 0, door_u + 7, -7.5)):
                    continue
                if cu - ww / 2 < u0 + 1.0 or cu + ww / 2 > u1 - 1.0:
                    continue
                if wp.intersects(rect(u0, 0, u1, y_top).difference(wall)):
                    continue
                openings.append({"poly": wp, "type": "window", "block": bi})
        # gable attic window
        if b.roof == "gable" and not ridge_u and rng.random() < 0.4:
            cu = (u0 + u1) / 2
            openings.append({"poly": rect(cu - 1.2, y_top - 1.0, cu + 1.2, y_top - 3.2), "type": "window",
                             "block": bi})
        # openings must lie on this block's *visible* wall
        keep = []
        for o in openings:
            if o.get("block") == bi:
                if occluder is not None and o["poly"].intersection(occluder).area > 0.05 * o["poly"].area:
                    continue
            keep.append(o)
        openings = keep
        silhouettes.append(sil)
        zorder += 1
    # porch on the front: nearest of all
    porch = None
    if house["porch"] and view == "front":
        mb = house["blocks"][0]
        u0, u1, _, _, _ = block_uv(mb, view)
        pw = rng.uniform(8, min(24, mb.w - 4))
        pu0 = min(max(u0 + house["front_door_u"] * mb.w - pw / 2, u0 + 1), u1 - pw - 1)
        top = -(fh - rng.uniform(0.2, 1.0))
        proof = rect(pu0 - 1, top, pu0 + pw + 1, top - rng.uniform(1.5, 3.5))
        porch = {"roof": proof, "posts": [rect(pu0 + 0.3, 0, pu0 + 0.9, top),
                                          rect(pu0 + pw - 0.9, 0, pu0 + pw - 0.3, top)],
                 "fascia": rect(pu0 - 1, top, pu0 + pw + 1, top + 0.5)}
        surfaces = [(f, p.difference(proof.union(porch["fascia"])), bi, k) for (f, p, bi, k) in surfaces]
        surfaces = [(f, q, bi, k) for (f, p, bi, k) in surfaces for q in polys_of(p)]
        for p in polys_of(proof):
            surfaces.append(("roof", p, -1, "porch"))
        trims.append(porch["fascia"])
        for post in porch["posts"]:
            trims.append(post)
    return {"surfaces": surfaces, "trims": trims, "openings": openings, "porch": porch,
            "extent": unary_union([p for (_, p, _, _) in surfaces] + trims).bounds}


# ----------------------------------------------------------------------------
# roof plan
# ----------------------------------------------------------------------------
def build_roof_plan(house, rng):
    blocks = house["blocks"]
    fh = house["fh"]
    facets = []          # (poly, angle_deg for rows)
    ridge_lines = []
    order = sorted(blocks, key=lambda b: -(b.stories * fh + (b.pitch * min(b.w, b.d) / 2)))
    taller = []
    outlines = []
    flat_polys = []
    for b in order:
        ov = b.ov
        R = rect(b.x0 - ov, b.y0 - ov, b.x1 + ov, b.y1 + ov)
        cover = unary_union(taller) if taller else None
        Rv = R if cover is None else R.difference(cover)
        outlines.append(R)
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        if b.roof == "flat":
            flat_polys.append(Rv)
        elif b.roof in ("gable", "shed"):
            if b.roof == "shed":
                facets.append((Rv, 0 if b.ridge == "x" else 90))
            elif b.ridge == "x":
                facets.append((Rv.intersection(rect(R.bounds[0], R.bounds[1], R.bounds[2], cy)), 0))
                facets.append((Rv.intersection(rect(R.bounds[0], cy, R.bounds[2], R.bounds[3])), 0))
                ridge_lines.append(LineString([(R.bounds[0], cy), (R.bounds[2], cy)]))
            else:
                facets.append((Rv.intersection(rect(R.bounds[0], R.bounds[1], cx, R.bounds[3])), 90))
                facets.append((Rv.intersection(rect(cx, R.bounds[1], R.bounds[2], R.bounds[3])), 90))
                ridge_lines.append(LineString([(cx, R.bounds[1]), (cx, R.bounds[3])]))
        else:  # hip
            x0, y0, x1, y1 = R.bounds
            W_, D_ = x1 - x0, y1 - y0
            if W_ >= D_:
                h = D_ / 2
                r0, r1 = (x0 + h, cy), (x1 - h, cy)
                tri_l = Polygon([(x0, y0), r0, (x0, y1)])
                tri_r = Polygon([(x1, y0), r1, (x1, y1)])
                top = Polygon([(x0, y0), (x1, y0), r1, r0])
                bot = Polygon([(x0, y1), r0, r1, (x1, y1)])
                for f, ang in [(tri_l, 90), (tri_r, 90), (top, 0), (bot, 0)]:
                    facets.append((Rv.intersection(f), ang))
            else:
                h = W_ / 2
                r0, r1 = (cx, y0 + h), (cx, y1 - h)
                tri_t = Polygon([(x0, y0), (x1, y0), r0])
                tri_b = Polygon([(x0, y1), r1, (x1, y1)])
                left = Polygon([(x0, y0), r0, r1, (x0, y1)])
                right = Polygon([(x1, y0), (x1, y1), r1, r0])
                for f, ang in [(tri_t, 0), (tri_b, 0), (left, 90), (right, 90)]:
                    facets.append((Rv.intersection(f), ang))
            ridge_lines.append(LineString([r0, r1]))
            for corner in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                near = r0 if math.dist(corner, r0) < math.dist(corner, r1) else r1
                ridge_lines.append(LineString([corner, near]))
        if cover is not None:
            ridge_lines = [l.difference(cover) if i >= len(ridge_lines) - 5 else l
                           for i, l in enumerate(ridge_lines)]
        taller.append(R)
    footprint = unary_union([rect(b.x0, b.y0, b.x1, b.y1) for b in blocks])
    return {"facets": [(p, a) for (g, a) in facets for p in polys_of(g)],
            "ridges": ridge_lines, "outline": unary_union(outlines),
            "flat": flat_polys, "footprint": footprint}


# ----------------------------------------------------------------------------
# floor plan
# ----------------------------------------------------------------------------
ROOM_NAMES = ["LIVING ROOM", "KITCHEN", "DINING", "BEDROOM", "MASTER BEDROOM", "BATH",
              "MASTER BATH", "OFFICE", "LAUNDRY", "HALL", "CLOSET", "FAMILY ROOM",
              "ENTRY", "PANTRY", "MUD ROOM", "STUDY", "BEDROOM 2", "BEDROOM 3"]


def split_rooms(rng, x0, y0, x1, y1, depth=0):
    w, h = x1 - x0, y1 - y0
    if depth >= 4 or (w * h < rng.uniform(140, 260)) or min(w, h) < 9:
        return [(x0, y0, x1, y1)]
    if w > h * rng.uniform(0.9, 1.4):
        t = rng.uniform(0.35, 0.65)
        return split_rooms(rng, x0, y0, x0 + w * t, y1, depth + 1) + \
            split_rooms(rng, x0 + w * t, y0, x1, y1, depth + 1)
    t = rng.uniform(0.35, 0.65)
    return split_rooms(rng, x0, y0, x1, y0 + h * t, depth + 1) + \
        split_rooms(rng, x0, y0 + h * t, x1, y1, depth + 1)


def build_floor_plan(house, rng):
    mb = house["blocks"][0]
    rooms = split_rooms(rng, mb.x0, mb.y0, mb.x1, mb.y1)
    t_ext, t_int = rng.uniform(0.5, 0.8), rng.uniform(0.33, 0.5)
    fp = rect(mb.x0, mb.y0, mb.x1, mb.y1)
    room_polys = []
    for (x0, y0, x1, y1) in rooms:
        r = rect(x0, y0, x1, y1)
        inner = r.buffer(-t_int / 2, join_style=2)
        # exterior walls thicker
        inner = inner.intersection(fp.buffer(-t_ext, join_style=2))
        room_polys.append(inner)
    walls = fp.difference(unary_union(room_polys))
    garage = None
    for b in house["blocks"][1:]:
        if b.kind == "garage":
            g = rect(b.x0, b.y0, b.x1, b.y1)
            gin = g.buffer(-t_ext, join_style=2)
            walls = walls.union(g.difference(gin))
            garage = gin
            # shared wall opening not needed
    # door openings between adjacent rooms
    doors = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b_ = rooms[i], rooms[j]
            # shared vertical edge
            if abs(a[2] - b_[0]) < 1e-6 or abs(b_[2] - a[0]) < 1e-6:
                x = a[2] if abs(a[2] - b_[0]) < 1e-6 else a[0]
                lo, hi = max(a[1], b_[1]), min(a[3], b_[3])
                if hi - lo > 4.5 and rng.random() < 0.8:
                    c = rng.uniform(lo + 2, hi - 2)
                    doors.append(("v", x, c, 2.67))
            if abs(a[3] - b_[1]) < 1e-6 or abs(b_[3] - a[1]) < 1e-6:
                y = a[3] if abs(a[3] - b_[1]) < 1e-6 else a[1]
                lo, hi = max(a[0], b_[0]), min(a[2], b_[2])
                if hi - lo > 4.5 and rng.random() < 0.8:
                    c = rng.uniform(lo + 2, hi - 2)
                    doors.append(("h", y, c, 2.67))
    # front door + windows on the exterior
    fx = mb.x0 + house["front_door_u"] * mb.w
    doors.append(("h", mb.y0, fx, 3.0))
    windows = []
    for (x0, y0, x1, y1) in rooms:
        for (edge, is_ext) in [(("h", y0, x0, x1), abs(y0 - mb.y0) < 1e-6),
                               (("h", y1, x0, x1), abs(y1 - mb.y1) < 1e-6),
                               (("v", x0, y0, y1), abs(x0 - mb.x0) < 1e-6),
                               (("v", x1, y0, y1), abs(x1 - mb.x1) < 1e-6)]:
            if not is_ext:
                continue
            o, pos, lo, hi = edge
            n = max(0, int((hi - lo) / rng.uniform(6, 12)))
            for k in range(n):
                c = lo + (k + 0.5) * (hi - lo) / n
                ww = rng.choice([2.5, 3.0, 4.0, 5.0])
                if o == "h" and abs(pos - mb.y0) < 1e-6 and abs(c - fx) < 4:
                    continue
                windows.append((o, pos, c, ww))
    cut = []
    for (o, pos, c, w) in doors + windows:
        if o == "v":
            cut.append(rect(pos - 1.0, c - w / 2, pos + 1.0, c + w / 2))
        else:
            cut.append(rect(c - w / 2, pos - 1.0, c + w / 2, pos + 1.0))
    walls_cut = walls.difference(unary_union(cut)) if cut else walls
    names = rng.sample(ROOM_NAMES, min(len(rooms), len(ROOM_NAMES)))
    # exterior patio / deck
    patio = None
    if rng.random() < 0.4:
        pw, pd = rng.uniform(10, min(24, mb.w - 4)), rng.uniform(8, 16)
        px = rng.uniform(mb.x0, mb.x1 - pw)
        patio = rect(px, mb.y1, px + pw, mb.y1 + pd)
    return {"rooms": rooms, "room_polys": room_polys, "walls": walls_cut, "walls_full": walls,
            "doors": doors, "windows": windows, "names": names, "garage": garage, "patio": patio,
            "footprint": fp, "t_ext": t_ext}


# ----------------------------------------------------------------------------
# text / annotation helpers (PIL, on top at the end)
# ----------------------------------------------------------------------------
class TextQueue:
    def __init__(self):
        self.items = []

    def add(self, xy, s, size, colour, bold=False, anchor="la", angle=0):
        self.items.append((xy, s, size, colour, bold, anchor, angle))

    def flush(self, canvas_bgr):
        if not self.items:
            return canvas_bgr
        img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for (xy, s, size, colour, bold, anchor, angle) in self.items:
            f = font(max(8, int(size)), bold)
            if angle:
                tw = int(d.textlength(s, font=f)) + 4
                tmp = Image.new("RGBA", (tw, int(size * 1.4)), (0, 0, 0, 0))
                ImageDraw.Draw(tmp).text((2, 0), s, font=f, fill=colour + (255,))
                tmp = tmp.rotate(angle, expand=True)
                img.paste(tmp, (int(xy[0]), int(xy[1]) - tmp.height), tmp)
            else:
                d.text(xy, s, font=f, fill=colour, anchor=anchor)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def ft_label(v):
    ft_ = int(v)
    in_ = int(round((v - ft_) * 12))
    if in_ == 12:
        ft_, in_ = ft_ + 1, 0
    return f"{ft_}'-{in_}\""


def draw_dim_string(canvas, tq, V, pts_ft, y_ft, app, S, above=True, size=None):
    """Horizontal dimension chain through x positions pts_ft at height y_ft."""
    ink = app["ink"]
    lw = app["outline_lw"] * 0.7
    size = size or app["text_px"]
    pts_ft = sorted(pts_ft)
    a, b = V.px(pts_ft[0], y_ft), V.px(pts_ft[-1], y_ft)
    cv_line(canvas, a, b, ink, lw)
    for x in pts_ft:
        p = V.px(x, y_ft)
        cv_line(canvas, (p[0], p[1] - 0.25 * S), (p[0], p[1] + 0.25 * S), ink, lw)
        cv_line(canvas, (p[0] - 0.15 * S, p[1] + 0.15 * S), (p[0] + 0.15 * S, p[1] - 0.15 * S), ink, lw * 1.3)
    for x0, x1 in zip(pts_ft[:-1], pts_ft[1:]):
        p = V.px((x0 + x1) / 2, y_ft)
        tq.add((p[0], p[1] - (0.15 * S if above else -0.15 * S)), ft_label(x1 - x0), size, ink,
               anchor="ms" if above else "ma")


def draw_leader(canvas, tq, V, target_ft, text, app, S, side, y_off=0.0, size=None):
    ink = app["ink"]
    lw = app["outline_lw"] * 0.7
    size = size or app["text_px"]
    t = V.px(*target_ft)
    dx = (rng_sign(side)) * size * 4.0
    elbow = (t[0] + dx, t[1] - size * 2.5 + y_off * size)
    end = (elbow[0] + rng_sign(side) * size * 1.5, elbow[1])
    cv_line(canvas, t, elbow, ink, lw)
    cv_line(canvas, elbow, end, ink, lw)
    cv2.circle(canvas, _pt(t), int(0.06 * S * SCALE), bgr(ink), -1, lineType=cv2.LINE_AA, shift=SHIFT)
    anchor = "lm" if side > 0 else "rm"
    tq.add((end[0] + rng_sign(side) * 3, end[1]), text, size, ink, anchor=anchor)


def rng_sign(side):
    return 1 if side > 0 else -1


def draw_level_mark(canvas, tq, V, x_ft, y_ft, label, elev_ft, app, S, extent_u):
    ink = app["ink"]
    lw = app["outline_lw"] * 0.7
    p = V.px(x_ft, y_ft)
    q = V.px(extent_u, y_ft)
    # dashed line across the view
    dash = 0.6 * S
    x = min(p[0], q[0])
    xe = max(p[0], q[0])
    while x < xe:
        cv_line(canvas, (x, p[1]), (min(xe, x + dash), p[1]), mix(ink, (255, 255, 255), 0.3), lw)
        x += dash * 1.8
    tri = [(p[0], p[1]), (p[0] - 0.4 * S, p[1] - 0.4 * S), (p[0] + 0.4 * S, p[1] - 0.4 * S)]
    cv_polyline(canvas, tri, ink, lw, closed=True)
    size = app["text_px"] * 0.9
    anchor = "ls" if x_ft > extent_u else "rs"
    off = 0.5 * S if x_ft > extent_u else -0.5 * S
    tq.add((p[0] + off, p[1] - 0.5 * S), label, size, ink, anchor=anchor)
    tq.add((p[0] + off, p[1] + 0.9 * S), elev_ft, size, ink, anchor="la" if x_ft > extent_u else "ra")


# ----------------------------------------------------------------------------
# rendering one elevation view into the canvas
# ----------------------------------------------------------------------------
def render_elevation_view(canvas, tq, V, elev, styles, house, app, rng, S, W, H, label_fams, hole_mode,
                          view_name, title):
    ink, lw = app["ink"], app["outline_lw"]
    out = []   # (family, poly_px_with_holes)
    opening_polys = unary_union([o["poly"] for o in elev["openings"]]) if elev["openings"] else None
    # ground line
    ext = elev["extent"]
    gx0, gx1 = ext[0] - rng.uniform(2, 8), ext[2] + rng.uniform(2, 8)
    a, b = V.px(gx0, 0), V.px(gx1, 0)
    cv_line(canvas, a, b, ink, app["outline_lw"] * 1.8)
    if rng.random() < 0.35:
        # ground hatch
        x = a[0]
        while x < b[0]:
            cv_line(canvas, (x, a[1] + 2), (x - 0.35 * S, a[1] + 0.35 * S), ink, 1)
            x += 0.45 * S
    # unlabelled distractors: a neighbour "beyond" in dashed outline, a fence or
    # retaining wall with its own hatch, a tree; real sheets carry these and the
    # model must not select them when asked for a wall material
    if rng.random() < 0.3:
        side = rng.choice([-1, 1])
        nx0 = (ext[2] + 2) if side > 0 else (ext[0] - 2 - rng.uniform(14, 26))
        nw_, nh_ = rng.uniform(14, 26), rng.uniform(9, 20)
        pts = [V.px(nx0, 0), V.px(nx0, -nh_), V.px(nx0 + nw_ / 2, -nh_ - rng.uniform(3, 7)), V.px(nx0 + nw_, -nh_),
               V.px(nx0 + nw_, 0)]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            L = math.dist(a_, b_); n = max(1, int(L / (0.5 * S)))
            for k in range(0, n, 2):
                t0, t1 = k / n, min(1, (k + 1) / n)
                cv_line(canvas, (a_[0] + (b_[0] - a_[0]) * t0, a_[1] + (b_[1] - a_[1]) * t0),
                        (a_[0] + (b_[0] - a_[0]) * t1, a_[1] + (b_[1] - a_[1]) * t1), mix(ink, (255, 255, 255), 0.4), lw)
    if rng.random() < 0.25:
        # fence / retaining wall band with a hatch of its own, never labelled
        fx0 = rng.choice([ext[0] - rng.uniform(4, 12), ext[2] + rng.uniform(1, 3)])
        fw, fhh = rng.uniform(4, 10), rng.uniform(2.5, 5)
        fpoly = rect(fx0, 0, fx0 + fw, -fhh)
        fst = Style(rng.choice(["vertical", "hatch", "grid", "block"]), app["paper"] if not app["colour"] else
                    rng.choice(BASE_PALETTE), app["ink_fill"] if not app["colour"] else app["ink"],
                    app["outline_lw"] * 0.6, {"sp": rng.choice([0.33, 0.5, 1.0]), "angle": 45}, rng.randrange(1 << 20), "fence")
        draw_fill(canvas, V.geom(fpoly), fst, S, W, H)
        cv_outline(canvas, V.geom(fpoly), ink, lw)
    if rng.random() < 0.2:
        # tree: a scribbled canopy over a trunk
        tx = rng.choice([ext[0] - rng.uniform(3, 9), ext[2] + rng.uniform(3, 9)])
        th = rng.uniform(10, 24)
        cv_line(canvas, V.px(tx, 0), V.px(tx, -th * 0.45), ink, lw * 1.5)
        c = V.px(tx, -th * 0.7)
        rr = rng.uniform(4, 8) * S
        for k in range(28):
            ang = k * 360 / 28
            q = (c[0] + rr * math.cos(math.radians(ang)) * rng.uniform(0.75, 1.05),
                 c[1] + rr * 0.8 * math.sin(math.radians(ang)) * rng.uniform(0.75, 1.05))
            cv2.circle(canvas, _pt(q), int(rr * 0.12 * SCALE), bgr(ink), max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
    # surfaces
    for (fam, poly, bi, kind) in elev["surfaces"]:
        if poly.is_empty:
            continue
        st = styles[fam]
        p = poly
        if opening_polys is not None:
            p = p.difference(opening_polys)          # visually windows always sit in front
        ppx = V.geom(p)
        draw_fill(canvas, ppx, st, S, W, H)
        # outline
        cv_outline(canvas, ppx, app["outline"], lw)
        if fam in label_fams:
            lab = poly.difference(opening_polys) if (hole_mode and opening_polys is not None) else poly
            for q in polys_of(lab):
                out.append((fam, V.geom(q)))
    # eave and window-head shadows: BIM exports darken the wall under every
    # overhang; the shadow is still the wall material, so labels are unchanged
    if app["colour"] and house.get("shadows"):
        shade = np.zeros((H, W), dtype=np.uint8)
        for (fam, poly, bi, kind) in elev["surfaces"]:
            if kind not in ("roof", "porch"):
                continue
            x0, y0, x1, y1 = poly.bounds
            band = rect(x0, y1, x1, y1 + house["shadow_h"])
            m = poly_mask(V.geom(band), W, H)
            shade = np.maximum(shade, m)
        for o in elev["openings"]:
            x0, y0, x1, y1 = o["poly"].bounds
            m = poly_mask(V.geom(rect(x0 - 0.3, y0 - 0.3, x1 + 0.3, y0 - 0.3 + house["shadow_h"] * 0.5)), W, H)
            shade = np.maximum(shade, m)
        wall_mask = np.zeros((H, W), dtype=np.uint8)
        for (fam, poly, bi, kind) in elev["surfaces"]:
            if kind in ("wall", "dormer", "found", "chimney"):
                wall_mask = np.maximum(wall_mask, poly_mask(V.geom(poly), W, H))
        shade = (shade.astype(np.float32) / 255.0) * (wall_mask.astype(np.float32) / 255.0)
        shade = cv2.GaussianBlur(shade, (0, 0), max(1.0, 0.08 * S))
        canvas[:] = (canvas.astype(np.float32) * (1 - house["shadow_k"] * shade[..., None])).astype(np.uint8)
    # trims
    trim_colour = styles["trim"].base if "trim" in styles else app["trim"]
    trim_polys = []
    for tp in elev["trims"]:
        for p in polys_of(tp):
            ppx = V.geom(p)
            cv_fill(canvas, ppx, trim_colour)
            cv_outline(canvas, ppx, app["outline"], lw)
            trim_polys.append(p)
    # corner boards
    if house["corner_boards"]:
        for (fam, poly, bi, kind) in elev["surfaces"]:
            if kind != "wall":
                continue
            x0, y0, x1, y1 = poly.bounds
            for x in (x0, x1):
                cbp = rect(x - 0.2, y1, x + 0.2, y0 + 0.01)
                cb = V.geom(cbp)
                cv_fill(canvas, cb, trim_colour)
                cv_outline(canvas, cb, app["outline"], lw)
                trim_polys.append(cbp)
    # openings
    for o in elev["openings"]:
        draw_opening(canvas, V, o, house, app, S, rng, trim_colour)
        if o["type"] in ("window", "door"):
            x0, y0, x1, y1 = o["poly"].bounds
            casing = rect(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + (0.3 if o["type"] == "window" else 0)).difference(o["poly"])
            trim_polys.append(casing)
    if "trim" in label_fams:
        for p in polys_of(unary_union(trim_polys)):
            if p.area > 0.3:
                out.append(("trim", V.geom(p)))
    # roof material callouts etc.
    notes = []
    for fam in ("main", "accent", "roof", "chimney", "foundation"):
        if fam in styles and any(f == fam for (f, _, _, _) in elev["surfaces"]):
            if rng.random() < (0.7 if fam in ("main", "accent") else 0.45):
                cand = [p for (f, p, _, _) in elev["surfaces"] if f == fam and p.area > 6]
                if cand:
                    p = rng.choice(cand)
                    pt = p.representative_point()
                    side = 1 if pt.x > (ext[0] + ext[2]) / 2 else -1
                    notes.append((pt.x, pt.y, styles[fam].callout, side))
    for i, (x, y, text, side) in enumerate(notes[:5]):
        draw_leader(canvas, tq, V, (x + rng.uniform(-1, 1), y + rng.uniform(-1, 1)), text, app, S, side,
                    y_off=-i * 0.9)
    # level marks
    if rng.random() < 0.5:
        fh = house["fh"]
        side_x = ext[2] + rng.uniform(1.5, 4.0)
        labels = [("FIN. FLOOR", 0.0), ("T.O. PLATE", -(fh))]
        if house["blocks"][0].stories >= 2:
            labels.append(("SECOND FLOOR", -fh))
            labels.append(("T.O. PLATE", -(2 * fh)))
        for (lab, y) in labels[:rng.randint(2, len(labels))]:
            draw_level_mark(canvas, tq, V, side_x, y, lab, f"{'+' if y < 0 else ''}{ft_label(-y)}", app, S, ext[0])
    # dimension string across the top or bottom
    if rng.random() < 0.5:
        us = [ext[0], ext[2]]
        for b in house["blocks"]:
            u0, u1, *_ = block_uv(b, view_name)
            us += [u0, u1]
        us = sorted(set(round(u, 2) for u in us))
        y = ext[1] - rng.uniform(1.5, 3.0) if rng.random() < 0.5 else 2.0
        draw_dim_string(canvas, tq, V, us, y, app, S, above=y < 0)
    # title
    if title:
        p = V.px(ext[0] + rng.uniform(0, 3), rng.uniform(2.6, 3.6))
        size = app["text_px"] * 1.5
        tq.add(p, title, size, ink, bold=rng.random() < 0.5)
        sc = rng.choice(["1/4\" = 1'-0\"", "1/8\" = 1'-0\"", "3/16\" = 1'-0\""])
        tq.add((p[0], p[1] + size * 1.2), f"SCALE: {sc}", size * 0.6, ink)
        if rng.random() < 0.6:
            cv_line(canvas, (p[0], p[1] + size * 1.15), (p[0] + size * 9, p[1] + size * 1.15), ink, 1.5)
    return out


def draw_opening(canvas, V, o, house, app, S, rng, trim_colour=None):
    ink, lw = app["ink"], app["outline_lw"]
    trim_c = trim_colour if trim_colour is not None else app["trim"]
    p = o["poly"]
    x0, y0, x1, y1 = p.bounds
    glass = (225, 236, 246) if app["colour"] else app["paper"]
    if o["type"] == "window":
        trim = V.geom(rect(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3))
        cv_fill(canvas, trim, trim_c)
        cv_outline(canvas, trim, ink, lw)
        g = V.geom(p)
        cv_fill(canvas, g, glass)
        cv_outline(canvas, g, ink, lw)
        cols, rows = {"2x2": (2, 2), "1x2": (1, 2), "1x1": (1, 1), "3x2": (3, 2), "6lite": (3, 2)}[house["window_style"]]
        if (x1 - x0) > 4.5 and cols == 1:
            cols = 2
        for c in range(1, cols):
            x = x0 + (x1 - x0) * c / cols
            cv_line(canvas, V.px(x, y0), V.px(x, y1), ink, lw)
        for r in range(1, rows):
            y = y0 + (y1 - y0) * r / rows
            cv_line(canvas, V.px(x0, y), V.px(x1, y), ink, lw)
        # sill
        sill = V.geom(rect(x0 - 0.45, y1, x1 + 0.45, y1 + 0.25))
        cv_fill(canvas, sill, trim_c)
        cv_outline(canvas, sill, ink, lw)
    elif o["type"] == "door":
        trim = V.geom(rect(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1))
        cv_fill(canvas, trim, trim_c)
        cv_outline(canvas, trim, ink, lw)
        g = V.geom(p)
        cv_fill(canvas, g, mix(app["paper"], ink, 0.08) if not app["colour"] else (200, 205, 210))
        cv_outline(canvas, g, ink, lw)
        # panels
        n = 2 if (x1 - x0) > 4 else 1
        for i in range(n):
            px0 = x0 + (x1 - x0) * i / n
            px1 = x0 + (x1 - x0) * (i + 1) / n
            for (a, b, c, d) in [(px0 + 0.4, y1 - 5.8, px1 - 0.4, y1 - 3.3), (px0 + 0.4, y1 - 2.8, px1 - 0.4, y1 - 0.8)]:
                cv_outline(canvas, V.geom(rect(a, b, c, d)), ink, lw)
        if (x1 - x0) > 2 and rng.random() < 0.5:
            cv2.circle(canvas, _pt(V.px(x1 - 0.5, y1 - 3.3)), int(0.08 * S * SCALE), bgr(ink), -1,
                       lineType=cv2.LINE_AA, shift=SHIFT)
    elif o["type"] == "garage":
        g = V.geom(p)
        cv_fill(canvas, g, app["trim"] if rng.random() < 0.5 else (
            (215, 218, 222) if app["colour"] else mix(app["paper"], ink, 0.08)))
        cv_outline(canvas, g, ink, lw)
        rows = 4
        for r in range(1, rows):
            y = y0 + (y1 - y0) * r / rows
            cv_line(canvas, V.px(x0, y), V.px(x1, y), ink, lw)
        cols = max(2, int((x1 - x0) / 2.0))
        for r in range(rows):
            for c in range(cols):
                a = x0 + (x1 - x0) * (c + 0.12) / cols
                b = x0 + (x1 - x0) * (c + 0.88) / cols
                ya = y0 + (y1 - y0) * (r + 0.2) / rows
                yb = y0 + (y1 - y0) * (r + 0.8) / rows
                cv_outline(canvas, V.geom(rect(a, ya, b, yb)), ink, max(1, lw * 0.8))


# ----------------------------------------------------------------------------
# roof plan rendering
# ----------------------------------------------------------------------------
def render_roof_plan(canvas, tq, V, rp, styles, app, rng, S, W, H, label_mode):
    ink, lw = app["ink"], app["outline_lw"]
    out = []
    # facets
    st = styles["roof"]
    facet_polys = []
    for (p, ang) in rp["facets"]:
        st_f = Style("roof_rows", st.base, st.line, st.lw, {"row": st.params.get("row", 0.5),
                                                              "angle": ang,
                                                              "joints": st.params.get("joints", False)},
                     st.seed, st.label_name)
        draw_fill(canvas, V.geom(p), st_f, S, W, H)
        facet_polys.append(p)
    skylights = []
    if rng.random() < 0.3 and facet_polys:
        for _ in range(rng.randint(1, 3)):
            f = rng.choice(facet_polys)
            x0, y0, x1, y1 = f.bounds
            cx, cy = rng.uniform(x0 + 3, x1 - 3), rng.uniform(y0 + 3, y1 - 3)
            sk = rect(cx - 1.5, cy - 1.2, cx + 1.5, cy + 1.2)
            if f.contains(sk):
                skylights.append(sk)
    for sk in skylights:
        g = V.geom(sk)
        cv_fill(canvas, g, app["trim"] if not app["colour"] else (225, 236, 246))
        cv_outline(canvas, g, ink, lw)
        cv_outline(canvas, V.geom(sk.buffer(-0.3, join_style=2)), ink, lw)
    for fp in rp["flat"]:
        g = V.geom(fp)
        cv_fill(canvas, g, (232, 232, 232) if app["colour"] else app["paper"])
        cv_outline(canvas, g, ink, lw)
        if "flat" in styles:
            draw_fill(canvas, g, styles["flat"], S, W, H)
    cv_outline(canvas, V.geom(rp["outline"]), ink, lw * 1.5)
    for l in rp["ridges"]:
        for seg in lines_of(l):
            c = list(seg.coords)
            cv_polyline(canvas, [V.px(*q) for q in c], ink, lw)
    # walls below, dashed
    for p in polys_of(rp["footprint"]):
        coords = [V.px(*q) for q in p.exterior.coords]
        for a, b in zip(coords[:-1], coords[1:]):
            L = math.dist(a, b)
            n = max(1, int(L / (0.6 * S)))
            for k in range(0, n, 2):
                t0, t1 = k / n, min(1, (k + 1) / n)
                cv_line(canvas, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                        (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), mix(ink, (255, 255, 255), 0.3), lw)
    # slope arrows
    for (p, ang) in rp["facets"]:
        if p.area < 40 or rng.random() < 0.4:
            continue
        c = p.representative_point()
        L = 2.0
        d = (0, 1) if ang == 0 else (1, 0)
        d = (d[0] * rng.choice([-1, 1]), d[1] * rng.choice([-1, 1]))
        a = V.px(c.x - d[0] * L, c.y - d[1] * L)
        b = V.px(c.x + d[0] * L, c.y + d[1] * L)
        cv_line(canvas, a, b, ink, lw)
        cv2.circle(canvas, _pt(b), int(0.12 * S * SCALE), bgr(ink), -1, lineType=cv2.LINE_AA, shift=SHIFT)
        tq.add((b[0] + 0.3 * S, b[1]), rng.choice(["4:12", "6:12", "8:12", "5:12"]), app["text_px"] * 0.9, ink, anchor="lm")
    # labels
    labels = []
    if label_mode == "one":
        u = unary_union(facet_polys)
        for q in polys_of(u):
            if skylights:
                q = q.difference(unary_union(skylights))
            for qq in polys_of(q):
                labels.append(("roof", V.geom(qq)))
    else:
        # per block: facets grouped by the block are not tracked; group by touching
        u = unary_union(facet_polys)
        for q in polys_of(u):
            if skylights:
                q = q.difference(unary_union(skylights))
            for qq in polys_of(q):
                labels.append(("roof", V.geom(qq)))
    if "flat" in styles:
        for fp in rp["flat"]:
            for q in polys_of(fp):
                labels.append(("flat", V.geom(q)))
    out.extend(labels)
    # dims + title + north arrow
    x0, y0, x1, y1 = rp["outline"].bounds
    if rng.random() < 0.6:
        xs = sorted(set([round(b.x0, 2) for b in [None] if False] + [x0, x1] +
                        [round(v, 2) for p in polys_of(rp["footprint"]) for v in [p.bounds[0], p.bounds[2]]]))
        draw_dim_string(canvas, tq, V, xs, y0 - rng.uniform(2, 4), app, S)
    if rng.random() < 0.7:
        c = V.px(x1 + rng.uniform(3, 6), y0 + rng.uniform(0, 4))
        r = 1.2 * S
        cv2.circle(canvas, _pt(c), int(r * SCALE), bgr(ink), max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
        cv_polyline(canvas, [(c[0], c[1] - r), (c[0] - r * 0.35, c[1] + r * 0.3), (c[0], c[1] + r * 0.05),
                             (c[0] + r * 0.35, c[1] + r * 0.3)], ink, lw, closed=True)
        tq.add((c[0], c[1] - r - 0.2 * S), "N", app["text_px"] * 1.2, ink, anchor="ms", bold=True)
    p = V.px(x0, y1 + rng.uniform(3, 4.5))
    size = app["text_px"] * 1.5
    tq.add(p, "ROOF PLAN", size, ink, bold=True)
    tq.add((p[0], p[1] + size * 1.2), f"SCALE: {rng.choice(['1/8', '3/16', '1/4'])}\" = 1'-0\"", size * 0.6, ink)
    for fam in ("roof",):
        if rng.random() < 0.6 and facet_polys:
            q = rng.choice(facet_polys).representative_point()
            draw_leader(canvas, tq, V, (q.x, q.y), styles["roof"].callout, app, S, 1 if q.x > (x0 + x1) / 2 else -1)
    return out


# ----------------------------------------------------------------------------
# floor plan rendering
# ----------------------------------------------------------------------------
def render_floor_plan(canvas, tq, V, fp, styles, app, rng, S, W, H, plan_cfg):
    ink, lw = app["ink"], app["outline_lw"]
    out = []
    finishes = plan_cfg["finishes"]           # list of (family, [room polys])
    if plan_cfg.get("room_tint"):
        tint = plan_cfg["room_tint"]
        for rp in fp["room_polys"]:
            cv_fill(canvas, V.geom(rp), tint)
        if fp["garage"] is not None:
            cv_fill(canvas, V.geom(fp["garage"]), tint)
    for fam, polys in finishes:
        st = styles[fam]
        for p in polys:
            g = V.geom(p)
            draw_fill(canvas, g, st, S, W, H)
            if fam in plan_cfg["label"]:
                out.append((fam, g))
    # walls
    wg = V.geom(fp["walls"])
    if plan_cfg["poche"]:
        cv_fill(canvas, wg, styles["walls"].base)
        if "walls" in plan_cfg["label"]:
            for q in polys_of(fp["walls"]):
                out.append(("walls", V.geom(q)))
    else:
        cv_fill(canvas, wg, app["paper"])
    cv_outline(canvas, wg, ink, lw * (1.6 if not plan_cfg["poche"] else 1.0))
    # windows in plan: three thin lines across the opening
    for (o, pos, c, w) in fp["windows"]:
        t = fp["t_ext"]
        if o == "h":
            for k in (-0.5, 0, 0.5):
                cv_line(canvas, V.px(c - w / 2, pos + k * t * 0.8), V.px(c + w / 2, pos + k * t * 0.8), ink, lw)
            cv_line(canvas, V.px(c - w / 2, pos - t / 2), V.px(c - w / 2, pos + t / 2), ink, lw)
            cv_line(canvas, V.px(c + w / 2, pos - t / 2), V.px(c + w / 2, pos + t / 2), ink, lw)
        else:
            for k in (-0.5, 0, 0.5):
                cv_line(canvas, V.px(pos + k * t * 0.8, c - w / 2), V.px(pos + k * t * 0.8, c + w / 2), ink, lw)
            cv_line(canvas, V.px(pos - t / 2, c - w / 2), V.px(pos + t / 2, c - w / 2), ink, lw)
            cv_line(canvas, V.px(pos - t / 2, c + w / 2), V.px(pos + t / 2, c + w / 2), ink, lw)
    # doors: leaf + arc
    for (o, pos, c, w) in fp["doors"]:
        sgn = rng.choice([-1, 1])
        if o == "h":
            hinge = (c - w / 2, pos)
            leaf_end = (hinge[0], pos + sgn * w)
            cv_line(canvas, V.px(*hinge), V.px(*leaf_end), ink, lw * 1.3)
            ctr = V.px(*hinge)
            ang0, ang1 = (0, 90) if sgn > 0 else (270, 360)
            cv2.ellipse(canvas, _pt(ctr), (int(w * S * SCALE), int(w * S * SCALE)), 0, ang0, ang1, bgr(ink),
                        max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
        else:
            hinge = (pos, c - w / 2)
            leaf_end = (pos + sgn * w, hinge[1])
            cv_line(canvas, V.px(*hinge), V.px(*leaf_end), ink, lw * 1.3)
            ctr = V.px(*hinge)
            ang0, ang1 = (0, 90) if sgn > 0 else (90, 180)
            cv2.ellipse(canvas, _pt(ctr), (int(w * S * SCALE), int(w * S * SCALE)), 0, ang0, ang1, bgr(ink),
                        max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
    # fixtures + labels
    size = app["text_px"]
    for (r, name, rp) in zip(fp["rooms"], fp["names"], fp["room_polys"]):
        x0, y0, x1, y1 = r
        c = V.px((x0 + x1) / 2, (y0 + y1) / 2)
        tq.add(c, name, size, ink, anchor="mm")
        if rng.random() < 0.6:
            tq.add((c[0], c[1] + size * 1.2), f"{ft_label(x1 - x0)} x {ft_label(y1 - y0)}", size * 0.8, ink, anchor="mm")
        draw_fixtures(canvas, V, name, r, ink, lw, S, rng)
    if fp["garage"] is not None:
        c = fp["garage"].representative_point()
        tq.add(V.px(c.x, c.y), "GARAGE", size, ink, anchor="mm")
    if fp["patio"] is not None:
        c = fp["patio"].representative_point()
        tq.add(V.px(c.x, c.y), rng.choice(["PATIO", "DECK", "COVERED PATIO"]), size, ink, anchor="mm")
        cv_outline(canvas, V.geom(fp["patio"]), ink, lw)
    # MEP overlay
    if plan_cfg["mep"]:
        x0, y0, x1, y1 = fp["footprint"].bounds
        col = ink if not app["colour"] else (60, 60, 60)
        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            r = rng.uniform(2, 6) * S
            ctr = V.px(cx, cy)
            for a in range(0, 360, 20):
                cv2.ellipse(canvas, _pt(ctr), (int(r * SCALE), int(r * SCALE)), 0, a, a + 10, bgr(col), max(1, int(lw)),
                            lineType=cv2.LINE_AA, shift=SHIFT)
            tq.add((ctr[0], ctr[1]), rng.choice(["S", "$", "B1", "F", "WH"]), size * 0.8, col, anchor="mm")
        for _ in range(rng.randint(2, 5)):
            a = V.px(rng.uniform(x0, x1), rng.uniform(y0, y1))
            b = V.px(rng.uniform(x0, x1), rng.uniform(y0, y1))
            L = math.dist(a, b)
            n = max(2, int(L / (0.5 * S)))
            for k in range(0, n, 2):
                t0, t1 = k / n, min(1, (k + 1) / n)
                cv_line(canvas, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                        (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), col, lw)
    # dims
    x0, y0, x1, y1 = fp["footprint"].bounds
    if rng.random() < 0.7:
        xs = sorted(set([round(v, 2) for r in fp["rooms"] for v in (r[0], r[2])]))
        draw_dim_string(canvas, tq, V, xs, y0 - rng.uniform(2, 3.5), app, S)
        if rng.random() < 0.5:
            draw_dim_string(canvas, tq, V, [x0, x1], y0 - rng.uniform(4.5, 6), app, S)
    if rng.random() < 0.6:
        for rr in [fp["footprint"]] + ([fp["patio"]] if fp["patio"] is not None else []):
            pass
    p = V.px(x0, max(y1, fp["patio"].bounds[3] if fp["patio"] is not None else y1) + rng.uniform(3, 4.5))
    tsize = app["text_px"] * 1.5
    tq.add(p, rng.choice(["FLOOR PLAN", "FIRST FLOOR PLAN", "MAIN FLOOR PLAN", "PROPOSED FLOOR PLAN",
                          "UNDER-FLOOR PLAN"]), tsize, ink, bold=True)
    tq.add((p[0], p[1] + tsize * 1.2), f"SCALE: {rng.choice(['1/4', '1/8', '3/16'])}\" = 1'-0\"", tsize * 0.6, ink)
    for fam, polys in finishes:
        if rng.random() < 0.6 and polys:
            q = rng.choice(polys).representative_point()
            draw_leader(canvas, tq, V, (q.x, q.y), styles[fam].callout, app, S, 1 if q.x > (x0 + x1) / 2 else -1)
    return out


def draw_fixtures(canvas, V, name, r, ink, lw, S, rng):
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    if "BED" in name and w > 9 and h > 9:
        bw, bh = (5.0, 6.5) if "MASTER" not in name else (6.0, 6.8)
        bx = x0 + 1.0
        by = y0 + (h - bh) / 2
        cv_outline(canvas, V.geom(rect(bx, by, bx + bw, by + bh)), ink, lw)
        cv_outline(canvas, V.geom(rect(bx + 0.3, by + 0.3, bx + bw / 2 - 0.2, by + 1.5)), ink, lw)
        cv_outline(canvas, V.geom(rect(bx + bw / 2 + 0.2, by + 0.3, bx + bw - 0.3, by + 1.5)), ink, lw)
    elif "BATH" in name and w > 5 and h > 5:
        # tub along a wall, toilet, sink
        cv_outline(canvas, V.geom(rect(x0 + 0.5, y0 + 0.5, x0 + 0.5 + min(5, w - 1), y0 + 0.5 + 2.5)), ink, lw)
        cv_outline(canvas, V.geom(rect(x0 + 0.8, y0 + 0.8, x0 + 0.2 + min(5, w - 1), y0 + 0.2 + 2.5).buffer(-0.1)), ink, lw)
        ctr = V.px(x1 - 1.2, y1 - 1.2)
        cv2.ellipse(canvas, _pt(ctr), (int(0.7 * S * SCALE), int(0.9 * S * SCALE)), 0, 0, 360, bgr(ink), max(1, int(lw)),
                    lineType=cv2.LINE_AA, shift=SHIFT)
        cv_outline(canvas, V.geom(rect(x1 - 1.9, y1 - 2.9, x1 - 0.5, y1 - 2.1)), ink, lw)
    elif "KITCHEN" in name and w > 8 and h > 8:
        cv_outline(canvas, V.geom(rect(x0 + 0.4, y0 + 0.4, x1 - 0.4, y0 + 2.4)), ink, lw)
        cv_outline(canvas, V.geom(rect(x0 + 0.4, y0 + 0.4, x0 + 2.4, y1 - 0.4)), ink, lw)
        cv_outline(canvas, V.geom(rect(x0 + 3, y0 + 0.7, x0 + 5.2, y0 + 2.1)), ink, lw)     # sink
        for dx, dy in [(0.5, 0.4), (1.5, 0.4), (0.5, 1.1), (1.5, 1.1)]:
            cv2.circle(canvas, _pt(V.px(x1 - 3.0 + dx, y0 + 0.6 + dy)), int(0.28 * S * SCALE), bgr(ink), max(1, int(lw)),
                       lineType=cv2.LINE_AA, shift=SHIFT)
        if w > 12 and h > 12:
            cv_outline(canvas, V.geom(rect((x0 + x1) / 2 - 2, (y0 + y1) / 2 - 1, (x0 + x1) / 2 + 2, (y0 + y1) / 2 + 1)), ink, lw)
    elif "DINING" in name and w > 8 and h > 8:
        c = V.px((x0 + x1) / 2, (y0 + y1) / 2)
        cv2.circle(canvas, _pt(c), int(2.2 * S * SCALE), bgr(ink), max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
        for a in range(0, 360, 60):
            q = (c[0] + 3.0 * S * math.cos(math.radians(a)), c[1] + 3.0 * S * math.sin(math.radians(a)))
            cv2.circle(canvas, _pt(q), int(0.7 * S * SCALE), bgr(ink), max(1, int(lw)), lineType=cv2.LINE_AA, shift=SHIFT)
    elif ("LIVING" in name or "FAMILY" in name) and w > 10 and h > 10:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cv_outline(canvas, V.geom(rect(cx - 4, cy + 1.5, cx + 4, cy + 4.5)), ink, lw)
        cv_outline(canvas, V.geom(rect(cx - 4, cy + 2.2, cx + 4, cy + 4.5)), ink, lw)
        cv_outline(canvas, V.geom(rect(cx - 2, cy - 2, cx + 2, cy)), ink, lw)


# ----------------------------------------------------------------------------
# sheet composition
# ----------------------------------------------------------------------------
def compose(image_id, seed, mode_weights):
    rng = random.Random(seed * 1_000_003 + image_id)
    mode = rng.choices(list(mode_weights.keys()), weights=list(mode_weights.values()))[0]
    app = make_appearance(rng)
    house = make_house(rng)
    fam_seed = rng.randrange(1 << 30)

    # ---- material schedule
    styles = {}
    S_guess = 60.0
    main_kind = rng.choice(WALL_KINDS)
    styles["main"] = make_style(rng, main_kind, app, fam_seed + 1, S_guess, "main")
    if house["accent"]:
        acc_kind = rng.choice([k for k in WALL_KINDS if k != main_kind])
        base_override = None
        if app["colour"] and rng.random() < SAME_COLOUR_PAIR_PROB:
            base_override = tuple(int(max(0, min(255, v + rng.randint(-14, 14)))) for v in styles["main"].base)
        styles["accent"] = make_style(rng, acc_kind, app, fam_seed + 2, S_guess, "accent",
                                      base_override=base_override)
    if house.get("accent2"):
        used = {main_kind, styles["accent"].kind if "accent" in styles else None}
        styles["accent2"] = make_style(rng, rng.choice([k for k in WALL_KINDS if k not in used]), app,
                                       fam_seed + 12, S_guess, "accent2")
    if house.get("label_trim"):
        tc = rng.choice(TRIM_PALETTE) if app["colour"] else app["trim"]
        styles["trim"] = Style("flat", tc, tc, 1.0, {}, fam_seed + 13, "trim")
        styles["trim"].callout = rng.choice(["TRIM", "PAINTED TRIM", "1x4 TRIM", "TRIM BOARD"])
    roof_kind = rng.choice(ROOF_KINDS)
    styles["roof"] = make_style(rng, roof_kind, app, fam_seed + 3, S_guess, "roof", roof=True)
    styles["chimney"] = make_style(rng, rng.choice(["brick", "stone", "stucco"]), app, fam_seed + 4, S_guess, "chimney")
    # the base band is brick / block / stone / concrete, and 30% of the time it
    # shares the roof's colour: two dark bands at the top and bottom of the wall
    # that only their fills tell apart (eval image 18)
    found_kind = rng.choice(["concrete", "concrete", "brick", "block", "stone"])
    found_base = None
    if app["colour"] and rng.random() < FOUND_ROOF_COLOUR_PROB:
        found_base = tuple(int(max(0, min(255, v + rng.randint(-12, 12)))) for v in styles["roof"].base)
    styles["foundation"] = make_style(rng, found_kind, app, fam_seed + 5, S_guess, "foundation",
                                      base_override=found_base)
    if not app["colour"] and rng.random() < 0.5:
        styles["foundation"].kind = "flat"

    # which families are labelled
    label_fams = {"main"}
    if "accent" in styles and rng.random() < 0.97:
        label_fams.add("accent")
    if "accent2" in styles and rng.random() < 0.97:
        label_fams.add("accent2")
    # a flat fill in mono mode is blank paper with an outline; the eval set never
    # labels those (21, 22), and labelling them teaches "select blank white"
    if not app["colour"]:
        for fam in ("main", "accent", "accent2"):
            if fam in label_fams and styles[fam].kind == "flat":
                label_fams.discard(fam)
        if styles["main"].kind == "flat" and not (label_fams & {"accent", "accent2"}):
            # give the sheet something to select: re-roll the main material
            styles["main"] = make_style(rng, rng.choice([k for k in WALL_KINDS if k != "stucco"]), app,
                                        fam_seed + 1, S_guess, "main")
            label_fams.add("main")
    if rng.random() < 0.9 and styles["roof"].kind != "flat":
        label_fams.add("roof")
    if rng.random() < 0.7:
        label_fams.add("chimney")
    if rng.random() < FOUND_LABEL_PROB and styles["foundation"].kind != "flat":
        label_fams.add("foundation")
    if "trim" in styles:
        label_fams.add("trim")
    hole_mode = rng.random() < WINDOW_HOLE_PROB

    # ---- build the drawings in feet
    drawings = []     # (kind, data, extent(bounds), title)
    if mode == "elevation":
        nviews = rng.choices(list(VIEW_COUNT_WEIGHTS.keys()), weights=list(VIEW_COUNT_WEIGHTS.values()))[0]
        allv = ["front", "rear", "left", "right"]
        if nviews == 1:
            views = ["front"] if rng.random() < 0.6 else [rng.choice(allv)]
        elif nviews == 2:
            views = rng.choice([["front", "rear"], ["left", "right"], ["front", "right"], ["front", "left"]])
        else:
            views = allv
        titles = {"front": "FRONT ELEVATION", "rear": "REAR ELEVATION", "left": "LEFT ELEVATION",
                  "right": "RIGHT ELEVATION"}
        tstyle = rng.choice(["name", "compass", "proposed", "none"])
        comp = {"front": "SOUTH", "rear": "NORTH", "left": "WEST", "right": "EAST"}
        for v in views:
            e = build_elevation(house, v, rng)
            if tstyle == "name":
                t = titles[v]
            elif tstyle == "compass":
                t = f"{comp[v]} ELEVATION"
            elif tstyle == "proposed":
                t = f"PROPOSED {titles[v]}"
            else:
                t = ""
            drawings.append(("elev", (e, v), e["extent"], t))
    elif mode == "roof_plan":
        rp = build_roof_plan(house, rng)
        if rp["flat"] and rng.random() < 0.5:
            styles["flat"] = make_style(rng, "flat_roof", app, fam_seed + 6, S_guess, "flat", roof=True)
            if rng.random() < 0.6:
                label_fams.add("flat")
        if styles["roof"].kind == "flat":
            styles["roof"].kind = "asphalt"
            styles["roof"].params = {"row": 0.5, "unit": 1.0}
        if styles["roof"].kind in ("asphalt", "shingle"):
            styles["roof"].params["joints"] = rng.random() < 0.5
        elif styles["roof"].kind == "seam":
            styles["roof"].params["row"] = styles["roof"].params["sp"]
        elif styles["roof"].kind == "tile_roof":
            styles["roof"].params["row"] = styles["roof"].params.get("row", 1.0)
        label_fams = {"roof"} | ({"flat"} if "flat" in label_fams else set())
        b = rp["outline"].bounds
        drawings.append(("roof", rp, (b[0], b[1], b[2], b[3] + 5), "ROOF PLAN"))
    else:
        fp = build_floor_plan(house, rng)
        # finishes
        finishes = []
        wet = [(i, r) for i, (r, n) in enumerate(zip(fp["rooms"], fp["names"]))
               if any(k in n for k in ("BATH", "KITCHEN", "LAUNDRY", "ENTRY", "MUD"))]
        dry = [(i, r) for i, (r, n) in enumerate(zip(fp["rooms"], fp["names"]))
               if not any(k in n for k in ("BATH", "KITCHEN", "LAUNDRY", "ENTRY", "MUD"))]
        slab_plan = rng.random() < 0.2
        if slab_plan:
            styles["f1"] = make_style(rng, rng.choice(["concrete", "flat", "hatch"]), app, fam_seed + 7, S_guess, "f1")
            if styles["f1"].kind == "flat" and not app["colour"]:
                styles["f1"].kind = "concrete"
                styles["f1"].params = {"density": 1.5}
            finishes.append(("f1", polys_of(unary_union(fp["room_polys"]))))
        else:
            if wet and rng.random() < 0.8:
                styles["f1"] = make_style(rng, rng.choice(["grid", "grid", "cross"]), app, fam_seed + 7, S_guess, "f1")
                k = rng.randint(1, len(wet))
                finishes.append(("f1", [fp["room_polys"][i] for i, _ in rng.sample(wet, k)]))
            if dry and rng.random() < 0.75:
                styles["f2"] = make_style(rng, rng.choice(["plank", "plank", "stipple", "grid"]), app, fam_seed + 8,
                                          S_guess, "f2")
                k = rng.randint(1, min(3, len(dry)))
                finishes.append(("f2", [fp["room_polys"][i] for i, _ in rng.sample(dry, k)]))
        if fp["patio"] is not None and rng.random() < 0.8:
            styles["patio"] = make_style(rng, rng.choice(["stone", "grid", "plank", "concrete"]), app, fam_seed + 9,
                                         S_guess, "patio")
            finishes.append(("patio", [fp["patio"]]))
        if fp["garage"] is not None and rng.random() < 0.4:
            styles["garage"] = make_style(rng, "concrete", app, fam_seed + 10, S_guess, "garage")
            finishes.append(("garage", [fp["garage"]]))
        poche = rng.random() < 0.6
        styles["walls"] = make_style(rng, "solid", app, fam_seed + 11, S_guess, "walls")
        if app["colour"] and rng.random() < 0.5:
            styles["walls"].base = rng.choice([(40, 40, 40), (80, 80, 80), (120, 120, 120), (60, 60, 70)])
        label = set()
        for fam, _ in finishes:
            if rng.random() < 0.85:
                label.add(fam)
        if poche and rng.random() < 0.5:
            label.add("walls")
        if not label:
            label.add(finishes[0][0] if finishes else "walls")
            if not finishes:
                poche = True
        plan_cfg = {"finishes": finishes, "label": label, "poche": poche, "mep": rng.random() < 0.15,
                    "room_tint": ((rng.randint(222, 240),) * 3 if rng.random() < 0.35 else None)}
        label_fams = label
        b = fp["footprint"].bounds
        if fp["patio"] is not None:
            pb = fp["patio"].bounds
            b = (min(b[0], pb[0]), min(b[1], pb[1]), max(b[2], pb[2]), max(b[3], pb[3]))
        if fp["garage"] is not None:
            gb = fp["garage"].bounds
            b = (min(b[0], gb[0]), min(b[1], gb[1]), max(b[2], gb[2]), max(b[3], gb[3]))
        drawings.append(("plan", (fp, plan_cfg), (b[0] - 1, b[1] - 6, b[2] + 1, b[3] + 5), "FLOOR PLAN"))

    # ---- layout in feet
    gap = rng.uniform(5, 10)
    margin = rng.uniform(2.5, 7)
    side_room = rng.uniform(3.5, 6.0)          # level marks / leaders
    placements = []
    if len(drawings) <= 2 or (len(drawings) == 3):
        x = margin
        row_h = 0
        for (kind, data, ext, title) in drawings:
            w = (ext[2] - ext[0]) + side_room * 2
            h = (ext[3] - ext[1]) + 6
            placements.append((x - ext[0] + side_room, -ext[1] + margin + 3))
            x += w + gap
            row_h = max(row_h, h)
        total_w, total_h = x - gap + margin, row_h + 2 * margin
    else:
        # 2x2
        ws = [(e[2] - e[0]) + side_room * 2 for (_, _, e, _) in drawings]
        hs = [(e[3] - e[1]) + 6 for (_, _, e, _) in drawings]
        cw = [max(ws[0], ws[2]), max(ws[1], ws[3])]
        rh = [max(hs[0], hs[1]), max(hs[2], hs[3])]
        for i, (kind, data, ext, title) in enumerate(drawings):
            col, row = i % 2, i // 2
            x = margin + (cw[0] + gap if col else 0)
            y = margin + (rh[0] + gap if row else 0)
            placements.append((x - ext[0] + side_room, y - ext[1] + 3))
        total_w = margin * 2 + cw[0] + cw[1] + gap
        total_h = margin * 2 + rh[0] + rh[1] + gap
    long_px = rng.uniform(*LONG_SIDE_PX)
    S = long_px / max(total_w, total_h)
    S = min(S, 130.0)
    W, H = int(total_w * S), int(total_h * S)
    W, H = max(W, 800), max(H, 500)
    # re-scale line weights and text now S and W are known: a PDF export keeps
    # line weight and lettering roughly proportional to the sheet, not the house
    long_side = max(W, H)
    app["outline_lw"] = min(max(1.2, S * 0.022), long_side * rng.uniform(0.0006, 0.0011))
    app["text_px"] = min(max(9.0, S * 0.25), long_side * rng.uniform(0.007, 0.012))
    for st in styles.values():
        st.lw = min(max(1.0, S * 0.016), app["outline_lw"] * rng.uniform(0.5, 0.9))
        if st.kind == "brick" and S < 45:
            st.params["row"] = 0.333

    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = bgr(app["paper"])
    if rng.random() < GRAPH_PAPER_PROB:
        g = (rng.randint(222, 238),) * 3
        step = rng.choice([1.0, 2.0]) * S
        _hlines(canvas, 0, 0, 1.0, step, g, 1)
        _vlines(canvas, 0, 0, 1.0, step, g, 1)
    tq = TextQueue()
    labelled = []       # (family, poly_px)
    for (kind, data, ext, title), (ox, oy) in zip(drawings, placements):
        V = View(S, ox * S, oy * S)
        if kind == "elev":
            e, vname = data
            labelled += render_elevation_view(canvas, tq, V, e, styles, house, app, rng, S, W, H, label_fams,
                                              hole_mode, vname, title)
        elif kind == "roof":
            labelled += render_roof_plan(canvas, tq, V, data, styles, app, rng, S, W, H,
                                         "one" if rng.random() < 0.6 else "block")
        else:
            fp, plan_cfg = data
            labelled += render_floor_plan(canvas, tq, V, fp, styles, app, rng, S, W, H, plan_cfg)
    # notes block
    if rng.random() < 0.2:
        x = W - rng.uniform(0.12, 0.2) * W
        y = rng.uniform(0.05, 0.3) * H
        size = app["text_px"]
        tq.add((x, y), "NOTES:", size, app["ink"], bold=True)
        notes = ["1. ALL WORK PER LOCAL CODE.", "2. VERIFY ALL DIMENSIONS IN FIELD.",
                 "3. FINISHES AS SELECTED BY OWNER.", "4. PROVIDE FLASHING AT ALL OPENINGS.",
                 "5. SEE STRUCTURAL FOR FRAMING."]
        for i, n in enumerate(notes[:rng.randint(2, 5)]):
            tq.add((x, y + size * 1.5 * (i + 1)), n, size * 0.85, app["ink"])
    # markup
    if labelled and rng.random() < HIGHLIGHT_PROB:
        fam, p = rng.choice(labelled)
        x0, y0, x1, y1 = p.bounds
        r = rect(x0 - 0.02 * S, y0 - 0.02 * S, x1 + 0.02 * S, y1 + 0.02 * S)
        layer = canvas.copy()
        cv_fill(layer, r, rng.choice(HIGHLIGHT_COLOURS))
        m = poly_mask(r, W, H)
        a = (m.astype(np.float32) / 255.0 * 0.35)[..., None]
        canvas[:] = (canvas * (1 - a) + layer * a).astype(np.uint8)
    canvas = tq.flush(canvas)
    if rng.random() < DOT_MARKUP_PROB:
        for _ in range(rng.randint(2, 6)):
            c = (rng.uniform(0.1, 0.9) * W, rng.uniform(0.2, 0.8) * H)
            cv2.circle(canvas, _pt(c), int(rng.uniform(0.35, 0.6) * S * SCALE), bgr((235, 60, 220)), -1,
                       lineType=cv2.LINE_AA, shift=SHIFT)
    if rng.random() < CLOUD_PROB and labelled:
        fam, p = rng.choice(labelled)
        x0, y0, x1, y1 = p.bounds
        col = (220, 40, 40)
        pts = list(rect(x0 - S, y0 - S, x1 + S, y1 + S).exterior.coords)
        for a, b in zip(pts[:-1], pts[1:]):
            L = math.dist(a, b)
            n = max(1, int(L / (0.9 * S)))
            for k in range(n):
                t0, t1 = k / n, (k + 1) / n
                p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                c = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                cv2.ellipse(canvas, _pt(c), (int(math.dist(p0, p1) / 2 * SCALE), int(0.35 * S * SCALE)), ang, 180, 360,
                            bgr(col), max(1, int(S * 0.02)), lineType=cv2.LINE_AA, shift=SHIFT)
    # excerpt crop
    crop = None
    if rng.random() < EXCERPT_CROP_PROB and labelled:
        cx0 = int(rng.uniform(0, 0.25) * W) if rng.random() < 0.5 else 0
        cy0 = int(rng.uniform(0, 0.25) * H) if rng.random() < 0.5 else 0
        cx1 = W - (int(rng.uniform(0, 0.25) * W) if rng.random() < 0.5 else 0)
        cy1 = H - (int(rng.uniform(0, 0.25) * H) if rng.random() < 0.5 else 0)
        if cx1 - cx0 > 0.5 * W and cy1 - cy0 > 0.5 * H and (cx0 or cy0 or cx1 < W or cy1 < H):
            crop = (cx0, cy0, cx1, cy1)
    if crop:
        cx0, cy0, cx1, cy1 = crop
        canvas = np.ascontiguousarray(canvas[cy0:cy1, cx0:cx1])
        W, H = cx1 - cx0, cy1 - cy0
        cb = box(cx0, cy0, cx1, cy1)
        labelled = [(f, affinity.translate(p.buffer(0).intersection(cb), -cx0, -cy0)) for (f, p) in labelled]
        labelled = [(f, q) for (f, p) in labelled for q in polys_of(p)]
    # soft raster / jpeg
    if rng.random() < SOFT_RASTER_PROB:
        f = rng.uniform(0.6, 0.85)
        small = cv2.resize(canvas, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < JPEG_PROB:
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 90)])
        if ok:
            canvas = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ---- annotations
    fam_ids = {}
    anns = []
    min_area = (0.02 * S) ** 2 * 100      # ~ (2 ft)^2
    for fam, p in labelled:
        if p.is_empty or p.area < max(400, min_area):
            continue
        if fam not in fam_ids:
            fam_ids[fam] = len(fam_ids) + 1
        p = p.simplify(0.5)
        if p.is_empty or not isinstance(p, Polygon):
            for q in polys_of(p):
                labelled.append((fam, q))
            continue
        outer = [round(v, 2) for c in p.exterior.coords[:-1] for v in c]
        holes = [[round(v, 2) for c in ring.coords[:-1] for v in c] for ring in p.interiors]
        holes = [h for h in holes if len(h) >= 6]
        x0, y0, x1, y1 = p.bounds
        anns.append({"id": len(anns) + 1, "category_id": fam_ids[fam],
                     "category_name": f"pattern{fam_ids[fam]}", "family": fam,
                     "segmentation": [outer] + holes, "num_holes": len(holes),
                     "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                     "area": round(p.area, 1)})
    return canvas, {"image": {"file_name": f"synth6_{image_id:06d}.png", "width": W, "height": H},
                    "mode": mode, "appearance": "colour" if app["colour"] else f"mono_{app['level']}",
                    "px_per_ft": round(S, 2), "annotations": anns}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
_CFG = {}


def _init(out, seed, mode_weights):
    _CFG.update({"out": out, "seed": seed, "mw": mode_weights})


def _job(image_id):
    try:
        canvas, ann = compose(image_id, _CFG["seed"], _CFG["mw"])
    except Exception as exc:          # a bad draw must not kill the batch
        import traceback
        traceback.print_exc()
        return (image_id, False, str(exc))
    if not ann["annotations"]:
        return (image_id, False, "no annotations")
    cv2.imwrite(os.path.join(_CFG["out"], "images", ann["image"]["file_name"]), canvas,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    with open(os.path.join(_CFG["out"], "annotations", f"synth6_{image_id:06d}.json"), "w") as f:
        json.dump(ann, f)
    return (image_id, True, ann["mode"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--start", type=int, default=0, help="first image id")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--mode-weights", default=None, help="elevation,roof_plan,freeform e.g. 60,20,20")
    args = ap.parse_args()
    mw = dict(MODE_WEIGHTS)
    if args.mode_weights:
        e, r, f = [float(v) for v in args.mode_weights.split(",")]
        mw = {"elevation": e, "roof_plan": r, "freeform": f}
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    ids = list(range(args.start, args.start + args.n))
    t0 = time.time()
    ok = 0
    modes = {}
    with Pool(args.workers, initializer=_init, initargs=(args.out, args.seed, mw)) as pool:
        for i, (iid, good, info) in enumerate(pool.imap_unordered(_job, ids, chunksize=2)):
            if good:
                ok += 1
                modes[info] = modes.get(info, 0) + 1
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                el = time.time() - t0
                print(f"{i + 1}/{len(ids)} ok={ok} {el:.0f}s ({el / (i + 1):.2f}s/img) {modes}", flush=True)
    with open(os.path.join(args.out, "generation_manifest.json"), "w") as f:
        json.dump({"generator": "generate_synthetic_v6.py", "n": args.n, "seed": args.seed, "start": args.start,
                   "ok": ok, "modes": modes, "mode_weights": mw}, f, indent=2)


if __name__ == "__main__":
    main()
