"""FreeCAD-geometry pilot: v6 sheets whose elevations come from a fused 3D model.

Everything a sheet looks like -- materials and hatching, colour, ink, text,
annotation, labels, crop -- is generate_synthetic_v6.py with the given flags.
Only elevation geometry changes: each v6 house (same blocks, pitches,
overhangs, chimney) is built as 3D solids in FreeCAD (scripts/fc_massing.py),
fused, and every view is the set of faces actually visible from that side
(planar depth test between overlapping faces). Wings and garages now meet the
main roof properly, hip ends foreshorten correctly, rakes and fascias are the
real roof slab edges, and side views show the roofs of blocks behind.
Roof plans and floor plans are v6's own.

  python scripts/generate_synthetic_fc.py --out data/synthetic/fc_probe300 --n 300 --start 200000 --r8
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_synthetic_v6 as G  # noqa: E402

FREECADCMD = os.environ.get("FREECADCMD", "/opt/fc/bin/freecadcmd")
HERE = os.path.dirname(os.path.abspath(__file__))
R8 = dict(val_details=True, material_mix=True, val_fills=True, res_degrade=0.15, neutral_palette=True,
          fill_scale=True, real_labelling=True)
TOWARD = {"front": (0, -1), "rear": (0, 1), "left": (-1, 0), "right": (1, 0)}   # plan dir to the viewer

_FACES = {}
_CUR = [None]


# ----------------------------------------------------------------------------
# houses: reproduce compose()'s rng prefix so the FreeCAD model is the house
# compose() will draw
# ----------------------------------------------------------------------------
# --cross-gables P (default 0.4 since 2026-10-20a): a front cross gable -- a gabled bay projecting from the front wall with
# its ridge perpendicular to the main ridge (one 65%, a symmetric pair 35%; steep "Tudor" pitch 40%) -- on houses whose main
# roof is a side-gable or hip. Gemini round 5: cross gables / steep Tudor on 21, several front gables on 19, hip + gable
# mixes on 20 of 111 elevations, none drawable before. The bay reaches back to the main ridge so the two roofs meet in the
# fused solid; it is block kind "xgable" (own walls, own gable triangle: a gable accent lands in it). Own RNG stream seeded
# from the house itself, so every caller (massing, elevations, plans) sees the same house; P = 0 = earlier pools.
CROSS_GABLES = [0.4]
# --base-bands (default 1 since 2026-10-21a): the generator drew a foundation band on 65% of houses and a wainscot line
# (accent or accent2) on ~40%; Gemini's elevations show them on 25% and 20% (150 described, r4 + r5), and the user called
# the line across the bottom third overused. 1 = keep a drawn foundation with p 0.38 and a wainscot accent with p 0.5
# (own stream seeded from the house); 0 = earlier pools.
BASE_BANDS = [1.0]
# --spec: image id -> house keys forced on the drawn house instead of searched for -- the main block's stories and roof
# (3+ storeys come only with townhouse rows, flat roofs on 8% of houses: a search rarely meets them with the other keys)
# and the cross-gable count (on an eligible house, 0 = none). Own seed from the house; empty = no change.
HOUSE_FORCE = {}
FORCED_KEYS = ("stories", "roof", "cross_gables")


def _force_of(spec):
    h = spec.get("house", {})
    f = {}
    if isinstance(h.get("stories"), int) and 1 <= h["stories"] <= 4:
        f["stories"] = h["stories"]
    if h.get("roof") in ("gable", "hip", "flat", "shed"):
        f["roof"] = h["roof"]
    if h.get("cross_gables") is not None:
        f["cross_gables"] = min(int(h["cross_gables"]), 2)
    if spec.get("facade_aspect"):
        f["aspect"] = float(spec["facade_aspect"])
    if spec.get("bay") is not None:
        f["bay"] = bool(spec["bay"])
    if spec.get("apartment"):
        f["apartment"] = True
        f["balconies"] = bool(spec.get("balconies"))
    if spec.get("garage") == "integrated":
        f["garage_int"] = max(1, min(3, int(spec.get("garage_doors") or 1)))
    for k in ("recessed_balconies", "sections", "recessed_entry", "upper_setback"):
        if spec.get(k):
            f[k] = True
    return f


def _apply_force(house, f):
    m = house["blocks"][0]
    fr = random.Random(int(house["fh"] * 1e9) * 5 + 3)
    if "stories" in f:
        m.stories = f["stories"]
    if "roof" in f and f["roof"] != m.roof:
        m.roof = f["roof"]
        m.pitch = {"gable": fr.uniform(0.3, 0.6), "hip": fr.uniform(0.25, 0.5), "shed": fr.uniform(0.15, 0.35),
                   "flat": 0.0}[m.roof]
        if m.roof == "flat":
            house["dormers"] = False
    if "aspect" in f and (not house.get("units") or f.get("apartment")):   # (an apartment does not use the unit template)
        # front-view width / height: stretch the plan along x (every block, so attachments stay attached), x0.35-2.2
        k = max(0.35, min(2.2, f["aspect"] / facade_aspect(house)))
        for b in house["blocks"]:
            b.x0, b.w = b.x0 * k, b.w * k
    if f.get("sections") and m.roof == "flat":
        # projecting sections: 1-3 flat-roofed bays, 2-4 ft proud of the main facade, full height, as Gemini articulates
        # apartment fronts; the accent "block" material lands on them (v6: a non-main block)
        n = fr.choice([1, 2, 2, 3])
        sw = fr.uniform(0.14, 0.26) * m.w
        p = fr.uniform(2.0, 4.0)
        xs = [m.x0 + m.w / 2] if n == 1 else [m.x0 + 0.5 * sw + 1, m.x1 - 0.5 * sw - 1] if n == 2 else \
            [m.x0 + 0.5 * sw + 1, m.x0 + m.w / 2, m.x1 - 0.5 * sw - 1]
        for c in xs:
            house["blocks"].append(G.Block(c - sw / 2, m.y0 - p, sw, p + 1.0, m.stories, "flat", "x", 0.0, m.ov, "section"))


def _cut_plan(house, image_id):
    """--spec recesses as boxes cut from the massing (fc_massing "cuts"), plan feet, z up from grade: recessed balconies on
    the upper storeys at the apartment window columns, a recessed entry around the front door, the top storey set back
    from the front. Deterministic per sheet (own stream), so the massing and the elevations agree."""
    f = HOUSE_FORCE.get(image_id, {}) if HOUSE_FORCE else {}
    if not any(f.get(k) for k in ("recessed_balconies", "recessed_entry", "upper_setback")):
        return []
    m, fh = house["blocks"][0], house["fh"]
    cr = random.Random(image_id * 883 + 41)
    front = min(b.y0 for b in house["blocks"]) - 1.0
    out = []
    if f.get("recessed_balconies") and m.stories >= 2:
        allc = _apt_cols(house, "front", image_id)
        cols = allc
        pick = cr.choice(["all", "alt", "ends"])
        cols = cols if pick == "all" or len(cols) < 3 else cols[cr.randrange(2)::2] if pick == "alt" else [cols[0], cols[-1]]
        w = min(cr.uniform(7, 10), 0.9 * m.w / max(1, len(allc)))
        dep = cr.uniform(4, 6)
        for st in range(1, m.stories):
            for cu in cols:
                out.append({"kind": "balcony", "st": st, "x0": cu - w / 2, "y0": front, "z0": st * fh,
                            "w": w, "d": m.y0 + dep - front, "h": fh - 1.0})
    if f.get("recessed_entry"):
        w, dep = cr.uniform(6, 9), cr.uniform(3, 5)
        cu = m.x0 + min(max(house["front_door_u"], 0.15), 0.85) * m.w
        if f.get("apartment"):
            cu = m.x0 + m.w / 2
        out.append({"kind": "entry", "st": 0, "x0": cu - w / 2, "y0": front, "z0": 0.0, "w": w,
                    "d": m.y0 + dep - front, "h": min(fh - 1.0, 8.5)})
    if f.get("upper_setback") and m.stories >= 2 and m.roof == "flat":
        dep = min(cr.uniform(5, 9), 0.35 * m.d)
        out.append({"kind": "setback", "st": m.stories - 1, "x0": m.x0 - 5, "y0": front, "z0": (m.stories - 1) * fh + 0.05,
                    "w": m.w + 10, "d": m.y0 + dep - front, "h": 60.0})
    return out


def _cross_gables(make_house):
    def wrapped(rng):
        house = make_house(rng)
        f = HOUSE_FORCE.get(_CUR[0]) if HOUSE_FORCE else None
        if f:
            _apply_force(house, f)
        if BASE_BANDS[0] > 0:
            br = random.Random(int(house["fh"] * 1e9) * 13 + int(house["plate"] * 1e9) + 29)
            if house["foundation"] > 0 and br.random() >= 0.38:
                house["foundation"] = 0.0
            if br.random() >= 0.5:
                if house["accent"] == "wainscot":
                    house["accent"], house["accent2"] = house["accent2"], None
                if house["accent2"] == "wainscot":
                    house["accent2"] = None
        if CROSS_GABLES[0] <= 0:
            return house
        m = house["blocks"][0]
        cr = random.Random(int(house["fh"] * 1e9) * 7 + int(m.w * 1e9) * 3 + 11)
        force = f.get("cross_gables") if f else None
        if force == 0:
            return house
        if (cr.random() >= CROSS_GABLES[0] and force is None) or house.get("units") or m.ridge != "x" or m.roof not in ("gable", "hip") \
                or m.w < 24:
            return house
        main_rise = m.pitch * m.d / 2
        steep = cr.random() < 0.4
        if (cr.random() < 0.65 if force is None else force == 1):
            cw = cr.uniform(0.22, 0.4) * m.w
            centres = [m.x0 + cr.uniform(0.25, 0.75) * m.w]
        else:
            cw = cr.uniform(0.18, 0.28) * m.w
            off = cr.uniform(0.2, 0.3) * m.w
            centres = [m.x0 + m.w / 2 - off, m.x0 + m.w / 2 + off]
        p = cr.uniform(1.0, 7.0)
        pitch = cr.uniform(0.9, 1.4) if steep else m.pitch * cr.uniform(1.0, 1.3)
        pitch = min(pitch, (main_rise + 2.0) / (cw / 2))
        for c in centres:
            x0 = max(m.x0 + 0.5, min(c - cw / 2, m.x1 - 0.5 - cw))
            house["blocks"].append(G.Block(x0, m.y0 - p, cw, p + m.d / 2, m.stories, "gable", "y", pitch, m.ov, "xgable"))
        house["cross_gables"] = len(centres)
        return house
    wrapped._cross = True
    return wrapped


if not getattr(G.make_house, "_cross", False):
    G.make_house = _cross_gables(G.make_house)

# --wall-splits P (default 0.3 since 2026-10-20b): a vertical band of the main block's front / rear wall in a second material
# from grade to eave -- an end bay (50%) or an interior entry bay, 20-45% of the wall wide, a trim board at each edge,
# mirrored on the rear view -- as the "accent2" family (only when accent2 is free). Gemini round 5: a material change
# mid-wall / vertical split on 16 of 111 elevations; the generator only zoned materials in horizontal bands. Own RNG seeded
# from the house; P = 0 = earlier pools.
WALL_SPLITS = [0.3]


def _wall_splits(make_house):
    def wrapped(rng):
        house = make_house(rng)
        if WALL_SPLITS[0] <= 0:
            return house
        m = house["blocks"][0]
        sr = random.Random(int(house["fh"] * 1e9) * 13 + int(m.d * 1e9) * 5 + 23)
        if sr.random() >= WALL_SPLITS[0] or house.get("accent2") or m.w < 20:
            return house
        frac = sr.uniform(0.2, 0.45)
        if sr.random() < 0.5:
            fa, fb = (0.0, frac) if sr.random() < 0.5 else (1.0 - frac, 1.0)
        else:
            fa = sr.uniform(0.1, 0.9 - frac)
            fb = fa + frac
        house["accent2"], house["split"] = "split", (fa, fb)
        return house
    wrapped._split = True
    return wrapped


if not getattr(G.make_house, "_split", False):
    G.make_house = _wall_splits(G.make_house)

# --panel-siding P (default 0.12 since 2026-10-20c): a wall family (main / accent / accent2) is drawn as fibre-cement panels
# with reveal joints instead of its drawn kind -- panels 2-4 ft wide, 4-10 ft tall, aligned (65%) or staggered half a panel --
# decided per family from its own style seed. Gemini round 5: fibre-cement panels with reveal grids on ~13-17 of 111
# elevations, and nothing in WALL_KINDS draws a two-way rectangular grid at panel scale. P = 0 = earlier pools.
PANEL_SIDING = [0.12]
GEMINI_LOOK = [0.0]     # --gemini-look P (2026-10-20e): see revit_render.compose_revit
WALL_ROLES = ("main", "accent", "accent2")


def _panel_siding(make_style):
    def wrapped(rng, kind, appearance, seed, S, label_name, roof=False, base_override=None):
        st = make_style(rng, kind, appearance, seed, S, label_name, roof=roof, base_override=base_override)
        if PANEL_SIDING[0] <= 0 or roof or label_name not in WALL_ROLES:
            return st
        pr = random.Random(seed * 211 + 97)
        if pr.random() >= PANEL_SIDING[0]:
            return st
        return _to_panel(st, pr)
    wrapped._panel = True
    return wrapped


def _to_panel(st, pr):
    """A wall style redrawn as fibre-cement panels (pw x ph ft, reveals aligned or staggered)."""
    st.kind = "panel"
    pw = pr.choice([2.0, 3.0, 4.0, 4.0])
    # "sp" kept (= panel width) for code that adjusts a wall's spacing (roof-lines, pairs); drawing uses pw / ph
    st.params = {"pw": pw, "ph": pr.choice([4.0, 6.0, 8.0, 9.0, 10.0]), "stagger": pr.random() < 0.35, "sp": pw}
    st.callout = pr.choice(["FIBRE-CEMENT PANEL", "FIBER CEMENT PANEL SIDING", "PANEL SIDING W/ REVEALS",
                            "FIBRE-CEMENT PANEL (A)"])
    return st


def house_for(image_id, seed, mw):
    _CUR[0] = image_id                         # --spec forcing (HOUSE_FORCE) is keyed by the id
    rng = random.Random(seed * 1_000_003 + image_id)
    rng.choices(list(mw.keys()), weights=list(mw.values()))
    G.make_appearance(rng)
    return G.make_house(rng)


# Bump GEN_VERSION whenever a DEFAULT changes what a pool contains (images or labels). It is
# written to generation_manifest.json; the run scripts (scripts/pool_guard.sh) refuse to reuse a
# pool built by an older version instead of silently training on it.
GEN_VERSION = "2026-10-21a"      # 21a: --base-bands 1 (0 = 20d); 20a: --cross-gables 0.4 (0 = 2026-10-17b); 20b: --wall-splits 0.3 (0 = 20a); 20c: --panel-siding 0.12 (0 = 20b);
                                 # 20d: --pattern-drift 0.3 (0 = 20c)
OPENINGS_PER_FACE = [True]      # --openings-per-face (default 1), set in _init
# --cut-standing (default 0 since 2026-10-04): posts standing in front of a wall (porch posts,
# railing and deck posts, newel posts) are drawn as before but no longer cut out of the wall
# label -- the wall continues behind them, as every hand-labelled sheet draws it (real 02, 27;
# Gemini). Flat trim (belts, fascia, rim board, stair stringer) is still cut. 1 = earlier pools.
CUT_STANDING = [False]
# --downspouts P (2026-10-04, default 0.05): probability a wall end gets a downspout from the
# eave to grade, drawn over the siding with the wall label running behind it, as on the real
# rendered elevations (02, 03, 05). Own RNG per view; 0 = earlier pools.
DOWNSPOUTS = [0.05]
# --ref-exclude (default 1, 2026-10-04): each annotation file carries "ref_exclude", pixel
# polygons of things standing in front of a labelled surface with no pattern of their own --
# posts, railing panels, exterior stairs, downspouts, solid furniture and rugs. They stay inside
# the labels (the material continues behind them); training and mining draw REFERENCE boxes
# from the label minus these, so a reference is never a crop of a post or a bed. 0 = omit.
REF_EXCLUDE = [True]
# --near-pairs P (default 0.6 since 2026-10-14): a colour elevation whose house has a side wing gets, with
# probability P, the wing as a second family in the main wall's line pattern but another hue (CIEDE2000 10-25),
# or another hue plus spacing 1.4-2x (~14% of elevations); see revit_render.plant_near_pair. 0 = no pairs.
# --near-pairs-plan P (default 0.3 since 2026-10-15; 0 on 2026-10-14..14f; 0.6 on 2026-10-07..13): finish floor
# plans, two rooms' floors differing in one attribute, the attributes in --near-pairs-plan-kinds (default "direction"
# since 2026-10-15: orthogonal lines vs 45-degree hatch, same spacing). 2026-10-14 all three kinds were turned off: they
# taught "another room = another family" (Gemini30 5 drops a garage). Back for direction only because --turn-boards
# taught the opposite with no counter-example and _distinct never lets direction separate two families, so a
# diagonal hatch next to orthogonal lines at the same spacing was always one family in synthetic data: the GEN 14f
# model merged them on Gemini30 4 (q4 0.95 -> 0.72, q2 0.41 -> 0.32). Turned boards turn 0 <-> 90 (lines) or
# 45 <-> -45 (hatch), never orthogonal <-> diagonal, so the two rules do not contradict.
# Old pools: --near-pairs-plan 0.6 --near-pairs-plan-kinds direction,spacing,tone (2026-10-07..13), 0 (14..14f).
# The 2026-10-07 / 2026-10-13 pair pools (direction / spacing / tone, B on the roof) are not rebuildable from
# here: check out the generator at a73ab1f (2026-10-07) / eb67479 (2026-10-13).
NEAR_PAIRS = [0.0]      # default 0 since 2026-10-14c: net zero in the best recipe (synth_progress "wing pairs off")
NEAR_PAIRS_PLAN = [0.3]
NEAR_PAIRS_PLAN_KINDS = [("direction",)]
# --balcony-doors P (default 0.7 since 2026-10-07): an upper-floor balcony railing gets a door in the
# wall behind it (cut out of the wall label like every door; the railing still draws over it), as on
# Gemini r4 #098. Before, doors existed only at grade, so every synthetic railing stood in front of
# siding. Needs --railings > 0. 0 = earlier pools.
BALCONY_DOORS = [0.7]
# --gable-trim P (default 0.5 since 2026-10-14b): a house whose gables are NOT an accent family gets a trim board
# across the base of every gable (frieze / gable-end band), the same family continuing above it, as on Gemini r2 #004.
# Before, the generator drew that band ONLY under accent gables, so in synthetic data a trim board under a gable
# always started a new family; the wing long run then split one-family houses at it (Gemini30 1: 0.85 -> 0.23 with
# the reference in the gable). Own RNG per house (all views agree). 0 = earlier pools.
GABLE_TRIM = [0.5]
# --turn-boards P (default 0.9 since 2026-10-15b; 0.6 on 2026-10-14d..15): on floor plans, a directional floor family (planks, plank lines,
# joists, 45-degree hatch) is laid per room in a random direction on a sheet chosen with P; see
# revit_plans.compose_floor. 0 = earlier pools.
TURN_BOARDS = [0.9]     # 0.6 on 2026-10-14d..15
# --pair-turns 0/1 (default 1 since 2026-10-15b): a plan with a direction pair always turns its boards (when
# --turn-boards > 0), so "orthogonal vs diagonal = two families" and "0 <-> 90 = one family" share a sheet.
# 0 = the 2026-10-15 sheets (the pairs then cost HF14 12 0.89 -> 0.81).
PAIR_TURNS = [True]
# --heavy-pieces P (default 0.4 since 2026-10-14e): on an elevation sheet chosen with P, some surface pieces (thin bands
# more often) are boxed in a heavy line; see revit_render.compose_revit. 0 = earlier pools.
HEAVY_PIECES = [0.4]
# --gable-pairs P (default 0 since 2026-10-17b; 0.4 on 2026-10-17 -- single18kg: Gemini30 26 +0.11 but 1 -0.32, val 9 -0.06): a colour elevation whose house has a gable (else a second storey) gets,
# with P, the gable / upper storey in the main wall's style at another chroma (CIEDE2000 6-12, two families: Gemini30 26)
# or, half the time, a within-family drift (CIEDE2000 0.4-2, one family); see revit_render.gable_pair_planned. 0 = earlier pools.
GABLE_PAIRS = [0.0]
SHAPED = [False]      # --shaped: notched / U / chamfered / bayed houses (3D), set in main()


def _rect_block(b, x0, y0, x1, y1, parent):
    nb = dict(b, x0=x0, y0=y0, w=x1 - x0, d=y1 - y0, parent=parent)
    nb.pop("poly", None)
    if b["roof"] in ("gable", "hip"):
        nb["ridge"] = "x" if (x1 - x0) >= (y1 - y0) else "y"
    return nb


def _poly_block(b, pts, sloped, parent, **kw):
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    q = orient(Polygon(pts), 1.0)
    cc = [tuple(round(v, 4) for v in p) for p in list(q.exterior.coords)[:-1]]
    x0, y0, x1, y1 = q.bounds
    nb = dict(b, x0=x0, y0=y0, w=x1 - x0, d=y1 - y0, parent=parent, poly=cc, **kw)
    n = len(cc)
    nb["sloped"] = [i for i in range(n) if sloped(cc[i], cc[(i + 1) % n])]
    return nb


def shape_blocks(blocks, house, r, bay=None):
    """--shaped: turn the main block into an L (notch), a U (rear courtyard) or a chamfered
    polygon, and add a 45-degree bay. Pieces keep `parent` (the v6 block they came from)
    so elevation zoning still uses the right Block."""
    from shapely.geometry import box as sbox
    main, rest = blocks[0], blocks[1:]
    x0, y0, x1, y1 = main["x0"], main["y0"], main["x0"] + main["w"], main["y0"] + main["d"]
    others = [sbox(b["x0"], b["y0"], b["x0"] + b["w"], b["y0"] + b["d"]).buffer(0.5) for b in rest]

    def free(g):
        return not any(g.intersects(o) for o in others)
    pieces = [dict(main, parent=0)]
    op = r.random()
    w, d = x1 - x0, y1 - y0
    if op < 0.30 and w >= 28 and d >= 24:                         # L: notch a corner
        u, v = r.uniform(0.25, 0.42) * w, r.uniform(0.3, 0.48) * d
        opts = []
        for right in (True, False):
            for front in (True, False):
                nx = (x1 - u, x1) if right else (x0, x0 + u)
                ny = (y0, y0 + v) if front else (y1 - v, y1)
                if free(sbox(nx[0], ny[0], nx[1], ny[1])):
                    opts.append((right, front, nx, ny))
        if opts:
            right, front, nx, ny = r.choice(opts)
            A = (x0, y0, x1 - u, y1) if right else (x0 + u, y0, x1, y1)
            B = (nx[0], y0 + v, nx[1], y1) if front else (nx[0], y0, nx[1], y1 - v)
            pieces = [_rect_block(main, *A, 0), _rect_block(main, *B, 0)]
    elif op < 0.42 and w >= 34 and d >= 26:                       # U: courtyard into the rear
        cw, cd = r.uniform(0.3, 0.45) * w, r.uniform(0.35, 0.5) * d
        cx = x0 + (w - cw) / 2 + r.uniform(-0.1, 0.1) * w
        if free(sbox(cx, y1 - cd, cx + cw, y1)):
            pieces = [_rect_block(main, x0, y0, x1, y1 - cd, 0), _rect_block(main, x0, y1 - cd, cx, y1, 0),
                      _rect_block(main, cx + cw, y1 - cd, x1, y1, 0)]
    elif op < 0.65 and main["roof"] in ("hip", "gable", "flat") and not any(b.get("xgable") for b in rest):
        # chamfered corners (not under a cross gable: the slope-plane roof will not fuse with the bay's, 950005)
        corners = [(x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)]
        cut = [c for c in corners if free(sbox(min(c[0], c[0] + 6 * c[2]), min(c[1], c[1] + 6 * c[3]),
                                                max(c[0], c[0] + 6 * c[2]), max(c[1], c[1] + 6 * c[3])))]
        r.shuffle(cut)
        cut = cut[:r.choice([1, 1, 2])]
        pts = []
        for (cx, cy, sx, sy) in corners:
            if (cx, cy, sx, sy) in cut:
                k = r.uniform(3, 6)
                a = (cx, cy + sy * k) if (sx * sy) > 0 else (cx + sx * k, cy)
                b = (cx + sx * k, cy) if (sx * sy) > 0 else (cx, cy + sy * k)
                pts += [a, b]
            else:
                pts.append((cx, cy))
        if cut:
            ridge_x = main["ridge"] == "x"

            def sl(p, q):
                if main["roof"] == "hip":
                    return True
                dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
                if dx > 1e-6 and dy > 1e-6:
                    return True                                      # chamfer edges carry a small hip
                return (dy < 1e-6) if ridge_x else (dx < 1e-6)       # eave edges run parallel to the ridge
            pieces = [_poly_block(main, pts, sl, 0)]
    # 45-degree bay on the front or rear wall of the largest piece
    # (bay: --spec forces it on the front wall (True) or off (False); the stream is consumed as before either way)
    draw = r.random()
    if (draw < 0.25 if bay is None else bay) and main["roof"] != "flat":
        P = max(pieces, key=lambda b: b["w"] * b["d"])
        px0, px1 = P["x0"], P["x0"] + P["w"]
        wy = r.choice([P["y0"], P["y0"] + P["d"]])
        if bay:
            wy = P["y0"]
        out = -1 if wy == P["y0"] else 1
        bw, bd = r.uniform(7, 12), r.uniform(2.5, 4)
        if bay:                                 # a described bay: wide enough that its front face takes a window
            bw, bd = 10 + 0.8 * (bw - 7), min(bd, 3.0)
        bd = min(bd, (bw - 2.0) / 2)                                # keep the trapezoid's outer face >= 2 ft
        if px1 - px0 > bw + 6:
            a = r.uniform(px0 + 2, px1 - 2 - bw)
            pts = [(a, wy), (a + bd, wy + out * bd), (a + bw - bd, wy + out * bd), (a + bw, wy)]
            if free(sbox(a, min(wy, wy + out * bd), a + bw, max(wy, wy + out * bd))):
                Hb = min(main["H"], house["fh"] + house["plate"])

                def sl_bay(p, q):
                    return not (abs(p[1] - wy) < 1e-3 and abs(q[1] - wy) < 1e-3)   # not the house-side edge
                pieces.append(_poly_block(main, pts, sl_bay, 0, H=Hb, roof="hip", ov=0.6))
    for i, b in enumerate(rest):
        b.setdefault("parent", i + 1)
    return pieces + rest


def massing_spec(image_id, seed, house):
    r = random.Random(seed * 1_000_003 + image_id * 151 + 7)
    blocks = []
    for b in house["blocks"]:
        H = b.stories * house["fh"] + house["plate"]
        # wings of a flat-roofed house inherit pitch 0 (v6 draws them as a flat line)
        pitch = b.pitch if b.roof == "flat" or b.pitch >= 0.1 else r.uniform(0.25, 0.45)
        blocks.append({"x0": b.x0, "y0": b.y0, "w": b.w, "d": b.d, "H": H, "roof": b.roof, "ridge": b.ridge,
                       "pitch": pitch, "ov": b.ov, "parapet": r.uniform(1.0, 2.5)})
        if b.kind == "xgable":
            blocks[-1]["xgable"] = True
    hf = HOUSE_FORCE.get(image_id, {}) if HOUSE_FORCE else {}
    if SHAPED[0] and not (hf.get("apartment") or hf.get("sections")):     # (a described apartment facade stays planar)
        # own rng stream: the rest of the spec (chimney) is unchanged by the flag
        blocks = shape_blocks(blocks, house, random.Random(seed * 1_000_003 + image_id * 211 + 3),
                              bay=HOUSE_FORCE.get(image_id, {}).get("bay"))
    spec = {"id": image_id, "blocks": blocks}
    cuts = _cut_plan(house, image_id)
    if cuts:
        spec["cuts"] = [{k: c[k] for k in ("x0", "y0", "z0", "w", "d", "h")} for c in cuts]
    if house["chimney"]:
        m = blocks[0]
        cw, cd = r.uniform(2.0, 3.5), r.uniform(1.8, 3.0)
        along_x = m["ridge"] == "x"
        span = m["d"] if along_x else m["w"]
        rise = {"gable": m["pitch"] * span / 2, "hip": m["pitch"] * span / 2, "shed": m["pitch"] * span}.get(m["roof"], 0)
        if along_x:
            cx = r.choice([m["x0"] + r.uniform(2, 6), m["x0"] + m["w"] - r.uniform(2, 6) - cw])
            cy = m["y0"] + m["d"] / 2 + r.uniform(-0.25, 0.1) * m["d"]
        else:
            cy = r.choice([m["y0"] + r.uniform(2, 6), m["y0"] + m["d"] - r.uniform(2, 6) - cd])
            cx = m["x0"] + m["w"] / 2 + r.uniform(-0.25, 0.1) * m["w"]
        spec["chimney"] = {"x0": cx, "y0": cy, "w": cw, "d": cd, "z0": m["H"] - 2,
                           "z1": m["H"] + rise + r.uniform(1.5, 3.5)}
    return spec


def run_freecad(specs, procs=1):
    """Each house is built independently, so `procs` freecadcmd processes on contiguous
    chunks give the same faces as one process (a single one uses ~2 cores)."""
    if procs > 1 and len(specs) > 1:
        size = -(-len(specs) // min(procs, len(specs)))
        chunks = [specs[k:k + size] for k in range(0, len(specs), size)]
        with ThreadPool(len(chunks)) as tp:
            parts = tp.map(run_freecad, chunks)
        return {k: v for part in parts for k, v in part.items()}
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        json.dump(specs, open(src, "w"))
        code = f"import sys; sys.argv=['x', {src!r}, {dst!r}]; exec(open({os.path.join(HERE, 'fc_massing.py')!r}).read())"
        p = subprocess.run([FREECADCMD, "-c", code], capture_output=True, text=True)
        if not os.path.exists(dst):
            raise RuntimeError("freecadcmd failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
        print([l for l in p.stdout.splitlines() if l.startswith("fc_massing")][-1], flush=True)
        return json.load(open(dst))


# ----------------------------------------------------------------------------
# projection + visibility
# ----------------------------------------------------------------------------
def _uvt(view, x, y, z):
    u, t = G.view_map(view)(x, y)
    return u, -z, t


def visible_faces(faces, view, specs_blocks, chim):
    tx, ty = TOWARD[view]
    cand = []
    for f in faces:
        n = f["n"]
        dot = n[0] * tx + n[1] * ty
        if dot < 0.02:
            continue
        rings = [[_uvt(view, *p) for p in lp] for lp in f["loops"]]
        try:
            poly = Polygon([(u, v) for u, v, _ in rings[0]], [[(u, v) for u, v, _ in r] for r in rings[1:]]).buffer(0)
        except Exception:
            continue
        if poly.is_empty or poly.area < 1e-3:
            continue
        P = np.array([p for r in rings for p in r])
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        plane = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        cand.append({"poly": poly, "plane": plane, "n": n, "c": f["c"], "frontal": dot > 0.98})

    def depth(fc, pt):
        return fc["plane"][0] * pt.x + fc["plane"][1] * pt.y + fc["plane"][2]

    out = []
    for i, F in enumerate(cand):
        occ = []
        for j, Gf in enumerate(cand):
            if i == j or not F["poly"].intersects(Gf["poly"]):
                continue
            ov = F["poly"].intersection(Gf["poly"])
            if ov.area < 1e-4:
                continue
            pt = ov.representative_point()
            if depth(Gf, pt) < depth(F, pt) - 1e-4:
                occ.append(Gf["poly"])
        vis = F["poly"].difference(unary_union(occ)) if occ else F["poly"]
        if vis.area < 1e-3:
            continue
        F = dict(F, vis=vis)
        F["kind"] = classify(F, specs_blocks, chim)
        out.append(F)
    return out


def classify(F, blocks, chim):
    n, (cx, cy, cz) = F["n"], F["c"]
    if abs(n[2]) > 0.05:
        return ("roof", None)
    if chim and chim["x0"] - 0.05 <= cx <= chim["x0"] + chim["w"] + 0.05 and \
            chim["y0"] - 0.05 <= cy <= chim["y0"] + chim["d"] + 0.05 and cz > chim["z0"]:
        return ("chimney", None)
    for bi, b in enumerate(blocks):
        if b.get("poly"):
            from shapely.geometry import Point, Polygon
            if Polygon(b["poly"]).buffer(0.05).contains(Point(cx, cy)):
                return ("wall", b.get("parent", bi))
        elif b["x0"] - 0.05 <= cx <= b["x0"] + b["w"] + 0.05 and b["y0"] - 0.05 <= cy <= b["y0"] + b["d"] + 0.05:
            return ("wall", b.get("parent", bi))
    return ("trim", None)           # rake boards / fascia: slab edges outside every footprint


def ref_exclude_rings(geoms_px, W, H):
    """--ref-exclude: pixel polygons (already in image coordinates) -> the annotation file's
    "ref_exclude" list of flat outer rings, clipped to the image; [] when off or empty."""
    if not REF_EXCLUDE[0] or not geoms_px:
        return []
    u = unary_union([g.buffer(0) for g in geoms_px]).intersection(box(0, 0, W, H))
    out = []
    for q in G.polys_of(u):
        if q.area >= 16:
            out.append([round(v, 1) for c in q.exterior.coords[:-1] for v in c])
    return out


def _downspouts(house, view, walls_by_block, openings):
    """--downspouts P (2026-10-04): at each end of each visible block wall, with probability P, a
    downspout runs from the top of the wall (under the eave) to grade, with a kick-out shoe at the
    bottom. Drawn over the siding (revit_render), never cut from the wall label -- the wall
    continues behind it, as on the real rendered elevations. Skips an end where an opening is
    within a foot. Own RNG per view: P = 0 is byte-identical."""
    dr = random.Random(_CUR[0] * 991 + {"front": 1, "rear": 2, "left": 3, "right": 4}[view] * 37 + 11)
    occ = unary_union([o["poly"] for o in openings]).buffer(1.0) if openings else None
    out = []
    for bi in sorted(walls_by_block):
        wall = unary_union([F["vis"] for F in walls_by_block[bi]])
        if wall.is_empty or wall.bounds[2] - wall.bounds[0] < 6:
            continue
        u0, _, u1, _ = wall.bounds
        pw = dr.uniform(0.22, 0.32)
        for end in (0, 1):
            if dr.random() >= DOWNSPOUTS[0]:
                continue
            x = u0 + dr.uniform(0.15, 0.5) if end == 0 else u1 - dr.uniform(0.15, 0.5) - pw
            col = wall.intersection(G.rect(x, 1, x + pw, -200))
            if col.is_empty:
                continue
            y_top = col.bounds[1] + 0.05                    # just under the wall's top at this x
            pipe = G.rect(x, y_top, x + pw, -0.35)
            if occ is not None and pipe.intersects(occ):
                continue
            shoe_dx = pw * 1.6 * (1 if end == 0 else -1)    # kick-out away from the corner
            shoe = G.rect(min(x, x + shoe_dx), -0.35 - pw * 0.9, max(x + pw, x + pw + shoe_dx), -0.05)
            out.append(unary_union([pipe, shoe]))
    return out


# ----------------------------------------------------------------------------
# the v6 elevation dict, from visible faces
# ----------------------------------------------------------------------------
def _railing(house, view, walls_by_block, fh, rr):
    """--railings (2026-10-02): a balcony or raised deck with a railing (and half the time an
    exterior stair) standing in front of a wall, as on HF14 27. Solid boards -- rim joist,
    posts, stair stringer -- are drawn as trims; the rim and stringer are cut out of the labels,
    the posts ("standing") only under --cut-standing 1. The rails, balusters / cables, treads
    and handrail are drawn over the siding and the wall label runs on behind them, as the
    annotators drew it. `depth` = distance in front of the
    wall, so revit_render casts the shadow by the same sun vector as wall_shadows."""
    walls = {bi: unary_union([F["vis"] for F in Fs]) for bi, Fs in walls_by_block.items()}
    walls = {bi: w for bi, w in walls.items() if not w.is_empty and w.bounds[2] - w.bounds[0] > 14}
    if not walls:
        return None
    bi = max(walls, key=lambda k: walls[k].area)
    B, wall = house["blocks"][bi], walls[bi]
    u0, _, u1, _ = wall.bounds
    deck_y = -fh if B.stories >= 2 and rr.random() < 0.55 else -rr.uniform(1.5, 4.0)
    w = max(8.0, rr.uniform(0.35, 0.85) * (u1 - u0))
    w = min(w, u1 - u0 - 1)
    a = rr.uniform(u0 + 0.5, u1 - w - 0.5)
    cut, rails, lines, standing, excl = [], [], [], [], []
    rim_h = rr.uniform(0.7, 1.1)
    cut.append(G.rect(a, deck_y, a + w, deck_y + rim_h))                          # rim joist / deck edge
    rail_h = rr.uniform(3.0, 3.6)
    top = deck_y - rail_h
    rails.append(G.rect(a, top, a + w, top + rr.uniform(0.2, 0.35)))               # top rail
    rails.append(G.rect(a, deck_y - 0.55, a + w, deck_y - 0.4))                    # bottom rail
    pw = rr.uniform(0.35, 0.5)
    n_p = max(1, round(w / rr.uniform(5, 8)))
    pxs = [a + pw / 2 + (w - pw) * k / n_p for k in range(n_p + 1)]
    for px in pxs:
        cut.append(G.rect(px - pw / 2, top - 0.15, px + pw / 2, deck_y))           # railing posts
        standing.append(cut[-1])
        if deck_y > -fh + 0.1:                                                     # ground deck: posts to grade
            cut.append(G.rect(px - pw / 2 * 0.9, deck_y + rim_h, px + pw / 2 * 0.9, 0))
            standing.append(cut[-1])
    excl.append(G.rect(a, top - 0.15, a + w, deck_y))                              # the railing panel
    infill = rr.choices(["pickets", "cable"], weights=[75, 25])[0]
    y_lo, y_hi = deck_y - 0.55, top + 0.35
    if infill == "pickets":
        sp = rr.uniform(0.3, 0.5)
        x = a + sp
        while x < a + w - 0.1:
            lines.append(((x, y_hi), (x, y_lo)))
            x += sp
    else:
        sp = rr.uniform(0.25, 0.4)
        y = y_hi + sp
        while y < y_lo - 0.05:
            lines.append(((a, y), (a + w, y)))
            y += sp
    stair = None
    if -deck_y > 1.5 and rr.random() < 0.5:
        run = -deck_y * rr.uniform(1.0, 1.4)
        if a + w + run <= u1 + 2:
            xt, xb = a + w, a + w + run
        elif a - run >= u0 - 2:
            xt, xb = a, a - run
        else:
            xt = None
        if xt is not None:
            stair = (xt, xb)
            n = max(2, math.ceil(-deck_y / rr.uniform(0.55, 0.65)))
            dx, dy = (xb - xt) / n, -deck_y / n
            strg = LineString([(xt, deck_y + 0.4), (xb, 0.4)]).buffer(0.45, cap_style=2).intersection(
                G.rect(min(xt, xb) - 1, deck_y, max(xt, xb) + 1, 0))
            if not strg.is_empty:
                cut.append(strg)                                                    # stringer
            for k in range(n):                                                      # treads + risers
                x0, y0 = xt + dx * k, deck_y + dy * k
                lines.append(((x0, y0), (x0 + dx, y0)))
                lines.append(((x0 + dx, y0), (x0 + dx, y0 + dy)))
            hr = rr.uniform(2.8, 3.2)
            excl.append(Polygon([(xt, deck_y - hr - 0.15), (xb, -hr - 0.15), (xb, 0.0), (xt, deck_y)]).buffer(0))
            lines.append(((xt, deck_y - hr), (xb, -hr)))                            # handrail
            lines.append(((xt, deck_y - hr + 0.12), (xb, -hr + 0.12)))
            for x in (xb,):
                cut.append(G.rect(x - pw / 2, -hr - 0.15, x + pw / 2, 0))           # newel post
                standing.append(cut[-1])
            sp = rr.uniform(0.35, 0.5) * (1 if xb > xt else -1)
            x = xt + sp
            while (x - xb) * (1 if xb > xt else -1) < -0.1:
                f = (x - xt) / (xb - xt)
                lines.append(((x, deck_y * (1 - f) - hr), (x, deck_y * (1 - f))))   # stair balusters
                x += sp
    depth = rr.uniform(4, 9) if deck_y <= -fh + 0.1 else rr.uniform(6, 12)
    return {"cut": cut, "rails": rails, "lines": lines, "depth": depth, "deck": (a, a + w, deck_y),
            "stair": stair, "bi": bi, "standing": standing, "ref_excl": standing + excl}


def _recess_features(house, view, openings, fh):
    """What sits in the --spec recesses (_cut_plan) on the front view: a door or slider on each recessed balcony's back
    wall with a railing flush with the facade (pickets, cable or glass; no posts -- the reveals hold it), and a door in a
    recessed entry. Openings the recess swallows are dropped. Returns a railing dict (as --railings) or None."""
    cuts = [c for c in _cut_plan(house, _CUR[0]) if c["kind"] in ("balcony", "entry")]
    if view != "front" or not cuts:
        return None
    rr = random.Random(_CUR[0] * 757 + 13)
    infill = rr.choice(["pickets", "pickets", "cable", "glass"])
    rail_h = rr.uniform(3.2, 3.6)
    bi = next((o["block"] for o in openings), 0)
    cut, rails, lines, excl = [], [], [], []
    for c in cuts:
        a, b = c["x0"], c["x0"] + c["w"]
        z0 = -c["z0"]
        hole = G.rect(a, z0 + 0.01, b, z0 - c["h"])
        openings[:] = [o for o in openings if not o["poly"].intersects(hole) and
                       not (c["kind"] == "entry" and o["type"] == "door" and o["poly"].bounds[3] > -0.2)]   # one front door
        dw = 3.0 if c["kind"] == "balcony" and rr.random() < 0.4 else min(6.0, c["w"] - 1.5) if rr.random() < 0.5 else 3.0
        cu = (a + b) / 2
        openings.append({"poly": G.rect(cu - dw / 2, z0, cu + dw / 2, z0 - 6.8), "type": "door", "block": bi})
        if c["kind"] != "balcony":
            continue
        top = z0 - rail_h
        rails.append(G.rect(a, top, b, top + 0.22))
        excl.append(G.rect(a, top, b, z0))
        if infill == "glass":
            rails.append(G.rect(a, z0 - 0.4, b, z0 - 0.28))
            lines.append(((a + 0.3, top + 0.6), (b - 0.3, top + 0.6)))
        elif infill == "pickets":
            rails.append(G.rect(a, z0 - 0.5, b, z0 - 0.38))
            x = a + 0.4
            while x < b - 0.2:
                lines.append(((x, top + 0.22), (x, z0 - 0.5))); x += 0.4
        else:
            y = top + 0.5
            while y < z0 - 0.3:
                lines.append(((a, y), (b, y))); y += 0.33
    if not rails:
        return None
    return {"cut": cut, "rails": rails, "lines": lines, "depth": 0.3, "deck": (cuts[0]["x0"], cuts[0]["x0"] + cuts[0]["w"], -fh),
            "stair": None, "bi": bi, "standing": [], "ref_excl": excl}


def _apartment_balconies(house, view, openings, fh):
    """--spec apartment with balconies (describer tag): stacked balconies on the main block's upper storeys at chosen
    window columns (every bay, alternate bays, or the two ends) -- slab edge, end posts, top / bottom rail and pickets,
    cable or a glass panel -- each with a door or slider behind it in place of that window. One railing dict, so it is
    drawn, shadowed and cut from the labels exactly like --railings. Own stream per sheet and view."""
    B = house["blocks"][0]
    if view not in ("front", "rear") or B.stories < 2:
        return None
    br = random.Random(_CUR[0] * 739 + {"front": 1, "rear": 2}[view] * 17 + 9)
    # (block indices follow the shaped massing pieces, so the columns come from every upper-storey window)
    ups = [o for o in openings if o["type"] == "window" and o["poly"].bounds[3] < -fh]
    cols = sorted({round((o["poly"].bounds[0] + o["poly"].bounds[2]) / 2, 1) for o in ups})
    bi = ups[0]["block"] if ups else 0
    if not cols:
        return None
    pick = br.choice(["all", "alt", "ends"])
    cols = cols if pick == "all" or len(cols) < 3 else cols[br.randrange(2)::2] if pick == "alt" else [cols[0], cols[-1]]
    w = br.uniform(7, 11)
    infill = br.choice(["pickets", "pickets", "cable", "glass"])
    rail_h, pw, slab = br.uniform(3.2, 3.6), br.uniform(0.3, 0.45), br.uniform(0.6, 1.0)
    dw = br.choice([3.0, 5.0, 6.0])
    cut, rails, lines, standing, excl = [], [], [], [], []
    for st in range(1, B.stories):
        deck_y = -st * fh
        for cu in cols:
            a = cu - w / 2
            door = G.rect(cu - dw / 2, deck_y, cu + dw / 2, deck_y - 6.8)
            openings[:] = [o for o in openings if not o["poly"].intersects(door.buffer(0.3))]
            openings.append({"poly": door, "type": "door", "block": bi})
            top = deck_y - rail_h
            cut.append(G.rect(a, deck_y, a + w, deck_y + slab))                      # slab edge
            rails.append(G.rect(a, top, a + w, top + 0.25))                          # top rail
            for px in (a + pw / 2, a + w - pw / 2):
                cut.append(G.rect(px - pw / 2, top - 0.1, px + pw / 2, deck_y)); standing.append(cut[-1])
            excl.append(G.rect(a, top - 0.1, a + w, deck_y))
            if infill == "glass":
                rails.append(G.rect(a, deck_y - 0.45, a + w, deck_y - 0.3))
                lines.append(((a + pw, top + 0.6), (a + w - pw, top + 0.6)))
            elif infill == "pickets":
                rails.append(G.rect(a, deck_y - 0.55, a + w, deck_y - 0.4))
                x = a + pw + 0.4
                while x < a + w - pw - 0.1:
                    lines.append(((x, top + 0.25), (x, deck_y - 0.55))); x += 0.4
            else:
                y = top + 0.55
                while y < deck_y - 0.3:
                    lines.append(((a + pw, y), (a + w - pw, y))); y += 0.33
    if not cut:
        return None
    return {"cut": cut, "rails": rails, "lines": lines, "depth": br.uniform(4, 6), "deck": (cols[0] - w / 2, cols[0] + w / 2, -fh),
            "stair": None, "bi": bi, "standing": standing, "ref_excl": standing + excl}


def _balcony_door(railing, walls_by_block, openings, fh, view):
    """--balcony-doors P: an upper-floor balcony gets a door (or a slider) in the wall behind its
    railing, from the deck up 6.8 ft. It is an ordinary opening -- cut from the wall label, drawn as a
    door -- with the railing drawn over its lower part, so the labelled siding stops at the door frame
    behind the pickets. Windows it would overlap are removed. Own RNG per view."""
    a, b, deck_y = railing["deck"]
    if deck_y > -fh + 0.1:                   # ground decks: the doors there are the grade doors
        return
    dr = random.Random(_CUR[0] * 911 + {"front": 1, "rear": 2, "left": 3, "right": 4}[view] * 13 + 5)
    if dr.random() >= BALCONY_DOORS[0]:
        return
    bi = railing["bi"]
    wall = unary_union([F["vis"] for F in walls_by_block[bi]])
    dw = 3.0 if dr.random() < 0.6 else 6.0
    if b - a < dw + 2.0:
        dw = 3.0
    if b - a < dw + 2.0:
        return
    du = dr.uniform(a + 1.0, b - 1.0 - dw)
    p = G.rect(du, deck_y, du + dw, deck_y - 6.8)
    if not p.within(wall.buffer(0.01)):
        return
    openings[:] = [o for o in openings if not (o["block"] == bi and o["poly"].buffer(0.5).intersects(p))]
    openings.append({"poly": p, "type": "door", "block": bi})


def build_elevation_fc(house, view, rng):
    spec, faces = _FACES[_CUR[0]]
    blocks, chim = spec["blocks"], spec.get("chimney")
    vf = visible_faces(faces, view, blocks, chim)
    fh = house["fh"]
    surfaces, trims, openings = [], [], []
    # view-space order of v6's Block objects matches spec["blocks"]
    walls_by_block = {}
    for F in vf:
        kind, bi = F["kind"]
        if kind == "roof":
            for p in G.polys_of(F["vis"]):
                surfaces.append(("roof", p, bi if bi is not None else 0, "roof"))
            # fascia band along the eave (lowest, horizontal edge of the plane), seen from the eave side
            if F["frontal"] is False and abs(F["n"][2]) < 0.97:
                x0, y0, x1, y1 = F["poly"].bounds
                eave = box(x0, y1 - 0.02, x1, y1 + rng.uniform(0.35, 0.7)).intersection(F["vis"].buffer(0.8))
                if not eave.is_empty and eave.area > 0.2:
                    trims.append(eave)
        elif kind == "chimney":
            for p in G.polys_of(F["vis"]):
                surfaces.append(("chimney", p, 0, "chimney"))
        elif kind == "trim":
            trims.append(F["vis"])
        else:
            walls_by_block.setdefault(bi, []).append(F)
    if chim:
        cs = [p for (f, p, _, _) in surfaces if f == "chimney"]
        if cs:
            b = unary_union(cs).bounds
            trims.append(box(b[0] - 0.3, b[1], b[2] + 0.3, b[1] + 0.5))
    acc = house["accent"]
    gt = random.Random(_CUR[0] * 929 + 17)            # --gable-trim: own stream, one decision per house
    gable_trim = acc != "gable" and gt.random() < GABLE_TRIM[0]
    gable_trim_h = gt.uniform(0.4, 0.8)
    for bi, Fs in walls_by_block.items():
        B = house["blocks"][bi]
        wall = unary_union([F["vis"] for F in Fs])
        if wall.is_empty:
            continue
        u0, y_top_all, u1, _ = wall.bounds
        H = B.stories * fh + house["plate"]
        y_top = -H
        belt_polys = []
        if house["belt"] and B.stories >= 2:
            for s in range(1, B.stories):
                yb = -s * fh
                belt_polys.append(G.rect(u0 - 0.2, yb - house["belt_h"] / 2, u1 + 0.2, yb + house["belt_h"] / 2))

        if gable_trim and B.roof == "gable" and wall.bounds[1] < y_top - 1.0:
            belt_polys.append(G.rect(u0 - 0.2, y_top - 0.1, u1 + 0.2, y_top + gable_trim_h))

        def zone_for(placement):
            if placement == "gable" and wall.bounds[1] < y_top - 1.0 and (B.roof == "gable" or not G.TRIM_CUT):
                belt_polys.append(G.rect(u0 - 0.2, y_top - 0.1, u1 + 0.2, y_top + rng.uniform(0.4, 0.8)))
                return G.rect(u0 - 1, y_top, u1 + 1, y_top - 100)
            if placement == "wainscot":
                belt_polys.append(G.rect(u0 - 0.2, -house["wainscot_h"] - 0.3, u1 + 0.2, -house["wainscot_h"]))
                return G.rect(u0 - 1, 0.5, u1 + 1, -house["wainscot_h"])
            if placement == "upper" and B.stories >= 2:
                if not house["belt"]:
                    belt_polys.append(G.rect(u0 - 0.2, -fh - 0.3, u1 + 0.2, -fh + 0.3))
                return G.rect(u0 - 1, -fh, u1 + 1, -200)
            if placement == "block" and B.kind != "main":
                return G.rect(u0 - 1, 1, u1 + 1, -200)
            if placement == "split" and B.kind == "main" and view in ("front", "rear") and house.get("split"):
                fa, fb = house["split"]
                if view == "rear":
                    fa, fb = 1.0 - fb, 1.0 - fa
                wb = u1 - u0
                sa, sb = u0 + fa * wb, u0 + fb * wb
                for e in (sa, sb):
                    if u0 + 0.3 < e < u1 - 0.3:
                        belt_polys.append(G.rect(e - 0.2, 0.0, e + 0.2, y_top))
                return G.rect(sa - (1 if fa <= 0 else 0), 1, sb + (1 if fb >= 1 else 0), -200)
            return None

        accent_zone = zone_for(acc) if acc else None
        accent2_zone = zone_for(house["accent2"]) if house.get("accent2") else None
        found = G.rect(u0 - 0.1, 0, u1 + 0.1, -house["foundation"]) if house["foundation"] > 0 else None
        belts = unary_union(belt_polys).intersection(wall.buffer(0.2)) if belt_polys else None
        main_zone = wall
        if found is not None:
            main_zone = main_zone.difference(found)
        if belts is not None:
            main_zone = main_zone.difference(belts)
        for zname, zone in (("accent", accent_zone), ("accent2", accent2_zone)):
            if zone is not None:
                for p in G.polys_of(main_zone.intersection(zone)):
                    surfaces.append((zname, p, bi, "wall"))
                main_zone = main_zone.difference(zone)
        for p in G.polys_of(main_zone):
            surfaces.append(("main", p, bi, "wall"))
        if found is not None:
            for p in G.polys_of(wall.intersection(found)):
                surfaces.append(("foundation", p, bi, "found"))
        if belts is not None:
            trims += G.polys_of(belts)
        openings += place_openings(house, B, bi, view, wall, Fs, rng)
    # porch on the front: nearest of all (v6's 2D porch)
    porch = None
    if house["porch"] and view == "front" and 0 in walls_by_block:
        mb = house["blocks"][0]
        u0, u1 = mb.x0, mb.x1
        pw = rng.uniform(8, min(24, mb.w - 4))
        pu0 = min(max(u0 + house["front_door_u"] * mb.w - pw / 2, u0 + 1), u1 - pw - 1)
        top = -(fh - rng.uniform(0.2, 1.0))
        proof = G.rect(pu0 - 1, top, pu0 + pw + 1, top - rng.uniform(1.5, 3.5))
        porch = {"roof": proof, "posts": [G.rect(pu0 + 0.3, 0, pu0 + 0.9, top), G.rect(pu0 + pw - 0.9, 0, pu0 + pw - 0.3, top)],
                 "fascia": G.rect(pu0 - 1, top, pu0 + pw + 1, top + 0.5)}
        cut = proof.union(porch["fascia"])
        if G.TRIM_CUT:      # the porch is in front: belts / fascia behind it are hidden
            trims = [q for t in trims for q in G.polys_of(t.difference(cut))]
        surfaces = [(f, q, bi, k) for (f, p, bi, k) in surfaces for q in G.polys_of(p.difference(cut))]
        surfaces.append(("roof", proof, -1, "porch"))
        trims.append(porch["fascia"])
        trims += porch["posts"]
    standing = list(porch["posts"]) if porch else []
    railing = None
    hf = HOUSE_FORCE.get(_CUR[0], {}) if HOUSE_FORCE else {}
    recessed = _recess_features(house, view, openings, fh) if hf else None
    if recessed is not None:
        railing = recessed
    elif hf.get("apartment") and hf.get("balconies") and not hf.get("recessed_balconies"):
        railing = _apartment_balconies(house, view, openings, fh)
        if railing is not None:
            trims += railing["cut"]
            standing += railing["standing"]
    if railing is None and not hf.get("apartment") and getattr(G, "RAILINGS", 0) > 0 and walls_by_block:
        rr = random.Random(_CUR[0] * 977 + {"front": 1, "rear": 2, "left": 3, "right": 4}[view] * 31 + 7)
        if rr.random() < G.RAILINGS:
            railing = _railing(house, view, walls_by_block, fh, rr)
            if railing is not None:
                trims += railing["cut"]
                standing += railing["standing"]
                if BALCONY_DOORS[0] > 0:
                    _balcony_door(railing, walls_by_block, openings, fh, view)
    downspouts = _downspouts(house, view, walls_by_block, openings) if DOWNSPOUTS[0] > 0 else []
    ref_excl = (list(porch["posts"]) if porch else []) + (railing["ref_excl"] if railing else []) + downspouts
    # labels are cut by every trim except the standing posts (unless --cut-standing 1); the
    # drawing still uses every trim, so images do not change
    label_trims = trims if CUT_STANDING[0] else [t for t in trims if not any(t is q for q in standing)]
    trims = [q for t in trims for q in G.polys_of(t) if q.area > 0.05]
    label_trims = [q for t in label_trims for q in G.polys_of(t) if q.area > 0.05]
    allp = [p for (_, p, _, _) in surfaces] + trims
    return {"surfaces": surfaces, "trims": trims, "label_trims": label_trims, "openings": openings,
            "porch": porch, "extent": unary_union(allp).bounds, "vf": vf, "railing": railing,
            "downspouts": downspouts, "ref_excl": ref_excl}


def _apt_cols(house, view, image_id=None):
    """The apartment window columns of a view (view-u feet): a regular bay spacing over the main block's span, own
    stream per sheet and view, so every block of the facade (projecting sections included) and the recess cuts share it."""
    m = house["blocks"][0]
    span = {"front": (m.x0, m.x1), "rear": (-m.x1, -m.x0), "left": (-m.y1, -m.y0), "right": (m.y0, m.y1)}[view]
    iid = _CUR[0] if image_id is None else image_id
    ar = random.Random(iid * 613 + {"front": 1, "rear": 2, "left": 3, "right": 4}[view] * 29 + 3)
    bay = ar.uniform(9, 13) if view in ("front", "rear") else ar.uniform(8, 12)
    n = max(1, int((span[1] - span[0]) / bay))
    return [span[0] + (i + 0.5) * (span[1] - span[0]) / n for i in range(n)]


def _apartment_openings(house, B, bi, view, fu0, width, ok):
    """--spec apartment block (the describer's apartment_or_multistorey_block): window columns aligned through every
    storey (_apt_cols, shared by all blocks of the facade), wider units on the long faces, and on the front a glazed
    entry (single or double door) in place of the ground-floor window of one or two bays -- not the townhouse
    template's door + garage per unit. Own stream per sheet and view."""
    fh = house["fh"]
    ar = random.Random(_CUR[0] * 617 + {"front": 1, "rear": 2, "left": 3, "right": 4}[view] * 31 + 5)
    long_face = view in ("front", "rear")
    cols = _apt_cols(house, view)
    n = len(cols)
    pitch_u = (cols[1] - cols[0]) if n > 1 else width
    ww = min(pitch_u - 1.5, ar.choice([3.5, 4.0, 5.0, 6.0]) if long_face else ar.choice([3.0, 3.5, 4.0]))
    wh = ar.choice([4.0, 4.5, 5.0])
    entries = set()
    if view == "front" and not any(c["kind"] == "entry" for c in _cut_plan(house, _CUR[0])):
        mid = (n - 1) / 2
        entries = {int(mid)} if n % 2 or n < 4 else {n // 2 - 1, n // 2} if ar.random() < 0.3 else {n // 2}
        if n >= 6 and ar.random() < 0.4:
            entries = {1, n - 2}
    out = []
    for i, cu in enumerate(cols):
        if not fu0 - 0.5 <= cu <= fu0 + width + 0.5:
            continue                            # this column is on another block of the facade
        for st in range(B.stories):
            if st == 0 and i in entries:
                dw = 3.0 if ar.random() < 0.5 else 5.5
                p = G.rect(cu - dw / 2, 0, cu + dw / 2, -7.2)
                if ok(p, 0.0) or p.within(G.rect(fu0, 0.01, fu0 + width, -B.stories * fh)):
                    out.append({"poly": p, "type": "door", "block": bi})
                continue
            head = -st * fh - 7.0
            p = G.rect(cu - ww / 2, head, cu + ww / 2, head + wh)
            if ok(p):
                out.append({"poly": p, "type": "window", "block": bi})
    return out


def place_openings(house, B, bi, view, wall, Fs, rng):
    fh = house["fh"]
    out = []
    inner = wall.buffer(-0.6)
    # frontal wall faces only (each is one planar facade)
    faces = [F for F in Fs if F["frontal"]]
    if not faces:
        return out
    fu0, _, fu1, _ = unary_union([F["vis"] for F in faces]).bounds
    width = fu1 - fu0
    front_main = view == "front" and B.kind == "main"
    garage_face = view == "front" and B.kind == "garage"

    # --openings-per-face (2026-10-02, default on): an opening sits on ONE planar face, never
    # across the edge where a bay / wing steps forward (windows and doors drawn straight through
    # a projecting bay's corners). 0 reproduces earlier pools.
    per_face = OPENINGS_PER_FACE[0]
    fvis = [F["vis"] for F in faces]

    def one_face(p, m):
        return any(p.within(v.buffer(m)) for v in fvis)

    def ok(p, margin=0.0):
        return p.within(inner) and not any(p.buffer(0.8).intersects(o["poly"]) for o in out) and \
            (not per_face or one_face(p, -0.3))

    def fit(p):
        """Grade-level door: as is, slid into the face under its centre, or None."""
        if not per_face or one_face(p, 0.01):
            return p
        x0, y0, x1, y1 = p.bounds
        c = Point((x0 + x1) / 2, (y0 + y1) / 2)
        for v in sorted(fvis, key=lambda v: v.distance(c)):     # the face under it, else the nearest that fits
            a, _, b, _ = v.bounds
            if b - a < (x1 - x0) + 1.0:
                continue
            vb = v.buffer(0.01)
            room = v.buffer(-0.5).union(vb.intersection(G.rect(a - 1, -0.6, b + 1, 0.5))).buffer(0.01)   # 0.5 ft clear, sill at grade
            for k in range(int((b - a) / 0.5) + 2):          # nearest free spot on this face, either way
                for dx in ((k * 0.5, -k * 0.5) if k else (0.0,)):
                    q = affinity.translate(p, dx, 0)
                    if q.within(room) and \
                            not any(q.buffer(0.8).intersects(o["poly"]) for o in out):
                        return q
        return None

    if HOUSE_FORCE.get(_CUR[0], {}).get("apartment") and B.kind in ("main", "section"):
        return out + _apartment_openings(house, B, bi, view, fu0, width, ok)
    if house.get("units") and B.kind == "main" and view in ("front", "rear"):
        n = house["units"]
        uw = width / n
        tmpl = house["unit_template"]
        head0 = -rng.uniform(6.8, 7.2)
        for i in range(n):
            base = fu0 + i * uw
            for kind, off, sw in tmpl["ground"]:
                if kind == "door" and view == "front":
                    p, t = G.rect(base + off, 0, base + off + sw, -6.8), "door"
                elif kind == "garage" and view == "front":
                    p, t = G.rect(base + off, 0, base + off + sw, -7.0), "garage"
                else:
                    ww = sw if kind == "win" else 3.0
                    p, t = G.rect(base + off, head0, base + off + ww, head0 + 4.0), "window"
                if p.within(wall.buffer(0.01)) and (not per_face or one_face(p, 0.01)):
                    out.append({"poly": p, "type": t, "block": bi})
            for st in range(1, B.stories):
                for kind, off, sw in tmpl["upper"]:
                    head = -st * fh - 7.0
                    p = G.rect(base + off, head, base + off + sw, head + 4.0)
                    if ok(p):
                        out.append({"poly": p, "type": "window", "block": bi})
        return out
    gint = HOUSE_FORCE.get(_CUR[0], {}).get("garage_int") if (HOUSE_FORCE and front_main) else None
    if gint:
        # --spec integrated garage: 1-3 garage doors in the main block's own front wall at grade, at one end (own stream)
        gr = random.Random(_CUR[0] * 659 + 23)
        gw = (16.0 if gr.random() < 0.5 else 9.0) if gint == 1 else 9.0
        run = gint * gw + (gint - 1) * 2.0
        fa, _, fb, _ = max(fvis, key=lambda v: v.bounds[2] - v.bounds[0]).bounds    # all on one planar face
        if run > fb - fa - 3 and gint > 1:
            gint, run = 1, gw
        if run < fb - fa - 3:
            g0 = fa + 1.5 if gr.random() < 0.5 else fb - 1.5 - run
            for k in range(gint):
                p = G.rect(g0 + k * (gw + 2.0), 0, g0 + k * (gw + 2.0) + gw, -7.5)
                if p.within(wall.buffer(0.01)) and (p := fit(p)) is not None:
                    out.append({"poly": p, "type": "garage", "block": bi})
    if front_main:
        dw = 3.0 if rng.random() < 0.7 else 6.0
        du = min(max(fu0 + house["front_door_u"] * width, fu0 + 2.5), fu1 - 2.5 - dw)
        if gint and any(G.rect(du - 1, 0, du + dw + 1, -6.8).intersects(o["poly"]) for o in out):
            du = min(max(fu0 + fu1 - du - dw, fu0 + 2.5), fu1 - 2.5 - dw)    # the other side of the facade
        p = G.rect(du, 0, du + dw, -6.8)
        if p.within(wall.buffer(0.01)) and (p := fit(p)) is not None:
            out.append({"poly": p, "type": "door", "block": bi})
    if garage_face and width > 11:
        gdw = min(8.0 if width < 20 or rng.random() < 0.4 else 16.0, width - 3)
        gu = fu0 + (width - gdw) / 2 + rng.uniform(-1, 1)
        p = G.rect(gu, 0, gu + gdw, -7.0)
        if p.within(wall.buffer(0.01)) and (p := fit(p)) is not None:
            out.append({"poly": p, "type": "garage", "block": bi})
    for s in range(B.stories):
        if B.kind == "garage" and s == 0 and garage_face:
            continue
        n = max(0, int(width / rng.uniform(6, 11)))
        if B.kind == "garage":
            n = min(n, 2)
        for i in range(n):
            cu = fu0 + (i + 0.5) * width / n + rng.uniform(-1.0, 1.0)
            ww = rng.choice([2.5, 3.0, 3.0, 4.0, 5.0, 6.0])
            wh = rng.choice([3.0, 3.5, 4.0, 4.5, 5.0])
            head = -s * fh - rng.uniform(6.6, 7.4)
            if s == 0 and house["accent"] == "wainscot" and head + wh > -house["wainscot_h"] + 0.3:
                wh = min(wh, -house["wainscot_h"] - head - 0.3)
                if wh < 2.0:
                    continue
            p = G.rect(cu - ww / 2, head, cu + ww / 2, head + wh)
            if ok(p):
                out.append({"poly": p, "type": "window", "block": bi})
    if per_face and not house.get("units"):      # a bay / narrow wing face left empty: one centred window per storey
        for v in fvis:
            a_, _, b_, _ = v.bounds
            fw = b_ - a_
            if not 3.8 <= fw <= 12 or any(o["poly"].intersects(v) for o in out):
                continue
            cu, ww = (a_ + b_) / 2, min(fw - 1.4, 4.0 if fw < 8 else 5.0)
            for s_ in range(B.stories):
                head = -s_ * fh - 7.0
                p = G.rect(cu - ww / 2, head, cu + ww / 2, head + 4.5)
                if ok(p):
                    out.append({"poly": p, "type": "window", "block": bi})
    # gable-end attic window where the facade rises into a gable
    x0, y0, x1, _ = wall.bounds
    H = B.stories * fh + house["plate"]
    if y0 < -H - 3.5 and rng.random() < 0.4:
        cu = (x0 + x1) / 2
        p = G.rect(cu - 1.2, -H - 1.0, cu + 1.2, -H - 3.2)
        if ok(p):
            out.append({"poly": p, "type": "window", "block": bi})
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --spec FILE (2026-10-20): render a GIVEN house + look instead of drawing one -- facsimiles of Gemini images
# (codex_doc "synthetic facsimiles"). A spec is a JSON object:
#   {"name": "gemini30_16", "mode": "elevation" | "roof_plan" | "freeform", "shaded": true (colour) | false (line-only),
#    "views": 1 | 2 | 4 (elevation views on the sheet: one-hot view-count weights for that image),
#    "view_list": ["front", "rear"] (which views, in order; forced after the sheet's draws), "facade_aspect": width / height
#    of the front view (the closest of the first 25 fully matching ids), house "ridge": "side" | "front", "cross_gables",
#    "split",
#    "house": {"stories": 2, "roof": "gable", "units": null | 2..5, "wing": bool, "ell": bool, "porch": bool,
#              "chimney": bool, "dormers": bool, "accent": null | "gable" | "wainscot" | "upper" | "block",
#              "accent2": null | ..., "label_trim": bool},
#    "styles": {"main": {"kind": "lap", "base": [r, g, b], "params": {"sp": 0.5}}, "accent": {...}, "roof": {...},
#               "foundation": {...}, "chimney": {...}}}
# Discrete choices are met by SEARCH: every image id fixes its own mode, house and shaded draw, so the first id (from
# --start) whose draws match every given "house" / "mode" / "shaded" key is used, and the FreeCAD massing, labels and
# ref_exclude stay consistent by construction. Styles are then forced by role (kind locked against later restyling;
# "stipple" / "flat" pick which way stucco comes out,
# colour base on colour sheets, params merged). Keys the generator cannot express are reported, not faked: a spec
# whose discrete keys no id meets within --spec-search ids takes the best partial match and lists the unmet keys in
# spec_matches.json (the gap list). Elevations are the supported mode; plans match mode + house only.
# ----------------------------------------------------------------------------
SPEC_BY_ID = {}


def _prefix_draws(image_id, seed, mw):
    _CUR[0] = image_id
    rng = random.Random(seed * 1_000_003 + image_id)
    mode = rng.choices(list(mw.keys()), weights=list(mw.values()))[0]
    G.make_appearance(rng)
    house = G.make_house(rng)
    shaded = random.Random(seed * 1_000_003 + image_id * 173 + 41).random() < 0.4
    return mode, house, shaded


def _house_value(house, key):
    blocks = house["blocks"]
    if key == "stories":
        return blocks[0].stories
    if key == "roof":
        return blocks[0].roof
    if key == "wing":
        return any(b.kind not in ("main", "ell", "xgable", "section") for b in blocks[1:])
    if key == "ell":
        return any(b.kind == "ell" for b in blocks)
    if key == "ridge":                      # "side": eaves face the front view (ridge along x); "front": gable end faces it
        return "side" if blocks[0].ridge == "x" else "front"
    if key == "cross_gables":
        return house.get("cross_gables", 0)
    if key == "split":
        return bool(house.get("split"))
    return house.get(key)


def facade_aspect(house):
    """Front view of the house: overall width / height from grade to the top of the main roof (estimate)."""
    bl = house["blocks"]
    width = max(b.x1 for b in bl) - min(b.x0 for b in bl)
    m = bl[0]
    rise = {"gable": m.pitch * (m.d if m.ridge == "x" else m.w) / 2, "hip": m.pitch * min(m.w, m.d) / 2,
            "shed": m.pitch * m.d, "flat": 0.0}.get(m.roof, 0.0)
    return width / max(1.0, m.stories * house["fh"] + house["plate"] + rise)


def find_spec_id(spec, seed, mw, start, search):
    want = [("mode", spec["mode"])] if "mode" in spec else []
    if "shaded" in spec:
        want.append(("shaded", bool(spec["shaded"])))
    want += [("house." + k, v) for k, v in spec.get("house", {}).items()]
    target = spec.get("facade_aspect")
    force = _force_of(spec)
    best, full = (-1, None, None), []
    for iid in range(start, start + search):
        if force:
            HOUSE_FORCE[iid] = force
        mode, house, shaded = _prefix_draws(iid, seed, mw)
        got = {"mode": mode, "shaded": shaded}
        unmet = [k for k, v in want
                 if (got[k] if not k.startswith("house.") else _house_value(house, k[6:])) != v]
        if not unmet:
            if not target:
                return iid, []
            full.append((abs(math.log(facade_aspect(house) / float(target))), iid))
            if len(full) >= 25:                 # the closest facade proportions among the first 25 full matches
                break
        elif len(want) - len(unmet) > best[0] and "mode" not in unmet:     # a partial match never changes the mode
            best = (len(want) - len(unmet), iid, unmet)
    if full:
        return min(full)[1], []
    return best[1], best[2]


class _LockedStyle(G.Style):
    """A spec family: later code (plain-wall draws, roof fix-ups) cannot change its kind."""
    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, v):
        if not getattr(self, "_locked", False):
            self._kind = v


def _spec_make_style(orig):
    def make_style(rng, kind, appearance, seed, S, label_name, roof=False, base_override=None):
        o = SPEC_BY_ID.get(_CUR[0], {}).get("styles", {}).get(label_name)
        if not o:
            return orig(rng, kind, appearance, seed, S, label_name, roof=roof, base_override=base_override)
        kind = o.get("kind", kind)
        neutral = G.NEUTRAL_PALETTE
        if "base" in o and appearance["colour"]:
            base_override = tuple(int(v) for v in o["base"])
            G.NEUTRAL_PALETTE = False           # (it replaces any base) the described colour wins
        try:
            return _spec_style(orig, o, rng, kind, appearance, seed, S, label_name, roof, base_override)
        finally:
            G.NEUTRAL_PALETTE = neutral
    return make_style


def _spec_style(orig, o, rng, kind, appearance, seed, S, label_name, roof, base_override):
    if kind in ("stipple", "flat") and not roof:
        # "stucco" comes out stipple or flat at random: draw from a side stream until it is the asked one
        # (the sheet's main stream is consumed once, as before)
        st = orig(rng, "stucco", appearance, seed, S, label_name, roof=roof, base_override=base_override)
        for t in range(40):
            if st.kind == kind:
                break
            st = orig(random.Random(seed * 7919 + t), "stucco", appearance, seed, S, label_name, roof=roof,
                      base_override=base_override)
    elif kind == "panel" and not roof:      # not a make_style kind: a lap wall turned into panels (as --panel-siding)
        st = _to_panel(orig(rng, "lap", appearance, seed, S, label_name, roof=roof, base_override=base_override),
                       random.Random(seed * 211 + 97))
    else:
        st = orig(rng, kind, appearance, seed, S, label_name, roof=roof, base_override=base_override)
    k = st.kind
    st.__class__ = _LockedStyle
    st._kind, st._locked = k, False
    st.params.update(o.get("params", {}))
    st._locked = "kind" in o
    return st


def _init(out, seed, mw, flags, faces, revit=False, revit_plans=False, shaped=False, spec_by_id=None):
    SHAPED[0] = shaped
    if spec_by_id:
        SPEC_BY_ID.update(spec_by_id)
        HOUSE_FORCE.update({i: _force_of(s) for i, s in spec_by_id.items() if _force_of(s)})
    flags = dict(flags)
    OPENINGS_PER_FACE[0] = bool(flags.pop("openings_per_face", 1))
    CUT_STANDING[0] = bool(flags.pop("cut_standing", 0))
    DOWNSPOUTS[0] = float(flags.pop("downspouts", 0.05))
    REF_EXCLUDE[0] = bool(flags.pop("ref_exclude", 1))
    NEAR_PAIRS[0] = float(flags.pop("near_pairs", 0.0))
    NEAR_PAIRS_PLAN[0] = float(flags.pop("near_pairs_plan", 0.3))
    NEAR_PAIRS_PLAN_KINDS[0] = tuple(flags.pop("near_pairs_plan_kinds", "direction").split(","))
    BALCONY_DOORS[0] = float(flags.pop("balcony_doors", 0.7))
    GABLE_TRIM[0] = float(flags.pop("gable_trim", 0.5))
    TURN_BOARDS[0] = float(flags.pop("turn_boards", 0.9))
    PAIR_TURNS[0] = bool(flags.pop("pair_turns", 1))
    HEAVY_PIECES[0] = float(flags.pop("heavy_pieces", 0.4))
    GABLE_PAIRS[0] = float(flags.pop("gable_pairs", 0.0))
    CROSS_GABLES[0] = float(flags.pop("cross_gables", 0.4))
    BASE_BANDS[0] = float(flags.pop("base_bands", 1.0))
    WALL_SPLITS[0] = float(flags.pop("wall_splits", 0.3))
    PANEL_SIDING[0] = float(flags.pop("panel_siding", 0.12))
    G.PATTERN_DRIFT = float(flags.pop("pattern_drift", 0.3))
    GEMINI_LOOK[0] = float(flags.pop("gemini_look", 0.0))
    G.TIGHT_PAD = tuple(flags.pop("tight_pad", (0.015, 0.05)))
    G._init(out, seed, mw, **flags)
    _FACES.update(faces)
    G.build_elevation = build_elevation_fc
    if revit:
        import revit_render
        revit_render.install(G, sys.modules[__name__], plans=revit_plans)
    if not getattr(G.make_style, "_panel", False):
        G.make_style = _panel_siding(G.make_style)
    if SPEC_BY_ID and not getattr(G.make_style, "_spec", False):
        G.make_style = _spec_make_style(G.make_style)
        G.make_style._spec = True


def _job(image_id):
    if image_id not in _FACES:
        return (image_id, False, "no massing")
    _CUR[0] = image_id
    nv = SPEC_BY_ID.get(image_id, {}).get("views")
    if nv:                                 # one-hot view-count weights: the same single draw, so nothing else moves
        old = getattr(G, "REVIT_VIEWS", None)
        G.REVIT_VIEWS = [float(nv == 1), float(nv == 2), float(nv == 4)]
        try:
            return G._job(image_id)
        finally:
            G.REVIT_VIEWS = old
    return G._job(image_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--mode-weights", default="86,7,7")
    ap.add_argument("--r8", action="store_true", help="the r8 flag set (best probe pool so far)")
    ap.add_argument("--tight-crop", action="store_true")
    ap.add_argument("--paper", type=float, default=0.0,
                    help="2026-10-06, Gemini-like: P of sheets photographed / scanned (paper tone, uneven light, "
                         "blur, noise, JPEG); labels unchanged. 0 = byte-identical")
    ap.add_argument("--pale-ink", type=float, default=0.0,
                    help="2026-10-06, Gemini-like: P of line-only sheets drawn in pale grey ink. 0 = byte-identical")
    ap.add_argument("--revit-view-weights", default=None,
                    help="2026-10-06: Revit elevation 1 / 2 / 4 view weights, e.g. 50,45,5 (Gemini: almost all "
                         "1-2 large views). Default 30,40,30")
    ap.add_argument("--real-labelling", action="store_true",
                    help="v6's --real-labelling alone (walls labelled, mostly not trim/chimney/foundation)")
    ap.add_argument("--revit", action="store_true",
                    help="elevations drawn Revit-style from the 3D model: cast shadows, line-weight "
                         "hierarchy, level datums, view titles, restrained materials")
    ap.add_argument("--revit-plans", action="store_true",
                    help="with --revit: roof plans (from the 3D model's roof faces) and floor plans drawn "
                         "Revit-style too (scripts/revit_plans.py); off = v6 plans, byte-identical")
    ap.add_argument("--window-hole-prob", type=float, default=1.0,
                    help="per sheet: cut windows/doors out of wall labels (1.0 = always, the eval convention; "
                         "0.8 reproduces the pre-2026-09-29 revit pools)")
    ap.add_argument("--casing-holes", type=int, default=1, choices=[0, 1],
                    help="1 = wall-label holes cover window/door casing and head (default); "
                         "0 = holes at the glass only, the pre-2026-09-29 pools")
    ap.add_argument("--trim-cut", type=int, default=1, choices=[0, 1],
                    help="1 = white trim cut out of every surface label, trim slivers dropped, porch roof hides "
                         "the belt behind it, no gable accent on gable-less houses (default); 0 = pre-2026-09-29 pools")
    ap.add_argument("--distinct-looks", type=int, default=1, choices=[0, 1],
                    help="1 = Revit elevations never label two look-alike families or a plain (patternless) "
                         "family (default); 0 = pre-2026-09-29 pools")
    ap.add_argument("--plan-v2", type=int, default=2, choices=[0, 1, 2],
                    help="--revit-plans floor plans v2: covered patios in footprint notches, grey paving beside "
                         "grey interiors, stronger interior texture, and a ceiling / electrical sheet type (15%%) "
                         "with a labelled soffit band; 0 reproduces pools before 2026-10-01. "
                         "2 (default since 2026-10-04) = v3 on top: furniture in every room (solid with shadows on "
                         "rendered sheets, ghosted under ceiling plans), patio furniture, framing cropped to the "
                         "drawing, and floor labels that leave out wall-fixed pieces (counters, tubs, vanities); "
                         "1 reproduces pools from 2026-10-01 to 2026-10-03")
    ap.add_argument("--plan-source", default=None,
                    help="--revit-plans floor plans: a layouts file from scripts/swiss_plans.py (real Swiss Dwellings "
                         "apartments, CC BY 4.0 -- credit Archilyse AG) used instead of the procedural layout; "
                         "off = byte-identical")
    ap.add_argument("--plan-source-p", type=float, default=1.0,
                    help="with --plan-source: probability a floor plan uses a real layout (rest procedural)")
    ap.add_argument("--masonry-base", type=float, default=0.35,
                    help="Revit elevations: probability the base band is a brick / stone / block wainscot "
                         "instead of concrete or flat. Default 0.35 since 2026-10-01; 0 reproduces earlier pools")
    ap.add_argument("--faint-lines", type=float, default=0.2,
                    help="Revit line-only elevations: probability every pattern is drawn in pale grey "
                         "(190-228, finer pen), as on HF14 14; default 0.2 since 2026-10-14f (every run passed it); 0 = earlier pools")
    ap.add_argument("--soft-shadows", type=float, default=0.5,
                    help="Revit elevations: probability shadows are lighter (x0.76-0.88), neutral grey and two-level "
                         "(overlap darker), as on HF14 02. Default 0.5 since 2026-10-13 (every run since 2026-10-02 "
                         "passed it); 0 = earlier pools")
    ap.add_argument("--roof-lines", type=float, default=0.3,
                    help="Revit elevations: probability the roof is plain tight horizontal lines, 60%% of them over "
                         "looser vertical boards, as on HF14 25 / 27; default 0.3 since 2026-10-14f (every run passed it); 0 = earlier pools")
    ap.add_argument("--openings-per-face", type=int, choices=[0, 1], default=1,
                    help="1 (default since 2026-10-02) = every window / door sits on one planar wall face, never "
                         "across the corner of a projecting bay or wing; 0 reproduces earlier pools")
    ap.add_argument("--cut-standing", type=int, choices=[0, 1], default=0,
                    help="0 (default since 2026-10-04) = porch / railing / newel posts stay inside the wall label "
                         "(the wall continues behind them); 1 cuts them out, reproducing earlier pools")
    ap.add_argument("--downspouts", type=float, default=0.05,
                    help="Revit elevations: probability a wall end gets a downspout (eave to grade) drawn over the "
                         "siding, the wall label running behind it. Default 0.05 since 2026-10-04; 0 = earlier pools")
    ap.add_argument("--ref-exclude", type=int, choices=[0, 1], default=1,
                    help="1 (default) = annotation files carry ref_exclude (posts, railings, stairs, downspouts, solid "
                         "furniture, rugs) so reference boxes avoid them; 0 omits it (earlier pools' files)")
    ap.add_argument("--near-pairs", type=float, default=0.0,
                    help="Revit colour elevations with a side wing: probability the wing is a second family in the "
                         "main wall's pattern but another hue (or hue + spacing). 0.6 on 2026-10-14..14b; default 0 since 2026-10-14c "
                         "(no net effect on real sheets, breaks Gemini30 1)")
    ap.add_argument("--near-pairs-plan", type=float, default=0.3,
                    help="Revit finish floor plans: probability of two floor families differing in ONE attribute "
                         "from --near-pairs-plan-kinds. Default 0.3 since 2026-10-15 (0 on 2026-10-14..14f, "
                         "0.6 on 2026-10-07..13)")
    ap.add_argument("--pair-turns", type=int, default=1,
                    help="1 (default since 2026-10-15b): a plan with a direction pair always turns its boards "
                         "(needs --turn-boards > 0); 0 = 2026-10-15 sheets")
    ap.add_argument("--near-pairs-plan-kinds", default="direction",
                    help="comma list of direction (orthogonal vs diagonal lines), spacing (1.4-2x), tone (colour "
                         "plans). Default direction since 2026-10-15; direction,spacing,tone = 2026-10-07..13")
    ap.add_argument("--balcony-doors", type=float, default=0.7,
                    help="Revit elevations with --railings: probability an upper-floor balcony has a door behind its "
                         "railing (cut from the wall label). Default 0.7 since 2026-10-07; 0 = earlier pools")
    ap.add_argument("--gable-trim", type=float, default=0.5,
                    help="Revit elevations: probability a house with non-accent gables gets a trim board across each "
                         "gable base, the same family above and below. Default 0.5 since 2026-10-14b; 0 = earlier pools")
    ap.add_argument("--turn-boards", type=float, default=0.9,
                    help="Revit floor plans: probability a sheet lays each room of a directional floor family "
                         "(planks / plank lines / joists / 45-deg hatch) in its own direction. Default 0.9 since "
                         "2026-10-15b (0.6 on 2026-10-14d..15); 0 = earlier pools")
    ap.add_argument("--heavy-pieces", type=float, default=0.4,
                    help="Revit elevations: probability a sheet boxes some surface pieces (thin bands more often) in a "
                         "heavy line, labels unchanged. Default 0.4 since 2026-10-14e; 0 = earlier pools")
    ap.add_argument("--gable-pairs", type=float, default=0.0,
                    help="Revit colour elevations with a gable or second storey: probability the gable / upper storey "
                         "is drawn in the main wall's style at another chroma (two families) or, half the time, a "
                         "within-family drift (one family). 0.4 on 2026-10-17 (single18kg); default 0 since 2026-10-17b (breaks Gemini30 1)")
    ap.add_argument("--railings", type=float, default=0.35,
                    help="Revit elevations: probability a view gets a balcony / raised deck railing (half with an "
                         "exterior stair) in front of a wall, the wall label running on behind it, as on HF14 27. "
                         "Default 0.35 since 2026-10-14f (every run passed 0.15 before; HF14 27 q1 0.865 -> 0.928 with "
                         "its railing removed); 0 = byte-identical to pools built without it")
    ap.add_argument("--cedar-shingle", type=float, default=0.5,
                    help="probability a shingle style is drawn as cedar shingles (narrow, vertical joints "
                         "dominant, broken butt lines, keyway tabs) instead of coursed units; real elevations "
                         "use both. Default 0.5 since 2026-10-01; 0 reproduces earlier pools exactly")
    ap.add_argument("--markup", type=float, default=0.0,
                    help="Revit line-only elevations: probability of colour markup (flat see-through fills "
                         "over the finished drawing, labels unchanged); 0 = off")
    ap.add_argument("--colour-pairs", type=float, default=0.0,
                    help="Revit colour elevations: probability of planting a labelled same-texture pair at matched "
                         "lightness but a different hue (brick band under an asphalt roof); 0 = off")
    ap.add_argument("--shaped", action="store_true",
                    help="3D houses beyond boxes: L (notch), U (rear courtyard), chamfered corners, 45-degree bays; "
                         "roofs over convex polygons as the lower envelope of edge slope planes (fc_massing.py). "
                         "Off = byte-identical")
    ap.add_argument("--base-bands", type=float, default=1.0,
                    help="1 = foundation bands / wainscot lines at Gemini's rates (BASE_BANDS); 0 = the 2026-10-20d pools")
    ap.add_argument("--cross-gables", type=float, default=0.4,
                    help="probability a side-gable / hip house gets one or two front cross gables (CROSS_GABLES); "
                         "0 = the 2026-10-17b pools")
    ap.add_argument("--wall-splits", type=float, default=0.3,
                    help="probability the main front / rear wall gets a vertical band in a second material (accent2, "
                         "when free); 0 = the 2026-10-20a pools")
    ap.add_argument("--panel-siding", type=float, default=0.12,
                    help="probability a wall family is drawn as fibre-cement panels with reveal joints; 0 = the 2026-10-20b pools")
    ap.add_argument("--pattern-drift", type=float, default=0.3,
                    help="probability a family's drawn texture gets a smooth 0.6-1.8 px displacement (spacing wanders, lines "
                         "bow, as Gemini draws); 0 = the 2026-10-20c pools")
    ap.add_argument("--gemini-look", type=float, default=0.0,
                    help="probability an elevation sheet is annotated the way Gemini draws one (bigger text, lighter heavy "
                         "outlines, material callouts, overall dimension); 0 = off")
    ap.add_argument("--tight-pad", default="0.015,0.05",
                    help="--tight-crop padding range, fraction of the drawing (Gemini runs to the edge: ~0,0.012)")
    ap.add_argument("--spec", default=None,
                    help="JSON list of specs: render these houses / looks instead of drawing them (see SPEC_BY_ID); "
                         "--n is ignored, one sheet per spec")
    ap.add_argument("--spec-search", type=int, default=50000,
                    help="ids searched per spec for one whose drawn mode / shaded / house match the spec")
    args = ap.parse_args()
    SHAPED[0] = args.shaped
    if args.revit_plans and not args.revit:
        ap.error("--revit-plans needs --revit")
    e, r, f = [float(v) for v in args.mode_weights.split(",")]
    mw = {"elevation": e, "roof_plan": r, "freeform": f}
    flags = dict(R8) if args.r8 else {}
    if args.tight_crop:
        flags["tight_crop_"] = True
    if args.paper:
        flags["paper"] = args.paper
    if args.pale_ink:
        flags["pale_ink"] = args.pale_ink
    if args.revit_view_weights:
        flags["revit_views"] = [float(v) for v in args.revit_view_weights.split(",")]
    if args.real_labelling:
        flags["real_labelling"] = True
    flags["window_hole_prob"] = args.window_hole_prob
    flags["casing_holes"] = bool(args.casing_holes)
    flags["trim_cut"] = bool(args.trim_cut)
    flags["distinct_looks"] = bool(args.distinct_looks)
    if args.colour_pairs > 0:
        flags["colour_pairs"] = args.colour_pairs
    if args.markup > 0:
        flags["markup"] = args.markup
    flags["cedar_shingle"] = args.cedar_shingle
    flags["masonry_base"] = args.masonry_base
    if args.faint_lines > 0:
        flags["faint_lines"] = args.faint_lines
    if args.soft_shadows > 0:
        flags["soft_shadows"] = args.soft_shadows
    if args.roof_lines > 0:
        flags["roof_lines"] = args.roof_lines
    if args.railings > 0:
        flags["railings"] = args.railings
    flags["openings_per_face"] = args.openings_per_face
    flags["cut_standing"] = args.cut_standing
    flags["downspouts"] = args.downspouts
    flags["ref_exclude"] = args.ref_exclude
    flags["near_pairs"] = args.near_pairs
    flags["near_pairs_plan"] = args.near_pairs_plan
    flags["near_pairs_plan_kinds"] = args.near_pairs_plan_kinds
    flags["balcony_doors"] = args.balcony_doors
    flags["gable_trim"] = args.gable_trim
    flags["turn_boards"] = args.turn_boards
    flags["pair_turns"] = args.pair_turns
    flags["heavy_pieces"] = args.heavy_pieces
    flags["gable_pairs"] = args.gable_pairs
    flags["cross_gables"] = args.cross_gables
    flags["base_bands"] = args.base_bands
    BASE_BANDS[0] = args.base_bands
    CROSS_GABLES[0] = args.cross_gables       # the main process draws houses too (house_for, --spec search)
    flags["wall_splits"] = args.wall_splits
    WALL_SPLITS[0] = args.wall_splits
    flags["panel_siding"] = args.panel_siding
    flags["pattern_drift"] = args.pattern_drift
    flags["gemini_look"] = args.gemini_look
    flags["tight_pad"] = tuple(float(v) for v in args.tight_pad.split(","))
    flags["plan_v2"] = args.plan_v2
    if args.plan_source:
        flags["plan_source"] = os.path.abspath(args.plan_source)
        flags["plan_source_p"] = args.plan_source_p
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    ids = list(range(args.start, args.start + args.n))
    spec_by_id = {}
    if args.spec:
        if args.distinct_looks or args.masonry_base or args.roof_lines or args.faint_lines:
            print("--spec: forcing --distinct-looks 0 --masonry-base 0 --roof-lines 0 --faint-lines 0 "
                  "(they restyle families the spec fixes)", flush=True)
        flags["distinct_looks"] = False
        flags["masonry_base"] = 0.0
        flags["panel_siding"] = 0.0             # a spec names its materials
        flags["neutral_palette"] = True         # Gemini's muted colours for families the spec gives no colour
        flags.pop("roof_lines", None)
        flags.pop("faint_lines", None)
        G.TRIM_CUT = flags["trim_cut"]         # the only _init global make_house / make_appearance read
        specs_in = json.load(open(args.spec))
        ids, report, nxt = [], [], args.start
        for sp in specs_in:
            iid, unmet = find_spec_id(sp, args.seed, mw, nxt, args.spec_search)
            ids.append(iid)
            spec_by_id[iid] = sp
            report.append({"name": sp.get("name"), "image_id": iid, "unmet": unmet})
            nxt = iid + 1                      # ids stay unique and increasing
            for k in [k for k in HOUSE_FORCE if k < nxt and k != iid and k not in spec_by_id]:
                del HOUSE_FORCE[k]
            print(f"spec {sp.get('name')}: id {iid}" + (f"  UNMET {unmet}" if unmet else ""), flush=True)
        with open(os.path.join(args.out, "spec_matches.json"), "w") as fo:
            json.dump(report, fo, indent=1)
    t0 = time.time()
    specs = [massing_spec(i, args.seed, house_for(i, args.seed, mw)) for i in ids]
    raw = run_freecad(specs, procs=args.workers)
    faces = {s["id"]: (s, raw[str(s["id"])]) for s in specs if isinstance(raw.get(str(s["id"])), list)}
    print(f"massing {len(faces)}/{len(ids)} in {time.time() - t0:.0f}s", flush=True)
    ok, modes = 0, {}
    with Pool(args.workers, initializer=_init, initargs=(args.out, args.seed, mw, flags, faces, args.revit, args.revit_plans,
                                                                args.shaped, spec_by_id)) as pool:
        for i, (iid, good, info) in enumerate(pool.imap_unordered(_job, ids, chunksize=2)):
            if good:
                ok += 1
                modes[info] = modes.get(info, 0) + 1
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                print(f"{i + 1}/{len(ids)} ok={ok} {time.time() - t0:.0f}s {modes}", flush=True)
    with open(os.path.join(args.out, "generation_manifest.json"), "w") as fo:
        json.dump({"generator": "scripts/generate_synthetic_fc.py", "generator_version": GEN_VERSION,
                   "n": args.n, "seed": args.seed,
                   "start": args.start, "ok": ok, "modes": modes, "mode_weights": mw, "flags": flags, "revit": args.revit,
                   "revit_plans": args.revit_plans, "shaped": args.shaped}, fo, indent=2)


if __name__ == "__main__":
    main()
