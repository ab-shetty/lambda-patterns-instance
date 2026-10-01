"""Revit-style elevation sheets from the FreeCAD 3D model (--revit in generate_synthetic_fc.py).

What makes a Revit elevation read as one, all computed from the fused solids:
  * cast shadows: every face in front of a wall plane projected along the sun
    direction onto it (oblique projection), plus window-recess shadows;
  * line-weight hierarchy: heavy building profile, heavier ground line, thin
    interior edges, faint surface patterns;
  * restrained materials: hidden-line (white surfaces, grey pattern lines) or
    consistent-colours (low-chroma material colours) view styles;
  * annotation: dashed level datums with the quartered target head, view title
    with a number bubble line and scale, occasional material tags -- no trees,
    neighbours or scribbles.
Roof and floor plan sheets fall through to v6 unless --revit-plans (revit_plans.py). Labels follow v6's rules and the
annotation JSON is v6's format, so tight-crop / real-labelling / res-degrade in
v6's _job apply unchanged.
"""
import math
import random

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

G = None      # generate_synthetic_v6
FC = None     # generate_synthetic_fc
_ORIG_COMPOSE = None


PLANS = [False]      # --revit-plans: roof / floor plans drawn by revit_plans.py too

# ---- distinct looks (G.DISTINCT_LOOKS, --distinct-looks 1): two families on one sheet must
# not look the same, and a family with no pattern is not a pattern question.
# Same look = same LOOK class, spacing within 1.4x, colour within CIELAB dE 8 (a light grey
# vs cream lap pair, dE ~10, reads as two materials; white vs white never does).
LOOK = {"vertical": "vlines", "seam": "vlines"}          # seam = vertical lines + a faint twin
SPACING = {"lap": "sp", "bb": "sp", "seam": "sp", "vertical": "sp", "asphalt": "row", "brick": "row",
           "shingle": "row", "tile_roof": "row", "stipple": "density", "concrete": "density"}
KIND_POOL = {"roof": "ROOF_KINDS", "chimney": ["brick", "stone", "stucco"], "foundation": ["concrete", "flat"]}
PRIORITY = ["main", "roof", "accent", "accent2", "chimney", "foundation"]
MASONRY_PALETTE = [(142, 104, 94), (122, 92, 84), (158, 124, 112), (112, 86, 80),        # brick reds
                   (120, 128, 124), (98, 104, 108), (140, 140, 136), (160, 158, 150),    # greys
                   (110, 124, 126), (96, 120, 122), (126, 140, 138),                      # teal-greys
                   (150, 142, 130), (132, 126, 118), (176, 164, 146)]                      # tans / buff


def _lab(c):
    px = np.uint8([[list(c)[::-1]]])       # style colours are RGB
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return L * 100 / 255, a - 128, b - 128


def _same_look(s1, s2):
    if LOOK.get(s1.kind, s1.kind) != LOOK.get(s2.kind, s2.kind):
        return False
    k1, k2 = SPACING.get(s1.kind), SPACING.get(s2.kind)
    if k1 and k2 and s1.params.get(k1) and s2.params.get(k2):
        a, b = s1.params[k1], s2.params[k2]
        if max(a, b) / min(a, b) >= 1.4:
            return False
    return math.dist(_lab(s1.base), _lab(s2.base)) < 8


def _finish_style(st, shaded, app):
    if not shaded:
        st.base = (255, 255, 255)
        st.line = app["ink_fill"]
    else:
        lum = 0.299 * st.base[0] + 0.587 * st.base[1] + 0.114 * st.base[2]
        st.line = G.mix(st.base, (0, 0, 0), 0.35) if lum > 90 else G.mix(st.base, (255, 255, 255), 0.3)
    st.params.pop("grad", None)


def distinct_looks(styles, shaded, app, fam_seed, S_guess):
    """Re-draw the lower-priority family of every look-alike pair with another kind, from its
    own RNG so the sheet's main random stream is untouched."""
    fams = [f for f in PRIORITY if f in styles]
    for j, fb in enumerate(fams):
        for tries in range(12):
            if not any(_same_look(styles[fa], styles[fb]) for fa in fams[:j]):
                break
            rr = random.Random(fam_seed * 31 + j * 1009 + tries)
            pool = KIND_POOL.get(fb, "WALL_KINDS")
            pool = getattr(G, pool) if isinstance(pool, str) else pool
            styles[fb] = G.make_style(rr, rr.choice(pool), app, fam_seed + 50 + j * 7 + tries, S_guess, fb,
                                      roof=(fb == "roof"))
            if fb == "roof" and styles[fb].kind == "flat" and not shaded:
                styles[fb].kind, styles[fb].params = "asphalt", {"row": 0.5, "unit": 1.0}
            _finish_style(styles[fb], shaded, app)


# ---- colour pairs (G.COLOUR_PAIRS, --colour-pairs P): val 17's failure is a red brick band
# under a grey asphalt roof -- the same coursed texture at the same lightness, told apart only
# by hue. Every Revit-trained model selects the roof too at some epochs, and only ~3% of
# labelled pairs in the pool are like it. With probability P a colour sheet gets one: the accent
# (else the chimney) is re-drawn as a coursed kind unlike the roof's, at the roof's lightness
# (dL* <= 6) with a clearly different hue (dE >= 25), and both are labelled. Without an accent
# the chimney takes it if the house has one, else the foundation band.
# RESULT (2026-09-30, 2k ROI screens, seed 7): not adopted. 0.5 looked like a win (HF14 @4096
# +0.032 p=0.01, val 17 brick 0.09 -> 0.46) but 1.0 did not reproduce it (HF14 +0.011, brick
# 0.09-0.15), so it was run noise; the 10k pool fixed val 17's brick (0.89 @4096) without it.
COURSED = {"asphalt": ["brick", "block", "shingle", "lap"], "shingle": ["brick", "block", "lap"],
           "tile_roof": ["brick", "block", "shingle", "lap"], "seam": ["bb", "vertical"]}


def _rgb_from_lab(L, a, b):
    px = np.uint8([[[round(L * 255 / 100), round(a + 128), round(b + 128)]]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0][::-1])


def plant_colour_pair(styles, shaded, app, fam_seed, S_guess, has_chimney):
    rr = random.Random(fam_seed * 131 + 7)       # own stream: the sheet's RNG is untouched
    if not shaded or rr.random() >= G.COLOUR_PAIRS or styles["roof"].kind not in COURSED:
        return None
    # the accent zone if the house has one, else a drawn chimney, else the foundation band
    fam = "accent" if "accent" in styles else "chimney" if has_chimney else "foundation"
    L0, a0, b0 = _lab(styles["roof"].base)
    for _ in range(30):
        h, C = rr.uniform(0, 2 * math.pi), rr.uniform(25, 50)
        rgb = _rgb_from_lab(min(90, max(20, L0 + rr.uniform(-6, 6))), C * math.cos(h), C * math.sin(h))
        lab = _lab(rgb)
        if abs(lab[0] - L0) <= 6 and math.dist(lab, (L0, a0, b0)) >= 25 and \
                all(math.dist(lab, _lab(styles[f].base)) >= 15 for f in styles if f not in (fam, "roof")):
            break
    else:
        return None
    st = G.make_style(rr, rr.choice(COURSED[styles["roof"].kind]), app, fam_seed + 77, S_guess, fam,
                      base_override=rgb)
    _finish_style(st, shaded, app)
    styles[fam] = st
    return fam


# ---- colour markup (G.MARKUP, --markup P): 5 of HF14's 14 sheets are line drawings with
# flat see-through colour painted over them (lavender, green, pink...). Real labels follow the
# TEXTURE, not the paint: a painted plain stucco wall is not a question (HF14 20, 23), an
# unpainted vertical siding is (25, 27), and one paint colour can cover two families. So the
# paint is an overlay the model must see through; labels are unchanged. Own RNG stream.
MARKUP_PALETTE = [(186, 182, 226), (200, 196, 236), (168, 160, 220), (122, 196, 104), (150, 212, 128),
                  (236, 170, 204), (246, 214, 120), (160, 200, 236), (240, 190, 140), (140, 210, 200)]


def markup_plan(styles, mr):
    """Which families get paint, and how, for the whole sheet (one colour per family)."""
    cols = {}
    for f in [f for f in PRIORITY if f in styles]:
        if mr.random() < (0.75 if f == "main" else 0.5):
            used = list(cols.values())
            cols[f] = mr.choice(used) if used and mr.random() < 0.15 else \
                mr.choice([c for c in MARKUP_PALETTE if c not in used] or MARKUP_PALETTE)
    if not cols:
        cols["main"] = mr.choice(MARKUP_PALETTE)
    return {"cols": cols, "alpha": mr.uniform(0.55, 0.95), "skip_open": mr.random() < 0.8,
            "skip_trim": mr.random() < 0.6, "partial": mr.random() < 0.3, "rng": mr,
            "grid": mr.random() < 0.35, "work_box": mr.random() < 0.4}


def markup_sheet_extras(canvas, plan, S):
    """Whole-sheet markup furniture seen on HF14: engineering-grid paper under the drawing
    (18, 19) and dashed blue-grey 'work area' boxes over part of a view (25-27)."""
    rr = plan["rng"]
    H, W = canvas.shape[:2]
    if plan["grid"]:
        step = max(6, int(S * rr.uniform(0.5, 1.2)))
        g = np.zeros((H, W), bool)
        g[::step, :] = True
        g[:, ::step] = True
        paper = canvas.min(axis=2) >= 250
        canvas[g & paper] = rr.randint(215, 235)
    if plan["work_box"]:
        col = rr.choice([(150, 110, 70), (170, 150, 120), (160, 160, 160)])   # BGR blue-grey
        for _ in range(rr.randint(1, 2)):
            x0, y0 = rr.randint(0, W // 2), rr.randint(0, H // 2)
            x1, y1 = rr.randint(x0 + W // 5, W - 1), rr.randint(y0 + H // 5, H - 1)
            dash, lw = max(8, int(S * 0.6)), max(1, int(S * 0.03))
            for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (2 * dash)))
                for k in range(n):
                    t0, t1 = 2 * k / (2 * n), (2 * k + 1) / (2 * n)
                    cv2.line(canvas, (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0)),
                             (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1)), col, lw)


def apply_markup(canvas, V, e, plan, opening_polys, W, H):
    """Multiply-blend the plan's paint over one finished view: ink stays dark, white takes the colour."""
    rr = plan["rng"]
    trims = unary_union(e["trims"]) if e["trims"] else None
    for f, c in plan["cols"].items():
        polys = [p for (fam, p, _, _) in e["surfaces"] if fam == f]
        if not polys:
            continue
        if plan["partial"] and len(polys) > 1:
            polys = [p for p in polys if rr.random() < 0.6] or polys[:1]
        g = unary_union(polys)
        if plan["skip_open"] and opening_polys is not None:
            g = g.difference(opening_polys)
        if plan["skip_trim"] and trims is not None:
            g = g.difference(trims)
        m = np.zeros((H, W), np.uint8)
        for q in G.polys_of(g):
            m = np.maximum(m, G.poly_mask(V.geom(q), W, H))
        a = (m.astype(np.float32) / 255 * plan["alpha"])[..., None]
        tint = np.array(G.bgr(c), np.float32) / 255
        canvas[:] = (canvas.astype(np.float32) * (1 - a + a * tint)).round().clip(0, 255).astype(np.uint8)


def install(g, fc, plans=False):
    global G, FC, _ORIG_COMPOSE
    G, FC = g, fc
    _ORIG_COMPOSE = g.compose
    g.compose = compose
    PLANS[0] = plans
    if plans:
        import revit_plans
        revit_plans.install(g, fc)


def compose(image_id, seed, mode_weights):
    rng = random.Random(seed * 1_000_003 + image_id)
    mode = rng.choices(list(mode_weights.keys()), weights=list(mode_weights.values()))[0]
    if mode != "elevation":
        if PLANS[0]:
            import revit_plans
            return (revit_plans.compose_roof if mode == "roof_plan" else revit_plans.compose_floor)(
                image_id, seed, mode_weights)
        return _ORIG_COMPOSE(image_id, seed, mode_weights)
    return compose_revit(image_id, seed, mode_weights)


# ----------------------------------------------------------------------------
# shadows
# ----------------------------------------------------------------------------
def _clip_front(loop, t_w):
    """Sutherland-Hodgman: keep the part of a (u, v, t) polygon with t < t_w."""
    out = []
    n = len(loop)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        ia, ib = a[2] < t_w, b[2] < t_w
        if ia:
            out.append(a)
        if ia != ib:
            f = (t_w - a[2]) / (b[2] - a[2])
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), t_w))
    return out


def wall_shadows(view, faces, vf, sun):
    """Shadow polygons (view feet) on every frontal wall plane."""
    sx, sy = sun
    loops3 = [[FC._uvt(view, *p) for p in f["loops"][0]] for f in faces if f["loops"]]
    out = []
    for W in vf:
        if W["kind"][0] not in ("wall", "chimney") or not W["frontal"]:
            continue
        t_w = W["plane"][2]
        shade = []
        for lp in loops3:
            if min(p[2] for p in lp) >= t_w - 1e-3:
                continue
            c = _clip_front(lp, t_w - 1e-3)
            if len(c) < 3:
                continue
            pts = [(u + (t_w - t) * sx, v + (t_w - t) * sy) for u, v, t in c]
            try:
                q = Polygon(pts).buffer(0)
            except Exception:
                continue
            if not q.is_empty and q.area > 1e-3:
                shade.append(q)
        if shade:
            sh = unary_union(shade).intersection(W["vis"])
            if not sh.is_empty and sh.area > 0.01:
                out.append(sh)
    return unary_union(out) if out else None


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def _dashed(canvas, a, b, colour, w, dash, gap):
    L = math.dist(a, b)
    if L < 1:
        return
    n = int(L / (dash + gap)) + 1
    for k in range(n):
        t0, t1 = k * (dash + gap) / L, min(1.0, (k * (dash + gap) + dash) / L)
        if t0 >= 1:
            break
        G.cv_line(canvas, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                  (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), colour, w)


def _level_head(canvas, c, r, ink, lw):
    """Revit level head: circle, quarters, two opposite quarters filled."""
    cx, cy = c
    cv2.circle(canvas, G._pt(c), int(r * G.SCALE), G.bgr((255, 255, 255)), -1, cv2.LINE_AA, G.SHIFT)
    for a0 in (180, 0):
        cv2.ellipse(canvas, G._pt(c), (int(r * G.SCALE), int(r * G.SCALE)), 0, a0, a0 + 90, G.bgr(ink), -1,
                    cv2.LINE_AA, G.SHIFT)
    cv2.circle(canvas, G._pt(c), int(r * G.SCALE), G.bgr(ink), max(1, int(lw)), cv2.LINE_AA, G.SHIFT)
    G.cv_line(canvas, (cx - r, cy), (cx + r, cy), ink, lw)
    G.cv_line(canvas, (cx, cy - r), (cx, cy + r), ink, lw)


def _outline_geom(canvas, g, colour, lw):
    for p in G.polys_of(g):
        G.cv_outline(canvas, p, colour, lw)


# ----------------------------------------------------------------------------
# the sheet
# ----------------------------------------------------------------------------
def compose_revit(image_id, seed, mode_weights):
    rng = random.Random(seed * 1_000_003 + image_id)
    rng.choices(list(mode_weights.keys()), weights=list(mode_weights.values()))
    G.make_appearance(rng)
    house = G.make_house(rng)
    r = random.Random(seed * 1_000_003 + image_id * 173 + 41)
    spec, faces = FC._FACES[FC._CUR[0]]

    # ---- view style
    shaded = r.random() < 0.4                       # "consistent colours" vs hidden line
    ink_v = r.randint(0, 30)
    ink = (ink_v,) * 3
    pat_v = r.randint(110, 175) if not shaded else r.randint(60, 120)
    app = {"colour": shaded, "level": "normal", "ink": ink, "outline": ink, "ink_fill": (pat_v,) * 3,
           "paper": (255, 255, 255), "trim": (255, 255, 255), "outline_lw": 1.0}
    app["vd"] = {"glass": r.choice([(255, 255, 255)] * 3 + [(200, 208, 214), (150, 158, 166)]) if not shaded
                 else r.choice([(170, 180, 190), (120, 130, 140), (200, 212, 222)]),
                 "frame": None if not shaded else r.choice([None, (70, 70, 72), (240, 240, 238)]),
                 "panes": r.choice([(1, 1), (2, 1), (2, 2), (1, 2), (3, 2)]),
                 "head": r.random() < 0.35, "swing": r.random() < 0.55, "tags": r.random() < 0.4,
                 "seed": r.randrange(1 << 30), "p_dims": 0.0, "p_tags": 0.0, "p_slope": 0.0, "p_note": 0.0,
                 "door": (255, 255, 255) if not shaded else r.choice([(230, 230, 228), (90, 70, 60), (200, 205, 210)])}
    fam_seed = r.randrange(1 << 30)
    S_guess = 60.0
    styles = {}
    wall_plain = (not shaded) and r.random() < 0.35      # Boise-style: plain white walls
    main_kind = r.choice(G.WALL_KINDS)
    styles["main"] = G.make_style(r, main_kind, app, fam_seed + 1, S_guess, "main")
    if wall_plain:
        styles["main"].kind = "flat"
    if house["accent"]:
        styles["accent"] = G.make_style(r, r.choice([k for k in G.WALL_KINDS if k != main_kind]), app,
                                        fam_seed + 2, S_guess, "accent")
    if house.get("accent2"):
        styles["accent2"] = G.make_style(r, r.choice(G.WALL_KINDS), app, fam_seed + 12, S_guess, "accent2")
    styles["roof"] = G.make_style(r, r.choice(G.ROOF_KINDS), app, fam_seed + 3, S_guess, "roof", roof=True)
    if styles["roof"].kind == "flat" and not shaded:
        styles["roof"].kind, styles["roof"].params = "asphalt", {"row": 0.5, "unit": 1.0}
    styles["chimney"] = G.make_style(r, r.choice(["brick", "stone", "stucco"]), app, fam_seed + 4, S_guess, "chimney")
    styles["foundation"] = G.make_style(r, r.choice(["concrete", "flat"]), app, fam_seed + 5, S_guess, "foundation")
    # --masonry-base P (2026-10-01): real houses often carry a brick / stone / block wainscot
    # at the base of the wall (HF14 18, val 17: a teal running-bond band under grey running-
    # bond roof shingles). The Revit base band was only ever concrete or flat, so that pair --
    # two coursed textures a tint apart -- never occurred. Own RNG: P = 0 is byte-identical.
    masonry_base = None
    if getattr(G, "MASONRY_BASE", 0) > 0:
        mr = random.Random(fam_seed * 97 + 13)
        if mr.random() < G.MASONRY_BASE:
            masonry_base = mr
            # real masonry is muted: brick reds, greys, teal-greys, tans (RGB)
            styles["foundation"] = G.make_style(mr, mr.choice(["brick", "brick", "stone", "block"]), app,
                                                fam_seed + 5, S_guess, "foundation",
                                                base_override=mr.choice(MASONRY_PALETTE))
    for st in styles.values():
        if not shaded:
            st.base = (255, 255, 255)
            st.line = app["ink_fill"]
        else:
            lum = 0.299 * st.base[0] + 0.587 * st.base[1] + 0.114 * st.base[2]
            st.line = G.mix(st.base, (0, 0, 0), 0.35) if lum > 90 else G.mix(st.base, (255, 255, 255), 0.3)
        st.params.pop("grad", None)
    if G.DISTINCT_LOOKS:
        distinct_looks(styles, shaded, app, fam_seed, S_guess)
    pair_fam = plant_colour_pair(styles, shaded, app, fam_seed, S_guess, bool(spec.get("chimney"))) \
        if G.COLOUR_PAIRS > 0 else None

    label_fams = {"main"}
    for f in ("accent", "accent2"):
        if f in styles and r.random() < 0.97:
            label_fams.add(f)
    if not shaded:
        for f in ("main", "accent", "accent2"):
            if f in styles and styles[f].kind == "flat":
                label_fams.discard(f)
    if r.random() < 0.9:
        label_fams.add("roof")
    if r.random() < 0.7:
        label_fams.add("chimney")
    if r.random() < 0.4 and styles["foundation"].kind != "flat":
        label_fams.add("foundation")
    # a masonry wainscot is a prominent material real annotators label (HF14 18, val 17):
    # labelled ~80% overall (0.4 above, plus 2/3 of the rest from its own stream)
    if masonry_base is not None and masonry_base.random() < 0.67:
        label_fams.add("foundation")
    if G.DISTINCT_LOOKS:        # plain colour, no pattern: drawn, never a pattern question
        label_fams = {f for f in label_fams if styles.get(f) is None or styles[f].kind != "flat"}
    if pair_fam:                # the planted pair is always a question
        label_fams |= {pair_fam, "roof"}
    hole_mode = r.random() < G.WINDOW_HOLE_PROB
    mk = None
    if getattr(G, "MARKUP", 0) > 0 and not shaded:
        mr = random.Random(seed * 1_000_003 + image_id * 191 + 53)
        if mr.random() < G.MARKUP:
            mk = markup_plan(styles, mr)

    # ---- views
    nviews = r.choices([1, 2, 4], weights=[30, 40, 30])[0]
    allv = ["front", "rear", "left", "right"]
    views = (["front"] if r.random() < 0.6 else [r.choice(allv)]) if nviews == 1 else \
        r.choice([["front", "rear"], ["left", "right"], ["front", "right"]]) if nviews == 2 else allv
    tstyle = r.choice(["building", "plain", "compass"])
    names = {"front": "FRONT", "rear": "REAR", "left": "LEFT", "right": "RIGHT"}
    comp = {"front": "SOUTH", "rear": "NORTH", "left": "WEST", "right": "EAST"}
    elevs = []
    for v in views:
        e = FC.build_elevation_fc(house, v, r)
        t = {"building": f"{names[v]} BUILDING ELEVATION", "plain": f"{names[v]} ELEVATION",
             "compass": f"{comp[v]} ELEVATION"}[tstyle]
        elevs.append((v, e, t))

    # ---- layout (feet): room left for level heads, below for titles
    left_room, right_room, top_room, bot_room = r.uniform(7, 10), r.uniform(2, 4), r.uniform(3, 5), r.uniform(7, 9)
    gap = r.uniform(4, 9)
    cells = []
    for (v, e, t) in elevs:
        x0, y0, x1, y1 = e["extent"]
        cells.append((x1 - x0 + left_room + right_room, y1 - y0 + top_room + bot_room))
    if len(elevs) <= 2:
        cols = len(elevs)
    else:
        cols = 2
    rows = math.ceil(len(elevs) / cols)
    cw = [max(cells[i][0] for i in range(len(cells)) if i % cols == c) for c in range(cols)]
    rh = [max(cells[i][1] for i in range(len(cells)) if i // cols == rr) for rr in range(rows)]
    margin = r.uniform(1.5, 4)
    total_w = margin * 2 + sum(cw) + gap * (cols - 1)
    total_h = margin * 2 + sum(rh) + gap * (rows - 1)
    S = min(130.0, r.uniform(*G.LONG_SIDE_PX) / max(total_w, total_h))
    W, H = int(total_w * S), int(total_h * S)
    canvas = np.full((H, W, 3), 255, np.uint8)
    tq = G.TextQueue()
    app["text_px"] = max(12, S * r.uniform(0.28, 0.4))
    lw_thin = max(1.0, S * r.uniform(0.010, 0.016))
    lw_heavy = max(2.0, S * r.uniform(0.03, 0.045))
    lw_ground = lw_heavy * r.uniform(1.3, 1.8)
    app["outline_lw"] = lw_thin
    for st in styles.values():
        st.lw = max(0.8, lw_thin * r.uniform(0.6, 0.9))
    sun = (r.choice([1, -1]) * r.uniform(0.6, 1.2), r.uniform(0.9, 1.7))
    shadow_k = r.uniform(0.55, 0.75) if not shaded else r.uniform(0.55, 0.7)
    shadows_on = r.random() < 0.85
    labelled = []
    num = 1
    for i, (v, e, title) in enumerate(elevs):
        col, row = i % cols, i // cols
        cx = margin + sum(cw[:col]) + gap * col
        cy = margin + sum(rh[:row]) + gap * row
        x0, y0, x1, y1 = e["extent"]
        V = G.View(S, (cx + left_room - x0) * S, (cy + top_room - y0) * S)
        opening_polys = unary_union([o["poly"] for o in e["openings"]]) if e["openings"] else None
        label_holes = unary_union([G.label_hole(o, app) for o in e["openings"]]) if e["openings"] else None
        trim_cut = unary_union(e["trims"]) if (G.TRIM_CUT and e["trims"]) else None
        # surfaces: pattern fills
        for (fam, poly, bi, kind) in e["surfaces"]:
            p = poly.difference(opening_polys) if opening_polys is not None else poly
            if p.is_empty:
                continue
            G.draw_fill(canvas, V.geom(p), styles[fam], S, W, H)
            if fam in label_fams:
                lab = poly.difference(label_holes) if (hole_mode and opening_polys is not None) else poly
                if trim_cut is not None:
                    lab = lab.difference(trim_cut)
                for q in G.polys_of(lab):
                    if G.TRIM_CUT and G._thin(q):
                        continue
                    labelled.append((fam, V.geom(q)))
        # trims: white boards
        for tp in e["trims"]:
            G.cv_fill(canvas, V.geom(tp), (255, 255, 255) if not shaded else (242, 242, 240))
        # cast shadows (walls) + window recesses
        if shadows_on:
            sh = wall_shadows(v, faces, e["vf"], sun)
            recess = r.uniform(0.25, 0.5)
            rs = []
            for o in e["openings"]:
                p = o["poly"]
                rs.append(p.difference(affinity.translate(p, sun[0] * recess, sun[1] * recess)))
            parts = ([sh] if sh is not None else []) + rs
            if parts:
                m = np.zeros((H, W), np.uint8)
                for q in G.polys_of(unary_union(parts)):
                    m = np.maximum(m, G.poly_mask(V.geom(q), W, H))
                sel = m > 127
                canvas[sel] = (canvas[sel].astype(np.float32) * shadow_k).astype(np.uint8)
        # openings (frames, glass, sills, swing lines) over the shadow, recess shadow re-applied on glass
        for o in e["openings"]:
            G.draw_opening(canvas, V, o, house, app, S, r, (255, 255, 255))
        if shadows_on:
            m = np.zeros((H, W), np.uint8)
            for o in e["openings"]:
                p = o["poly"]
                q = p.difference(affinity.translate(p, sun[0] * recess, sun[1] * recess))
                for qq in G.polys_of(q):
                    m = np.maximum(m, G.poly_mask(V.geom(qq), W, H))
            sel = m > 127
            canvas[sel] = (canvas[sel].astype(np.float32) * shadow_k).astype(np.uint8)
        # thin edges: every visible face boundary and trim
        for (fam, poly, bi, kind) in e["surfaces"]:
            _outline_geom(canvas, V.geom(poly), ink, lw_thin)
        for tp in e["trims"]:
            _outline_geom(canvas, V.geom(tp), ink, lw_thin)
        if getattr(FC, "SHAPED", [False])[0]:
            # --shaped: pieces of one v6 block (L / U / bay) are zoned as one surface, so draw
            # every visible 3D wall face's edge too -- the step where a wall sets back reads
            for Wf in e["vf"]:
                if Wf["kind"][0] in ("wall", "chimney"):
                    _outline_geom(canvas, V.geom(Wf["vis"]), ink, lw_thin)
        # heavy profile
        sil = unary_union([p for (_, p, _, _) in e["surfaces"]] + list(e["trims"])).buffer(0.02)
        for p in G.polys_of(sil):
            G.cv_polyline(canvas, list(V.geom(p).exterior.coords), ink, lw_heavy, closed=True)
        if mk is not None:
            apply_markup(canvas, V, e, mk, opening_polys, W, H)
        # ground line
        ga, gb = V.px(x0 - r.uniform(1, 4), 0), V.px(x1 + r.uniform(1, 4), 0)
        G.cv_line(canvas, ga, gb, ink, lw_ground)
        # level datums
        fh, plate = house["fh"], house["plate"]
        B0 = house["blocks"][0]
        levels = [("LEVEL 01", 0.0)]
        if B0.stories >= 2:
            levels.append(("LEVEL 02", fh))
        levels.append((r.choice(["T.O. PLATE", "LEVEL 01 TP", "TOP OF PLATE"]), B0.stories * fh + plate))
        if r.random() < 0.5:
            levels.append(("GRADE", -0.5))
        if r.random() < 0.5 and y0 < -(B0.stories * fh + plate) - 2:
            levels.append(("ROOF", -y0))
        if r.random() < 0.9:
            hx = x0 - left_room + 1.2
            dcol = G.mix(ink, (255, 255, 255), 0.45)
            for (name, z) in levels:
                a, b = V.px(hx + 0.6, -z), V.px(x1 + r.uniform(0.5, 2), -z)
                _dashed(canvas, a, b, dcol, lw_thin * 0.8, 0.9 * S, 0.35 * S)
                c = V.px(hx, -z)
                _level_head(canvas, c, 0.32 * S, ink, lw_thin)
                sz = app["text_px"] * 0.95
                tq.add((c[0] + 0.55 * S, c[1] - 0.35 * S), name, sz, ink, bold=True, anchor="ls")
                tq.add((c[0] + 0.55 * S, c[1] + 0.15 * S), G.ft_label(abs(z)) if z >= 0 else "-" + G.ft_label(-z),
                       sz * 0.8, ink, anchor="la")
        # material tags
        if r.random() < 0.45:
            for fam in ("main", "roof", "accent"):
                cand = [p for (f, p, _, _) in e["surfaces"] if f == fam and p.area > 8]
                if cand and fam in styles and r.random() < 0.6:
                    pt = max(cand, key=lambda q: q.area).representative_point()
                    side = 1 if pt.x > (x0 + x1) / 2 else -1
                    G.draw_leader(canvas, tq, V, (pt.x, pt.y), styles[fam].callout, app, S, side)
        # view title: number, bold name, rule, scale
        ty = V.px(x0, 0)[1] + r.uniform(2.8, 4.2) * S
        tx = V.px(x0 - left_room + 1.0, 0)[0]
        big = app["text_px"] * 1.9
        tq.add((tx, ty), str(num), big, ink, bold=True, anchor="lm")
        tq.add((tx + big * 1.2, ty - big * 0.12), title, app["text_px"] * 1.15, ink, bold=True, anchor="ls")
        G.cv_line(canvas, (tx + big * 1.2, ty + big * 0.05), (V.px(x1, 0)[0], ty + big * 0.05), ink, lw_thin)
        tq.add((tx + big * 1.2, ty + big * 0.25), "SCALE: " + r.choice(['1/4" = 1\'-0"', '1/8" = 1\'-0"', '3/16" = 1\'-0"']),
               app["text_px"] * 0.75, ink, anchor="la")
        num += 1
    canvas = tq.flush(canvas)
    if mk is not None:
        markup_sheet_extras(canvas, mk, S)
    if r.random() < G.JPEG_PROB:
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, r.randint(60, 92)])
        if ok:
            canvas = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # ---- annotations (v6 format)
    fam_ids, anns = {}, []
    min_area = max(400, (0.02 * S) ** 2 * 100)
    for fam, p in labelled:
        for q in G.polys_of(p.buffer(0)):
            if q.area < min_area:
                continue
            fam_ids.setdefault(fam, len(fam_ids) + 1)
            q = q.simplify(0.5)
            if q.is_empty or not isinstance(q, Polygon):
                continue
            outer = [round(v, 2) for c in q.exterior.coords[:-1] for v in c]
            holes = [[round(v, 2) for c in ring.coords[:-1] for v in c] for ring in q.interiors]
            holes = [hh for hh in holes if len(hh) >= 6]
            bx0, by0, bx1, by1 = q.bounds
            anns.append({"id": len(anns) + 1, "category_id": fam_ids[fam], "category_name": f"pattern{fam_ids[fam]}",
                         "family": fam, "segmentation": [outer] + holes, "num_holes": len(holes),
                         "bbox": [round(bx0, 1), round(by0, 1), round(bx1 - bx0, 1), round(by1 - by0, 1)],
                         "area": round(q.area, 1)})
    return canvas, {"image": {"file_name": f"synth6_{image_id:06d}.png", "width": W, "height": H},
                    "mode": "elevation", "appearance": "colour" if shaded else "markup" if mk is not None else "mono_normal",
                    "px_per_ft": round(S, 2), "annotations": anns, "render": "revit"}
