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
import copy
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
    px = np.uint8([[[round(L * 255 / 100), round(min(127, max(-128, a)) + 128), round(min(127, max(-128, b)) + 128)]]])
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


# ---- near-miss pairs (FC.NEAR_PAIRS, --near-pairs P): two labelled families on one colour elevation, the
# main wall (A) and a side wing beside it (B), drawn in the SAME line pattern and pen weight but a different
# colour -- Gemini r4 #078 (tan vs olive lap, 2 families) is the case the synthetic model merges.
#   hue          fill and pen moved to another hue: CIEDE2000 10-25 between the fills, lightness within +-3,
#                hue angle >= 20 deg apart (or, from a near-grey A, chroma >= 15);
#   hue+spacing  the same at CIEDE2000 7-20, plus spacing 1.4-2x apart (#078 is tan vs olive at 1.8x:
#                CIEDE2000 8.9 and denser lines).
# History. 2026-10-07: direction / spacing / tone pairs, B on the roof (61%) or an upper-storey / wing accent;
# the screen lost val -0.100. 2026-10-13: tone made hue-only (lightness-only pairs were a cast shadow) -- the
# screen still lost and split single real families along exactly the planted boundaries (gables, upper
# storeys, wings, rooms: synth_progress.md "2026-10-13 (CPU) -- reading the hue screen"), and the colour gap,
# measured as CIELAB dE76, let near-identical pairs through (sage vs lime: dE76 16, CIEDE2000 5.9).
# Now (2026-10-14): colour pairs only, colour sheets only, B always a side wing -- the same storeys, the
# same height, beside A -- so position does not give the answer; direction / spacing-only pairs dropped.
# The pair is decided before the geometry (compose_revit forces the wing accent). Own RNG: P = 0 is
# byte-identical to pools without pairs.
LINE_KINDS = ("lap", "vertical", "bb")


def _copy_style(st, fam):
    out = copy.copy(st)          # every attribute (callout, ...), params copied so the two stay apart
    out.params, out.label_name = dict(st.params), fam
    return out


def _callout(st, rr):
    st.callout = rr.choice(G.CALLOUT.get(st.kind, ["FINISH"]))


def _de2000(l1, l2):
    """CIEDE2000 between two CIELAB colours (perceived difference; dE76 overrates chroma steps in saturated colours)."""
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    Cb = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1, a2 = a1 * (1 + g), a2 * (1 + g)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1, h2 = math.degrees(math.atan2(b1, a1)) % 360, math.degrees(math.atan2(b2, a2)) % 360
    dh = 0.0 if C1 * C2 == 0 else (h2 - h1 + 180) % 360 - 180
    dH = 2 * math.sqrt(C1 * C2) * math.sin(math.radians(dh / 2))
    Lb, Cp = (L1 + L2) / 2, (C1 + C2) / 2
    hb = h1 + h2 if C1 * C2 == 0 else ((h1 + h2) / 2 if abs(h1 - h2) <= 180 else (h1 + h2 + 360 * (1 if h1 + h2 < 360 else -1)) / 2)
    T = (1 - 0.17 * math.cos(math.radians(hb - 30)) + 0.24 * math.cos(math.radians(2 * hb))
         + 0.32 * math.cos(math.radians(3 * hb + 6)) - 0.2 * math.cos(math.radians(4 * hb - 63)))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / math.sqrt(20 + (Lb - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cp, 1 + 0.015 * Cp * T
    Rt = -math.sin(math.radians(60 * math.exp(-((hb - 275) / 25) ** 2))) * 2 * math.sqrt(Cp ** 7 / (Cp ** 7 + 25 ** 7))
    x, y, z = (L2 - L1) / Sl, (C2 - C1) / Sc, dH / Sh
    return math.sqrt(x * x + y * y + z * z + Rt * y * z)


def _shift_colour(rgb, rr, lo, hi):
    """Another HUE at CIEDE2000 lo..hi from rgb, lightness within +-3: hue angle >= 20 deg away, or (from a
    near-grey colour) chroma >= 15. Returns (colour, (da, db)) or None."""
    L0, a0, b0 = _lab(rgb)
    C0, h0 = math.hypot(a0, b0), math.degrees(math.atan2(b0, a0))
    for _ in range(80):
        C = rr.uniform(max(15.0, C0 * 0.5), max(30.0, C0 * 1.4))
        h = math.radians(h0 + rr.choice([-1, 1]) * rr.uniform(20, 150)) if C0 >= 8 else rr.uniform(0, 2 * math.pi)
        c = _rgb_from_lab(min(92, max(25, L0 + rr.uniform(-3, 3))), C * math.cos(h), C * math.sin(h))
        L1, a1, b1 = _lab(c)
        dh = abs((math.degrees(math.atan2(b1, a1)) - h0 + 180) % 360 - 180)
        if abs(L1 - L0) <= 3.5 and lo <= _de2000((L0, a0, b0), (L1, a1, b1)) <= hi and \
                (dh >= 20 if C0 >= 8 else math.hypot(a1, b1) >= 15):
            return c, (a1 - a0, b1 - b0)
    return None


def _move_ab(rgb, da, db):
    L, a, b = _lab(rgb)
    return _rgb_from_lab(L, a + da, b + db)


def near_pair_planned(seed, image_id, shaded, house):
    """Decided before the geometry: a colour sheet whose house has a side wing gets the pair with probability
    P, and the wing is then made the accent zone (compose_revit). Own RNG stream."""
    if FC.NEAR_PAIRS[0] <= 0 or not shaded or not any(b.kind != "main" for b in house["blocks"]):
        return False
    return random.Random(seed * 1_000_003 + image_id * 197 + 89).random() < FC.NEAR_PAIRS[0]


# ---- gable tone pairs (FC.GABLE_PAIRS, --gable-pairs P; 0.4 on 2026-10-17, default 0 since 2026-10-17b: single18kg lifted
# Gemini30 26 0.48 -> 0.59 but split one-family sheets anyway, Gemini30 1 0.65 -> 0.34, val 9 -0.06 -- the twins did not protect them): Gemini30 26 (r4 #078) is one
# lap siding at one spacing (12.5 vs 13-14 px), tan on the storeys and olive on the gable and dormer, two families:
# the olive is the tan desaturated (b* 31 -> 16, hue angle 6 deg apart, L* -3.5), CIEDE2000 8.5. No synthetic pair
# looks like it: wing pairs need a hue angle >= 20 deg, sit on side wings only, and reach 7-9 only with a spacing
# change, so every model merges the two (26 ~0.48, the largest per-sheet loss on Gemini30). Here a colour elevation
# whose house has a gable (else a second storey) gets, with probability P, the gable (or upper storey) as the accent
# zone in the main wall's style, and half of those are
#   tone  the accent's fill moved in chroma (x0.35-0.7 or x1.4-2.4), hue +-15 deg, L* +-4, CIEDE2000 6-12:
#         labelled as two families;
#   twin  the same zone and trim band, a within-family drift (CIEDE2000 0.4-2, as Gemini30 1's pieces differ by
#         0.3-1.6): labelled as ONE family.
# The twin is the point: the 2026-10-07 / 13 pairs (B on gables / upper storeys, no twin) taught "another zone = another
# family" and split one-family sheets at exactly those boundaries (Gemini30 1, 5, 14), and the wing pairs still broke
# Gemini30 1. With twins the zone carries no information; only the colour gap does (real within-family piece gaps:
# 85% < 3; Gemini30 98% < 3). Not with a wing pair on the same sheet. Own RNG streams: P = 0 is byte-identical.
def gable_pair_planned(seed, image_id, shaded, house, near_on):
    """Decided before the geometry: (zone, "tone" | "twin") or None; compose_revit makes the zone the accent."""
    if getattr(FC, "GABLE_PAIRS", [0.0])[0] <= 0 or not shaded or near_on:
        return None
    gable = any(b.roof == "gable" for b in house["blocks"])
    upper = house["blocks"][0].stories >= 2
    gr = random.Random(seed * 1_000_003 + image_id * 211 + 97)
    if not (gable or upper) or gr.random() >= FC.GABLE_PAIRS[0]:
        return None
    zone = "gable" if gable and (not upper or gr.random() < 0.6) else "upper"
    return zone, "twin" if gr.random() < 0.5 else "tone"


def _tone_shift(rgb, rr, lo, hi, twin):
    """rgb with its chroma (and a little hue / lightness) moved, CIEDE2000 lo..hi; twin = a small drift on all
    three. From a near-grey (chroma < 8) a tone pair takes chroma 8-20 at any hue. Returns (colour, CIEDE2000) or None."""
    L0, a0, b0 = _lab(rgb)
    C0, h0 = math.hypot(a0, b0), math.degrees(math.atan2(b0, a0))
    for _ in range(120):
        if twin:
            L, C, h = L0 + rr.uniform(-1.5, 1.5), C0 * rr.uniform(0.85, 1.15), h0 + rr.uniform(-5, 5)
        elif C0 >= 8:
            f = rr.uniform(0.35, 0.7) if rr.random() < 0.6 else rr.uniform(1.4, 2.4)
            L, C, h = L0 + rr.uniform(-4, 4), C0 * f, h0 + rr.uniform(-15, 15)
        else:
            L, C, h = L0 + rr.uniform(-4, 4), rr.uniform(8, 20), rr.uniform(0, 360)
        c = _rgb_from_lab(min(92, max(25, L)), C * math.cos(math.radians(h)), C * math.sin(math.radians(h)))
        de = _de2000((L0, a0, b0), _lab(c))
        if lo <= de <= hi:
            return c, de
    return None


def plant_gable_pair(styles, fam_seed, what):
    A = styles["main"]
    if A.kind == "flat":
        return None
    rr = random.Random(fam_seed * 179 + 67)       # own stream: the sheet's RNG is untouched
    got = _tone_shift(A.base, rr, *((0.4, 2.0) if what == "twin" else (6, 12)), twin=what == "twin")
    if got is None:
        return None
    B = _copy_style(A, "accent")
    B.base = got[0]
    _finish_style(B, True, None)                 # the pen follows the fill, as for every colour family
    styles["accent"] = B
    return what, round(got[1], 1)


def plant_near_pair(styles, fam_seed):
    rr = random.Random(fam_seed * 173 + 61)      # own stream: the sheet's RNG is untouched
    A = styles["main"]
    if A.kind not in LINE_KINDS:
        A.kind = rr.choice(["lap", "vertical"])
        A.params = {"sp": rr.choice([0.33, 0.42, 0.5, 0.58, 0.67])}
        _callout(A, rr)
    A.params["sp"] = A.params.get("sp", 0.5)
    if A.kind == "lap":
        A.params["shadow"] = False          # plain lines: the pair differs in the planted attributes only
    what = rr.choice(["hue", "hue+spacing"])
    got = _shift_colour(A.base, rr, *((10, 25) if what == "hue" else (7, 20)))
    if got is None:
        return None
    B = _copy_style(A, "accent")
    B.base, (da, db) = got
    B.line = _move_ab(A.line, da, db)          # same pen lightness
    if what == "hue+spacing":
        f, sp = rr.uniform(1.4, 2.0), A.params["sp"]
        B.params["sp"] = sp * f if (sp * f <= 1.4 and rr.random() < 0.5) or sp / f < 0.22 else sp / f
    styles["accent"] = B
    return "main", "accent", what


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
    near_on = near_pair_planned(seed, image_id, shaded, house)
    if near_on:                                     # the pair's B is a side wing: make the wing the accent zone
        house["accent"] = "block"
    gp = gable_pair_planned(seed, image_id, shaded, house, near_on)
    if gp:                                          # the gable (or upper storey) is the accent zone
        house["accent"] = gp[0]
        if house.get("accent2") in ("gable", gp[0]):
            house["accent2"] = None
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
    # --roof-lines P (2026-10-02): HF14 25 / 27 draw the roof in elevation as plain, tight
    # horizontal course lines (no joints) over walls of looser vertical boards; our roofs were
    # asphalt / shingle / tile rows with joints, or standing seam -- VERTICAL lines -- so a
    # horizontal-lines roof never met vertical siding, and the synthetic-only model merges the two
    # (25 q02 0.986 shipped -> 0.719, 27 q04 0.977 -> 0.563). P of the sheets: the roof is plain
    # horizontal lines at 0.25-0.45 ft; 60% of those also get a vertical-board wall at >= 1.6x the
    # spacing. Any lap wall is kept >= 1.6x apart from the roof. Own RNG: P = 0 is byte-identical.
    if getattr(G, "ROOF_LINES", 0) > 0:
        rl = random.Random(fam_seed * 71 + 43)
        if rl.random() < G.ROOF_LINES:
            ro = G.make_style(rl, "lap", app, fam_seed + 3, S_guess, "roof", roof=True)
            ro.params = {"sp": rl.uniform(0.25, 0.45), "shadow": False}
            _finish_style(ro, shaded, app)
            styles["roof"] = ro
            if rl.random() < 0.6:
                fam = "main" if styles["main"].kind != "flat" else ("accent" if "accent" in styles else None)
                if fam is not None:
                    vk = rl.choice(["vertical", "bb", "bb"])
                    wv = G.make_style(rl, vk, app, fam_seed + 1, S_guess, fam)
                    wv.params["sp"] = max(wv.params["sp"], ro.params["sp"] * rl.uniform(1.6, 2.6))
                    _finish_style(wv, shaded, app)
                    styles[fam] = wv
            for f, st in styles.items():
                if f != "roof" and st.kind == "lap" and max(st.params["sp"], ro.params["sp"]) / \
                        min(st.params["sp"], ro.params["sp"]) < 1.6:
                    st.params["sp"] = ro.params["sp"] * 1.6 if st.params["sp"] >= ro.params["sp"] else \
                        ro.params["sp"] / 1.6
    # --faint-lines P (2026-10-02): HF14 14 draws its textures in barely-there grey (brick tile
    # lines ~240 on white paper, stucco left blank); line-only Revit sheets only ever drew pattern
    # lines at 110-175, and the synthetic-only model takes the faint tile for the blank stucco
    # beside it (14 q06 0.987 shipped -> 0.555). P of the line-only sheets draw every line pattern in
    # a pale grey, a little apart per family, in a finer pen; dot patterns (stucco, concrete) keep
    # their weight -- faint sparse dots read as blank paper. Own RNG: P = 0 is byte-identical.
    if getattr(G, "FAINT_LINES", 0) > 0 and not shaded:
        fr = random.Random(fam_seed * 53 + 29)
        if fr.random() < G.FAINT_LINES:
            fv = fr.randint(190, 228)
            k_lw = fr.uniform(0.55, 0.9)
            for st in styles.values():
                v = max(170, min(236, fv + fr.randint(-8, 8)))
                if st.kind in ("stipple", "stucco", "concrete", "dots"):
                    continue        # sparse pale dots vanish at 2048: a labelled wall that reads blank
                st.line, st.lw = (v,) * 3, max(1.0, st.lw * k_lw)
    near = plant_near_pair(styles, fam_seed) if near_on else None
    gpair = plant_gable_pair(styles, fam_seed, gp[1]) if gp and "accent" in styles else None

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
    if near:                    # so is a near-miss pair: both families labelled, as two
        label_fams |= set(near[:2])
    relabel = {}
    if gpair:                   # a tone pair is two families; a twin's gable is the main family
        label_fams |= {"main", "accent"}
        if gpair[0] == "twin":
            relabel["accent"] = "main"
    hole_mode = r.random() < G.WINDOW_HOLE_PROB
    mk = None
    if getattr(G, "MARKUP", 0) > 0 and not shaded:
        mr = random.Random(seed * 1_000_003 + image_id * 191 + 53)
        if mr.random() < G.MARKUP:
            mk = markup_plan(styles, mr)

    # ---- views
    nviews = r.choices([1, 2, 4], weights=getattr(G, "REVIT_VIEWS", None) or [30, 40, 30])[0]
    allv = ["front", "rear", "left", "right"]
    views = (["front"] if r.random() < 0.6 else [r.choice(allv)]) if nviews == 1 else \
        r.choice([["front", "rear"], ["left", "right"], ["front", "right"]]) if nviews == 2 else allv
    spv = getattr(FC, "SPEC_BY_ID", {}).get(FC._CUR[0], {}).get("view_list")
    if spv:                                    # --spec: the views the described sheet shows (after the draws above)
        views = [v for v in spv if v in allv][:4] or views
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
    sp_aspect = getattr(FC, "SPEC_BY_ID", {}).get(FC._CUR[0], {}).get("sheet_aspect")
    if sp_aspect:                              # --spec: the column count whose sheet proportions are closest to the original's
        def _aspect(c):
            cwk = [max(cells[i][0] for i in range(len(cells)) if i % c == k) for k in range(c)]
            rhk = [max(cells[i][1] for i in range(len(cells)) if i // c == k) for k in range(math.ceil(len(cells) / c))]
            return (sum(cwk) + gap * (c - 1)) / (sum(rhk) + gap * (len(rhk) - 1))
        cols = min((c for c in (1, 2, 4) if c <= len(elevs) and len(elevs) % c == 0),
                   key=lambda c: abs(math.log(_aspect(c) / sp_aspect)))
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
    hp = random.Random(fam_seed * 241 + 113)              # --heavy-pieces: own stream
    heavy_on = hp.random() < getattr(FC, "HEAVY_PIECES", [0.0])[0]
    app["outline_lw"] = lw_thin
    # --gemini-look P (generate_synthetic_fc, 2026-10-20e): sheets drawn the way Gemini draws an elevation -- legible text
    # (1.6-2.4x), lighter heavy outlines (x0.45-0.7), material callouts with leaders on most families, an overall-width
    # dimension string. Own stream, applied after this sheet's draws: P = 0 = earlier pools.
    glr = random.Random(fam_seed * 389 + 211)
    glook = getattr(FC, "GEMINI_LOOK", [0.0])[0] > 0 and glr.random() < FC.GEMINI_LOOK[0]
    if glook:
        app["text_px"] *= glr.uniform(1.6, 2.4)
        kh = glr.uniform(0.45, 0.7)
        lw_heavy, lw_ground = max(1.5, lw_heavy * kh), max(2.0, lw_ground * kh)
    for st in styles.values():
        st.lw = max(0.8, lw_thin * r.uniform(0.6, 0.9))
    sun = (r.choice([1, -1]) * r.uniform(0.6, 1.2), r.uniform(0.9, 1.7))
    shadow_k = r.uniform(0.55, 0.75) if not shaded else r.uniform(0.55, 0.7)
    shadows_on = r.random() < 0.85
    # --soft-shadows P (2026-10-02): HF14 02's porch is cream lap siding (246, 242, 230) that
    # turns a lighter, NEUTRAL grey in shade (199, 200, 195: x0.81, tint gone) with a darker
    # second level where shadows overlap (167, 162, 156: x0.68); ours were one x0.55-0.75
    # multiply that keeps the hue, and the synthetic-only model misses the shaded siding
    # (02 q00 0.888 shipped -> 0.729). P of the sheets: shade x0.76-0.88 toward grey, plus a
    # second sun a little off the first whose overlap is the dark level. Own RNG: 0 = identical.
    soft = None
    if getattr(G, "SOFT_SHADOWS", 0) > 0:
        sr = random.Random(fam_seed * 61 + 37)
        if sr.random() < G.SOFT_SHADOWS:
            sun2 = (sun[0] * sr.uniform(0.4, 0.8) + sr.uniform(-0.4, 0.4), sun[1] * sr.uniform(1.2, 1.8))
            soft = {"k1": sr.uniform(0.76, 0.88), "k2": sr.uniform(0.62, 0.75), "desat": sr.uniform(0.3, 0.7),
                    "sun2": sun2, "two": shaded and sr.random() < 0.8}

    def _shade(sel, k):
        px = canvas[sel].astype(np.float32)
        if soft is not None:
            lum = px @ np.float32([0.114, 0.587, 0.299])          # canvas is BGR
            px = px * (1 - soft["desat"]) + lum[:, None] * soft["desat"]
        canvas[sel] = np.clip(px * k, 0, 255).astype(np.uint8)
    labelled = []
    excl_px = []        # --ref-exclude: posts, railings, stairs, downspouts (inside the labels)
    num = 1
    for i, (v, e, title) in enumerate(elevs):
        col, row = i % cols, i // cols
        cx = margin + sum(cw[:col]) + gap * col
        cy = margin + sum(rh[:row]) + gap * row
        x0, y0, x1, y1 = e["extent"]
        V = G.View(S, (cx + left_room - x0) * S, (cy + top_room - y0) * S)
        opening_polys = unary_union([o["poly"] for o in e["openings"]]) if e["openings"] else None
        label_holes = unary_union([G.label_hole(o, app) for o in e["openings"]]) if e["openings"] else None
        excl_px += [V.geom(q) for q in e.get("ref_excl", [])]
        lt = e.get("label_trims", e["trims"])          # every trim but the standing posts
        trim_cut = unary_union(lt) if (G.TRIM_CUT and lt) else None
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
                    labelled.append((relabel.get(fam, fam), V.geom(q)))
        # trims: white boards
        for tp in e["trims"]:
            G.cv_fill(canvas, V.geom(tp), (255, 255, 255) if not shaded else (242, 242, 240))
        # downspouts: on the wall, under the shadows and the railing; the wall label runs on behind
        for ds in e.get("downspouts", []):
            dsc = (255, 255, 255) if not shaded else G.mix(app["trim"], (60, 60, 60), 0.35)
            for qq in G.polys_of(ds):
                G.cv_fill(canvas, V.geom(qq), dsc)
                _outline_geom(canvas, V.geom(qq), ink, lw_thin)
        # cast shadows (walls) + window recesses
        if shadows_on:
            sh = wall_shadows(v, faces, e["vf"], sun)
            recess = r.uniform(0.25, 0.5)
            if glook:                       # Gemini draws windows nearly flat: a thin recess shadow, not a dark L-band
                recess *= 0.33
            rs = []
            for o in e["openings"]:
                p = o["poly"]
                rs.append(p.difference(affinity.translate(p, sun[0] * recess, sun[1] * recess)))
            parts = ([sh] if sh is not None else []) + rs
            rail = e.get("railing")
            if rail is not None:        # railing + deck slab shadow, cast `depth` in front of the wall
                d = rail["depth"]
                sil = unary_union(rail["cut"] + rail["rails"] +
                                  [G.rect(min(p0[0], p1[0]) - 0.03, min(p0[1], p1[1]) - 0.03,
                                          max(p0[0], p1[0]) + 0.03, max(p0[1], p1[1]) + 0.03)
                                   if (p0[0] == p1[0] or p0[1] == p1[1]) else
                                   Polygon([p0, p1, (p1[0], p1[1] + 0.06), (p0[0], p0[1] + 0.06)])
                                   for p0, p1 in rail["lines"]])
                da, db, dy = rail["deck"]
                slab = Polygon([(da, dy), (db, dy), (db + sun[0] * d, dy + sun[1] * d), (da + sun[0] * d, dy + sun[1] * d)])
                walls_u = unary_union([q for (f, q, bi, k) in e["surfaces"] if k in ("wall", "found")])
                rsh = unary_union([affinity.translate(sil, sun[0] * d, sun[1] * d), slab]).intersection(walls_u)
                if not rsh.is_empty:
                    parts.append(rsh)
            if parts:
                m = np.zeros((H, W), np.uint8)
                for q in G.polys_of(unary_union(parts)):
                    m = np.maximum(m, G.poly_mask(V.geom(q), W, H))
                sel = m > 127
                if soft is None:
                    canvas[sel] = (canvas[sel].astype(np.float32) * shadow_k).astype(np.uint8)
                else:
                    _shade(sel, soft["k1"])
                    sh2 = wall_shadows(v, faces, e["vf"], soft["sun2"]) if soft["two"] else None
                    if sh2 is not None and sh is not None:      # where both suns are blocked: the dark level
                        m2 = np.zeros((H, W), np.uint8)
                        for q in G.polys_of(sh2.intersection(sh)):
                            m2 = np.maximum(m2, G.poly_mask(V.geom(q), W, H))
                        sel2 = m2 > 127
                        canvas[sel2] = np.clip(canvas[sel2].astype(np.float32) * (soft["k2"] / soft["k1"]),
                                               0, 255).astype(np.uint8)
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
            if soft is None:
                canvas[sel] = (canvas[sel].astype(np.float32) * shadow_k).astype(np.uint8)
            else:
                _shade(sel, soft["k1"])
        # thin edges: every visible face boundary and trim
        for (fam, poly, bi, kind) in e["surfaces"]:
            _outline_geom(canvas, V.geom(poly), ink, lw_thin)
        # --heavy-pieces P (FC.HEAVY_PIECES, default 0.4 since 2026-10-14e): on a sheet chosen with P, some surface
        # pieces are boxed in a heavy line -- thin ones (a band under 1.5 ft) with probability 0.6, others 0.15 --
        # as real sheets draw fascia / band boards as heavy objects. Labels unchanged. HF14 14: a thin band of the
        # panels' lap siding, boxed in heavy lines, was dropped (0.55 / 0.66); with the heavy lines whitened, 0.90 /
        # 0.75. Synthetic sheets drew heavy lines only around the whole building. Own RNG: P = 0 is byte-identical.
        if heavy_on:
            for (fam, poly, bi, kind) in e["surfaces"]:
                for q in G.polys_of(poly):
                    x0q, y0q, x1q, y1q = q.bounds
                    if hp.random() < (0.6 if min(x1q - x0q, y1q - y0q) < 1.5 else 0.15):
                        _outline_geom(canvas, V.geom(q), ink, lw_heavy * hp.uniform(0.6, 1.0))
        for tp in e["trims"]:
            _outline_geom(canvas, V.geom(tp), ink, lw_thin)
        if getattr(FC, "SHAPED", [False])[0]:
            # --shaped: pieces of one v6 block (L / U / bay) are zoned as one surface, so draw
            # every visible 3D wall face's edge too -- the step where a wall sets back reads
            for Wf in e["vf"]:
                if Wf["kind"][0] in ("wall", "chimney"):
                    _outline_geom(canvas, V.geom(Wf["vis"]), ink, lw_thin)
        rail = e.get("railing")
        if rail is not None:            # railing in front of the openings and the shadows
            fillc = (255, 255, 255) if not shaded else (242, 242, 240)
            for q in rail["cut"] + rail["rails"]:
                for qq in G.polys_of(q):
                    G.cv_fill(canvas, V.geom(qq), fillc)
                    _outline_geom(canvas, V.geom(qq), ink, lw_thin)
            for p0, p1 in rail["lines"]:
                G.cv_line(canvas, V.px(*p0), V.px(*p1), ink, max(1.0, lw_thin * 0.7))
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
        if r.random() < 0.45 or glook:
            fams = ("main", "roof", "accent") if not glook else ("main", "roof", "accent", "accent2", "foundation", "chimney")
            for fam in fams:
                cand = [p for (f, p, _, _) in e["surfaces"] if f == fam and p.area > 8]
                if cand and fam in styles and (r.random() < 0.6 or (glook and glr.random() < 0.85)):
                    pt = max(cand, key=lambda q: q.area).representative_point()
                    side = 1 if pt.x > (x0 + x1) / 2 else -1
                    G.draw_leader(canvas, tq, V, (pt.x, pt.y), styles[fam].callout, app, S, side)
        if glook and glr.random() < 0.6:               # overall width, as Gemini dimensions its elevations
            G.draw_dim_string(canvas, tq, V, [x0, x1], 1.6, app, S, above=False)
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
    out = {"image": {"file_name": f"synth6_{image_id:06d}.png", "width": W, "height": H},
           "mode": "elevation", "appearance": "colour" if shaded else "markup" if mk is not None else "mono_normal",
           "px_per_ft": round(S, 2), "annotations": anns, "render": "revit"}
    if near:
        out["near_pair"] = {"families": list(near[:2]), "differs": near[2]}
    if gpair:
        out["gable_pair"] = {"zone": gp[0], "kind": gpair[0], "de2000": gpair[1]}
    rx = FC.ref_exclude_rings(excl_px, W, H)
    if rx:
        out["ref_exclude"] = rx
    return canvas, out
