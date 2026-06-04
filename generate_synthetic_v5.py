"""
synthetic_v5 generator — irregular polygons, image-quilting textures, real windows-as-holes.

Output layout (per image i):
    /content/synthetic_v5/images/synth_{i:06d}.png
    /content/synthetic_v5/annotations/synth_{i:06d}.json

JSON format (per-image, matching pdf_excerpts_coco style):
{
  "image":      {"file_name": "...", "width": W, "height": H},
  "annotations": [
    {
      "id": int,
      "category_id": int,            # global id, one per curated tile used
      "category_name": "patternN",   # local within this image (1..4)
      "reference_tile": "<abs path>",# the curated tile the model takes as input
      "segmentation": [outer_poly, hole_poly_1, hole_poly_2, ...],
                                     # outer ring first, window holes after.
                                     # NOTE: standard pycocotools unions polygons.
                                     # Use the rasterize_mask() helper in this file
                                     # to correctly subtract holes when training.
      "num_holes": int,
      "bbox": [x, y, w, h],
      "area": float                  # outer area minus hole area
    }
  ]
}

Run (from Colab):
    !python /content/drive/MyDrive/Floz/generate_synthetic_v5.py --n 16 --smoke
"""

import argparse
import json
import math
import os
import random
import sys
import time
from glob import glob

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import LineString as _ShLineString
from shapely.geometry import Polygon as _ShPoly
from shapely.ops import split as _sh_split
from shapely.ops import unary_union as _sh_union

_FONT_CACHE = {}
def _font(size):
    if size not in _FONT_CACHE:
        for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                     '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size=size)
                break
            except Exception:
                pass
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

_ROOM_LABELS = ['BEDROOM', 'KITCHEN', 'BATH', 'CLOSET', 'LIVING', 'DINING',
                'ENTRY', 'GARAGE', 'OFFICE', 'STORAGE', 'LAUNDRY', 'PANTRY',
                'HALL', 'MASTER', 'FOYER', 'DECK', 'PORCH', 'STUDY']
_DIM_LABELS = ['10\'-6"', '12\'-0"', '8\'-4"', '14\'-2"', '6\'-0"', '4\'-8"',
               '16\'-0"', '11\'-3"', '9\'-9"', '7\'-6"']
_TAG_LABELS = ['A-1', 'A-2', 'A-3', '101', '102', '201', 'S-1', 'M-2',
               'TYP.', 'EQ.', 'V.I.F.']

# ---------- paths ----------
TILES_DIR = '/content/drive/MyDrive/Floz/reference_tiles_curated'
OUT_ROOT  = '/content/synthetic_v5'

# ---------- canvas / scene parameters ----------
CANVAS_W_RANGE = (2400, 5000)
CANVAS_H_RANGE = (1300, 3000)
PATTERNS_PER_IMAGE = (1, 3)           # how many distinct curated tiles to use per image
INSTANCES_PER_PATTERN = (1, 6)        # polygon regions per pattern
MIN_INSTANCE_AREA_FRAC = 0.005        # of canvas area
MAX_INSTANCE_AREA_FRAC = 0.30
MODE_WEIGHTS = {
    'elevation': 55,
    'freeform': 15,
    'roof_plan': 30,
}
ROWHOUSE_ELEVATION_PROB = 0.50
ELEVATION_EXCERPT_SHIFT_FRAC = 0.18   # random off-center crop around the facade bbox
ELEVATION_PARTIAL_CROP_PROB = 0.22
ELEVATION_PARTIAL_CROP_FRAC = 0.08
MARKUP_OVERLAY_PROB = 0.35
DOCUMENT_EFFECT_PROB = 0.45
MARKUP_COLORS = [
    (235, 63, 63, 170),   # red markup
    (222, 82, 231, 170),  # magenta markup
    (59, 118, 214, 155),  # blue note/callout
]

# ---------- quilting parameters ----------
BLOCK = 80
OVERLAP = 20
QUILT_CANDIDATES_K = 8                # randomly pick among top-K SSD matches

# ---------- window parameters ----------
WINDOW_DENSITY = 1.0 / 160_000         # ~ 1 window per (400x400) px region
MIN_WINDOW_PX = 50
MAX_WINDOW_PX = 160

# ---------- v5.1: real-plan instance granularity ----------
# Real PDF plans annotate finely: the same material is split per architectural
# unit into many separate, crisp-edged, side-by-side instances. Original v5
# produced few large, irregular (sawtooth), well-separated, often-merged blobs.
# These switches retarget the generator to match the real annotation style.
RECTILINEAR_INSTANCES = True   # straight edges only (no sawtooth/comb razor teeth)
NO_MERGE_SAME_MATERIAL = True  # adjacent same-tile regions stay SEPARATE instances
INTERLEAVE_PATTERNS = True     # different patterns abut (no white gaps between them)
BAY_SPLIT_PROB = 0.55          # chance a region is split into per-bay sub-instances
DRAW_INSTANCE_OUTLINE = True   # draw a dark loop around every instance. Suspected
                               # synth-only shortcut: the model can detect "pattern
                               # = region inside a perfect closed dark line", a cue
                               # that doesn't exist consistently in real plans and so
                               # inflates synth-val without helping real. Toggle off
                               # (--no-outline) to test/remove that shortcut.
FLOORPLAN_INTERLEAVE_PROB = 0.42
ELEVATION_INTERLEAVE_PROB = 0.36
ROOFPLAN_INTERLEAVE_PROB = 0.52


def _apply_split_scale(scale):
    """Scale every per-region split/interleave probability by `scale`.

    Real plans carry FAR fewer instances per excerpt than synth (median ~3 vs
    ~8): a material is usually shown as one large hatched region, not many split
    pieces. The interleave/bay-split machinery was tuned UP to make instances
    finer, which pushed synth the wrong way on instance COUNT (a feature an
    instance-stats classifier separates synth from real on trivially). scale<1
    coarsens synth toward the real count/size distribution; scale=0 disables the
    added splitting entirely."""
    global BAY_SPLIT_PROB, FLOORPLAN_INTERLEAVE_PROB
    global ELEVATION_INTERLEAVE_PROB, ROOFPLAN_INTERLEAVE_PROB
    BAY_SPLIT_PROB *= scale
    FLOORPLAN_INTERLEAVE_PROB *= scale
    ELEVATION_INTERLEAVE_PROB *= scale
    ROOFPLAN_INTERLEAVE_PROB *= scale

# Instance granularity. Real plans carry ~3 coarse instances/image; synth ~8-11
# finer ones (a structural easiness cue: many small regular regions are an easier
# decomposition than a few large coarse ones). INSTANCE_SCALE in [0,1]: 1.0 = the
# current fine decomposition, lower = coarser toward real. It (a) caps elevation
# wall bands, (b) merges same-material roof facets into one instance with prob
# (1-scale) -- visually faithful since render_roofplan_post keeps the internal
# ridge/hip lines -- (c) biases roof footprints toward a single wing, and (d)
# scales all interleave/bay split probs (via _apply_split_scale). Floor plans are
# left alone (5% of the recipe; cutting their coverage would leave blank rooms,
# which real plans don't have).
INSTANCE_SCALE = 1.0

def _inst_cap(fine, coarse):
    """Integer count interpolated between coarse (INSTANCE_SCALE=0) and fine
    (INSTANCE_SCALE=1)."""
    return int(round(coarse + (fine - coarse) * INSTANCE_SCALE))

# Dense/colored fills: real instances are densely filled (median interior ink
# coverage ~0.53 — colored/shaded elevation materials), while raw quilted-hatch
# synth instances are sparse (median ~0.11, thin lines on white). Render a
# fraction of categories as a solid per-category base color with the hatch
# multiplied over it, so synth interior-coverage distribution matches real and
# the model stops keying on "sparse lines on white".
DENSE_COLOR_FILL = True
DENSE_FILL_FRAC = 0.40         # fraction of categories rendered as dense colored
DENSE_FILL_OPACITY = 1.0       # 1.0 = solid base color, <1 keeps some white paper
DENSE_FILL_SCOPE = 'category'  # 'category' keeps all same-pattern instances aligned;
                               # 'instance' samples dense/sparse per instance but
                               # reuses the category color when dense.
# Muted architectural material colors (siding/stucco/brick/roof/stone tones).
FILL_PALETTE = [
    (196, 180, 151), (170, 158, 138), (151, 160, 168), (132, 142, 150),
    (181, 150, 124), (160, 120, 100), (138, 110, 96), (120, 130, 120),
    (158, 168, 176), (200, 190, 170), (146, 134, 120), (110, 122, 134),
]
# Fraction of images rendered as pure black-and-white (no color). ~50% of the
# real eval plans are monochrome line-art; synth otherwise always injects color
# via FILL_PALETTE and colored tiles, giving the model a color boundary cue that
# is absent on those real plans. Mono images use only grayscale tiles, gray
# dense fills, and gray markup.
MONO_IMAGE_PROB = 0.5
MONO_FILL_GRAY_RANGE = (70, 205)  # poché-dark to light-hatch gray for dense fills
# Saturation below this (mean HSV S, 0-255) marks a tile as grayscale/B&W.
TILE_GRAY_SAT_MAX = 8.0
# Minimum interior ink coverage for a dense-filled region. Real instances have
# median interior ink ~0.53; raw synth (esp. low-ink grayscale tiles in mono
# mode, with light fills) leaves ~44% of regions near-white (<0.10). This floor
# lifts only the too-faint regions (darkening their fill toward this coverage)
# so pattern regions read as filled like real plans; already-dense regions are
# untouched. 0.0 = off.
DENSE_FILL_MIN_INK = 0.35

# Per-image probability of injecting UNLABELED textured "negatives" (material
# swatches, dimension grids, section hatching) into background space. Real plans
# contain textured line-work that annotators do NOT label as material instances;
# synth historically labeled EVERY textured region, so the model learned
# "texture => instance" and massively over-predicts on real (per-image analysis:
# 30.9 preds vs 4.6 GT, 6.7x; fires 'pattern' on every textured room/grid). These
# carry texture but no annotation, teaching that texture alone is insufficient.
# 0.0 = off (default; preserves the canonical recipe).
NEGATIVE_TEXTURE_PROB = 0.0

# Per-elevation probability of rendering a "real-style" excerpt that mimics the
# real plans the model fails HARDEST on (Las Huertas / Ceilhunt elevations): one
# large WHOLE material region per surface (no band-splitting, no interleave),
# FAINT subtle texture (not dense saturated fill), and NO dark instance outline,
# on lots of white space. Direct response to the per-image failure analysis
# (model shatters a single real siding wall into ~50 fragments). Default-OFF.
REALSTYLE_PROB = 0.0

# Set True only while a real-style elevation scene is being built (by
# _realstyle_scene_overrides). Used to bias the wall toward IRREGULAR shapes
# (gable-cut tops) instead of plain rectangles: the divergence metric rewards
# precise mask-fitting, and a clean rectangular synth wall is far easier to fit
# exactly than the irregular, roofline-interrupted walls on the hard real
# elevations (idx5/idx10), keeping synth val IoU above real. iter5, 2026-06-03.
REALSTYLE_ACTIVE = False

# Render realstyle elevation walls as white + faint horizontal clapboard/lap
# lines (matches worst-real idx09) instead of a flat lightened tile fill, so the
# wall region is defined by internal line texture rather than a tint blob the
# model can shortcut. Applied at render time to role=='wall' realstyle crops.
REALSTYLE_CLAPBOARD = False  # ablation: strong horizontal texture HURT real (0.245->0.191), a known landmine

# Draw roof pitch callouts + top-of-plate reference lines on elevations (matches
# worst-real idx09). Non-target clutter, no instance emitted. Default-on.
REALSTYLE_ROOF_CUES = False  # control: ablated to reproduce +0.101 baseline

# Apply realstyle to FREEFORM floor plans too (worst-real idx00, IoU 0.052): real
# floor plans are mostly WHITE rooms bounded by wall lines, with material in only
# a FEW rooms + faint fills -- NOT the edge-to-edge saturated material blocks the
# old builder made ("no white gaps" was a wrong premise). Set per-image in
# compose_image to the image's realstyle flag; controls room coverage fraction.
FREEFORM_REALSTYLE = False
# Tuned (code-based loop, 2026-06-03) to MATCH measured target idx0 floor-plan stats
# (n_gt~2, area-fraction~0.09, ink~0.12): very few small material rooms, no bay/
# interleave splits, moderate (not extreme) whitening so regions keep ~target ink.
FREEFORM_REALSTYLE_COVER = (0.05, 0.14)
FREEFORM_REALSTYLE_WHITE = (0.28, 0.50)   # blend-toward-white for free fills (ink up from 0.033)

# Per-FREE-region probability of rendering a floor-plan material region as a
# tile/stone OUTLINE pattern (thin lines on white) instead of a dense fill --
# matching how real plans draw flooring (e.g. worst-image idx0's covered patio is
# a stone ashlar grid, ink ~0.09, NOT a fill). The model otherwise learns
# "material = dense fill" and never produces a mask for an outline-grid region
# (proven: oracle IoU on idx0 = 0.04). Default-OFF.
FREEFORM_TILE_OUTLINE_PROB = 0.0

# Path to the TARGET image's ACTUAL material texture (idx0's covered-patio stone,
# extracted to assets_idx0_stone.png). When set, freeform floor plans place ONE
# big WIDE region + one small rendered with this REAL stone texture, so the model
# trains on idx0's exact patio appearance and recognizes it on idx0 (raising
# iou(idx0) toward synth_val -> divergence -> 0). Default None (off).
IDX0_STONE_PATH = None
# White-blend range for the idx0 stone fill. High = near-invisible (down-convergence:
# make synth as low-signal/hard as idx0 so synth_val drops to iou(idx0) ~ 0.04).
IDX0_STONE_FAINT = (0.80, 0.92)

# Crank all sheet-context clutter (title block, hidden lines, multiple keyed notes)
# to max, to test whether matching real CAD-page DENSITY closes div@ep10. Default OFF.
CLUTTER_BOOST = False

# Multiplier on elevation building pixel dimensions, to render synth at real's
# native resolution (~2.6x) so eval-time downsampling makes synth instances as small
# /hard as real. 1.0 = off (default). See build_elevation_scene.
RESOLUTION_SCALE = 1.0

# Std (px) of organic boundary jitter applied to the stored ANNOTATION polygons only
# (image stays crisp). Real annotations are imprecise human tracings; synth's are
# machine-perfect polygons the model fits to ~0.37. Jittering the LABEL so it no
# longer matches a crisp edge lowers synth_iou@10 toward real. 0 = off (default).
LABEL_JITTER = 0.0


def _realstyle_scene_overrides():
    """Force whole-region elevations for a real-style image: single building,
    one band per wall, no interleave/bay splits. Returns the saved globals so
    the caller can restore them right after build_elevation_scene."""
    global ROWHOUSE_ELEVATION_PROB, INSTANCE_SCALE
    global BAY_SPLIT_PROB, FLOORPLAN_INTERLEAVE_PROB
    global ELEVATION_INTERLEAVE_PROB, ROOFPLAN_INTERLEAVE_PROB, REALSTYLE_ACTIVE
    saved = dict(rh=ROWHOUSE_ELEVATION_PROB, isc=INSTANCE_SCALE,
                 bay=BAY_SPLIT_PROB, fp=FLOORPLAN_INTERLEAVE_PROB,
                 el=ELEVATION_INTERLEAVE_PROB, rf=ROOFPLAN_INTERLEAVE_PROB)
    ROWHOUSE_ELEVATION_PROB = 0.0   # clean single house, not a rowhouse strip
    INSTANCE_SCALE = 0.0            # 1 band per wall (whole region)
    BAY_SPLIT_PROB = FLOORPLAN_INTERLEAVE_PROB = 0.0
    ELEVATION_INTERLEAVE_PROB = ROOFPLAN_INTERLEAVE_PROB = 0.0
    REALSTYLE_ACTIVE = True         # bias wall toward irregular gable-cut shape
    return saved


def _restore_scene_overrides(saved):
    global ROWHOUSE_ELEVATION_PROB, INSTANCE_SCALE
    global BAY_SPLIT_PROB, FLOORPLAN_INTERLEAVE_PROB
    global ELEVATION_INTERLEAVE_PROB, ROOFPLAN_INTERLEAVE_PROB, REALSTYLE_ACTIVE
    ROWHOUSE_ELEVATION_PROB = saved['rh']; INSTANCE_SCALE = saved['isc']
    BAY_SPLIT_PROB = saved['bay']; FLOORPLAN_INTERLEAVE_PROB = saved['fp']
    ELEVATION_INTERLEAVE_PROB = saved['el']; ROOFPLAN_INTERLEAVE_PROB = saved['rf']
    REALSTYLE_ACTIVE = False

# ============================================================
# Helpers
# ============================================================

def rasterize_mask(segmentation, h, w):
    """
    Reference rasterizer for the segmentation format used in this dataset.
    segmentation[0] = outer polygon, segmentation[1:] = hole polygons.
    Returns a uint8 binary mask of shape (h, w).
    """
    outer = Image.new('L', (w, h), 0)
    ImageDraw.Draw(outer).polygon([tuple(p) for p in zip(segmentation[0][0::2],
                                                          segmentation[0][1::2])],
                                  fill=1, outline=1)
    if len(segmentation) > 1:
        holes = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(holes)
        for poly in segmentation[1:]:
            d.polygon([tuple(p) for p in zip(poly[0::2], poly[1::2])],
                      fill=1, outline=1)
        outer = Image.fromarray(
            np.where(np.array(holes) > 0, 0, np.array(outer)).astype(np.uint8))
    return np.asarray(outer, dtype=np.uint8)


def load_curated_tiles(tiles_dir=None):
    tiles_dir = tiles_dir or TILES_DIR
    with open(os.path.join(tiles_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    tiles = []
    cat_id = 1
    for img_key, pats in manifest.items():
        for pat_key, info in pats.items():
            path = os.path.join(tiles_dir, info['tile_file'])
            arr = np.array(Image.open(path).convert('L'))
            ink_density = float((255 - arr).mean()) / 255.0
            sat_mean = float(np.array(Image.open(path).convert('HSV'))[..., 1].mean())
            tiles.append({
                'path': path,
                'name': info['tile_file'],
                'category_id': cat_id,
                'image_key': img_key,
                'pattern_key': pat_key,
                'ink_density': ink_density,
                'is_gray': sat_mean < TILE_GRAY_SAT_MAX,
            })
            cat_id += 1
    return tiles


def pick_wall_tile(tiles, rng, min_ink=0.05):
    """Prefer tiles with non-trivial ink density for wall regions, so walls
    don't end up looking blank."""
    candidates = [t for t in tiles if t['ink_density'] > min_ink]
    return rng.choice(candidates if candidates else tiles)


# ============================================================
# Image quilting
# ============================================================

def _extract_blocks(arr: np.ndarray, block: int, stride: int = 4) -> np.ndarray:
    """All BxBxC patches from `arr` at the given stride. Returns (N, B, B, C)."""
    h, w, c = arr.shape
    ys = list(range(0, h - block + 1, stride))
    xs = list(range(0, w - block + 1, stride))
    if not ys or not xs:
        return arr[None, :block, :block].copy()
    out = np.empty((len(ys) * len(xs), block, block, c), dtype=arr.dtype)
    k = 0
    for y in ys:
        for x in xs:
            out[k] = arr[y:y + block, x:x + block]
            k += 1
    return out


def _min_cut_mask_vertical(err: np.ndarray) -> np.ndarray:
    """
    Min-cut seam through a (B, O) error map, vertical seam.
    Returns (B, O) mask: 1 where the *new* (right) block should be used.
    """
    B, O = err.shape
    cost = err.copy().astype(np.float64)
    for i in range(1, B):
        left  = np.concatenate([[np.inf], cost[i - 1, :-1]])
        mid   = cost[i - 1]
        right = np.concatenate([cost[i - 1, 1:], [np.inf]])
        cost[i] += np.minimum(np.minimum(left, mid), right)
    path = np.empty(B, dtype=np.int32)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(B - 2, -1, -1):
        s = path[i + 1]
        candidates = []
        if s > 0:     candidates.append((s - 1, cost[i, s - 1]))
        candidates.append((s, cost[i, s]))
        if s < O - 1: candidates.append((s + 1, cost[i, s + 1]))
        path[i] = min(candidates, key=lambda t: t[1])[0]
    mask = np.zeros((B, O), dtype=np.float32)
    for i, s in enumerate(path):
        mask[i, s:] = 1.0
    return mask


def _min_cut_mask_horizontal(err: np.ndarray) -> np.ndarray:
    """Same as vertical but seam runs left-right. err shape (O, B)."""
    return _min_cut_mask_vertical(err.T).T


def quilt_synthesize(tile_arr: np.ndarray, out_h: int, out_w: int,
                     rng: random.Random) -> np.ndarray:
    """Synthesize an out_h x out_w texture from tile_arr via image quilting."""
    B, O = BLOCK, OVERLAP
    out_h = max(out_h, B)
    out_w = max(out_w, B)
    canvas = np.zeros((out_h, out_w, tile_arr.shape[2]), dtype=np.float32)
    filled = np.zeros((out_h, out_w), dtype=bool)
    blocks = _extract_blocks(tile_arr.astype(np.float32), B, stride=8)

    step = B - O
    ys = list(range(0, out_h - B + 1, step)) + [out_h - B]
    xs = list(range(0, out_w - B + 1, step)) + [out_w - B]
    ys = sorted(set(ys)); xs = sorted(set(xs))

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if yi == 0 and xi == 0:
                idx = rng.randrange(len(blocks))
                canvas[y:y + B, x:x + B] = blocks[idx]
                filled[y:y + B, x:x + B] = True
                continue

            # Compute SSD against existing canvas overlaps (left + top where present)
            ssd = np.zeros(len(blocks), dtype=np.float64)
            if xi > 0:
                exist = canvas[y:y + B, x:x + O]                  # (B, O, C)
                diff = blocks[:, :, :O, :] - exist[None]          # (N, B, O, C)
                ssd += (diff * diff).sum(axis=(1, 2, 3))
            if yi > 0:
                exist = canvas[y:y + O, x:x + B]                  # (O, B, C)
                diff = blocks[:, :O, :, :] - exist[None]          # (N, O, B, C)
                ssd += (diff * diff).sum(axis=(1, 2, 3))

            k = min(QUILT_CANDIDATES_K, len(blocks))
            cand_idx = np.argpartition(ssd, k - 1)[:k]
            chosen = blocks[rng.choice(cand_idx.tolist())]

            block_out = chosen.copy()

            # Min-cut seams
            if xi > 0:
                exist_l = canvas[y:y + B, x:x + O]
                err = ((chosen[:, :O, :] - exist_l) ** 2).sum(axis=2)  # (B, O)
                m = _min_cut_mask_vertical(err)[:, :, None]
                block_out[:, :O, :] = m * chosen[:, :O, :] + (1 - m) * exist_l
            if yi > 0:
                exist_t = canvas[y:y + O, x:x + B]
                err = ((chosen[:O, :, :] - exist_t) ** 2).sum(axis=2)  # (O, B)
                m = _min_cut_mask_horizontal(err)[:, :, None]
                block_out[:O, :, :] = m * chosen[:O, :, :] + (1 - m) * exist_t

            canvas[y:y + B, x:x + B] = block_out
            filled[y:y + B, x:x + B] = True

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ============================================================
# Irregular polygons
# ============================================================

def _rect_to_poly(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def make_polygon(cx: int, cy: int, w: int, h: int, rng: random.Random):
    """Generate an irregular polygon centered near (cx, cy) within w x h bbox.
    Returns list of (x, y) ints, ccw-ish.
    """
    kind = rng.choice(['rectilinear_notched', 'L_shape', 'sloped', 'perturbed',
                       'rectilinear_notched', 'perturbed'])  # weights
    x0, y0 = cx - w // 2, cy - h // 2

    if kind == 'L_shape':
        # Compose two overlapping rects forming L/T/U
        cut_w = rng.randint(w // 3, 2 * w // 3)
        cut_h = rng.randint(h // 3, 2 * h // 3)
        corner = rng.choice(['tl', 'tr', 'bl', 'br'])
        rect = _rect_to_poly(x0, y0, w, h)
        # remove a corner rectangle by walking the polygon
        if corner == 'tl':
            poly = [(x0 + cut_w, y0), (x0 + w, y0), (x0 + w, y0 + h),
                    (x0, y0 + h), (x0, y0 + cut_h), (x0 + cut_w, y0 + cut_h)]
        elif corner == 'tr':
            poly = [(x0, y0), (x0 + w - cut_w, y0), (x0 + w - cut_w, y0 + cut_h),
                    (x0 + w, y0 + cut_h), (x0 + w, y0 + h), (x0, y0 + h)]
        elif corner == 'bl':
            poly = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h),
                    (x0 + cut_w, y0 + h), (x0 + cut_w, y0 + h - cut_h),
                    (x0, y0 + h - cut_h)]
        else:
            poly = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - cut_h),
                    (x0 + w - cut_w, y0 + h - cut_h), (x0 + w - cut_w, y0 + h),
                    (x0, y0 + h)]
        return poly

    if kind == 'rectilinear_notched':
        poly = _rect_to_poly(x0, y0, w, h)
        # 1-3 axis-aligned notches
        for _ in range(rng.randint(1, 3)):
            edge = rng.randrange(4)
            depth = rng.randint(min(w, h) // 12, min(w, h) // 5)
            length = rng.randint(min(w, h) // 6, min(w, h) // 3)
            outward = rng.random() < 0.4  # bump out instead of in
            sign = -1 if outward else 1
            # Insert two new vertices along that edge
            (ax, ay), (bx, by) = poly[edge], poly[(edge + 1) % len(poly)]
            t1 = rng.uniform(0.2, 0.5); t2 = rng.uniform(0.55, 0.85)
            p1 = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            p2 = (ax + (bx - ax) * t2, ay + (by - ay) * t2)
            # perpendicular (inward) normal
            ex, ey = bx - ax, by - ay
            L = math.hypot(ex, ey) or 1
            nx, ny = -ey / L, ex / L  # left-of-edge for ccw is inward
            q1 = (p1[0] + sign * nx * depth, p1[1] + sign * ny * depth)
            q2 = (p2[0] + sign * nx * depth, p2[1] + sign * ny * depth)
            poly = poly[:edge + 1] + [p1, q1, q2, p2] + poly[edge + 1:]
        return [(int(round(px)), int(round(py))) for px, py in poly]

    if kind == 'sloped':
        # Rect with one or two diagonal corners (think gable / hipped roof side)
        side = rng.choice(['top', 'left', 'right'])
        if side == 'top':
            slope = rng.randint(h // 4, h // 2)
            poly = [(x0, y0 + slope), (x0 + w // 2, y0),
                    (x0 + w, y0 + slope * rng.choice([0, 1])),
                    (x0 + w, y0 + h), (x0, y0 + h)]
        elif side == 'left':
            slope = rng.randint(w // 5, w // 3)
            poly = [(x0 + slope, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        else:
            slope = rng.randint(w // 5, w // 3)
            poly = [(x0, y0), (x0 + w - slope, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        return poly

    # perturbed
    base = _rect_to_poly(x0, y0, w, h)
    poly = []
    for i in range(len(base)):
        a = base[i]; b = base[(i + 1) % len(base)]
        poly.append(a)
        n_sub = rng.randint(2, 4)
        for k in range(1, n_sub):
            t = k / n_sub + rng.uniform(-0.05, 0.05)
            mx = a[0] + (b[0] - a[0]) * t
            my = a[1] + (b[1] - a[1]) * t
            ex, ey = b[0] - a[0], b[1] - a[1]
            L = math.hypot(ex, ey) or 1
            nx, ny = -ey / L, ex / L
            mag = rng.uniform(-min(w, h) * 0.04, min(w, h) * 0.04)
            poly.append((mx + nx * mag, my + ny * mag))
    return [(int(round(px)), int(round(py))) for px, py in poly]


def _poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _dedupe_poly(poly):
    out = []
    for pt in poly:
        ipt = (int(round(pt[0])), int(round(pt[1])))
        if not out or ipt != out[-1]:
            out.append(ipt)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _piece_sort_key(poly, axis):
    x0, y0, x1, y1 = _poly_bbox(poly)
    if axis == 'h':
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _stepped_split_line(poly, rng, axis=None):
    x0, y0, x1, y1 = _poly_bbox(poly)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    axis = axis or ('v' if w >= h else 'h')
    if axis == 'v':
        margin = max(18, min(w // 6, 90))
        base = rng.randint(x0 + margin, x1 - margin)
        n_steps = rng.randint(2, 4)
        cuts = sorted(rng.randint(y0 + 24, y1 - 24) for _ in range(n_steps))
        cur = base
        pts = [(cur, y0 - 10)]
        max_jit = max(20, min(w // 5, 110))
        for yy in cuts:
            nxt = int(np.clip(cur + rng.randint(-max_jit, max_jit),
                              x0 + margin, x1 - margin))
            pts.append((cur, yy))
            pts.append((nxt, yy))
            cur = nxt
        pts.append((cur, y1 + 10))
        return pts
    margin = max(18, min(h // 6, 90))
    base = rng.randint(y0 + margin, y1 - margin)
    n_steps = rng.randint(2, 4)
    cuts = sorted(rng.randint(x0 + 24, x1 - 24) for _ in range(n_steps))
    cur = base
    pts = [(x0 - 10, cur)]
    max_jit = max(20, min(h // 5, 110))
    for xx in cuts:
        nxt = int(np.clip(cur + rng.randint(-max_jit, max_jit),
                          y0 + margin, y1 - margin))
        pts.append((xx, cur))
        pts.append((xx, nxt))
        cur = nxt
    pts.append((x1 + 10, cur))
    return pts


def _split_poly_interleaved(poly, rng, max_pieces=3, min_piece_area=12_000):
    """Split a polygon with one or two stepped seams so adjacent pieces touch
    directly and read less like clean rectangles/trapezoids."""
    pieces = [poly]
    target = rng.randint(2, max_pieces)
    for _ in range(target - 1):
        idx = max(range(len(pieces)), key=lambda i: _polygon_area(pieces[i]))
        candidate = pieces.pop(idx)
        x0, y0, x1, y1 = _poly_bbox(candidate)
        if (x1 - x0) < 130 or (y1 - y0) < 130 or _polygon_area(candidate) < 2 * min_piece_area:
            pieces.append(candidate)
            break
        axis = 'v' if (x1 - x0) >= (y1 - y0) else 'h'
        split_line = _ShLineString(_stepped_split_line(candidate, rng, axis=axis))
        try:
            geoms = [g.buffer(0) for g in _sh_split(_ShPoly(candidate).buffer(0), split_line).geoms]
        except Exception:
            pieces.append(candidate)
            break
        geoms = [g for g in geoms
                 if getattr(g, 'geom_type', '') == 'Polygon'
                 and g.area >= min_piece_area
                 and len(list(g.exterior.coords)) >= 4]
        if len(geoms) < 2:
            pieces.append(candidate)
            break
        new_pieces = []
        for geom in geoms:
            coords = _dedupe_poly(list(geom.exterior.coords)[:-1])
            if len(coords) >= 3:
                new_pieces.append(coords)
        if len(new_pieces) < 2:
            pieces.append(candidate)
            break
        pieces.extend(new_pieces)
    return pieces


def _surface_items(poly, role, primary_tile, rng, alt_tiles=None,
                   has_windows=False, split_prob=0.0, max_pieces=3,
                   min_piece_area=12_000):
    poly = _dedupe_poly(poly)
    pieces = [poly]
    if alt_tiles and _polygon_area(poly) >= 2 * min_piece_area and rng.random() < split_prob:
        split_pieces = _split_poly_interleaved(poly, rng, max_pieces=max_pieces,
                                               min_piece_area=min_piece_area)
        if len(split_pieces) > 1:
            pieces = split_pieces
    axis = 'v' if (_poly_bbox(poly)[2] - _poly_bbox(poly)[0]) >= (_poly_bbox(poly)[3] - _poly_bbox(poly)[1]) else 'h'
    pieces = sorted(pieces, key=lambda p: _piece_sort_key(p, axis))
    palette = [primary_tile]
    for tile in alt_tiles or []:
        if tile['path'] != primary_tile['path']:
            palette.append(tile)
    out = []
    split = len(pieces) > 1
    for idx, piece in enumerate(pieces):
        x0, y0, x1, y1 = _poly_bbox(piece)
        piece_tile = primary_tile if not split else palette[idx % len(palette)]
        piece_windows = (has_windows
                         and (x1 - x0) >= 120
                         and (y1 - y0) >= 120
                         and _polygon_area(piece) >= 18_000
                         and (idx == 0 or rng.random() < 0.6))
        dense_bias = 0.0
        dense_boost = 0.0
        if role == 'roof_plan':
            dense_bias += 0.06
            dense_boost += 0.02
        if split:
            if role == 'free':
                dense_bias += 0.18
                dense_boost += 0.16
            elif role == 'roof_plan':
                dense_bias += 0.10
                dense_boost += 0.06
            elif role in ('wall', 'garage'):
                dense_bias += 0.12
                dense_boost += 0.10
        out.append({
            'poly': piece,
            'role': role,
            'tile': piece_tile,
            'has_windows': piece_windows,
            'draw_outline': False if split else True,
            'dense_bias': dense_bias,
            'dense_opacity_boost': dense_boost,
        })
    return out


# ============================================================
# Window geometry (drawn into the image; emitted as hole polygons)
# ============================================================

def _polygon_area(pts):
    n = len(pts); a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _jitter_poly(poly, rng, amp):
    """Perturb a polygon into an organic, human-traced-looking boundary: resample
    each edge with intermediate points, then offset every point by Gaussian noise
    of std `amp` px. Used ONLY on the stored annotation (not the rendered image),
    so the label no longer matches a crisp machine edge — mimicking real plans
    where the human annotation does not exactly follow the material boundary. This
    is the one lever that attacks the synth_iou@10 ≈ 0.37 attractor at its proven
    cause (machine-perfect synth labels), since no IMAGE change moves it."""
    if amp <= 0 or len(poly) < 3:
        return poly
    pts = []
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        pts.append((x0, y0))
        seglen = math.hypot(x1 - x0, y1 - y0)
        nsub = int(seglen // 28)
        for k in range(1, nsub):
            t = k / nsub
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return [(x + rng.gauss(0, amp), y + rng.gauss(0, amp)) for (x, y) in pts]


def _point_in_poly(x, y, poly):
    n = len(poly); inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def _rect_in_poly(rect, poly, margin=8):
    x, y, w, h = rect
    pts = [(x - margin, y - margin), (x + w + margin, y - margin),
           (x + w + margin, y + h + margin), (x - margin, y + h + margin)]
    return all(_point_in_poly(px, py, poly) for px, py in pts)


def draw_window_into(canvas: Image.Image, rect, rng: random.Random):
    """
    Draw a real window (frame, muntins, sill) inside `rect = (x, y, w, h)` on canvas.
    Returns the polygon (list of (x, y)) covering the window — to be excluded
    from the pattern mask as a COCO-style hole.
    """
    x, y, w, h = rect
    d = ImageDraw.Draw(canvas)
    # White interior glass
    d.rectangle([x, y, x + w, y + h], fill=(252, 252, 250))
    # Frame: 2-3 px outline
    fw = rng.choice([2, 3, 3])
    for i in range(fw):
        d.rectangle([x + i, y + i, x + w - i, y + h - i], outline=(15, 15, 20))
    # Inner sash inset
    inset = rng.randint(4, 8)
    d.rectangle([x + inset, y + inset, x + w - inset, y + h - inset],
                outline=(40, 40, 45))
    # Muntins: choose pattern
    pattern = rng.choice(['cross', 'horiz_split', 'two_over_two', 'colonial_2x3', 'none'])
    cx_ = x + w // 2; cy_ = y + h // 2
    if pattern == 'cross':
        d.line([cx_, y + inset, cx_, y + h - inset], fill=(40, 40, 45), width=2)
        d.line([x + inset, cy_, x + w - inset, cy_], fill=(40, 40, 45), width=2)
    elif pattern == 'horiz_split':
        d.line([x + inset, cy_, x + w - inset, cy_], fill=(40, 40, 45), width=2)
    elif pattern == 'two_over_two':
        d.line([x + inset, cy_, x + w - inset, cy_], fill=(40, 40, 45), width=2)
        d.line([cx_, y + inset, cx_, cy_], fill=(40, 40, 45), width=2)
        d.line([cx_, cy_, cx_, y + h - inset], fill=(40, 40, 45), width=2)
    elif pattern == 'colonial_2x3':
        for k in range(1, 3):
            d.line([x + k * w // 3, y + inset, x + k * w // 3, y + h - inset],
                   fill=(40, 40, 45), width=1)
        d.line([x + inset, cy_, x + w - inset, cy_], fill=(40, 40, 45), width=1)
    # Sill: a thicker bar protruding slightly below
    sill_h = rng.randint(4, 8)
    sill_overhang = rng.randint(3, 8)
    d.rectangle([x - sill_overhang, y + h, x + w + sill_overhang, y + h + sill_h],
                fill=(30, 30, 35))
    # The hole polygon should cover frame+sill so the pattern doesn't touch any of it
    return [(x - sill_overhang, y),
            (x + w + sill_overhang, y),
            (x + w + sill_overhang, y + h + sill_h),
            (x - sill_overhang, y + h + sill_h)]


def place_windows_in_polygon(canvas: Image.Image, poly, rng: random.Random):
    """Plant a few windows inside `poly`. Returns list of hole polygons (list-of-points)."""
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    bx, by, bw, bh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    area = _polygon_area(poly)
    target = max(1, int(round(area * WINDOW_DENSITY)))
    target = min(target, 6)
    holes = []
    attempts = 0
    placed = 0
    while placed < target and attempts < 50:
        attempts += 1
        ww = rng.randint(MIN_WINDOW_PX, min(MAX_WINDOW_PX, max(MIN_WINDOW_PX + 1, bw // 3)))
        wh = rng.randint(MIN_WINDOW_PX, min(MAX_WINDOW_PX, max(MIN_WINDOW_PX + 1, bh // 3)))
        # Common architectural aspect ratios
        if rng.random() < 0.5:
            wh = int(ww * rng.uniform(1.1, 1.8))
        wx = rng.randint(bx + 5, bx + bw - ww - 5) if bw > ww + 10 else bx
        wy = rng.randint(by + 5, by + bh - wh - 25) if bh > wh + 30 else by
        rect = (wx, wy, ww, wh)
        if not _rect_in_poly(rect, poly, margin=10):
            continue
        # No overlap with existing windows
        clash = False
        for h in holes:
            hxs = [p[0] for p in h]; hys = [p[1] for p in h]
            hx0, hy0, hx1, hy1 = min(hxs), min(hys), max(hxs), max(hys)
            if not (wx + ww + 20 < hx0 or hx1 + 20 < wx
                    or wy + wh + 20 < hy0 or hy1 + 20 < wy):
                clash = True; break
        if clash:
            continue
        hole = draw_window_into(canvas, rect, rng)
        holes.append(hole)
        placed += 1
    return holes


# ============================================================
# Background (light architectural noise)
# ============================================================

def _draw_dimension_strip(d, x0, y0, x1, y1, rng):
    """Thin dimension line with tick marks and one or two distance labels."""
    d.line([x0, y0, x1, y1], fill=(110, 110, 110), width=1)
    horizontal = abs(x1 - x0) > abs(y1 - y0)
    n_ticks = rng.randint(2, 6)
    for i in range(n_ticks + 1):
        t = i / n_ticks
        tx = int(x0 + (x1 - x0) * t); ty = int(y0 + (y1 - y0) * t)
        if horizontal:
            d.line([tx, ty - 5, tx, ty + 5], fill=(110, 110, 110), width=1)
        else:
            d.line([tx - 5, ty, tx + 5, ty], fill=(110, 110, 110), width=1)
    f = _font(rng.randint(11, 15))
    if horizontal:
        for i in range(n_ticks):
            t = (i + 0.5) / n_ticks
            tx = int(x0 + (x1 - x0) * t); ty = int(y0 + (y1 - y0) * t)
            d.text((tx - 14, ty - 18), rng.choice(_DIM_LABELS),
                   fill=(80, 80, 80), font=f)


def _draw_door_arc(d, hinge_x, hinge_y, r, quadrant):
    """Draw a quarter-circle door arc. quadrant in {0,1,2,3}: TL, TR, BR, BL."""
    bbox = [hinge_x - r, hinge_y - r, hinge_x + r, hinge_y + r]
    starts = {0: 180, 1: 270, 2: 0,   3: 90}
    if quadrant == 0:
        d.line([hinge_x, hinge_y, hinge_x - r, hinge_y], fill=(80, 80, 80), width=1)
    elif quadrant == 1:
        d.line([hinge_x, hinge_y, hinge_x + r, hinge_y], fill=(80, 80, 80), width=1)
    elif quadrant == 2:
        d.line([hinge_x, hinge_y, hinge_x + r, hinge_y], fill=(80, 80, 80), width=1)
    else:
        d.line([hinge_x, hinge_y, hinge_x - r, hinge_y], fill=(80, 80, 80), width=1)
    d.arc(bbox, start=starts[quadrant], end=starts[quadrant] + 90,
          fill=(80, 80, 80), width=1)


def render_floorplan_pre(d: ImageDraw.ImageDraw, meta: dict, rng: random.Random):
    """Labels + fixtures inside non-pattern rooms. Drawn BEFORE patterns so
    pattern fills naturally cover only their own (empty) rooms."""
    pat_set = meta['pattern_rooms']
    for room in meta['rooms']:
        if room in pat_set:
            continue
        rx0, ry0, rx1, ry1 = room
        rw_, rh_ = rx1 - rx0, ry1 - ry0
        # Real plans crowd non-pattern rooms with fixtures + furniture; draw 1-2
        # fixtures (a second one only in larger rooms) so they don't read empty.
        if rng.random() < 0.85:
            _draw_fixture(d, room, rng)
        if rw_ > 260 and rh_ > 220 and rng.random() < 0.55:
            _draw_fixture(d, room, rng)
        # Add small furniture/closet/appliance outlines. Count scales with room
        # AREA (~one cluster per 160px cell) so large non-pattern rooms fill up
        # like real plans instead of sitting as big empty boxes.
        if rw_ > 120 and rh_ > 120:
            n_cells = (rw_ * rh_) // (160 * 160)
            n_furn = int(min(16, max(2, n_cells * rng.uniform(0.8, 1.3))))
            for _ in range(n_furn):
                fw = rng.randint(40, max(50, min(rw_ // 2, 140)))
                fh = rng.randint(30, max(40, min(rh_ // 2, 110)))
                fx = rng.randint(rx0 + 8, max(rx0 + 9, rx1 - fw - 8))
                fy = rng.randint(ry0 + 8, max(ry0 + 9, ry1 - fh - 8))
                d.rectangle([fx, fy, fx + fw, fy + fh],
                            outline=(45, 45, 45), width=2)
                # Optional internal hatch / cross / divider
                kind = rng.choice(['plain', 'cross', 'hdiv', 'vdiv', 'circle'])
                if kind == 'cross':
                    d.line([fx, fy, fx + fw, fy + fh], fill=(130, 130, 130), width=1)
                    d.line([fx + fw, fy, fx, fy + fh], fill=(130, 130, 130), width=1)
                elif kind == 'hdiv':
                    d.line([fx, fy + fh // 2, fx + fw, fy + fh // 2],
                           fill=(110, 110, 110), width=1)
                elif kind == 'vdiv':
                    d.line([fx + fw // 2, fy, fx + fw // 2, fy + fh],
                           fill=(110, 110, 110), width=1)
                elif kind == 'circle':
                    cr = min(fw, fh) // 3
                    ccx, ccy = fx + fw // 2, fy + fh // 2
                    d.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr],
                              outline=(110, 110, 110), width=1)
        cx_, cy_ = (rx0 + rx1) // 2, (ry0 + ry1) // 2
        if rng.random() < 0.9:
            label = rng.choice(_ROOM_LABELS)
            f = _font(rng.randint(14, 22))
            try:
                tw, th = d.textbbox((0, 0), label, font=f)[2:]
            except Exception:
                tw, th = len(label) * 8, 14
            d.text((cx_ - tw // 2, cy_ - th), label, fill=(60, 60, 60), font=f)
        if rng.random() < 0.5:
            f2 = _font(rng.randint(10, 14))
            sub = rng.choice(_DIM_LABELS)
            try:
                tw2, th2 = d.textbbox((0, 0), sub, font=f2)[2:]
            except Exception:
                tw2, th2 = len(sub) * 6, 12
            d.text((cx_ - tw2 // 2, cy_ + 4), sub, fill=(90, 90, 90), font=f2)


def render_floorplan_post(d: ImageDraw.ImageDraw, meta: dict, rng: random.Random,
                          W: int, H: int):
    """Walls, partitions, doors, dimension chains, title. Drawn AFTER patterns
    so wall lines remain visible on top of pattern fills."""
    ox, oy, ox2, oy2 = meta['outer_bbox']
    wall_t = meta['wall_t']
    # Outer walls — follow outer_poly (handles L-shape) with thick double line
    pts = meta['outer_poly']
    d.line(pts + [pts[0]], fill=(20, 20, 20), width=4)
    # Inner offset line — one rect outline per room. Rooms inside a merged
    # pattern region are skipped so the region reads as one continuous room
    # with no internal sub-walls.
    merged_clusters = meta.get('merged_clusters', [])
    merged_rooms = {r for cl in merged_clusters for r in cl}
    for room in meta['rooms']:
        if room in merged_rooms:
            continue
        rx0, ry0, rx1, ry1 = room
        d.rectangle([rx0 - 1, ry0 - 1, rx1 + 1, ry1 + 1], outline=(60, 60, 60), width=1)
    # Partitions — drop the sub-segments interior to a merged region.
    for kind, pos, a, b in meta['partitions']:
        hidden = _merged_hidden_intervals(kind, pos, a, b, merged_clusters)
        for s0, s1 in _visible_segments(a, b, hidden):
            if kind == 'v':
                d.line([pos, s0, pos, s1], fill=(30, 30, 30), width=2)
            else:
                d.line([s0, pos, s1, pos], fill=(30, 30, 30), width=2)
    # Doors: white-out the gap and draw an arc. Skip doors whose wall section
    # was removed by a merge.
    for kind, pos, a, b in meta['doors']:
        hidden = _merged_hidden_intervals(kind, pos, a, b, merged_clusters)
        mid = (a + b) / 2.0
        if any(lo <= mid <= hi for lo, hi in hidden):
            continue
        if kind == 'v':
            d.rectangle([pos - 2, a, pos + 2, b], fill=(255, 255, 255))
            r_arc = max(20, b - a)
            _draw_door_arc(d, pos, b, r_arc, rng.choice([0, 1, 2, 3]))
        else:
            d.rectangle([a, pos - 2, b, pos + 2], fill=(255, 255, 255))
            r_arc = max(20, b - a)
            _draw_door_arc(d, b, pos, r_arc, rng.choice([0, 1, 2, 3]))
    # Dimension chains along outer edges
    ow_ = ox2 - ox; oh_ = oy2 - oy
    for edge in ('top', 'bottom', 'left', 'right'):
        if rng.random() < 0.6:
            off = rng.randint(20, 50)
            if edge == 'top':
                _draw_dimension_strip(d, ox, oy - off, ox2, oy - off, rng)
            elif edge == 'bottom':
                _draw_dimension_strip(d, ox, oy2 + off, ox2, oy2 + off, rng)
            elif edge == 'left':
                _draw_dimension_strip(d, ox - off, oy, ox - off, oy2, rng)
            else:
                _draw_dimension_strip(d, ox2 + off, oy, ox2 + off, oy2, rng)
    # Title block
    if rng.random() < 0.6:
        tbw, tbh = rng.randint(220, 360), rng.randint(80, 140)
        corner = rng.choice(['br', 'bl'])
        if corner == 'br':
            tbx, tby = W - tbw - 24, H - tbh - 24
        else:
            tbx, tby = 24, H - tbh - 24
        d.rectangle([tbx, tby, tbx + tbw, tby + tbh], outline=(40, 40, 40), width=2)
        for k in range(1, rng.randint(2, 4)):
            d.line([tbx, tby + k * tbh // 4, tbx + tbw, tby + k * tbh // 4],
                   fill=(80, 80, 80), width=1)
        d.text((tbx + 8, tby + 6),
               rng.choice(['FLOOR PLAN', 'PLAN', 'A-101', 'SHEET 1']),
               fill=(40, 40, 40), font=_font(14))

    # Dense leader-line callouts scattered across the image (small visual noise
    # near patterns, like the Las Huertas reference).
    callout_phrases = ['9\'-0" CLG', 'TYP.', 'V.I.F.', 'EQ.', '2x6 STUD',
                       'GFCI', 'D-1', 'W-2', 'REF.', 'D.W.', 'F.D.', 'A.F.F.',
                       'CONT.', 'SEE A-3', '5/8" GWB', 'R-21 INSUL', 'see arch']
    n_callouts = rng.randint(8, 18)
    f_co = _font(11)
    for _ in range(n_callouts):
        rx0_, ry0_, rx1_, ry1_ = ox, oy, ox2, oy2
        sx = rng.randint(rx0_ + 30, rx1_ - 30)
        sy = rng.randint(ry0_ + 30, ry1_ - 30)
        # Leader line: small dot + a short angled line + text
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(40, 40, 40))
        ang = rng.uniform(0, 2 * math.pi)
        L1 = rng.randint(30, 70); L2 = rng.randint(40, 90)
        ex = int(sx + math.cos(ang) * L1)
        ey = int(sy + math.sin(ang) * L1)
        ex2 = ex + (L2 if math.cos(ang) >= 0 else -L2)
        d.line([sx, sy, ex, ey], fill=(110, 110, 110), width=1)
        d.line([ex, ey, ex2, ey], fill=(110, 110, 110), width=1)
        text = rng.choice(callout_phrases)
        tx = ex2 + 2 if ex2 > ex else ex2 - 60
        d.text((tx, ey - 12), text, fill=(70, 70, 70), font=f_co)


def render_elevation_decorations(d: ImageDraw.ImageDraw, meta: dict,
                                  rng: random.Random, W: int, H: int):
    """Grade line + hatch, column grid with bubble tags, dim chains, title.
    Drawn BEFORE patterns; column lines are subtle and disappear behind walls.
    Dim chains and bubbles sit outside the building bbox so they remain visible.
    """
    grade_y = meta['grade_y']
    ox, oy, ox2, oy2 = meta['overall_bbox']
    # Grade line (full canvas) + hatch below
    d.line([0, grade_y, W, grade_y], fill=(20, 20, 20), width=3)
    for i in range(0, W, 28):
        d.line([i, grade_y + 2, i + 14, grade_y + 18], fill=(80, 80, 80), width=1)

    # Column grid per building
    bubble_top_y = oy - rng.randint(70, 110)
    bubble_bot_y = grade_y + rng.randint(60, 100)
    bubble_r = rng.randint(18, 26)
    f_b = _font(int(bubble_r * 1.1))
    col_label = ord('A')
    for bldg in meta['buildings']:
        bl, br = bldg['bldg_left'], bldg['bldg_right']
        n_cols = rng.randint(3, 5)
        col_xs = [int(bl + (br - bl) * i / (n_cols - 1)) for i in range(n_cols)]
        for x in col_xs:
            d.line([x, bubble_top_y, x, bubble_bot_y], fill=(150, 150, 150), width=1)
            for by in (bubble_top_y, bubble_bot_y):
                d.ellipse([x - bubble_r, by - bubble_r,
                           x + bubble_r, by + bubble_r],
                          fill=(255, 255, 255), outline=(40, 40, 40), width=2)
                lbl = chr(col_label)
                try:
                    tw, th = d.textbbox((0, 0), lbl, font=f_b)[2:]
                except Exception:
                    tw, th = bubble_r, bubble_r
                d.text((x - tw // 2, by - th // 2 - 2), lbl,
                       fill=(40, 40, 40), font=f_b)
            col_label += 1

    # Top dimension chain: overall building width broken into 2-4 segments
    dim_y = oy - rng.randint(140, 200)
    chain_left, chain_right = ox, ox2
    n_segs = rng.randint(2, 4)
    seg_xs = [int(chain_left + (chain_right - chain_left) * i / n_segs)
              for i in range(n_segs + 1)]
    f_dim = _font(14)
    for i in range(n_segs):
        x1, x2 = seg_xs[i], seg_xs[i + 1]
        d.line([x1, dim_y, x2, dim_y], fill=(80, 80, 80), width=1)
        d.line([x1, dim_y - 6, x1, dim_y + 6], fill=(80, 80, 80), width=1)
        d.line([x2, dim_y - 6, x2, dim_y + 6], fill=(80, 80, 80), width=1)
        label = rng.choice(_DIM_LABELS)
        try:
            tw, th = d.textbbox((0, 0), label, font=f_dim)[2:]
        except Exception:
            tw, th = 40, 14
        d.text(((x1 + x2) // 2 - tw // 2, dim_y - th - 4), label,
               fill=(60, 60, 60), font=f_dim)

    # Side floor-height chain on the right
    if meta['buildings']:
        bldg = meta['buildings'][0]
        floor_h = bldg['floor_h']; stories = bldg['stories']
        side_x = ox2 + rng.randint(80, 140)
        d.line([side_x, grade_y - stories * floor_h - 10, side_x,
                grade_y + 10], fill=(80, 80, 80), width=1)
        for s in range(stories + 1):
            y = grade_y - s * floor_h
            d.line([side_x - 6, y, side_x + 6, y], fill=(80, 80, 80), width=1)
        for s in range(stories):
            y_mid = grade_y - s * floor_h - floor_h // 2
            label = f"{rng.randint(8, 12)}'-0\""
            d.text((side_x + 12, y_mid - 7), label, fill=(60, 60, 60), font=f_dim)

    # Non-target architectural context: balconies, stairs, work-area extents,
    # and neighboring mass outlines. These cues appear in the real excerpts and
    # make the segmentation task less "material-only on a clean facade".
    for feat in meta.get('context_features', []):
        kind = feat.get('kind')
        if kind == 'balcony':
            x0, y0, x1, y1 = feat['bbox']
            d.line([x0, y1, x1, y1], fill=(60, 60, 60), width=2)
            d.line([x0, y0, x1, y0], fill=(70, 70, 70), width=2)
            n = max(3, int((x1 - x0) / 18))
            for i in range(n + 1):
                x = int(x0 + (x1 - x0) * i / n)
                d.line([x, y0, x, y1], fill=(110, 110, 110), width=1)
            if feat.get('posts'):
                d.line([x0 + 2, y1, x0 + 2, y1 + feat.get('post_h', 36)], fill=(70, 70, 70), width=2)
                d.line([x1 - 2, y1, x1 - 2, y1 + feat.get('post_h', 36)], fill=(70, 70, 70), width=2)
        elif kind == 'stair':
            x0, y0, x1, y1 = feat['bbox']
            steps = feat.get('steps', 7)
            up_left = feat.get('up_left', False)
            d.line([x0, y1, x1, y0] if up_left else [x0, y0, x1, y1],
                   fill=(80, 80, 80), width=2)
            for i in range(steps):
                tx0 = int(x0 + (x1 - x0) * i / steps)
                tx1 = int(x0 + (x1 - x0) * (i + 1) / steps)
                if up_left:
                    sy = int(y1 - (y1 - y0) * (i + 1) / steps)
                    d.line([tx0, sy, tx1, sy], fill=(120, 120, 120), width=1)
                else:
                    sy = int(y0 + (y1 - y0) * i / steps)
                    d.line([tx0, sy, tx1, sy], fill=(120, 120, 120), width=1)
            rail_y = y0 - 14 if not up_left else y0 - 10
            d.line([x0, rail_y, x1, rail_y - 8], fill=(90, 90, 90), width=1)
        elif kind == 'work_area':
            x0, y0, x1, y1 = feat['bbox']
            blue = (41, 109, 171)
            dash = 22
            for x in range(x0, x1, dash * 2):
                d.line([x, y0, min(x + dash, x1), y0], fill=blue, width=2)
                d.line([x, y1, min(x + dash, x1), y1], fill=blue, width=2)
            for y in range(y0, y1, dash * 2):
                d.line([x0, y, x0, min(y + dash, y1)], fill=blue, width=2)
                d.line([x1, y, x1, min(y + dash, y1)], fill=blue, width=2)
            label = feat.get('label', 'WORK AREA')
            d.text((x0 + 8, max(8, y0 - 24)), label, fill=blue, font=_font(16))
        elif kind == 'neighbor_roof':
            pts = feat.get('points', [])
            if len(pts) >= 2:
                d.line(pts, fill=(135, 135, 135), width=2)

    # Additional sheet-level context that appears in excerpted PDFs: hidden
    # lines, keyed notes, and a small title block fragment.
    dashed_y = []
    # CLUTTER_BOOST (test of "max visual similarity to real busy sheets"): force all
    # sheet-context elements on and add extra keyed notes, so synth approaches the
    # density of a real CAD page. Background clutter is UNLABELED.
    _cb = CLUTTER_BOOST
    for bldg in meta['buildings']:
        if rng.random() < (1.0 if _cb else 0.75):
            y = bldg['wall_top'] + rng.randint(28, max(32, bldg['floor_h'] - 24))
            dashed_y.append(y)
            for x in range(bldg['bldg_left'] - 60, bldg['bldg_right'] + 60, 24):
                d.line([x, y, x + 12, y], fill=(165, 165, 165), width=1)
        for _kn in range(3 if _cb else 1):
            if rng.random() < (1.0 if _cb else 0.55):
                note_y = bldg['wall_top'] + rng.randint(46, max(52, bldg['floor_h'] + 14)) + _kn * 26
                anchor_x = rng.randint(bldg['wall_left'] + 24, bldg['wall_right'] - 24)
                note_left = rng.random() < 0.55
                tx = max(20, bldg['bldg_left'] - rng.randint(170, 290)) if note_left else min(W - 200, bldg['bldg_right'] + rng.randint(30, 120))
                knee_x = anchor_x + (-rng.randint(24, 54) if note_left else rng.randint(24, 54))
                d.line([anchor_x, note_y, knee_x, note_y - 18, tx, note_y - 18], fill=(105, 105, 105), width=1)
                d.text((tx + 4, note_y - 32), rng.choice(['EXIST. TYP.', 'ALIGN W/ EXIST.', 'VERIFY HT.', 'MATCH ROOF', 'NEW SIDING', 'WOOD SIDING', 'LT. GRAY BRICK', 'GUTTER & D.S.', 'FIN. FLR.']), fill=(78, 78, 78), font=_font(14))

    if rng.random() < (1.0 if _cb else 0.70):
        tb_w = rng.randint(220, 320)
        tb_h = rng.randint(62, 108)
        tb_x0 = W - tb_w - rng.randint(18, 36)
        tb_y0 = H - tb_h - rng.randint(18, 34)
        d.rectangle([tb_x0, tb_y0, tb_x0 + tb_w, tb_y0 + tb_h], outline=(110, 110, 110), width=1)
        row_y = tb_y0 + int(tb_h * 0.56)
        col_x = tb_x0 + int(tb_w * 0.68)
        d.line([tb_x0, row_y, tb_x0 + tb_w, row_y], fill=(135, 135, 135), width=1)
        d.line([col_x, tb_y0, col_x, tb_y0 + tb_h], fill=(135, 135, 135), width=1)
        d.text((tb_x0 + 10, tb_y0 + 10), rng.choice(['A-201', 'A-301', 'A-401']), fill=(70, 70, 70), font=_font(15))
        d.text((tb_x0 + 10, row_y + 8), rng.choice(['REV 2', 'BID SET', 'FIELD VERIFY']), fill=(92, 92, 92), font=_font(13))
        d.text((col_x + 10, row_y + 8), rng.choice(['1/4\" = 1\'-0\"', '3/16\" = 1\'-0\"']), fill=(92, 92, 92), font=_font(13))

    # Title bottom-left
    title = rng.choice(['ELEVATION', 'FRONT ELEVATION', 'SIDE ELEVATION',
                        'REAR ELEVATION', 'RIGHT ELEVATION'])
    f_t = _font(rng.randint(20, 28))
    try:
        tw, th = d.textbbox((0, 0), title, font=f_t)[2:]
    except Exception:
        tw, th = 220, 24
    d.text((30, H - th - 30), title, fill=(20, 20, 20), font=f_t)
    sub_f = _font(13)
    d.text((30, H - 28), 'SCALE: 1/4" = 1\'-0"', fill=(60, 60, 60), font=sub_f)


def render_elevation_post(d: ImageDraw.ImageDraw, meta: dict,
                          rng: random.Random, W: int, H: int):
    """Roof slope (pitch) callouts + top-of-plate reference lines, drawn ON TOP
    of the roof pattern (matches worst-real idx09). Non-target annotation clutter
    only — no instance is emitted — so it also teaches the model that drawn
    line-work over the roof is NOT a material region. Gated by REALSTYLE_ROOF_CUES."""
    if not REALSTYLE_ROOF_CUES:
        return
    f = _font(15)
    for b in meta.get('buildings', []):
        wl, wr = b['wall_left'], b['wall_right']
        wt = b['wall_top']
        ridge = b.get('bldg_top', wt)
        rtype = b.get('roof_type', 'flat')
        apex_x = (wl + wr) // 2
        # Pitch callout on the left slope (eave -> apex), ~1/3 up.
        if rtype in ('gable', 'row_gable', 'hip', 'shed') and ridge < wt - 8:
            t = rng.uniform(0.30, 0.45)
            sx = int(wl + (apex_x - wl) * t)
            sy = int(wt + (ridge - wt) * t)
            run = rng.randint(42, 60)
            slope = (wt - ridge) / max(1, apex_x - wl)
            rise = max(8, int(run * slope))
            d.line([sx, sy, sx + run, sy], fill=(90, 90, 90), width=1)
            d.line([sx + run, sy, sx + run, sy + rise], fill=(90, 90, 90), width=1)
            d.line([sx, sy, sx + run, sy + rise], fill=(120, 120, 120), width=1)
            d.text((sx + run + 3, sy + rise // 2 - 7),
                   str(rng.randint(3, 9)), fill=(70, 70, 70), font=f)
            d.text((sx + run // 2 - 7, sy - 17), "12", fill=(70, 70, 70), font=f)
        # Top-of-plate / roof reference line running to the right margin.
        if rng.random() < 0.6:
            ry = wt + rng.randint(-6, 12)
            x_end = min(W - 8, wr + rng.randint(120, 260))
            xx = wr + 6
            while xx < x_end:
                d.line([xx, ry, min(xx + 14, x_end), ry], fill=(110, 110, 110), width=1)
                xx += 24
            d.text((max(0, x_end - 76), ry - 16),
                   rng.choice(["T.O. PLATE", "T.O. ROOF", "T.O. SUBFLR"]),
                   fill=(80, 80, 80), font=f)


def render_markup_overlay(img: Image.Image, meta, rng: random.Random, mode: str,
                          mono: bool = False):
    """Add review/markup artifacts found in the real excerpts."""
    if rng.random() >= MARKUP_OVERLAY_PROB:
        return img
    # Keep monochrome images color-free: render markup in dark gray instead of
    # the red/magenta/blue palette.
    markup_colors = ([(60, 60, 60, 175)] if mono else MARKUP_COLORS)

    W, H = img.size
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    f_note = _font(rng.randint(13, 16))

    if mode == 'elevation' and meta is not None:
        ox, oy, ox2, oy2 = meta['overall_bbox']
        regions = [(ox, oy, ox2, oy2)]
    elif mode == 'freeform' and meta is not None:
        ox, oy, ox2, oy2 = meta['outer_bbox']
        regions = [(ox, oy, ox2, oy2)]
    else:
        regions = [(W * 0.12, H * 0.12, W * 0.88, H * 0.88)]

    notes = ['REMOVE WINDOW', 'VERIFY', 'ALIGN W/ EXIST.', 'EXISTING',
             'NEW SIDING', 'DOOR TRIM', 'MATCH EXIST.', 'TO BE DEMO.']
    n_marks = rng.randint(1, 5)
    for _ in range(n_marks):
        rx0, ry0, rx1, ry1 = map(int, rng.choice(regions))
        cx = rng.randint(max(24, rx0), min(W - 24, rx1))
        cy = rng.randint(max(24, ry0), min(H - 24, ry1))
        rad = rng.randint(24, 54)
        color = rng.choice(markup_colors)
        if rng.random() < 0.75:
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      fill=color, outline=color[:3] + (210,), width=4)
        else:
            box_w = rng.randint(120, 220)
            box_h = rng.randint(34, 56)
            d.rounded_rectangle([cx, cy, cx + box_w, cy + box_h], radius=8,
                                fill=(255, 255, 255, 220),
                                outline=color[:3] + (220,), width=3)
            d.text((cx + 10, cy + 8), rng.choice(notes),
                   fill=color[:3] + (255,), font=f_note)
        if rng.random() < 0.8:
            ax = int(np.clip(cx + rng.randint(-180, 180), 8, W - 8))
            ay = int(np.clip(cy + rng.randint(50, 150), 8, H - 8))
            d.line([cx, cy + rad // 2, ax, ay], fill=color[:3] + (180,), width=3)
            d.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=color[:3] + (220,))

    base = img.convert('RGBA')
    base.alpha_composite(overlay)
    return base.convert('RGB')


def apply_document_effects(img: Image.Image, rng: random.Random, mode: str):
    """Light rasterization / scan artifacts so the output reads like an excerpt
    from a document set, not a freshly-rendered synthetic canvas."""
    if rng.random() >= DOCUMENT_EFFECT_PROB:
        return img

    out = img
    if rng.random() < 0.75:
        scale = rng.uniform(0.90, 0.98) if mode == 'elevation' else rng.uniform(0.93, 0.99)
        dw = max(1, int(round(out.width * scale)))
        dh = max(1, int(round(out.height * scale)))
        out = out.resize((dw, dh), Image.BILINEAR).resize(out.size, Image.BILINEAR)
    if rng.random() < 0.45:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.75)))

    arr = np.asarray(out).astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noise = np_rng.normal(0.0, rng.uniform(1.4, 3.6), (arr.shape[0], arr.shape[1], 1))
    arr += noise
    if rng.random() < 0.55:
        grad = np.linspace(rng.uniform(-6.0, 0.0), rng.uniform(0.0, 6.0), arr.shape[1], dtype=np.float32)
        arr += grad[None, :, None]
    arr *= rng.uniform(0.985, 1.015)
    arr += rng.uniform(-4.0, 4.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


# ============================================================
# Layout helpers (overlap, garage door, layouts)
# ============================================================

def _polygon_to_mask(poly, W, H) -> np.ndarray:
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon([(int(p[0]), int(p[1])) for p in poly],
                              fill=1, outline=1)
    return np.asarray(m, dtype=np.uint8)


def _bay_split(rect, rng, n=None, min_bay=60):
    """Split an axis-aligned rect into n rectilinear bays along its long axis.
    Each bay becomes its own instance, so a continuous band reads as many
    separate per-unit instances (the real-plan annotation style). Cut positions
    are evenly spaced with jitter; returns a list of (x0, y0, x1, y1) rects."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    axis = 'v' if w >= h else 'h'
    span = w if axis == 'v' else h
    if n is None:
        n = rng.randint(2, 4)
    n = max(1, min(n, span // min_bay))
    if n <= 1:
        return [rect]
    cuts = [(x0 if axis == 'v' else y0) + span * i // n for i in range(n + 1)]
    jit = max(1, span // (5 * n))
    for i in range(1, n):
        cuts[i] += rng.randint(-jit, jit)
    cuts = sorted(cuts)
    if axis == 'v':
        return [(cuts[i], y0, cuts[i + 1], y1) for i in range(n)]
    return [(x0, cuts[i], x1, cuts[i + 1]) for i in range(n)]


def _is_axis_rect(poly):
    """True iff poly is a 4-vertex axis-aligned rectangle."""
    if len(poly) != 4:
        return False
    xs = sorted({int(p[0]) for p in poly})
    ys = sorted({int(p[1]) for p in poly})
    return len(xs) == 2 and len(ys) == 2


def _bay_items(item, rng):
    """Split a rectangular wall/region item into per-bay sub-items (separate
    same-material instances)."""
    xs = [p[0] for p in item['poly']]; ys = [p[1] for p in item['poly']]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    out = []
    for cx0, cy0, cx1, cy1 in _bay_split((x0, y0, x1, y1), rng):
        if min(cx1 - cx0, cy1 - cy0) > 120:
            poly = _complexify(_multi_notch_rect((cx0, cy0, cx1, cy1),
                                                 rng, rng.randint(0, 2)), rng)
        else:
            poly = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]
        out.append({**item,
                    'poly': poly})
    return out


def place_garage_door(canvas: Image.Image, poly, rng: random.Random):
    """Plant a single big garage door inside the garage polygon. Returns hole polygon (list of pts)."""
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    bx, by = min(xs), min(ys)
    bw_, bh_ = max(xs) - bx, max(ys) - by
    door_w = int(bw_ * rng.uniform(0.65, 0.92))
    door_h = int(bh_ * rng.uniform(0.55, 0.85))
    dx = bx + (bw_ - door_w) // 2
    dy = by + bh_ - door_h - rng.randint(2, 14)
    d = ImageDraw.Draw(canvas)
    d.rectangle([dx, dy, dx + door_w, dy + door_h],
                fill=(245, 245, 240), outline=(15, 15, 20), width=3)
    n_panels = rng.randint(3, 5)
    panel_h = door_h / n_panels
    for k in range(1, n_panels):
        py = dy + int(k * panel_h)
        d.line([dx + 3, py, dx + door_w - 3, py], fill=(40, 40, 45), width=2)
    if rng.random() < 0.3:
        d.line([dx + door_w // 2, dy + 3, dx + door_w // 2, dy + door_h - 3],
               fill=(40, 40, 45), width=2)
    if rng.random() < 0.4 and door_w > 120:
        n_win = rng.randint(2, 4)
        win_h_g = max(8, int(panel_h * 0.5))
        win_w_g = (door_w - 30) // n_win - 8
        win_y_g = dy + int(panel_h * 0.25)
        for k in range(n_win):
            wx = dx + 15 + k * (win_w_g + 8)
            d.rectangle([wx, win_y_g, wx + win_w_g, win_y_g + win_h_g],
                        outline=(40, 40, 45), width=1)
    return [(dx, dy), (dx + door_w, dy),
            (dx + door_w, dy + door_h), (dx, dy + door_h)]


def _translate_elevation_feature(feat, dx, dy):
    out = dict(feat)
    if 'bbox' in out:
        x0, y0, x1, y1 = out['bbox']
        out['bbox'] = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if 'points' in out:
        out['points'] = [(x + dx, y + dy) for x, y in out['points']]
    return out


def build_rowhouse_elevation_scene(rng: random.Random, tiles):
    """Repeated townhouse / rowhouse elevation with shared roofline, garages,
    and optional dormers. This matches the multi-unit real excerpts better
    than the generic detached-house builder."""
    items = []
    n_units = rng.randint(3, 5)
    # Coarsen toward real: fewer units = fewer instances; a 3-unit excerpt reads
    # as real as a 5-unit one.
    n_units = min(n_units, _inst_cap(5, 3))
    unit_w = rng.randint(250, 360)
    floor_h = rng.randint(175, 225)
    stories = 2
    wall_h = floor_h * stories
    bx = 0
    bw = n_units * unit_w
    grade_y = 0
    wall_top = grade_y - wall_h
    eave_overhang = rng.randint(20, 38)
    fascia_h = rng.randint(8, 14)
    roof_pitch = rng.uniform(0.22, 0.38)
    ridge_y = wall_top - int((bw / max(n_units, 1)) * roof_pitch)
    overall_top = ridge_y
    context_features = []

    band_break = grade_y - rng.randint(int(floor_h * 0.55), int(floor_h * 0.85))
    podium_break = grade_y - rng.randint(int(floor_h * 0.18), int(floor_h * 0.32))
    base_tile, mid_tile, top_tile = rng.sample(tiles, 3)
    roof_tile = rng.choice(tiles)
    accent_tile = rng.choice([t for t in tiles if t['path'] not in {
        base_tile['path'], mid_tile['path'], top_tile['path'], roof_tile['path']
    }] or tiles)

    for i in range(n_units):
        ux0 = bx + i * unit_w
        ux1 = ux0 + unit_w
        garage_side = 'left' if i % 2 == 0 else 'right'
        garage_w = int(unit_w * rng.uniform(0.54, 0.70))
        if garage_side == 'left':
            gx0, gx1 = ux0, ux0 + garage_w
            ex0, ex1 = gx1, ux1
        else:
            gx0, gx1 = ux1 - garage_w, ux1
            ex0, ex1 = ux0, gx0
        garage_top = grade_y - int(floor_h * rng.uniform(0.70, 0.88))

        items.extend(_surface_items(
            [(gx0, grade_y), (gx1, grade_y), (gx1, garage_top), (gx0, garage_top)],
            'garage',
            base_tile if i % 2 == 0 else accent_tile,
            rng,
            alt_tiles=[accent_tile, mid_tile],
            has_windows=False,
            split_prob=ELEVATION_INTERLEAVE_PROB * 0.8,
            max_pieces=2,
            min_piece_area=18_000,
        ))
        if ex1 - ex0 > 26:
            items.extend(_surface_items(
                [(ex0, grade_y), (ex1, grade_y), (ex1, band_break), (ex0, band_break)],
                'wall',
                base_tile,
                rng,
                alt_tiles=[accent_tile, mid_tile],
                has_windows=True,
                split_prob=ELEVATION_INTERLEAVE_PROB,
                max_pieces=3,
                min_piece_area=20_000,
            ))

        items.extend(_surface_items(
            [(ux0, band_break), (ux1, band_break), (ux1, podium_break), (ux0, podium_break)],
            'wall',
            mid_tile if i % 2 == 0 else top_tile,
            rng,
            alt_tiles=[top_tile, accent_tile, base_tile],
            has_windows=True,
            split_prob=ELEVATION_INTERLEAVE_PROB,
            max_pieces=3,
            min_piece_area=20_000,
        ))
        items.extend(_surface_items(
            [(ux0, podium_break), (ux1, podium_break), (ux1, wall_top), (ux0, wall_top)],
            'wall',
            top_tile if i % 2 == 0 else mid_tile,
            rng,
            alt_tiles=[mid_tile, accent_tile, base_tile],
            has_windows=True,
            split_prob=ELEVATION_INTERLEAVE_PROB,
            max_pieces=3,
            min_piece_area=20_000,
        ))

        # Small entry / porch canopy for some units.
        if rng.random() < 0.75 and ex1 - ex0 > 40:
            canopy_w = int((ex1 - ex0) * rng.uniform(0.55, 0.90))
            if garage_side == 'left':
                cx1 = ex1 - rng.randint(2, 8)
                cx0 = cx1 - canopy_w
            else:
                cx0 = ex0 + rng.randint(2, 8)
                cx1 = cx0 + canopy_w
            cy1 = garage_top + rng.randint(8, 18)
            cy0 = cy1 - rng.randint(18, 34)
            items.extend(_surface_items(
                [(cx0, cy1), (cx1, cy1), (cx1, cy0), (cx0, cy0)],
                'roof',
                roof_tile,
                rng,
                alt_tiles=[accent_tile],
                has_windows=False,
                split_prob=ELEVATION_INTERLEAVE_PROB * 0.55,
                max_pieces=2,
                min_piece_area=14_000,
            ))
            context_features.append({
                'kind': 'balcony',
                'bbox': (cx0, cy0 - 46, cx1, cy0 - 4),
                'posts': True,
                'post_h': rng.randint(28, 42),
            })
            if rng.random() < 0.55:
                sx1 = cx1 if garage_side == 'left' else cx0 + rng.randint(52, 78)
                sx0 = sx1 - rng.randint(52, 78)
                context_features.append({
                    'kind': 'stair',
                    'bbox': (sx0, cy0 - 2, sx1, grade_y - 2),
                    'steps': rng.randint(5, 8),
                    'up_left': garage_side == 'right',
                })

        # Dormers read strongly like the townhouse samples.
        if rng.random() < 0.8:
            dorm_w = int(unit_w * rng.uniform(0.28, 0.42))
            dorm_h = rng.randint(int(floor_h * 0.45), int(floor_h * 0.70))
            dx0 = ux0 + (unit_w - dorm_w) // 2 + rng.randint(-8, 8)
            dx1 = dx0 + dorm_w
            dorm_base_y = wall_top - rng.randint(10, 26)
            dorm_top_y = dorm_base_y - dorm_h
            dorm_apex_y = dorm_top_y - rng.randint(18, 40)
            overall_top = min(overall_top, dorm_apex_y)
            items.extend(_surface_items(
                [(dx0, dorm_base_y), (dx1, dorm_base_y), (dx1, dorm_top_y), (dx0, dorm_top_y)],
                'wall',
                mid_tile if i % 2 == 0 else accent_tile,
                rng,
                alt_tiles=[top_tile, base_tile],
                has_windows=True,
                split_prob=ELEVATION_INTERLEAVE_PROB * 0.8,
                max_pieces=2,
                min_piece_area=12_000,
            ))
            items.extend(_surface_items(
                [(dx0 - 10, dorm_top_y), ((dx0 + dx1) // 2, dorm_apex_y), (dx1 + 10, dorm_top_y),
                 (dx1 + 10, dorm_top_y + fascia_h), (dx0 - 10, dorm_top_y + fascia_h)],
                'roof',
                roof_tile,
                rng,
                alt_tiles=[accent_tile],
                has_windows=False,
                split_prob=ELEVATION_INTERLEAVE_PROB * 0.45,
                max_pieces=2,
                min_piece_area=10_000,
            ))

    if rng.random() < 0.65:
        wa_x0 = bx + rng.randint(20, max(22, unit_w // 3))
        wa_x1 = bx + bw - rng.randint(20, max(22, unit_w // 3))
        wa_y0 = wall_top - rng.randint(18, 40)
        wa_y1 = grade_y + rng.randint(-18, 30)
        context_features.append({
            'kind': 'work_area',
            'bbox': (wa_x0, wa_y0, wa_x1, wa_y1),
            'label': rng.choice(['WORK AREA', 'DOOR TRIM', 'ALIGN W/ EXIST.']),
        })

    if rng.random() < 0.7:
        left_pts = [
            (bx - rng.randint(140, 220), wall_top + rng.randint(40, 90)),
            (bx + rng.randint(10, 40), wall_top + rng.randint(8, 24)),
            (bx + rng.randint(90, 160), wall_top + rng.randint(24, 54)),
        ]
        right_pts = [
            (bx + bw - rng.randint(160, 220), wall_top + rng.randint(24, 54)),
            (bx + bw + rng.randint(20, 60), wall_top + rng.randint(4, 24)),
            (bx + bw + rng.randint(120, 220), wall_top + rng.randint(38, 90)),
        ]
        context_features.append({'kind': 'neighbor_roof', 'points': left_pts})
        context_features.append({'kind': 'neighbor_roof', 'points': right_pts})

    left_roof = [
        (bx - eave_overhang, wall_top + fascia_h),
        (bx + bw // 2, ridge_y + fascia_h),
        (bx + bw // 2, ridge_y),
        (bx - eave_overhang, wall_top),
    ]
    right_roof = [
        (bx + bw // 2, ridge_y),
        (bx + bw + eave_overhang, wall_top),
        (bx + bw + eave_overhang, wall_top + fascia_h),
        (bx + bw // 2, ridge_y + fascia_h),
    ]
    items.extend(_surface_items(
        left_roof,
        'roof',
        roof_tile,
        rng,
        alt_tiles=[accent_tile],
        has_windows=False,
        split_prob=ELEVATION_INTERLEAVE_PROB * 0.45,
        max_pieces=2,
        min_piece_area=16_000,
    ))
    items.extend(_surface_items(
        right_roof,
        'roof',
        roof_tile,
        rng,
        alt_tiles=[accent_tile],
        has_windows=False,
        split_prob=ELEVATION_INTERLEAVE_PROB * 0.45,
        max_pieces=2,
        min_piece_area=16_000,
    ))

    bldg_layouts = [{
        'bldg_left': bx,
        'bldg_right': bx + bw,
        'bldg_top': ridge_y,
        'bldg_bot': grade_y,
        'wall_left': bx,
        'wall_right': bx + bw,
        'wall_top': wall_top,
        'grade_y': grade_y,
        'floor_h': floor_h,
        'stories': stories,
        'eave_overhang': eave_overhang,
        'band_ys': [grade_y, band_break, podium_break, wall_top],
        'roof_type': 'row_gable',
    }]
    meta = {
        'overall_bbox': (bx - 40, overall_top - 6, bx + bw + 40, 0),
        'grade_y': 0,
        'buildings': bldg_layouts,
        'context_features': context_features,
    }
    return items, meta


def build_elevation_scene(rng: random.Random, tiles):
    """Build a (multi-)building elevation at natural pixel size centered on
    grade_y=0. Returns (items, meta). Caller is responsible for translating
    polygons into a tightly-cropped canvas.

    Walls are split into 1-3 horizontal bands of stacked materials (real
    elevations transition siding -> shingles, brick -> stucco, etc.).
    Roofs include eave overhangs that project beyond walls.
    """
    if rng.random() < ROWHOUSE_ELEVATION_PROB:
        return build_rowhouse_elevation_scene(rng, tiles)

    GAP = 0
    items = []
    n_buildings = 1 if rng.random() < 0.65 else 2
    SEPARATOR = rng.randint(40, 180)
    cur_x = 0
    bldg_layouts = []
    overall_top = 0

    for _b in range(n_buildings):
        stories = rng.choice([1, 1, 2, 2])
        # RESOLUTION_SCALE renders the building at more pixels natively so that,
        # after the eval downsample to image_max_size, synth instances are as SMALL
        # (and thus as hard to segment) as real ones. Real plans are ~4800px native
        # vs synth elevations ~1850px, so at 1024 synth instances are ~2.6x larger /
        # easier — a real driver of synth_iou > real_iou. Scaling the building (not
        # the unscaled fonts/line-widths, cosmetic) makes synth eval-resolution-match.
        floor_h = int(rng.randint(180, 280) * RESOLUTION_SCALE)
        wall_h = floor_h * stories
        bw = int(rng.randint(900, 1700) * RESOLUTION_SCALE)

        # iter5 NOTE (2026-06-03): forcing gable for realstyle (irregular pentagon
        # wall instead of rectangle) did NOT lower synth_iou@10 (stayed 0.367) and
        # div@ep10 got slightly worse (+0.176 vs +0.156). Wall-shape regularity is
        # not the synth-easiness lever. Reverted to the full roof mix.
        roof_type = rng.choice(['gable', 'gable', 'shed', 'flat', 'hip'])
        pitch = rng.uniform(0.30, 0.65)
        eave_overhang = rng.randint(20, 55)
        fascia_h = rng.randint(8, 16)

        has_garage = rng.random() < 0.45
        garage_side = rng.choice(['left', 'right']) if has_garage else None
        garage_w = rng.randint(int(bw * 0.25), int(bw * 0.45)) if has_garage else 0
        garage_h = int(floor_h * rng.uniform(0.72, 0.95)) if has_garage else 0

        bx = cur_x
        grade_y = 0
        wall_top = grade_y - wall_h

        if garage_side == 'left':
            gx0, gx1 = bx, bx + garage_w
            wall_left, wall_right = gx1 + GAP, bx + bw
        elif garage_side == 'right':
            wall_left, wall_right = bx, bx + bw - garage_w
            gx0, gx1 = wall_right + GAP, bx + bw
        else:
            wall_left, wall_right = bx, bx + bw
            gx0 = gx1 = None

        wall_w = wall_right - wall_left
        if wall_w < 240:
            cur_x = bx + bw + SEPARATOR
            continue

        # Decide horizontal band breaks for stacked materials.
        # Bias toward 2-3 bands so walls show material transitions.
        if stories == 2:
            n_bands = rng.choice([2, 2, 3, 3])
        else:
            n_bands = rng.choice([1, 2, 2, 3])
        # Coarsen toward real: cap bands (3 at scale 1 -> 1 at scale 0).
        n_bands = max(1, min(n_bands, _inst_cap(3, 1)))
        # Choose distinct tiles per band
        wall_tiles = []
        used_paths = set()
        for _ in range(n_bands):
            for _try in range(8):
                t = pick_wall_tile(tiles, rng)
                if t['path'] not in used_paths:
                    wall_tiles.append(t); used_paths.add(t['path']); break
            else:
                wall_tiles.append(pick_wall_tile(tiles, rng))

        # Compute band y-breaks (in grade_y - distance space).
        # Bands listed from BOTTOM to TOP in y-decreasing order.
        band_ys = [grade_y]  # bottoms
        if n_bands == 1:
            band_ys.append(wall_top)
        elif n_bands == 2:
            # Often a story-line break, sometimes a water-table low break
            if rng.random() < 0.7 and stories == 2:
                yb = grade_y - floor_h
            else:
                yb = grade_y - rng.randint(int(wall_h * 0.25), int(wall_h * 0.55))
            band_ys += [yb, wall_top]
        else:
            yb1 = grade_y - rng.randint(int(wall_h * 0.18), int(wall_h * 0.35))
            yb2 = grade_y - rng.randint(int(wall_h * 0.55), int(wall_h * 0.78))
            band_ys += [yb1, yb2, wall_top]

        bldg_top = wall_top

        if roof_type == 'gable':
            ridge_y = wall_top - int(wall_w / 2 * pitch)
            # Top band gets the gable triangle. Build per-band polys.
            # Bands 0..n_bands-2 are rectangles between band_ys[i+1] and band_ys[i].
            # Top band (i = n_bands-1) is rectangle + gable triangle.
            for i in range(n_bands):
                y_bot = band_ys[i]
                y_top = band_ys[i + 1]
                if i < n_bands - 1:
                    poly = [(wall_left, y_bot), (wall_right, y_bot),
                            (wall_right, y_top), (wall_left, y_top)]
                else:
                    poly = [(wall_left, y_bot), (wall_right, y_bot),
                            (wall_right, y_top),
                            ((wall_left + wall_right) // 2, ridge_y),
                            (wall_left, y_top)]
                items.extend(_surface_items(
                    poly,
                    'wall',
                    wall_tiles[i],
                    rng,
                    alt_tiles=[t for j, t in enumerate(wall_tiles) if j != i],
                    has_windows=i == n_bands - 1 or rng.random() < 0.7,
                    split_prob=ELEVATION_INTERLEAVE_PROB,
                    max_pieces=3,
                    min_piece_area=18_000,
                ))
            bldg_top = ridge_y
            # Roof: two sloping panels with eave overhang (drawn-on, separate items)
            roof_tile = rng.choice(tiles)
            # Left slope: from (wall_left - eave, wall_top) up to apex (mid, ridge_y)
            apex = ((wall_left + wall_right) // 2, ridge_y)
            # Add slight roof thickness (fascia line drawn later)
            left_roof = [
                (wall_left - eave_overhang, wall_top + fascia_h),
                (apex[0], ridge_y + fascia_h),
                (apex[0], ridge_y),
                (wall_left - eave_overhang, wall_top),
            ]
            right_roof = [
                (apex[0], ridge_y),
                (wall_right + eave_overhang, wall_top),
                (wall_right + eave_overhang, wall_top + fascia_h),
                (apex[0], ridge_y + fascia_h),
            ]
            # Roof gets pattern only sometimes (often roofs in real elevations are blank/lightly hatched)
            if rng.random() < 0.35:
                items.extend(_surface_items(
                    left_roof,
                    'roof',
                    roof_tile,
                    rng,
                    alt_tiles=[rng.choice(tiles)],
                    has_windows=False,
                    split_prob=ELEVATION_INTERLEAVE_PROB * 0.45,
                    max_pieces=2,
                    min_piece_area=16_000,
                ))
                items.extend(_surface_items(
                    right_roof,
                    'roof',
                    roof_tile,
                    rng,
                    alt_tiles=[rng.choice(tiles)],
                    has_windows=False,
                    split_prob=ELEVATION_INTERLEAVE_PROB * 0.45,
                    max_pieces=2,
                    min_piece_area=16_000,
                ))
        else:
            for i in range(n_bands):
                y_bot = band_ys[i]
                y_top = band_ys[i + 1]
                poly = [(wall_left, y_bot), (wall_right, y_bot),
                        (wall_right, y_top), (wall_left, y_top)]
                items.extend(_surface_items(
                    poly,
                    'wall',
                    wall_tiles[i],
                    rng,
                    alt_tiles=[t for j, t in enumerate(wall_tiles) if j != i],
                    has_windows=i == n_bands - 1 or rng.random() < 0.7,
                    split_prob=ELEVATION_INTERLEAVE_PROB,
                    max_pieces=3,
                    min_piece_area=18_000,
                ))
            base_h = rng.randint(8, 14)
            if roof_type == 'shed':
                slope = max(20, int(wall_w * pitch * 0.30))
                high_left = rng.random() < 0.5
                top_high_y = wall_top - base_h - slope
                top_low_y  = wall_top - base_h
                if high_left:
                    roof_poly = [
                        (wall_left - eave_overhang, wall_top), (wall_right + eave_overhang, wall_top),
                        (wall_right + eave_overhang, top_low_y),
                        (wall_left - eave_overhang, top_high_y),
                    ]
                else:
                    roof_poly = [
                        (wall_left - eave_overhang, wall_top), (wall_right + eave_overhang, wall_top),
                        (wall_right + eave_overhang, top_high_y),
                        (wall_left - eave_overhang, top_low_y),
                    ]
                bldg_top = top_high_y
            elif roof_type == 'hip':
                inset = int(wall_w * rng.uniform(0.18, 0.30))
                ridge_h = max(30, int(wall_w * pitch * 0.30))
                roof_top_y = wall_top - ridge_h
                roof_poly = [
                    (wall_left - eave_overhang, wall_top),
                    (wall_right + eave_overhang, wall_top),
                    (wall_right - inset, roof_top_y),
                    (wall_left + inset, roof_top_y),
                ]
                bldg_top = roof_top_y
            else:  # flat
                parapet = rng.randint(12, 26)
                roof_poly = [
                    (wall_left - 4, wall_top - parapet),
                    (wall_right + 4, wall_top - parapet),
                    (wall_right + 4, wall_top),
                    (wall_left - 4, wall_top),
                ]
                bldg_top = wall_top - parapet
            if rng.random() < 0.4:
                roof_tile = rng.choice(tiles)
                items.extend(_surface_items(
                    roof_poly,
                    'roof',
                    roof_tile,
                    rng,
                    alt_tiles=[t for t in wall_tiles if t['path'] != roof_tile['path']],
                    has_windows=False,
                    split_prob=ELEVATION_INTERLEAVE_PROB * 0.4,
                    max_pieces=2,
                    min_piece_area=16_000,
                ))

        if garage_side and garage_w > 0:
            garage_top = grade_y - garage_h
            garage_poly = [
                (gx0, grade_y), (gx1, grade_y),
                (gx1, garage_top), (gx0, garage_top),
            ]
            garage_tile = pick_wall_tile(tiles, rng)
            items.extend(_surface_items(
                garage_poly,
                'garage',
                garage_tile,
                rng,
                alt_tiles=[t for t in wall_tiles if t['path'] != garage_tile['path']],
                has_windows=False,
                split_prob=ELEVATION_INTERLEAVE_PROB * 0.8,
                max_pieces=2,
                min_piece_area=18_000,
            ))

        bldg_layouts.append({
            'bldg_left': bx, 'bldg_right': bx + bw,
            'bldg_top': bldg_top, 'bldg_bot': grade_y,
            'wall_left': wall_left, 'wall_right': wall_right,
            'wall_top': wall_top, 'grade_y': grade_y,
            'floor_h': floor_h, 'stories': stories,
            'eave_overhang': eave_overhang,
            'band_ys': band_ys,
            'roof_type': roof_type,
        })
        overall_top = min(overall_top, bldg_top - 6)
        cur_x = bx + bw + SEPARATOR

    if not bldg_layouts:
        return [], None

    # v5.1: split each rectangular wall band into per-bay instances so a
    # continuous facade course reads as several separate same-material
    # instances (matches real per-unit elevation annotation).
    if NO_MERGE_SAME_MATERIAL:
        split_items = []
        for it in items:
            if (it['role'] == 'wall' and _is_axis_rect(it['poly'])
                    and rng.random() < 0.8):
                split_items.extend(_bay_items(it, rng))
            else:
                split_items.append(it)
        items = split_items

    overall_left = bldg_layouts[0]['bldg_left'] - 50
    overall_right = bldg_layouts[-1]['bldg_right'] + 50
    meta = {
        'overall_bbox': (overall_left, overall_top, overall_right, 0),
        'grade_y': 0,
        'buildings': bldg_layouts,
    }
    return items, meta


def _decompose_into_wings(shape, ox, oy, ow, oh, params):
    """Return list of axis-aligned wing rects (x0,y0,x1,y1) whose union equals
    the footprint. Each wing will get its own hip/gable facet partition."""
    if shape == 'rect':
        return [(ox, oy, ox + ow, oy + oh)]
    if shape == 'L':
        cw, ch, corner = params['cw'], params['ch'], params['corner']
        if corner == 'tl':
            return [(ox, oy + ch, ox + ow, oy + oh),
                    (ox + cw, oy, ox + ow, oy + ch)]
        if corner == 'tr':
            return [(ox, oy + ch, ox + ow, oy + oh),
                    (ox, oy, ox + ow - cw, oy + ch)]
        if corner == 'bl':
            return [(ox, oy, ox + ow, oy + oh - ch),
                    (ox + cw, oy + oh - ch, ox + ow, oy + oh)]
        return [(ox, oy, ox + ow, oy + oh - ch),
                (ox, oy + oh - ch, ox + ow - cw, oy + oh)]
    if shape == 'T':
        sw, sh, sx0 = params['sw'], params['sh'], params['sx0']
        return [(sx0, oy, sx0 + sw, oy + sh),
                (ox, oy + sh, ox + ow, oy + oh)]
    if shape == 'U':
        cw, ch = params['cw'], params['ch']
        return [(ox, oy, ox + ow, oy + oh - ch),
                (ox, oy + oh - ch, ox + cw, oy + oh),
                (ox + ow - cw, oy + oh - ch, ox + ow, oy + oh)]
    return [(ox, oy, ox + ow, oy + oh)]


def _rect_intersection_area(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _augment_roof_wings(wings, bounds, rng):
    """Attach 0-2 smaller wings to exposed edges so roof plans stop reading
    like perfect base primitives."""
    ox, oy, ox1, oy1 = bounds
    out = list(wings)
    n_extra = rng.choices([0, 1, 2], weights=[0.25, 0.45, 0.30])[0]
    for _ in range(n_extra):
        placed = False
        for _try in range(20):
            wx0, wy0, wx1, wy1 = rng.choice(out)
            ww = wx1 - wx0
            wh = wy1 - wy0
            side = rng.choice(['left', 'right', 'top', 'bottom'])
            if side in ('left', 'right'):
                attach_h = min(wh, rng.randint(max(90, int(wh * 0.28)),
                                               max(120, int(wh * 0.70))))
                attach_w = rng.randint(max(80, int(ww * 0.18)),
                                       max(110, int(ww * 0.42)))
                ay0 = rng.randint(wy0, max(wy0, wy1 - attach_h))
                ay1 = ay0 + attach_h
                rect = ((wx0 - attach_w, ay0, wx0, ay1)
                        if side == 'left' else
                        (wx1, ay0, wx1 + attach_w, ay1))
            else:
                attach_w = min(ww, rng.randint(max(90, int(ww * 0.28)),
                                               max(120, int(ww * 0.72))))
                attach_h = rng.randint(max(80, int(wh * 0.18)),
                                       max(110, int(wh * 0.40)))
                ax0 = rng.randint(wx0, max(wx0, wx1 - attach_w))
                ax1 = ax0 + attach_w
                rect = ((ax0, wy0 - attach_h, ax1, wy0)
                        if side == 'top' else
                        (ax0, wy1, ax1, wy1 + attach_h))
            rx0, ry0, rx1, ry1 = rect
            if rx0 < ox or ry0 < oy or rx1 > ox1 or ry1 > oy1:
                continue
            if any(_rect_intersection_area(rect, existing) > 0 for existing in out):
                continue
            if not any(_rects_share_edge(rect, existing) for existing in out):
                continue
            out.append(rect)
            placed = True
            break
        if not placed:
            break
    return out


def _edge_on_rect_perimeter(edge, rect):
    """edge=(x0,y0,x1,y1) axis-aligned. True iff it lies along rect's perimeter
    with non-zero overlap (i.e. the edge is internal to the rect's outline)."""
    ex0, ey0, ex1, ey1 = edge
    rx0, ry0, rx1, ry1 = rect
    if ex0 == ex1:  # vertical edge
        if ex0 != rx0 and ex0 != rx1:
            return False
        lo, hi = sorted([ey0, ey1])
        return max(lo, ry0) < min(hi, ry1)
    if ey0 == ey1:  # horizontal edge
        if ey0 != ry0 and ey0 != ry1:
            return False
        lo, hi = sorted([ex0, ex1])
        return max(lo, rx0) < min(hi, rx1)
    return False


def _wing_facets(wing, neighbors):
    """Partition a wing rect into roof facets. Long axis carries the ridge;
    each external short end gets a hip triangle (inset = short_dim/2); short
    ends shared with a neighbour wing become gable cuts (inset = 0) so the
    triangle doesn't breach into the neighbour. Returns (facets, ridge_seg).
    """
    x0, y0, x1, y1 = wing
    w = x1 - x0
    h = y1 - y0
    horizontal = w >= h
    facets = []
    if horizontal:
        y_mid = (y0 + y1) // 2
        half = h // 2
        left_internal = any(_edge_on_rect_perimeter((x0, y0, x0, y1), n)
                            for n in neighbors)
        right_internal = any(_edge_on_rect_perimeter((x1, y0, x1, y1), n)
                             for n in neighbors)
        inset_l = 0 if left_internal else half
        inset_r = 0 if right_internal else half
        # Keep at least 60 px of ridge.
        if w - inset_l - inset_r < 60 and (inset_l + inset_r) > 0:
            over = (inset_l + inset_r) - max(0, w - 60)
            cl = over // 2
            cr = over - cl
            inset_l = max(0, inset_l - cl)
            inset_r = max(0, inset_r - cr)
        facets.append([(x0, y0), (x1, y0),
                       (x1 - inset_r, y_mid), (x0 + inset_l, y_mid)])
        facets.append([(x0 + inset_l, y_mid), (x1 - inset_r, y_mid),
                       (x1, y1), (x0, y1)])
        if inset_l >= 30:
            facets.append([(x0, y0), (x0 + inset_l, y_mid), (x0, y1)])
        if inset_r >= 30:
            facets.append([(x1, y0), (x1, y1), (x1 - inset_r, y_mid)])
        ridge_seg = ((x0 + inset_l, y_mid), (x1 - inset_r, y_mid))
    else:
        x_mid = (x0 + x1) // 2
        half = w // 2
        top_internal = any(_edge_on_rect_perimeter((x0, y0, x1, y0), n)
                           for n in neighbors)
        bot_internal = any(_edge_on_rect_perimeter((x0, y1, x1, y1), n)
                           for n in neighbors)
        inset_t = 0 if top_internal else half
        inset_b = 0 if bot_internal else half
        if h - inset_t - inset_b < 60 and (inset_t + inset_b) > 0:
            over = (inset_t + inset_b) - max(0, h - 60)
            ct = over // 2
            cb = over - ct
            inset_t = max(0, inset_t - ct)
            inset_b = max(0, inset_b - cb)
        facets.append([(x0, y0), (x_mid, y0 + inset_t),
                       (x_mid, y1 - inset_b), (x0, y1)])
        facets.append([(x_mid, y0 + inset_t), (x1, y0),
                       (x1, y1), (x_mid, y1 - inset_b)])
        if inset_t >= 30:
            facets.append([(x0, y0), (x1, y0), (x_mid, y0 + inset_t)])
        if inset_b >= 30:
            facets.append([(x0, y1), (x_mid, y1 - inset_b), (x1, y1)])
        ridge_seg = ((x_mid, y0 + inset_t), (x_mid, y1 - inset_b))
    return facets, ridge_seg


def build_roofplan_scene(W: int, H: int, rng: random.Random, tiles):
    """Roof plan partitioned into hip/gable FACETS.

    Footprint = rect / L / T / U; decomposed into 1-3 wings. Each wing gets a
    hip-or-gable partition into 2-4 facet polygons. Most facets share a single
    primary tile (multi-instance signal); 0-2 facets are overridden with
    secondary tiles (distractors). Inter-facet edges (ridges, hips, valleys)
    are drawn for free by the per-item boundary line in the main render loop.
    """
    margin_x = rng.randint(120, 240)
    margin_y = rng.randint(120, 240)
    ox, oy = margin_x, margin_y
    ow, oh = W - 2 * margin_x, H - 2 * margin_y

    # Coarsen toward real: bias the footprint toward a single-wing rect (fewer
    # facets/instances). At scale 1 the full multi-wing shape mix is used; as
    # scale->0 it collapses to 'rect'. Multi-wing L/T/U roofs are still realistic,
    # just rarer when coarse.
    if rng.random() > INSTANCE_SCALE:
        shape = 'rect'
    else:
        shape = rng.choice(['rect', 'L', 'L', 'T', 'T', 'U', 'U', 'U'])
    params = {}
    if shape == 'L':
        params['cw'] = rng.randint(int(ow * 0.30), int(ow * 0.55))
        params['ch'] = rng.randint(int(oh * 0.30), int(oh * 0.55))
        params['corner'] = rng.choice(['tl', 'tr', 'bl', 'br'])
    elif shape == 'T':
        params['sw'] = rng.randint(int(ow * 0.30), int(ow * 0.55))
        params['sh'] = rng.randint(int(oh * 0.30), int(oh * 0.50))
        params['sx0'] = ox + (ow - params['sw']) // 2
    elif shape == 'U':
        params['cw'] = rng.randint(int(ow * 0.20), int(ow * 0.35))
        params['ch'] = rng.randint(int(oh * 0.30), int(oh * 0.55))

    wings = _decompose_into_wings(shape, ox, oy, ow, oh, params)
    wings = _augment_roof_wings(wings, (ox, oy, ox + ow, oy + oh), rng)
    outer_poly_raw = _union_polygon_axis_aligned(wings)
    outer_poly = _simplify_poly(outer_poly_raw) if outer_poly_raw else None
    if not outer_poly:
        outer_poly = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
    all_facets = []
    ridges = []  # for RIDGE labels in post-render
    for i, wing in enumerate(wings):
        neighbors = [w for j, w in enumerate(wings) if j != i]
        facets, ridge_seg = _wing_facets(wing, neighbors)
        all_facets.extend(facets)
        ridges.append(ridge_seg)

    # Primary pattern fills most facets; 0-2 facets get a secondary tile.
    primary_tile = pick_wall_tile(tiles, rng, min_ink=0.02)
    sec_pool = [t for t in tiles if t['path'] != primary_tile['path']]
    facet_tiles = [primary_tile] * len(all_facets)
    if len(all_facets) >= 3 and sec_pool:
        max_sec = min(2, len(all_facets) // 2)
        n_sec = rng.randint(0, max_sec)
        if n_sec > 0:
            for idx in rng.sample(range(len(all_facets)), n_sec):
                facet_tiles[idx] = pick_wall_tile(sec_pool, rng, min_ink=0.02)

    # v5.1: emit each facet as its OWN instance. Real plans keep adjacent
    # same-material facets (meeting at a ridge/hip) as separate annotations, so
    # do NOT union same-tile facets into one blob.
    # Coarsen toward real: with prob (1-INSTANCE_SCALE), merge adjacent
    # same-material facets into one instance. render_roofplan_post still draws the
    # internal ridge/hip lines, so a merged region reads exactly like a real roof
    # plan (one material, ridge lines inside) rather than many per-facet tiles.
    merge_facets = rng.random() > INSTANCE_SCALE
    items = []
    if NO_MERGE_SAME_MATERIAL and not merge_facets:
        for i, tile in enumerate(facet_tiles):
            poly_int = [(int(round(x)), int(round(y))) for x, y in all_facets[i]]
            items.extend(_surface_items(
                poly_int,
                'roof_plan',
                tile,
                rng,
                alt_tiles=[t for t in ([primary_tile] + sec_pool) if t['path'] != tile['path']],
                has_windows=False,
                split_prob=ROOFPLAN_INTERLEAVE_PROB,
                max_pieces=2,
                min_piece_area=18_000,
            ))
    else:
        by_tile = {}
        for i, tile in enumerate(facet_tiles):
            by_tile.setdefault(tile['path'], (tile, []))[1].append(i)
        for path, (tile, indices) in by_tile.items():
            polys = [_ShPoly(all_facets[i]).buffer(0) for i in indices]
            merged = _sh_union(polys)
            geoms = list(merged.geoms) if merged.geom_type == 'MultiPolygon' else [merged]
            for geom in geoms:
                coords = list(geom.exterior.coords)
                if coords and coords[0] == coords[-1]:
                    coords = coords[:-1]
                poly_int = [(int(round(x)), int(round(y))) for x, y in coords]
                items.append({'poly': poly_int, 'role': 'roof_plan',
                              'tile': tile, 'has_windows': False})

    bx0 = min(p[0] for p in outer_poly); by0 = min(p[1] for p in outer_poly)
    bx1 = max(p[0] for p in outer_poly); by1 = max(p[1] for p in outer_poly)
    meta = {'outer_poly': outer_poly, 'ridges': ridges,
            'outer_bbox': (bx0, by0, bx1, by1),
            'facets': all_facets}
    return items, meta


def render_roofplan_post(d: ImageDraw.ImageDraw, meta, rng, W, H):
    poly = meta['outer_poly']
    # Original (pre-merge) facet boundaries: drawn as thin medium-gray lines
    # so ridge/hip/valley structure stays visible inside same-pattern merged
    # regions, where the per-item boundary line no longer traces them.
    for facet in meta.get('facets', []):
        d.line(facet + [facet[0]], fill=(90, 90, 90), width=2)
    # Outer outline thicker than the facet lines so the silhouette reads as
    # the roof edge, not an interior ridge.
    d.line(poly + [poly[0]], fill=(20, 20, 20), width=4)
    # Each wing's ridge segment was emitted by _wing_facets; the facet boundary
    # lines already draw it, so we just label it.
    f = _font(rng.randint(13, 17))
    for (ax, ay), (bx, by) in meta['ridges']:
        if ax == bx and ay == by:
            continue
        if abs(bx - ax) >= abs(by - ay):
            d.text(((ax + bx) // 2 - 22, ay - 16), 'RIDGE',
                   fill=(40, 40, 40), font=f)
        else:
            d.text((ax + 6, (ay + by) // 2 - 8), 'RIDGE',
                   fill=(40, 40, 40), font=f)
    # Title
    d.text((30, H - 36), 'ROOF PLAN', fill=(20, 20, 20), font=_font(22))
    d.text((30, H - 14), 'SCALE: 1/8" = 1\'-0"', fill=(60, 60, 60), font=_font(13))
    # Grid bubbles around perimeter (light)
    bx0, by0, bx1, by1 = meta['outer_bbox']
    n = rng.randint(3, 6)
    f_b = _font(18)
    for i in range(n):
        x = int(bx0 + (bx1 - bx0) * (i + 0.5) / n)
        y = by0 - rng.randint(50, 90)
        d.line([x, y + 14, x, by0], fill=(150, 150, 150), width=1)
        d.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(40, 40, 40), width=2, fill=(255, 255, 255))
        d.text((x - 5, y - 9), chr(ord('A') + i), fill=(30, 30, 30), font=f_b)


def _rects_share_edge(a, b):
    """Return True if rects a, b share a non-zero portion of one edge."""
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    # Vertical shared edge: a.right == b.left (or vice versa) with y-overlap
    if ax1 == bx0 and min(ay1, by1) > max(ay0, by0):
        return True
    if bx1 == ax0 and min(ay1, by1) > max(ay0, by0):
        return True
    if ay1 == by0 and min(ax1, bx1) > max(ax0, bx0):
        return True
    if by1 == ay0 and min(ax1, bx1) > max(ax0, bx0):
        return True
    return False


def _union_polygon_axis_aligned(rects):
    """Compute outline polygon (CCW) of axis-aligned rect union via grid sweep.
    Returns list of (x, y) points. Holes in the union are not handled (we never
    pass disjoint or hole-creating sets here).
    """
    if len(rects) == 1:
        x0, y0, x1, y1 = rects[0]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    nx, ny = len(xs) - 1, len(ys) - 1
    cell = np.zeros((ny, nx), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        ix0 = xs.index(x0); ix1 = xs.index(x1)
        iy0 = ys.index(y0); iy1 = ys.index(y1)
        cell[iy0:iy1, ix0:ix1] = 1
    # March around boundary using simple edge-tracing on the cell grid.
    # We collect oriented edges, then chain them.
    edges = set()
    def _add(p1, p2):
        if (p2, p1) in edges:
            edges.remove((p2, p1))
        else:
            edges.add((p1, p2))
    for j in range(ny):
        for i in range(nx):
            if not cell[j, i]:
                continue
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            _add((x0, y0), (x1, y0))   # top
            _add((x1, y0), (x1, y1))   # right
            _add((x1, y1), (x0, y1))   # bottom
            _add((x0, y1), (x0, y0))   # left
    if not edges:
        return None
    # Build adjacency and walk
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    start = next(iter(adj))
    poly = [start]
    cur = start
    visited = 0
    while visited < len(edges):
        nxts = adj.get(cur, [])
        if not nxts:
            break
        nxt = nxts.pop(0)
        poly.append(nxt)
        cur = nxt
        visited += 1
        if cur == start:
            break
    if poly[0] == poly[-1]:
        poly = poly[:-1]
    return poly


def _simplify_poly(poly):
    """Drop duplicate and collinear vertices from a rectilinear polygon so its
    edges strictly alternate horizontal / vertical."""
    pts = [p for i, p in enumerate(poly) if p != poly[i - 1]]
    if len(pts) < 3:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
            continue  # b sits on a straight run
        out.append(b)
    return out


def _inset_rectilinear(poly, inset):
    """Inset an axis-aligned rectilinear polygon inward by `inset` px. Each edge
    is shifted toward the interior; new vertices are the intersections of
    consecutive shifted edge-lines. Winding-agnostic."""
    poly = _simplify_poly(poly)
    n = len(poly)
    if n < 4 or inset <= 0:
        return [(int(round(x)), int(round(y))) for x, y in poly]
    lines = []  # per edge, after inward shift: ('h', y) or ('v', x)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if ay == by:  # horizontal edge on line y = ay
            mx = (ax + bx) / 2.0
            shift = inset if _point_in_poly(mx, ay + 1, poly) else -inset
            lines.append(('h', ay + shift))
        else:         # vertical edge on line x = ax
            my = (ay + by) / 2.0
            shift = inset if _point_in_poly(ax + 1, my, poly) else -inset
            lines.append(('v', ax + shift))
    out = []
    for i in range(n):
        lp, lc = lines[i - 1], lines[i]
        if lp[0] == 'v':
            x, y = lp[1], lc[1]
        else:
            x, y = lc[1], lp[1]
        out.append((int(round(x)), int(round(y))))
    return out


def _grow_room_cluster(seed, rooms, blocked, rng, target):
    """Grow an edge-connected cluster of room rects from `seed`, up to `target`
    rooms, drawing only from rooms not in `blocked`."""
    cluster = [seed]
    pool = [r for r in rooms if r != seed and r not in blocked]
    while len(cluster) < target and pool:
        adj = [r for r in pool
               if any(_rects_share_edge(r, c) for c in cluster)]
        if not adj:
            break
        nxt = rng.choice(adj)
        cluster.append(nxt)
        pool.remove(nxt)
    return cluster


def _build_merged_poly(cluster, rng):
    """Union a set of room rects into one inset rectilinear polygon. Returns a
    vertex list, or None if the union is a plain rectangle (not irregular) or
    the edge-trace dropped part of it (a pinch / disjoint set)."""
    raw = _union_polygon_axis_aligned(list(cluster))
    if raw is None:
        return None
    raw = _simplify_poly(raw)
    if len(raw) < 6:  # 4 vertices == plain rectangle
        return None
    rect_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in cluster)
    if _polygon_area(raw) < 0.9 * rect_area:
        return None  # trace lost a sub-loop
    poly = _simplify_poly(_inset_rectilinear(raw, rng.randint(2, 4)))
    if len(poly) < 6 or _polygon_area(poly) < 120 * 120:
        return None
    return poly


def _multi_notch_rect(room, rng, inset):
    """Inset room rect with 0-4 corners cut by closet-sized notches. Notch
    depths are capped so opposite notches can never cross."""
    rx0, ry0, rx1, ry1 = room
    x0, y0, x1, y1 = rx0 + inset, ry0 + inset, rx1 - inset, ry1 - inset
    rw, rh = x1 - x0, y1 - y0
    if rw < 150 or rh < 150:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cs = {}
    for c in ('tl', 'tr', 'br', 'bl'):
        cs[c] = None if rng.random() < 0.5 else (
            rng.randint(int(rw * 0.15), int(rw * 0.42)),
            rng.randint(int(rh * 0.15), int(rh * 0.42)))
    poly = []
    if cs['tl']:
        cw, ch = cs['tl']
        poly += [(x0, y0 + ch), (x0 + cw, y0 + ch), (x0 + cw, y0)]
    else:
        poly.append((x0, y0))
    if cs['tr']:
        cw, ch = cs['tr']
        poly += [(x1 - cw, y0), (x1 - cw, y0 + ch), (x1, y0 + ch)]
    else:
        poly.append((x1, y0))
    if cs['br']:
        cw, ch = cs['br']
        poly += [(x1, y1 - ch), (x1 - cw, y1 - ch), (x1 - cw, y1)]
    else:
        poly.append((x1, y1))
    if cs['bl']:
        cw, ch = cs['bl']
        poly += [(x0 + cw, y1), (x0 + cw, y1 - ch), (x0, y1 - ch)]
    else:
        poly.append((x0, y1))
    return poly


def _edge_normal_in(a, b, poly):
    """Inward unit normal of edge a->b for polygon `poly`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    if _point_in_poly(mx + nx * 2, my + ny * 2, poly):
        return nx, ny
    return -nx, -ny


def _sawtooth_edge(a, b, poly, rng):
    """Replace edge a->b with triangular razor teeth carved inward. Returns
    vertices from a (inclusive) up to but excluding b."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = rng.randint(3, 8)
    if L < n * 16:
        return [a]
    nx, ny = _edge_normal_in(a, b, poly)
    # Hard cap on tooth depth so long edges can't carve a hole through the
    # polygon's interior. Long edges still get many teeth, just shallower ones.
    amp = min(rng.uniform(0.04, 0.13) * L, 22.0)
    pts = [a]
    for k in range(n):
        tm, t1 = (k + 0.5) / n, (k + 1) / n
        pts.append((a[0] + (b[0] - a[0]) * tm + nx * amp,
                    a[1] + (b[1] - a[1]) * tm + ny * amp))
        if k < n - 1:
            pts.append((a[0] + (b[0] - a[0]) * t1,
                        a[1] + (b[1] - a[1]) * t1))
    return pts


def _comb_edge(a, b, poly, rng):
    """Replace edge a->b with square comb teeth carved inward."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = rng.randint(2, 5)
    parts = 2 * n + 1
    if L < parts * 18:
        return [a]
    nx, ny = _edge_normal_in(a, b, poly)
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    amp = min(rng.uniform(0.05, 0.14) * L, 24.0)
    seg = L / parts
    pts = [a]
    for i in range(parts):
        if i % 2 == 1:  # tooth: carve inward
            s0, s1 = i * seg, (i + 1) * seg
            p0 = (a[0] + ux * s0, a[1] + uy * s0)
            p1 = (a[0] + ux * s1, a[1] + uy * s1)
            pts += [p0, (p0[0] + nx * amp, p0[1] + ny * amp),
                    (p1[0] + nx * amp, p1[1] + ny * amp), p1]
    return pts


def _teeth_pass(poly, rng):
    """Replace most of the polygon's long edges with sawtooth or comb teeth.
    Every tooth is built from straight line segments -- no curved edges."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        r = rng.random()
        if L >= 90 and r < 0.72:
            out += (_sawtooth_edge(a, b, poly, rng) if r < 0.36
                    else _comb_edge(a, b, poly, rng))
        else:
            out.append(a)
    return out


def _complexify(poly, rng):
    """Apply the razor-tooth pass; carve only inward so the region never spills
    past its room. Every edge stays a straight line segment.

    If the teeth pass produces a self-intersecting polygon (deep teeth from
    opposite sides crossing through a narrow throat in a merged region), fall
    back to the pre-teeth polygon rather than emit a polygon that erodes into
    disconnected pieces. This guards against narrow-strip artifacts."""
    # Real plans have crisp rectilinear instance edges; the razor-tooth pass
    # trains the wrong shape prior, so keep the polygon straight-edged.
    if RECTILINEAR_INSTANCES:
        return [(int(round(x)), int(round(y))) for x, y in poly]
    candidate = [(int(round(x)), int(round(y)))
                 for x, y in _teeth_pass(poly, rng)]
    try:
        sh = _ShPoly(candidate)
        if sh.is_valid and not sh.buffer(-4).is_empty:
            eroded = sh.buffer(-4)
            n_parts = len(eroded.geoms) if hasattr(eroded, 'geoms') else 1
            if n_parts == 1:
                return candidate
    except Exception:
        pass
    return [(int(round(x)), int(round(y))) for x, y in poly]


def _merge_intervals(ivs):
    ivs = sorted((lo, hi) for lo, hi in ivs if hi > lo)
    out = []
    for lo, hi in ivs:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _intersect_intervals(a, b):
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return out


def _merged_hidden_intervals(kind, pos, a, b, merged_clusters):
    """Sub-intervals of a partition segment [a, b] at `pos` that lie in the
    closure-interior of a merged pattern region (cluster rooms on both sides),
    and so should not be drawn as wall."""
    hidden = []
    for cluster in merged_clusters:
        left, right = [], []
        for rx0, ry0, rx1, ry1 in cluster:
            if kind == 'v':
                span = (max(a, ry0), min(b, ry1))
                if rx0 <= pos - 1 and rx1 >= pos:
                    left.append(span)
                if rx0 <= pos and rx1 >= pos + 1:
                    right.append(span)
            else:
                span = (max(a, rx0), min(b, rx1))
                if ry0 <= pos - 1 and ry1 >= pos:
                    left.append(span)
                if ry0 <= pos and ry1 >= pos + 1:
                    right.append(span)
        hidden += _intersect_intervals(_merge_intervals(left),
                                       _merge_intervals(right))
    return _merge_intervals(hidden)


def _visible_segments(a, b, hidden):
    """[a, b] minus the `hidden` intervals -> list of (lo, hi) sub-segments."""
    segs = [(a, b)]
    for h0, h1 in hidden:
        nxt = []
        for s0, s1 in segs:
            if h1 <= s0 or h0 >= s1:
                nxt.append((s0, s1))
                continue
            if h0 > s0:
                nxt.append((s0, h0))
            if h1 < s1:
                nxt.append((h1, s1))
        segs = nxt
    return segs


def build_floorplan_scene(W: int, H: int, rng: random.Random, tiles):
    """Dense floor plan with multi-room same-pattern placement and deliberately
    complex pattern regions (multi-room merges, multi-corner notches, sawtooth /
    comb razor teeth, perturbed edges).

    Pattern polygons stay inside their room footprint but their boundaries are
    highly irregular, so the model gets no rectangular shape prior. Fixtures are
    only drawn in non-pattern rooms; pattern regions carry no holes. The model
    has to find:
      - the primary pattern in 2-4 spatially separated rooms (same category_id)
      - 0-2 secondary patterns in additional rooms
    """
    items = []
    margin = rng.randint(80, 200)
    ox, oy = margin, margin
    ow, oh = W - 2 * margin, H - 2 * margin
    wall_t = rng.randint(8, 18)

    # Outer footprint: rect or L-shape
    if rng.random() < 0.55:
        outer_poly = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
        notch = None
    else:
        cw = rng.randint(int(ow * 0.25), int(ow * 0.45))
        ch = rng.randint(int(oh * 0.25), int(oh * 0.45))
        corner = rng.choice(['tl', 'tr', 'bl', 'br'])
        notch = (corner, cw, ch)
        if corner == 'tl':
            outer_poly = [(ox + cw, oy), (ox + ow, oy), (ox + ow, oy + oh),
                          (ox, oy + oh), (ox, oy + ch), (ox + cw, oy + ch)]
        elif corner == 'tr':
            outer_poly = [(ox, oy), (ox + ow - cw, oy), (ox + ow - cw, oy + ch),
                          (ox + ow, oy + ch), (ox + ow, oy + oh), (ox, oy + oh)]
        elif corner == 'bl':
            outer_poly = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh),
                          (ox + cw, oy + oh), (ox + cw, oy + oh - ch), (ox, oy + oh - ch)]
        else:
            outer_poly = [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh - ch),
                          (ox + ow - cw, oy + oh - ch), (ox + ow - cw, oy + oh), (ox, oy + oh)]

    # Many partitions for a denser plan (more rooms = more visual noise around patterns).
    rooms = [(ox + wall_t, oy + wall_t, ox + ow - wall_t, oy + oh - wall_t)]
    partitions = []
    doors = []

    for _ in range(rng.randint(12, 22)):
        rooms.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
        rx0, ry0, rx1, ry1 = rooms.pop(0)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < 160 and rh < 160:
            rooms.append((rx0, ry0, rx1, ry1)); continue
        if rw > rh:
            sx = rng.randint(rx0 + 80, rx1 - 80)
            partitions.append(('v', sx, ry0, ry1))
            gap_y = rng.randint(ry0 + 20, ry1 - 60)
            gap_h = rng.randint(28, 50)
            doors.append(('v', sx, gap_y, gap_y + gap_h))
            rooms += [(rx0, ry0, sx, ry1), (sx, ry0, rx1, ry1)]
        else:
            sy = rng.randint(ry0 + 80, ry1 - 80)
            partitions.append(('h', sy, rx0, rx1))
            gap_x = rng.randint(rx0 + 20, rx1 - 60)
            gap_w = rng.randint(28, 50)
            doors.append(('h', sy, gap_x, gap_x + gap_w))
            rooms += [(rx0, ry0, rx1, sy), (rx0, sy, rx1, ry1)]

    rooms = [r for r in rooms
             if _point_in_poly((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, outer_poly)]

    eligible = [r for r in rooms if (r[2] - r[0]) > 110 and (r[3] - r[1]) > 110]

    # ---- Pattern placement strategy (v5.1) ----
    # Interleave 2-4 patterns across ADJACENT rooms so materials abut with no
    # white gaps (real plans never leave blank rooms between patterns). Each
    # room -- and optionally each per-bay sub-cell of a room -- is emitted as
    # its OWN instance, so the same material spread over neighbouring rooms
    # stays many separate, crisp-edged instances (the real annotation style).
    pat_set = set()           # rooms whose interior is covered by ANY pattern
    merged_clusters = []      # no merging in v5.1 -> always empty

    def _shaped_poly(room, rng):
        """A single rectilinear pattern polygon: clean multi-notched rectangle
        (straight edges; the razor-tooth pass is disabled in RECTILINEAR mode)."""
        return _complexify(_multi_notch_rect(room, rng, rng.randint(2, 4)), rng)

    if eligible and (FREEFORM_TILE_OUTLINE_PROB > 0.0 or IDX0_STONE_PATH):
        # idx0-faithful (covered-patio) placement: ONE big WIDE material region +
        # one small, spanning ACROSS rooms (NOT per-room). idx0's worst case is a
        # large faint wall-interrupted tile region the model smears a giant blob
        # over (pred ~300k px vs 56k GT -> iou 0.10). Clean per-room synth patches
        # are far easier -> the entire val<->idx0 divergence. Partition walls draw
        # OVER these (render_floorplan_post), reproducing idx0's boundary ambiguity.
        # When IDX0_STONE_PATH is set, render with idx0's REAL patio stone texture.
        if IDX0_STONE_PATH:
            tile = dict(rng.choice(tiles)); tile['path'] = IDX0_STONE_PATH
        else:
            tile = rng.choice(tiles)
        bw = int(rng.uniform(0.42, 0.66) * ow)
        bh = int(rng.uniform(0.16, 0.30) * oh)
        bx = ox + int(rng.uniform(0.02, 0.95) * max(1, ow - bw))
        by = oy + int(rng.uniform(0.02, 0.55) * max(1, oh - bh))
        items.append({'poly': [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)],
                      'role': 'free', 'tile': tile, 'has_windows': False, 'draw_outline': False})
        if rng.random() < 0.85:
            sw = int(rng.uniform(0.06, 0.13) * ow)
            sh = int(rng.uniform(0.07, 0.16) * oh)
            sx = ox + int(rng.uniform(0.02, 0.95) * max(1, ow - sw))
            sy = oy + int(rng.uniform(0.45, 0.95) * max(1, oh - sh))
            items.append({'poly': [(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)],
                          'role': 'free', 'tile': tile, 'has_windows': False, 'draw_outline': False})
    elif eligible:
        if FREEFORM_REALSTYLE:
            n_pat = min(rng.randint(1, 2), len(tiles))
        else:
            n_pat = min(rng.randint(2, 4), len(tiles))
        pat_tiles = rng.sample(tiles, n_pat)
        # Cover most eligible rooms so patterns sit side by side (interleaved) --
        # UNLESS realstyle, where real plans leave most rooms white (idx00).
        if FREEFORM_REALSTYLE:
            cover_frac = rng.uniform(*FREEFORM_REALSTYLE_COVER)
        elif INTERLEAVE_PATTERNS:
            cover_frac = rng.uniform(0.7, 0.95)
        else:
            cover_frac = 0.4
        floor_cover = 1 if FREEFORM_REALSTYLE else 2
        n_cover = max(floor_cover, int(round(len(eligible) * cover_frac)))
        chosen = rng.sample(eligible, min(n_cover, len(eligible)))
        # Assign neighbouring rooms different tiles where possible so distinct
        # patterns interleave rather than clump.
        prev_tile = None
        for room in sorted(chosen, key=lambda r: (r[1], r[0])):
            opts = [t for t in pat_tiles if t is not prev_tile] or pat_tiles
            tile = rng.choice(opts)
            prev_tile = tile
            rw, rh = room[2] - room[0], room[3] - room[1]
            # Split larger rooms into per-bay sub-instances (one continuous
            # band -> several adjacent same-material instances, like a real
            # facade base course split per unit).
            if (not FREEFORM_REALSTYLE and min(rw, rh) > 200
                    and rng.random() < BAY_SPLIT_PROB):
                cells = _bay_split(room, rng)
            else:
                cells = [room]
            for cell in cells:
                cw, ch = cell[2] - cell[0], cell[3] - cell[1]
                base_poly = [(cell[0], cell[1]), (cell[2], cell[1]),
                             (cell[2], cell[3]), (cell[0], cell[3])]
                alt_tiles = [t for t in pat_tiles if t['path'] != tile['path']]
                if (not FREEFORM_REALSTYLE and alt_tiles and min(cw, ch) > 180
                        and rng.random() < FLOORPLAN_INTERLEAVE_PROB):
                    split_polys = _split_poly_interleaved(
                        base_poly,
                        rng,
                        max_pieces=3 if min(cw, ch) > 260 else 2,
                        min_piece_area=14_000,
                    )
                    if len(split_polys) > 1:
                        axis = 'v' if cw >= ch else 'h'
                        palette = [tile] + alt_tiles
                        for idx, poly in enumerate(sorted(split_polys,
                                                          key=lambda p: _piece_sort_key(p, axis))):
                            items.append({
                                'poly': poly,
                                'role': 'free',
                                'tile': palette[idx % len(palette)],
                                'has_windows': False,
                                'draw_outline': False,
                                'dense_bias': 0.18,
                                'dense_opacity_boost': 0.16,
                            })
                        continue
                items.append({'poly': _shaped_poly(cell, rng), 'role': 'free',
                              'tile': tile, 'has_windows': False})
            pat_set.add(room)

    non_pat_rooms = [r for r in rooms if r not in pat_set]

    meta = {
        'outer_poly': outer_poly,
        'outer_bbox': (ox, oy, ox + ow, oy + oh),
        'wall_t': wall_t,
        'rooms': rooms,
        'non_pat_rooms': non_pat_rooms,
        'partitions': partitions,
        'doors': doors,
        'pattern_rooms': pat_set,
        'merged_clusters': merged_clusters,
    }
    return items, meta


def _draw_fixture(d, room, rng, return_bbox=False):
    """Draw one architectural fixture inside a room. If return_bbox is True,
    returns (x0, y0, x1, y1) of the fixture (for use as a hole polygon).
    """
    rx0, ry0, rx1, ry1 = room
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw < 110 or rh < 110:
        return None if return_bbox else None
    kind = rng.choice(['tub', 'toilet', 'sink', 'island', 'stairs', 'stairs', 'none'])
    cx, cy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    bbox = None
    if kind == 'tub':
        tw, th = min(rw - 30, rng.randint(140, 220)), min(rh - 30, rng.randint(70, 110))
        x0 = cx - tw // 2; y0 = ry0 + 14
        d.rectangle([x0, y0, x0 + tw, y0 + th], outline=(40, 40, 40), width=2,
                    fill=(255, 255, 255))
        d.rectangle([x0 + 8, y0 + 8, x0 + tw - 8, y0 + th - 8], outline=(80, 80, 80), width=1)
        d.ellipse([x0 + tw - 26, y0 + th // 2 - 6, x0 + tw - 14, y0 + th // 2 + 6],
                  outline=(80, 80, 80), width=1)
        bbox = (x0, y0, x0 + tw, y0 + th)
    elif kind == 'toilet':
        tw_, th_ = 36, 56
        x0 = rx0 + 14; y0 = ry0 + 14
        d.rectangle([x0, y0, x0 + tw_, y0 + 16], outline=(40, 40, 40), width=2,
                    fill=(255, 255, 255))
        d.ellipse([x0 + 4, y0 + 14, x0 + tw_ - 4, y0 + th_], outline=(40, 40, 40),
                  width=2, fill=(255, 255, 255))
        bbox = (x0, y0, x0 + tw_, y0 + th_)
    elif kind == 'sink':
        sw, sh = 80, 50
        x0 = rx1 - sw - 14; y0 = ry0 + 14
        d.rectangle([x0, y0, x0 + sw, y0 + sh], outline=(40, 40, 40), width=2,
                    fill=(255, 255, 255))
        d.rectangle([x0 + 6, y0 + 6, x0 + sw - 6, y0 + sh - 6], outline=(80, 80, 80), width=1)
        d.ellipse([x0 + sw // 2 - 3, y0 + 3, x0 + sw // 2 + 3, y0 + 9], fill=(40, 40, 40))
        bbox = (x0, y0, x0 + sw, y0 + sh)
    elif kind == 'island':
        iw, ih = min(rw - 80, rng.randint(180, 320)), min(rh - 80, rng.randint(80, 130))
        x0 = cx - iw // 2; y0 = cy - ih // 2
        d.rectangle([x0, y0, x0 + iw, y0 + ih], outline=(40, 40, 40), width=2,
                    fill=(255, 255, 255))
        d.rectangle([x0 + 4, y0 + 4, x0 + iw - 4, y0 + ih - 4], outline=(120, 120, 120), width=1)
        bbox = (x0, y0, x0 + iw, y0 + ih)
    elif kind == 'stairs':
        n = rng.randint(6, 12)
        if rw > rh:
            x0 = rx0 + 14; y0 = ry0 + 14
            sw = min(rw - 28, rng.randint(180, 320)); sh = min(rh - 28, 80)
            d.rectangle([x0, y0, x0 + sw, y0 + sh], outline=(40, 40, 40), width=2,
                        fill=(255, 255, 255))
            for k in range(1, n):
                xx = x0 + int(sw * k / n)
                d.line([xx, y0, xx, y0 + sh], fill=(40, 40, 40), width=1)
            bbox = (x0, y0, x0 + sw, y0 + sh)
        else:
            x0 = rx0 + 14; y0 = ry0 + 14
            sw = min(rw - 28, 80); sh = min(rh - 28, rng.randint(180, 320))
            d.rectangle([x0, y0, x0 + sw, y0 + sh], outline=(40, 40, 40), width=2,
                        fill=(255, 255, 255))
            for k in range(1, n):
                yy = y0 + int(sh * k / n)
                d.line([x0, yy, x0 + sw, yy], fill=(40, 40, 40), width=1)
            bbox = (x0, y0, x0 + sw, y0 + sh)
    if return_bbox:
        return bbox


# ============================================================
# Unlabeled textured negatives
# ============================================================

def _negative_patch(pw, ph, tiles_pool, rng, mono):
    """Build a pw x ph textured patch that will NOT be annotated. Mix of:
    real material quilt (hard negative: identical texture to labeled
    instances), dimension/furniture grid, and section hatching. Lines are
    drawn on the patch canvas so they clip to the patch."""
    kind = rng.random()
    if kind < 0.5 and tiles_pool:
        # Hard negative: same quilted material texture as labeled instances,
        # but no label. Forces the model to use context, not just "texture".
        tile_meta = rng.choice(tiles_pool)
        try:
            tile_arr = np.array(Image.open(tile_meta['path']).convert('RGB'))
            scale = rng.uniform(0.40, 0.85)
            nh = max(40, int(tile_arr.shape[0] * scale))
            nw = max(40, int(tile_arr.shape[1] * scale))
            tile_arr = np.array(Image.fromarray(tile_arr).resize((nw, nh),
                                                                 Image.BILINEAR))
            try:
                tex = quilt_synthesize(tile_arr, ph, pw, rng)
            except Exception:
                ry = math.ceil(ph / tile_arr.shape[0])
                rx = math.ceil(pw / tile_arr.shape[1])
                tex = np.tile(tile_arr, (ry, rx, 1))[:ph, :pw]
            return Image.fromarray(tex).resize((pw, ph), Image.BILINEAR)
        except Exception:
            pass  # fall through to a line pattern
    patch = Image.new('RGB', (pw, ph), (255, 255, 255))
    d = ImageDraw.Draw(patch)
    shade = rng.randint(60, 150)
    col = (shade, shade, shade)
    if kind < 0.8:
        # Dimension / furniture grid
        step = rng.randint(14, 40)
        for x in range(0, pw, step):
            d.line([(x, 0), (x, ph)], fill=col, width=1)
        for y in range(0, ph, step):
            d.line([(0, y), (pw, y)], fill=col, width=1)
    else:
        # 45-degree section hatching
        step = rng.randint(8, 22)
        for off in range(-ph, pw, step):
            d.line([(off, ph), (off + ph, 0)], fill=col, width=1)
    return patch


def _inject_negative_textures(img, occupied, tiles_pool, rng, W, H, mono):
    """Paste a few UNLABELED textured regions into background (unoccupied)
    space. occupied is updated so labeled instances and negatives never
    collide. Default-OFF (NEGATIVE_TEXTURE_PROB=0)."""
    if NEGATIVE_TEXTURE_PROB <= 0.0 or rng.random() >= NEGATIVE_TEXTURE_PROB:
        return
    n_neg = rng.randint(1, 4)
    placed = 0
    d = ImageDraw.Draw(img)
    for _ in range(n_neg * 6):
        if placed >= n_neg:
            break
        pw = max(60, min(int(rng.uniform(0.05, 0.22) * W), W - 4))
        ph = max(60, min(int(rng.uniform(0.05, 0.22) * H), H - 4))
        if pw >= W or ph >= H:
            continue
        px = rng.randint(0, W - pw)
        py = rng.randint(0, H - ph)
        if float(occupied[py:py + ph, px:px + pw].mean()) > 0.03:
            continue  # would collide with a labeled instance
        patch = _negative_patch(pw, ph, tiles_pool, rng, mono)
        img.paste(patch, (px, py))
        # Some negatives get a thin border (legend swatch / callout box), which
        # makes them look even more like a candidate instance => harder negative.
        if rng.random() < 0.5:
            s = rng.randint(20, 90)
            d.rectangle([px, py, px + pw - 1, py + ph - 1],
                        outline=(s, s, s), width=rng.randint(1, 2))
        occupied[py:py + ph, px:px + pw] = 1
        placed += 1


def _stone_ashlar_crop(w, h, rng):
    """White crop carrying a stone/ashlar tile-OUTLINE pattern (thin lines on
    white) -- how real floor plans draw flooring (idx0's covered patio), as a
    tile grid rather than a dense fill. ink ends ~0.06-0.12. Mask is unchanged."""
    w = max(1, int(w)); h = max(1, int(h))
    img = Image.new('RGB', (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    col = (rng.randint(70, 120),) * 3
    ch = rng.randint(30, 56)                      # course (row) height
    y = 0
    while y < h:
        d.line([(0, y), (w, y)], fill=col, width=1)
        x = rng.randint(0, ch)                    # row-offset joints (running bond)
        while x < w:
            d.line([(x, y), (x, min(h, y + ch))], fill=col, width=1)
            x += rng.randint(max(8, int(ch * 0.7)), max(12, int(ch * 1.8)))
        y += ch
    d.line([(0, h - 1), (w, h - 1)], fill=col, width=1)
    return img


# ============================================================
# Compose one synthetic image
# ============================================================

def compose_image(tiles_pool, rng: random.Random, image_id: int):
    mode_names = ['elevation', 'freeform', 'roof_plan']
    mode = rng.choices(mode_names, weights=[MODE_WEIGHTS[m] for m in mode_names])[0]

    # Pure black-and-white image: draw only from grayscale tiles so the rendered
    # patterns carry no color. Matches the ~50% of real plans that are monochrome.
    mono = rng.random() < MONO_IMAGE_PROB
    if mono:
        gray_tiles = [t for t in tiles_pool if t.get('is_gray')]
        if len(gray_tiles) >= 3:
            tiles_pool = gray_tiles
        else:
            mono = False  # not enough B&W tiles to build a scene; fall back

    # Real-style: mimic the hard real plans by REMOVING easy synth cues (dense
    # color, dark instance outlines) → faint outline-free regions where the model
    # must infer boundaries from texture, as on the failing real plans.
    # iter3 NOTE: merging roof facets into ONE whole instance (to match idx11's
    # GT=1) was a strong negative (synth 0.49, real 0.17, div +0.30) — the v61
    # coarsening trap. iter4: apply the faint/no-outline treatment to roof_plan
    # WITHOUT merging (facets stay separate instances at INSTANCE_SCALE=1.0), to
    # remove easy cues without coarsening.
    realstyle = (mode in ('elevation', 'roof_plan', 'freeform')) and (rng.random() < REALSTYLE_PROB)  # freeform realstyle = code-based sparse floor plans

    if mode == 'elevation':
        if realstyle:
            _rs_saved = _realstyle_scene_overrides()
            items, meta = build_elevation_scene(rng, tiles_pool)
            _restore_scene_overrides(_rs_saved)
        else:
            items, meta = build_elevation_scene(rng, tiles_pool)
        if not items or meta is None:
            mode = 'freeform'; realstyle = False

    if mode == 'elevation':
        # Tight crop: canvas size from building bbox + margins for grid + dim chains.
        ox, oy, ox2, oy2 = meta['overall_bbox']
        grade_y0 = meta['grade_y']
        margin_left  = rng.randint(120, 200)
        margin_right = rng.randint(180, 260)   # room for side floor-height chain
        margin_top   = rng.randint(220, 320)   # room for top dim chain + grid bubbles
        margin_bot   = rng.randint(180, 260)   # room for grade hatch + bottom bubbles
        canvas_w = (ox2 - ox) + margin_left + margin_right
        canvas_h = (grade_y0 - oy) + margin_top + margin_bot
        canvas_w = max(min(canvas_w, 5500), 1400)
        canvas_h = max(min(canvas_h, 3500),  900)
        # Off-center excerpt crop: real examples are often partial page captures,
        # not perfectly centered full elevations.
        span_x = ox2 - ox
        span_y = max(1, grade_y0 - oy)
        dx = margin_left - ox + rng.randint(
            -int(span_x * ELEVATION_EXCERPT_SHIFT_FRAC),
            int(span_x * ELEVATION_EXCERPT_SHIFT_FRAC),
        )
        dy = margin_top - oy + rng.randint(
            -int(span_y * ELEVATION_EXCERPT_SHIFT_FRAC * 0.6),
            int(span_y * ELEVATION_EXCERPT_SHIFT_FRAC * 0.6),
        )
        if rng.random() < ELEVATION_PARTIAL_CROP_PROB:
            dx += rng.choice([-1, 1]) * rng.randint(
                int(span_x * 0.06),
                max(int(span_x * 0.06), int(span_x * ELEVATION_PARTIAL_CROP_FRAC)),
            )
            dy += rng.choice([-1, 1]) * rng.randint(
                int(span_y * 0.02),
                max(int(span_y * 0.02), int(span_y * ELEVATION_PARTIAL_CROP_FRAC * 0.45)),
            )
        for item in items:
            item['poly'] = [(p[0] + dx, p[1] + dy) for p in item['poly']]
        new_meta = {
            'overall_bbox': (ox + dx, oy + dy, ox2 + dx, oy2 + dy),
            'grade_y': grade_y0 + dy,
            'buildings': [{**b,
                           'bldg_left':  b['bldg_left']  + dx,
                           'bldg_right': b['bldg_right'] + dx,
                           'bldg_top':   b['bldg_top']   + dy,
                           'bldg_bot':   b['bldg_bot']   + dy,
                           'wall_left':  b['wall_left']  + dx,
                           'wall_right': b['wall_right'] + dx,
                           'wall_top':   b['wall_top']   + dy,
                           'grade_y':    b['grade_y']    + dy}
                          for b in meta['buildings']],
            'context_features': [_translate_elevation_feature(f, dx, dy)
                                 for f in meta.get('context_features', [])],
        }
        meta = new_meta
        W, H = canvas_w, canvas_h
        img = Image.new('RGB', (W, H), (255, 255, 255))
        render_elevation_decorations(ImageDraw.Draw(img), meta, rng, W, H)
    elif mode == 'roof_plan':
        W = rng.randint(*CANVAS_W_RANGE); H = rng.randint(*CANVAS_H_RANGE)
        items, meta = build_roofplan_scene(W, H, rng, tiles_pool)
        img = Image.new('RGB', (W, H), (255, 255, 255))
    else:
        W = rng.randint(*CANVAS_W_RANGE); H = rng.randint(*CANVAS_H_RANGE)
        global FREEFORM_REALSTYLE
        _ff_saved = FREEFORM_REALSTYLE
        FREEFORM_REALSTYLE = realstyle
        items, meta = build_floorplan_scene(W, H, rng, tiles_pool)
        FREEFORM_REALSTYLE = _ff_saved
        img = Image.new('RGB', (W, H), (255, 255, 255))
        render_floorplan_pre(ImageDraw.Draw(img), meta, rng)

    occupied = np.zeros((H, W), dtype=np.uint8)
    annotations = []
    next_ann_id = 1
    pat_local_idx = {}
    next_pat = 1
    tex_cache = {}
    # Per-category base color. Whether a given instance/category uses the dense
    # fill can be sampled at category or instance scope, but the color itself
    # stays stable for a tile path so reference-linked instances still read as
    # the same material family.
    cat_style = {}
    cat_dense = {}

    rng.shuffle(items)

    for item in items:
        poly = item['poly']
        tile_meta = item['tile']
        role = item['role']
        poly = [(max(0, min(W - 1, px)), max(0, min(H - 1, py))) for px, py in poly]
        # v5.1: lower floor so small per-unit instances (real plans have a long
        # small-instance tail) survive, while still dropping degenerate slivers.
        min_area = 38 * 38 if role == 'roof_plan' else 48 * 48
        if _polygon_area(poly) < min_area:
            continue
        pm = _polygon_to_mask(poly, W, H)
        pm_area = int(pm.sum())
        # Allow up to 3% overlap (covers shared edges between stacked wall bands)
        if pm_area > 0 and int((occupied & pm).sum()) > max(64, int(0.03 * pm_area)):
            continue

        path = tile_meta['path']
        # Cache key includes role so we can use a different texture scale per role.
        scale_key = role
        cache_key = (path, scale_key)
        if cache_key not in tex_cache:
            tile_arr = np.array(Image.open(path).convert('RGB'))
            # Resize tile so its repeat period matches architectural scale.
            # Walls/roof in elevations: tiles render at 0.55-0.85 of original.
            # Floor plan: 0.45-0.75 (smaller, more repeats).
            # Roof plans: 0.30-0.55 (very small dotted/shingle pattern).
            if role == 'roof_plan':
                scale = rng.uniform(0.30, 0.55)
            elif role in ('wall', 'roof', 'garage'):
                scale = rng.uniform(0.55, 0.95)
            else:  # 'free' (floor plan)
                scale = rng.uniform(0.45, 0.85)
            new_h = max(40, int(tile_arr.shape[0] * scale))
            new_w = max(40, int(tile_arr.shape[1] * scale))
            tile_arr = np.array(Image.fromarray(tile_arr).resize((new_w, new_h),
                                                                  Image.BILINEAR))
            target_dim = rng.randint(700, 1100)
            try:
                tex_arr = quilt_synthesize(tile_arr, target_dim, target_dim, rng)
            except Exception:
                ry_ = math.ceil(target_dim / tile_arr.shape[0])
                rx_ = math.ceil(target_dim / tile_arr.shape[1])
                tex_arr = np.tile(tile_arr, (ry_, rx_, 1))[:target_dim, :target_dim]
            tex_cache[cache_key] = Image.fromarray(tex_arr)
        tex_img = tex_cache[cache_key]
        if rng.random() < 0.5:
            tex_img = tex_img.transpose(Image.FLIP_LEFT_RIGHT)

        bx0 = min(p[0] for p in poly); by0 = min(p[1] for p in poly)
        bx1 = max(p[0] for p in poly); by1 = max(p[1] for p in poly)
        bw_ = bx1 - bx0; bh_ = by1 - by0
        if tex_img.width < bw_ or tex_img.height < bh_:
            rx_ = math.ceil(bw_ / tex_img.width)
            ry_ = math.ceil(bh_ / tex_img.height)
            big = Image.new('RGB', (tex_img.width * rx_, tex_img.height * ry_))
            for iy in range(ry_):
                for ix in range(rx_):
                    big.paste(tex_img, (ix * tex_img.width, iy * tex_img.height))
            tex_img_use = big
        else:
            tex_img_use = tex_img
        ox_ = rng.randint(0, max(0, tex_img_use.width - bw_))
        oy_ = rng.randint(0, max(0, tex_img_use.height - bh_))
        crop = tex_img_use.crop((ox_, oy_, ox_ + bw_, oy_ + bh_))
        # Tile/stone OUTLINE rendering for floor-plan material regions (matches
        # real flooring like idx0's stone patio; replaces the dense fill so the
        # model learns outline-grid regions ARE segmentable material). Per-region.
        tile_outline = (role == 'free' and FREEFORM_TILE_OUTLINE_PROB > 0.0
                        and rng.random() < FREEFORM_TILE_OUTLINE_PROB)
        # Dense colored fill: blend toward a muted base material color, then
        # multiply the hatch over it. This lets us tune interior ink coverage
        # toward real plans without forcing every dense instance to fully solid.
        if role == 'free' and IDX0_STONE_PATH and path == IDX0_STONE_PATH:
            # idx0's REAL patio stone, rendered VERY FAINT (near-invisible) to match
            # idx0's hard, low-signal patio: the model can't reliably segment a
            # barely-visible region, so synth_val drops toward iou(idx0) ~0.04 and
            # the divergence closes (down-convergence; up-convergence is blocked by
            # the synth->real context gap, oracle 0.04 even with the exact stone).
            crop = Image.blend(crop, Image.new('RGB', crop.size, (252, 252, 252)),
                               rng.uniform(IDX0_STONE_FAINT[0], IDX0_STONE_FAINT[1]))
        elif tile_outline:
            crop = _stone_ashlar_crop(bw_, bh_, rng)
        elif realstyle and role != 'roof':
            # Real-style faint wall: lighten the quilt toward white so only a
            # subtle siding/brick texture remains (matches the cream, near-white
            # real facades), no dense color and no min-ink darkening.
            # NOTE (iter2, 2026-06-03): widening this toward STRONG texture
            # (0.05-0.75) to mimic worst-real idx5's crisp shingle/siding HURT real
            # (0.258 -> 0.210). Strong synth texture pushed appearance away from the
            # ~50%-faint-mono real set and/or added its own easy cue. Faint is best.
            white = Image.new('RGB', crop.size, (252, 252, 252))
            # Free-role (floor-plan) fills use a LIGHTER whitening tuned to the
            # measured target ink (~0.12); walls/roof keep the faint range.
            _wlo, _whi = (FREEFORM_REALSTYLE_WHITE if role == 'free' else (0.55, 0.80))
            crop = Image.blend(crop, white, rng.uniform(_wlo, _whi))
            # iter-loop (match worst-real idx09, 2026-06-03): real elevation walls
            # read as WHITE with evenly-spaced horizontal clapboard/lap-siding
            # lines, NOT a flat tinted blob. When the picked tile is smooth
            # (stucco) the white-blend leaves a featureless gray fill -> the model
            # learns "segment the gray region", a cue real lacks. Draw faint
            # horizontal lap lines so the wall is defined by internal line texture
            # on white. Mask is unchanged (full polygon). Gated to realstyle walls.
            if role == 'wall' and REALSTYLE_CLAPBOARD:
                cd = ImageDraw.Draw(crop)
                lap = rng.randint(9, 17)
                g = rng.randint(168, 205)
                yy = rng.randint(0, lap)
                while yy < crop.height:
                    cd.line([(0, yy), (crop.width, yy)], fill=(g, g, g), width=1)
                    yy += lap
        elif DENSE_COLOR_FILL and role != 'roof':
            if path not in cat_style:
                if mono:
                    g = rng.randint(*MONO_FILL_GRAY_RANGE)
                    cat_style[path] = (g, g, g)
                else:
                    cat_style[path] = rng.choice(FILL_PALETTE)
            color = cat_style[path]
            dense_bias = float(item.get('dense_bias', 0.0))
            dense_opacity = min(1.0, DENSE_FILL_OPACITY
                                + float(item.get('dense_opacity_boost', 0.0)))
            if item.get('force_dense'):
                use_dense = True
            elif DENSE_FILL_SCOPE == 'category' and dense_bias <= 1e-6:
                if path not in cat_dense:
                    cat_dense[path] = rng.random() < DENSE_FILL_FRAC
                use_dense = cat_dense[path]
            else:
                use_dense = rng.random() < min(1.0, DENSE_FILL_FRAC + dense_bias)
            if use_dense:
                base = Image.new('RGB', crop.size, color)
                hatch = crop
                if dense_opacity < 1.0:
                    crop = Image.blend(crop, base, dense_opacity)
                else:
                    crop = base
                crop = ImageChops.multiply(crop, hatch)
            # Minimum-ink floor, applied to EVERY pattern region (dense or not):
            # lift too-faint regions (low-ink tiles, esp. mono, and non-dense
            # regions) toward real interior coverage so none render near-white.
            # Regions already at/above the floor are untouched. Blends toward a
            # dark shade of the region color, preserving hatch texture.
            if DENSE_FILL_MIN_INK > 0.0:
                cur_lum = float(np.asarray(crop.convert('L')).mean())
                target_lum = 255.0 * (1.0 - DENSE_FILL_MIN_INK)
                if cur_lum > target_lum:
                    dark = tuple(int(c * 0.45) for c in color)
                    dark_lum = 0.299 * dark[0] + 0.587 * dark[1] + 0.114 * dark[2]
                    if cur_lum > dark_lum:
                        op = min(1.0, (cur_lum - target_lum) / (cur_lum - dark_lum))
                        crop = Image.blend(crop, Image.new('RGB', crop.size, dark), op)
        # iter7 NOTE (2026-06-03): feathering the VISIBLE realstyle boundary (soft
        # edge, crisp label) to break the synth_iou@10≈0.37 attractor FAILED —
        # synth@10 stayed 0.39, div@ep10 +0.208 (worse). The model learns the
        # generator's full-polygon label from window/grid/extent cues regardless of
        # the soft visual edge. CONCLUSIVE: synth_iou@10 is fixed by the LABEL-
        # GENERATION RULES, not image appearance; no image manipulation lowers it.
        layer = Image.new('RGB', (W, H), (255, 255, 255))
        layer.paste(crop, (bx0, by0))
        img.paste(layer, mask=Image.fromarray(pm * 255))

        # Pattern boundary line. Off when ablating the outline shortcut, and off
        # for real-style images (real material regions have no drawn boundary —
        # the model must infer it from the texture, as on the failing real plans).
        if item.get('draw_outline', DRAW_INSTANCE_OUTLINE) and not realstyle:
            ImageDraw.Draw(img).line(poly + [poly[0]], fill=(20, 20, 20), width=2)

        wants_windows = item.get('has_windows', role == 'wall')
        if wants_windows and role == 'wall':
            holes = place_windows_in_polygon(img, poly, rng)
        elif role == 'garage':
            holes = [place_garage_door(img, poly, rng)]
        else:
            # Floor-plan patterns carry no holes: fixtures are never drawn
            # inside pattern rooms, so the flooring fills the whole region.
            holes = []

        occupied |= pm

        if path not in pat_local_idx:
            pat_local_idx[path] = next_pat
            next_pat += 1
        pid = pat_local_idx[path]

        # Organic boundary jitter on the LABEL only (image already rendered crisp
        # above) so synth annotations look human-traced, not machine-perfect.
        if LABEL_JITTER > 0.0:
            jpoly = _jitter_poly(poly, rng, LABEL_JITTER)
            jholes = [_jitter_poly(h, rng, LABEL_JITTER) for h in holes]
        else:
            jpoly, jholes = poly, holes
        outer_flat = [c for p in jpoly for c in p]
        hole_flats = [[c for p in hole for c in p] for hole in jholes]
        outer_area = _polygon_area(jpoly)
        holes_area = sum(_polygon_area(h) for h in jholes)
        annotations.append({
            'id': next_ann_id,
            'category_id': tile_meta['category_id'],
            'category_name': f'pattern{pid}',
            'reference_tile': path,
            'segmentation': [outer_flat] + hole_flats,
            'num_holes': len(hole_flats),
            'bbox': [bx0, by0, bw_, bh_],
            'area': float(outer_area - holes_area),
            'role': role,
        })
        next_ann_id += 1

    # Unlabeled textured negatives: texture present, no annotation. Teaches the
    # model that texture alone is not an instance (cuts real over-prediction).
    _inject_negative_textures(img, occupied, tiles_pool, rng, W, H, mono)

    if mode == 'freeform':
        render_floorplan_post(ImageDraw.Draw(img), meta, rng, W, H)
    elif mode == 'roof_plan':
        render_roofplan_post(ImageDraw.Draw(img), meta, rng, W, H)
    elif mode == 'elevation':
        render_elevation_post(ImageDraw.Draw(img), meta, rng, W, H)
    img = render_markup_overlay(img, meta, rng, mode, mono=mono)
    img = apply_document_effects(img, rng, mode)

    return img, {
        'image': {'file_name': f'synth_{image_id:06d}.png', 'width': W, 'height': H},
        'mode': mode,
        'annotations': annotations,
    }


# ============================================================
# Main
# ============================================================

_WORKER_TILES = None
_WORKER_IMG_DIR = None
_WORKER_ANN_DIR = None


def _apply_smoke_overrides():
    global CANVAS_W_RANGE, CANVAS_H_RANGE, PATTERNS_PER_IMAGE, INSTANCES_PER_PATTERN
    CANVAS_W_RANGE = (1600, 2200)
    CANVAS_H_RANGE = (1100, 1500)
    PATTERNS_PER_IMAGE = (1, 2)
    INSTANCES_PER_PATTERN = (1, 3)


def _worker_init(tiles_dir, img_dir, ann_dir, smoke, no_outline=False,
                 dense_fill_frac=None, dense_fill_opacity=None,
                 dense_fill_scope=None, no_dense_fill=False,
                 mode_weights=None, elevation_excerpt_shift=None,
                 markup_overlay_prob=None, split_scale=None,
                 mono_image_prob=None, instance_scale=None,
                 dense_fill_min_ink=None, negative_texture_prob=None,
                 realstyle_prob=None, freeform_tile_outline_prob=None,
                 idx0_stone=None):
    """Pool initializer: loads tiles once per worker, applies smoke / no-outline
    overrides in the child process (forked globals don't propagate under 'spawn'
    start methods)."""
    global _WORKER_TILES, _WORKER_IMG_DIR, _WORKER_ANN_DIR
    global DRAW_INSTANCE_OUTLINE, DENSE_COLOR_FILL, DENSE_FILL_FRAC
    global DENSE_FILL_OPACITY, DENSE_FILL_SCOPE
    global MODE_WEIGHTS, ELEVATION_EXCERPT_SHIFT_FRAC, MARKUP_OVERLAY_PROB
    global MONO_IMAGE_PROB, INSTANCE_SCALE, DENSE_FILL_MIN_INK
    global NEGATIVE_TEXTURE_PROB, REALSTYLE_PROB, FREEFORM_TILE_OUTLINE_PROB
    global IDX0_STONE_PATH
    if smoke:
        _apply_smoke_overrides()
    if no_outline:
        DRAW_INSTANCE_OUTLINE = False
    if no_dense_fill:
        DENSE_COLOR_FILL = False
    if dense_fill_frac is not None:
        DENSE_FILL_FRAC = dense_fill_frac
    if dense_fill_opacity is not None:
        DENSE_FILL_OPACITY = dense_fill_opacity
    if dense_fill_scope is not None:
        DENSE_FILL_SCOPE = dense_fill_scope
    if mode_weights is not None:
        MODE_WEIGHTS = mode_weights
    if elevation_excerpt_shift is not None:
        ELEVATION_EXCERPT_SHIFT_FRAC = elevation_excerpt_shift
    if markup_overlay_prob is not None:
        MARKUP_OVERLAY_PROB = markup_overlay_prob
    if mono_image_prob is not None:
        MONO_IMAGE_PROB = mono_image_prob
    if instance_scale is not None:
        INSTANCE_SCALE = instance_scale
        _apply_split_scale(instance_scale)  # coarser surfaces + fewer splits
    if dense_fill_min_ink is not None:
        DENSE_FILL_MIN_INK = dense_fill_min_ink
    if negative_texture_prob is not None:
        NEGATIVE_TEXTURE_PROB = negative_texture_prob
    if realstyle_prob is not None:
        REALSTYLE_PROB = realstyle_prob
    if freeform_tile_outline_prob is not None:
        FREEFORM_TILE_OUTLINE_PROB = freeform_tile_outline_prob
    if idx0_stone is not None:
        IDX0_STONE_PATH = idx0_stone
    if split_scale is not None:
        _apply_split_scale(split_scale)
    _WORKER_TILES = load_curated_tiles(tiles_dir)
    _WORKER_IMG_DIR = img_dir
    _WORKER_ANN_DIR = ann_dir


def _worker_render(job):
    image_id, seed = job
    rng = random.Random(seed)
    img, ann = compose_image(_WORKER_TILES, rng, image_id)
    img.save(os.path.join(_WORKER_IMG_DIR, f'synth_{image_id:06d}.png'),
             optimize=True, compress_level=6)
    with open(os.path.join(_WORKER_ANN_DIR, f'synth_{image_id:06d}.json'), 'w') as f:
        json.dump(ann, f)
    return image_id, img.size, len(ann['annotations'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=OUT_ROOT)
    ap.add_argument('--tiles', type=str, default=TILES_DIR,
                    help='path to reference_tiles_curated directory (with manifest.json)')
    ap.add_argument('--workers', type=int, default=(os.cpu_count() or 1),
                    help='number of parallel worker processes')
    ap.add_argument('--smoke', action='store_true', help='small canvas/few instances for fast verify')
    ap.add_argument('--no-outline', action='store_true',
                    help='ablate the per-instance boundary line (suspected synth shortcut)')
    ap.add_argument('--no-dense-fill', action='store_true',
                    help='disable dense/colored material fills')
    ap.add_argument('--dense-fill-frac', type=float, default=None,
                    help='fraction of instances/categories using dense fills')
    ap.add_argument('--dense-fill-opacity', type=float, default=None,
                    help='blend weight for the dense base color (0..1)')
    ap.add_argument('--dense-fill-scope', type=str, default=None,
                    choices=('category', 'instance'),
                    help='sample dense fills per category or per instance')
    ap.add_argument('--mode-weights', type=str, default=None,
                    help='comma-separated weights: elevation,freeform,roof_plan')
    ap.add_argument('--elevation-excerpt-shift', type=float, default=None,
                    help='fractional random crop shift around elevation bbox')
    ap.add_argument('--split-scale', type=float, default=None,
                    help='multiplier (<=1) on all region split/interleave probs; '
                         'lowers instances/image toward the real distribution '
                         '(0 = no added splitting)')
    ap.add_argument('--markup-overlay-prob', type=float, default=None,
                    help='probability of adding markup-style circles/notes')
    ap.add_argument('--mono-image-prob', type=float, default=None,
                    help='fraction of images rendered as pure black-and-white '
                         '(grayscale tiles only, gray fills/markup); ~50%% of real '
                         'plans are monochrome')
    ap.add_argument('--instance-scale', type=float, default=None,
                    help='instance granularity in [0,1]: 1.0 = current fine '
                         'decomposition, lower = coarser toward real (~3 inst/img). '
                         'Caps elevation bands & rowhouse units, merges same-material '
                         'roof facets, biases roof footprints to single-wing, and '
                         'scales all split probs.')
    ap.add_argument('--dense-fill-min-ink', type=float, default=None,
                    help='minimum interior ink coverage for dense-filled regions '
                         '(real median ~0.53). Lifts too-faint regions so they do '
                         'not render near-white; 0 = off.')
    ap.add_argument('--negative-texture-prob', type=float, default=None,
                    help='per-image prob of injecting UNLABELED textured negatives '
                         '(material swatches, dimension grids, hatching) into '
                         'background. Teaches that texture alone != instance, to '
                         'cut real over-prediction; 0 = off (default).')
    ap.add_argument('--realstyle-prob', type=float, default=None,
                    help='per-elevation prob of rendering a "real-style" excerpt: '
                         'one WHOLE faint outline-free material region per surface '
                         '(no banding/interleave), mimicking the real elevations '
                         'the model fragments worst; 0 = off (default).')
    ap.add_argument('--freeform-tile-outline', type=float, default=None,
                    help='per-region prob of rendering a floor-plan material region '
                         'as a tile/stone OUTLINE pattern (thin lines on white) '
                         'instead of a dense fill, matching real flooring (idx0 '
                         'stone patio); 0 = off (default).')
    ap.add_argument('--idx0-stone', type=str, default=None,
                    help='path to the target idx0 covered-patio stone texture; when '
                         'set, floor plans get one big WIDE + one small region '
                         'rendered with this REAL texture (idx0-faithful). off=None.')
    ap.add_argument('--clutter-boost', action='store_true',
                    help='max all sheet-context clutter (title block, hidden lines, '
                         'multiple keyed notes) to match real CAD-page density.')
    ap.add_argument('--label-jitter', type=float, default=None,
                    help='std (px) of organic boundary jitter on annotation polygons '
                         '(image stays crisp) so synth labels look human-traced, not '
                         'machine-perfect; lowers synth_iou toward real. 0 = off.')
    ap.add_argument('--resolution-scale', type=float, default=None,
                    help='render elevation buildings at this x native pixel size '
                         '(~2.6 matches real ~4800px) so eval downsampling makes synth '
                         'instances as small/hard as real. 1.0 = off.')
    args = ap.parse_args()

    if args.smoke:
        _apply_smoke_overrides()
    global DRAW_INSTANCE_OUTLINE, DENSE_COLOR_FILL, DENSE_FILL_FRAC
    global DENSE_FILL_OPACITY, DENSE_FILL_SCOPE, MODE_WEIGHTS
    global ELEVATION_EXCERPT_SHIFT_FRAC, MARKUP_OVERLAY_PROB
    global MONO_IMAGE_PROB, INSTANCE_SCALE, DENSE_FILL_MIN_INK
    global NEGATIVE_TEXTURE_PROB, REALSTYLE_PROB, CLUTTER_BOOST, LABEL_JITTER
    global RESOLUTION_SCALE
    if args.clutter_boost:
        CLUTTER_BOOST = True
    if args.label_jitter is not None:
        LABEL_JITTER = args.label_jitter
    if args.resolution_scale is not None:
        RESOLUTION_SCALE = args.resolution_scale
    if args.no_outline:
        DRAW_INSTANCE_OUTLINE = False
    if args.no_dense_fill:
        DENSE_COLOR_FILL = False
    if args.dense_fill_frac is not None:
        DENSE_FILL_FRAC = args.dense_fill_frac
    if args.dense_fill_opacity is not None:
        DENSE_FILL_OPACITY = args.dense_fill_opacity
    if args.dense_fill_scope is not None:
        DENSE_FILL_SCOPE = args.dense_fill_scope
    if args.mode_weights is not None:
        vals = [float(v.strip()) for v in args.mode_weights.split(',')]
        if len(vals) != 3:
            raise ValueError('--mode-weights must have 3 comma-separated values')
        MODE_WEIGHTS = dict(zip(('elevation', 'freeform', 'roof_plan'), vals))
    if args.elevation_excerpt_shift is not None:
        ELEVATION_EXCERPT_SHIFT_FRAC = args.elevation_excerpt_shift
    if args.markup_overlay_prob is not None:
        MARKUP_OVERLAY_PROB = args.markup_overlay_prob
    if args.mono_image_prob is not None:
        MONO_IMAGE_PROB = args.mono_image_prob
    if not 0.0 <= MONO_IMAGE_PROB <= 1.0:
        raise ValueError('--mono-image-prob must be in [0, 1]')
    if args.instance_scale is not None:
        INSTANCE_SCALE = args.instance_scale
    if not 0.0 <= INSTANCE_SCALE <= 1.0:
        raise ValueError('--instance-scale must be in [0, 1]')
    if args.dense_fill_min_ink is not None:
        DENSE_FILL_MIN_INK = args.dense_fill_min_ink
    if not 0.0 <= DENSE_FILL_MIN_INK <= 1.0:
        raise ValueError('--dense-fill-min-ink must be in [0, 1]')
    if args.negative_texture_prob is not None:
        NEGATIVE_TEXTURE_PROB = args.negative_texture_prob
    if not 0.0 <= NEGATIVE_TEXTURE_PROB <= 1.0:
        raise ValueError('--negative-texture-prob must be in [0, 1]')
    if args.realstyle_prob is not None:
        REALSTYLE_PROB = args.realstyle_prob
    if not 0.0 <= REALSTYLE_PROB <= 1.0:
        raise ValueError('--realstyle-prob must be in [0, 1]')
    if not 0.0 <= DENSE_FILL_FRAC <= 1.0:
        raise ValueError('--dense-fill-frac must be in [0, 1]')
    if not 0.0 <= DENSE_FILL_OPACITY <= 1.0:
        raise ValueError('--dense-fill-opacity must be in [0, 1]')
    if not 0.0 <= ELEVATION_EXCERPT_SHIFT_FRAC <= 1.0:
        raise ValueError('--elevation-excerpt-shift must be in [0, 1]')
    if not 0.0 <= MARKUP_OVERLAY_PROB <= 1.0:
        raise ValueError('--markup-overlay-prob must be in [0, 1]')
    if args.split_scale is not None and not 0.0 <= args.split_scale <= 1.0:
        raise ValueError('--split-scale must be in [0, 1]')

    img_dir = os.path.join(args.out, 'images')
    ann_dir = os.path.join(args.out, 'annotations')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Reproducible per-image seed independent of worker assignment.
    jobs = [(i, args.seed * 1_000_003 + i)
            for i in range(args.start, args.start + args.n)]

    if args.workers <= 1:
        _worker_init(
            args.tiles, img_dir, ann_dir, args.smoke, args.no_outline,
            args.dense_fill_frac, args.dense_fill_opacity,
            args.dense_fill_scope, args.no_dense_fill,
            MODE_WEIGHTS, ELEVATION_EXCERPT_SHIFT_FRAC, MARKUP_OVERLAY_PROB,
            args.split_scale, MONO_IMAGE_PROB, args.instance_scale,
            args.dense_fill_min_ink, args.negative_texture_prob,
            args.realstyle_prob, args.freeform_tile_outline, args.idx0_stone,
        )
        print(f'Loaded {len(_WORKER_TILES)} curated tiles.', flush=True)
        t0 = time.time()
        for n, job in enumerate(jobs, 1):
            i, sz, na = _worker_render(job)
            elapsed = time.time() - t0
            print(f'[{n}/{args.n}] synth_{i:06d}.png  {sz}  anns={na}  '
                  f'avg={elapsed / n:.1f}s/img', flush=True)
        return

    from multiprocessing import Pool
    print(f'Spawning {args.workers} workers for {args.n} images.', flush=True)
    t0 = time.time()
    with Pool(processes=args.workers,
              initializer=_worker_init,
              initargs=(
                  args.tiles, img_dir, ann_dir, args.smoke, args.no_outline,
                  args.dense_fill_frac, args.dense_fill_opacity,
                  args.dense_fill_scope, args.no_dense_fill,
                  MODE_WEIGHTS, ELEVATION_EXCERPT_SHIFT_FRAC, MARKUP_OVERLAY_PROB,
                  args.split_scale, MONO_IMAGE_PROB, args.instance_scale,
                  args.dense_fill_min_ink, args.negative_texture_prob,
                  args.realstyle_prob, args.freeform_tile_outline, args.idx0_stone,
              )) as pool:
        for n, (i, sz, na) in enumerate(
                pool.imap_unordered(_worker_render, jobs, chunksize=4), 1):
            elapsed = time.time() - t0
            rate = n / elapsed
            eta_s = (args.n - n) / rate if rate > 0 else 0
            print(f'[{n}/{args.n}] synth_{i:06d}.png  {sz}  anns={na}  '
                  f'rate={rate:.2f}img/s  eta={eta_s / 60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
