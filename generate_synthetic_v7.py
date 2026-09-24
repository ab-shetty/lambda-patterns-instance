#!/usr/bin/env python3
"""synthetic v7 -- v6 re-tuned by looking at Gemini plans side by side.

v6d was compared visually with the labelled Gemini r2-r4 plans (2026-09-23,
contact sheets of 18 each). The differences, and what v7 changes for each:

  1. Framing. Gemini drawings fill the frame; v6d puts a small drawing on a
     sheet that is ~70% empty paper, so text, fills and regions are all tiny
     relative to the image. v7 crops every sheet to its ink bounding box plus a
     small pad and resamples to Gemini's long side (median 3168 px).
  2. Massing. Gemini houses have front cross-gable projections, stepped
     volumes, dormers, porches and chimneys; v6d is mostly one box with one
     roof. v7 adds 1-2 front projections and a set-back upper volume, and raises
     dormer / chimney / porch rates.
  3. Materials by volume. Gemini changes material per volume (board-and-batten
     gable over shingle over brick), with 2-4 families per elevation. v7 almost
     always places an accent and more often a second one.
  4. Colour. Gemini's colour is muted -- olive, tan, beige, warm grey, with
     brick-red / charcoal / slate roofs and white trim. v6d's palette includes
     saturated teal, red, purple, royal blue and green/purple trim. v7 swaps in
     a muted palette.

v6 itself is untouched (v6d stays bit-for-bit reproducible): this module
patches v6's module globals and wraps its compose().

    python3 generate_synthetic_v7.py --n 48 --out data/synthetic/v7_smoke --workers 64
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

import cv2
import json
import numpy as np
from shapely.geometry import Polygon, box
from shapely import affinity

import generate_synthetic_v6 as v6

GEMINI_LONG_SIDE = (2400, 3600)
CROP_PAD = (0.015, 0.05)          # fraction of the content's long side

# muted, realistic BIM-render colours (RGB)
v6.BASE_PALETTE = [
    (150, 148, 110), (140, 142, 104), (120, 128, 96), (168, 160, 120),   # olive / sage
    (196, 176, 140), (210, 190, 160), (188, 170, 140), (222, 206, 176),  # tan / beige
    (232, 222, 200), (240, 232, 214), (245, 238, 224),                   # cream
    (200, 150, 90), (212, 168, 110), (186, 140, 96),                     # ochre / wood
    (170, 110, 90), (160, 100, 84), (180, 128, 108),                     # brick
    (160, 160, 156), (180, 180, 176), (200, 200, 196), (136, 138, 140),  # warm grey
    (150, 160, 170), (170, 180, 188), (120, 132, 144),                   # slate blue-grey
    (190, 178, 160), (160, 150, 136), (128, 116, 100),                   # taupe / stone
    (214, 214, 206), (226, 226, 220),
]
v6.TRIM_PALETTE = [(255, 255, 255)] * 6 + [(240, 238, 232), (230, 222, 206), (90, 90, 90)]
v6.ROOF_PALETTE = [
    (150, 80, 70), (160, 90, 80), (140, 70, 60), (170, 100, 88),         # brick-red
    (80, 80, 84), (96, 96, 100), (110, 110, 112), (70, 72, 76),          # charcoal
    (120, 128, 140), (140, 150, 160), (100, 110, 124),                   # slate / metal
    (120, 96, 80), (140, 110, 90),                                       # brown
]

# Gemini lettering is ~1.6x v6's relative to the drawing (callouts, level marks
# and titles are readable at thumbnail size)
TEXT_SCALE = 1.6
_font_v6 = v6.font
v6.font = lambda size, bold=False: _font_v6(int(size * TEXT_SCALE), bold)

# Gemini's mono sheets are crisp dark linework; v6's 50% "light"/"faint" ink
# levels read as washed-out grey next to them
_make_appearance_v6 = v6.make_appearance


def make_appearance(rng):
    app = _make_appearance_v6(rng)
    if app["level"] != "normal" and rng.random() < 0.7:
        ink = rng.randint(15, 60)
        fv = min(200, ink + rng.randint(0, 60))
        app.update(level="normal", ink=(ink,) * 3, outline=(ink,) * 3, ink_fill=(fv,) * 3)
    return app


v6.make_appearance = make_appearance
# 2-view sheets are wide strips with a small building; Gemini is mostly 1 view
v6.VIEW_COUNT_WEIGHTS = {1: 62, 2: 34, 4: 4}
v6.CORNER_BOARD_CLIP = True
# Gemini annotators cut windows and doors out as remove polygons, consistently
v6.WINDOW_HOLE_PROB = 0.97
# Gemini sheets are whole drawings, rarely cut at the page edge
v6.EXCERPT_CROP_PROB = 0.10
# Gemini is mostly elevations (~70% by eye over 27 sheets)
v6.MODE_WEIGHTS = {"elevation": 72, "roof_plan": 15, "freeform": 13}

_make_house_v6 = v6.make_house


def make_house(rng):
    house = _make_house_v6(rng)
    blocks = house["blocks"]
    main = blocks[0]
    if not house.get("units"):
        # front cross-gable projection(s): gable faces the viewer, its own volume
        n_proj = rng.choices([0, 1, 2], weights=[25, 55, 20])[0]
        used = []
        for k in range(n_proj):
            pw = rng.uniform(0.25, 0.45) * main.w
            for _ in range(8):
                px = rng.uniform(0.02 * main.w, main.w - pw - 0.02 * main.w)
                if all(px + pw < a - 2 or px > b + 2 for a, b in used):
                    break
            else:
                continue
            used.append((px, px + pw))
            pd = rng.uniform(3, 10)
            st = main.stories if rng.random() < 0.6 else max(1, main.stories - 1)
            roof = rng.choices(["gable", "hip"], weights=[80, 20])[0]
            blocks.append(v6.Block(px, -pd, pw, pd + 0.5 * main.d, st, roof, "y",
                                   min(0.9, main.pitch * rng.uniform(1.0, 1.4)), main.ov, "wing"))
        # set-back upper volume on a one-storey main (split-level / pop-up)
        if main.stories == 1 and rng.random() < 0.3:
            uw = rng.uniform(0.35, 0.6) * main.w
            ux = rng.uniform(0, main.w - uw)
            blocks.append(v6.Block(ux, 0.25 * main.d, uw, 0.6 * main.d, 2,
                                   rng.choice(["gable", "gable", "hip"]),
                                   rng.choice(["x", "y"]), main.pitch, main.ov, "wing"))
        if not house["porch"]:
            house["porch"] = rng.random() < 0.3
        if not house["chimney"]:
            house["chimney"] = rng.random() < 0.3
        if not house["dormers"]:
            house["dormers"] = rng.random() < 0.3
        if main.stories == 1 and rng.random() < 0.25:
            main.stories = 2
    # materials change by volume, and most elevations carry 2-3 families
    if house["accent"] is None and rng.random() < 0.8:
        house["accent"] = rng.choice(["gable", "gable", "block", "wainscot", "upper"])
    if house["accent"] and not house.get("accent2") and rng.random() < 0.35:
        other = [z for z in ["gable", "wainscot", "upper", "block"] if z != house["accent"]]
        house["accent2"] = rng.choice(other)
    return house


v6.make_house = make_house


def tight_crop(canvas, ann, rng):
    """Crop to the ink bounding box + a small pad, then resample to Gemini size."""
    paper = np.median(np.concatenate([canvas[0], canvas[-1], canvas[:, 0], canvas[:, -1]]), axis=0)
    diff = np.abs(canvas.astype(np.int16) - paper.astype(np.int16)).max(axis=2) > 45
    rows = np.where(diff.sum(axis=1) >= 3)[0]
    cols = np.where(diff.sum(axis=0) >= 3)[0]
    H, W = canvas.shape[:2]
    if len(rows) == 0 or len(cols) == 0:
        return canvas, ann
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pad = int(rng.uniform(*CROP_PAD) * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    canvas = canvas[y0:y1, x0:x1]
    h, w = canvas.shape[:2]
    target = rng.uniform(*GEMINI_LONG_SIDE)
    f = target / max(h, w)
    nw, nh = max(1, round(w * f)), max(1, round(h * f))
    canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    fx, fy = nw / w, nh / h
    clip = box(0, 0, nw, nh)
    anns = []
    for a in ann["annotations"]:
        rings = [list(zip(s[0::2], s[1::2])) for s in a["segmentation"]]
        try:
            p = Polygon(rings[0], rings[1:]).buffer(0)
        except Exception:
            continue
        p = affinity.scale(affinity.translate(p, -x0, -y0), fx, fy, origin=(0, 0)).intersection(clip)
        for q in v6.polys_of(p):
            if q.area < 400:
                continue
            outer = [round(v, 2) for c in q.exterior.coords[:-1] for v in c]
            holes = [[round(v, 2) for c in r.coords[:-1] for v in c] for r in q.interiors]
            holes = [hh for hh in holes if len(hh) >= 6]
            bx0, by0, bx1, by1 = q.bounds
            anns.append(dict(a, id=len(anns) + 1, segmentation=[outer] + holes, num_holes=len(holes),
                             bbox=[round(bx0, 1), round(by0, 1), round(bx1 - bx0, 1), round(by1 - by0, 1)],
                             area=round(q.area, 1)))
    ann = dict(ann, annotations=anns, px_per_ft=round(ann["px_per_ft"] * fx, 2))
    ann["image"] = dict(ann["image"], width=nw, height=nh, file_name=ann["image"]["file_name"].replace("synth6", "synth7"))
    return canvas, ann


# Label like the Gemini annotators do. In elevations they label the wall
# materials and leave the roof, trim, chimney and foundation unlabelled (a roof
# is labelled on roof plans, where it is the subject). v6 labels every family it
# paints: 3.0 families / 14.4 regions per image against Gemini's 1.7 / 6.4.
# Families below are still drawn -- they become negatives, as in Gemini.
ELEV_LABEL_KEEP = {"roof": 0.15, "trim": 0.05, "chimney": 0.15, "foundation": 0.25}


def gemini_labelling(ann, rng):
    if ann["mode"] != "elevation":
        return ann
    keep = {f: rng.random() < p for f, p in ELEV_LABEL_KEEP.items()}
    anns = [a for a in ann["annotations"] if keep.get(a["family"], True)]
    ids = {}
    for a in anns:
        ids.setdefault(a["family"], len(ids) + 1)
    anns = [dict(a, id=i + 1, category_id=ids[a["family"]], category_name=f"pattern{ids[a['family']]}")
            for i, a in enumerate(anns)]
    return dict(ann, annotations=anns)


_CFG = {}


def _init(out, seed, mw, label_roofs=False):
    if label_roofs:
        # ablation: label elevation roofs as v6d does (v6's own 90% roof gate
        # still applies); the keep draws happen in the same order either way,
        # so every other family's labels are unchanged
        ELEV_LABEL_KEEP["roof"] = 1.0
    _CFG.update(out=out, seed=seed, mw=mw)


def _job(image_id):
    try:
        canvas, ann = v6.compose(image_id, _CFG["seed"], _CFG["mw"])
        rng = random.Random(_CFG["seed"] * 7_000_003 + image_id)
        canvas, ann = tight_crop(canvas, ann, rng)
        ann = gemini_labelling(ann, rng)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (image_id, False, str(exc))
    if not ann["annotations"]:
        return (image_id, False, "no annotations")
    cv2.imwrite(os.path.join(_CFG["out"], "images", ann["image"]["file_name"]), canvas,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    with open(os.path.join(_CFG["out"], "annotations", f"synth7_{image_id:06d}.json"), "w") as f:
        json.dump(ann, f)
    return (image_id, True, ann["mode"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--label-roofs", action="store_true",
                    help="label elevation roofs as v6d does (ablation)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    t = time.time()
    ids = range(args.start, args.start + args.n)
    with Pool(args.workers, initializer=_init, initargs=(args.out, args.seed, v6.MODE_WEIGHTS, args.label_roofs)) as pool:
        res = pool.map(_job, ids, chunksize=1)
    ok = sum(r[1] for r in res)
    modes = {}
    for r in res:
        if r[1]:
            modes[r[2]] = modes.get(r[2], 0) + 1
    print(f"{args.n}/{args.n} ok={ok} {time.time() - t:.0f}s {modes}")


if __name__ == "__main__":
    main()
