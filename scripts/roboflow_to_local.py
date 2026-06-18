#!/usr/bin/env python3
"""Convert a Roboflow COCO-segmentation export into the repo's local-data format.

The real-plan pool is labelled in Roboflow project `perceive-ai/floz-real-pool`
(instance-segmentation). This turns a downloaded COCO-segmentation export into an
`images/ + annotations/` dir consumable by `--local-data` and by
`scripts/build_mix.py` (which globs `images/*.png`).

Class handling (matches the existing 28-real format, which is class-agnostic /
reference-conditioned):
- `pattern`, `pattern1`..`pattern4`  -> generic material-region INSTANCES.
- `remove`                           -> HOLES. Roboflow can't draw holes inside a
  shape, so a `remove` polygon means "subtract this area from the pattern polygon
  it sits inside." The repo's `render_instance_mask` already treats
  `segmentation = [outer, hole1, hole2, ...]` (first poly outer, rest holes), so
  each `remove` is appended as an extra polygon to its parent instance.

A `remove` is matched to its parent pattern by centroid containment (point in
the pattern's outer polygon); ties broken by smallest containing area. If no
pattern contains the centroid, it falls back to the pattern with the largest
mask overlap, and is dropped (with a warning) if it overlaps nothing.

Images are re-saved as PNG (the downstream globs expect `*.png`).

Usage:
    python scripts/roboflow_to_local.py \
        --coco /workspace/roboflow_dl/train/_annotations.coco.json \
        --img-dir /workspace/roboflow_dl/train \
        --out /workspace/real_roboflow_local
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def _poly_xy(seg_flat):
    """COCO flat [x,y,x,y,...] -> Nx2 int32 array (first polygon only)."""
    return np.array(seg_flat, dtype=np.float32).reshape(-1, 2).astype(np.int32)


def _centroid(seg_flat):
    pts = np.array(seg_flat, dtype=np.float32).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _poly_area(seg_flat):
    return abs(cv2.contourArea(_poly_xy(seg_flat)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="path to _annotations.coco.json")
    ap.add_argument("--img-dir", required=True, help="dir holding the export images")
    ap.add_argument("--out", required=True, help="output local-data dir")
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/annotations", exist_ok=True)

    n_img = n_inst = n_hole = n_hole_dropped = 0
    for img_id, im in images.items():
        anns = anns_by_img.get(img_id, [])
        # COCO segmentation is a list of polygons; take only the first ring per
        # annotation (this export is single-polygon per instance).
        patterns, removes = [], []
        for a in anns:
            seg = a["segmentation"]
            if not isinstance(seg, list) or not seg:
                continue  # skip RLE / empty
            ring = seg[0]
            name = id2name.get(a["category_id"], "pattern")
            (removes if name == "remove" else patterns).append(ring)

        # Each pattern -> instance with its outer ring; holes attached below.
        inst_segs = [[p] for p in patterns]
        outer_polys = [_poly_xy(p) for p in patterns]
        areas = [_poly_area(p) for p in patterns]

        for rem in removes:
            cx, cy = _centroid(rem)
            # containing patterns, prefer smallest area
            containing = [j for j, poly in enumerate(outer_polys)
                          if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0]
            if containing:
                j = min(containing, key=lambda k: areas[k])
            else:
                # fallback: largest mask overlap
                rmask = np.zeros((im["height"], im["width"]), np.uint8)
                cv2.fillPoly(rmask, [_poly_xy(rem)], 1)
                best, best_ov = -1, 0
                for k, poly in enumerate(outer_polys):
                    pm = np.zeros_like(rmask)
                    cv2.fillPoly(pm, [poly], 1)
                    ov = int(np.logical_and(pm, rmask).sum())
                    if ov > best_ov:
                        best, best_ov = k, ov
                if best < 0:
                    n_hole_dropped += 1
                    print(f"  WARN: remove polygon in {im['file_name']} overlaps "
                          f"no pattern; dropped")
                    continue
                j = best
            inst_segs[j].append(rem)
            n_hole += 1

        out_anns = [{"segmentation": segs, "category_name": "pattern"}
                    for segs in inst_segs]

        stem = os.path.splitext(im["file_name"])[0]
        png = f"{stem}.png"
        src = os.path.join(args.img_dir, im["file_name"])
        Image.open(src).convert("RGB").save(f"{args.out}/images/{png}")
        json.dump({"image": {"file_name": png,
                             "width": im["width"], "height": im["height"]},
                   "mode": "freeform", "annotations": out_anns},
                  open(f"{args.out}/annotations/{stem}.json", "w"))
        n_img += 1
        n_inst += len(out_anns)

    print(f"wrote {n_img} images, {n_inst} instances, {n_hole} holes attached"
          f"{f', {n_hole_dropped} holes dropped' if n_hole_dropped else ''}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
