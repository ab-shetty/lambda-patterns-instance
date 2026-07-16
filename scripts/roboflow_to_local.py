#!/usr/bin/env python3
"""Convert a Roboflow COCO-segmentation export into the repo's local-data format.

The real-plan pool is labelled in Roboflow project `perceive-ai/floz-real-pool`
(instance-segmentation). This turns a downloaded COCO-segmentation export into an
`images/ + annotations/` dir consumable by `--local-data` and by
`scripts/build_mix.py` (which globs `images/*.png`).

Class handling (matches the existing 28-real reference-conditioned format):
- `pattern`, `pattern1`..`pattern4`  -> material-region instances. The class
  name is preserved so separate occurrences of the same pattern remain positive
  reference matches while different patterns remain negatives.
- `remove`                           -> HOLES. Roboflow can't draw holes inside a
  shape, so a `remove` polygon means "subtract this area from the pattern polygon
  it sits inside." The repo's `render_instance_mask` already treats
  `segmentation = [outer, hole1, hole2, ...]` (first poly outer, rest holes), so
  each `remove` is appended as an extra polygon to every containing instance.

A `remove` is matched to its parent patterns by centroid containment (point in
the pattern's outer polygon). Attaching to every containing pattern is important
when labeled material regions overlap. If no pattern contains the centroid, it
falls back to the pattern with the largest mask overlap, and is dropped (with a
warning) if it overlaps nothing.

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


def _usable_pattern(seg_flat):
    """Reject accidental edge slivers that vanish under training resize."""
    pts = np.array(seg_flat, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or _poly_area(seg_flat) < 16:
        return False
    span = pts.max(axis=0) - pts.min(axis=0)
    return bool(span[0] >= 2 and span[1] >= 2)


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

    n_img = n_img_skipped = n_inst = n_pattern_dropped = 0
    n_hole = n_hole_dropped = 0
    n_hole_attachments = n_multi_parent = 0
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
            if name == "remove":
                removes.append(ring)
            elif not _usable_pattern(ring):
                n_pattern_dropped += 1
                print(f"  WARN: degenerate pattern polygon in {im['file_name']} "
                      "was dropped")
            else:
                patterns.append((ring, name))

        # A Roboflow version may include images that have not been labeled yet.
        # They cannot produce a valid reference-conditioned training example.
        if not patterns:
            n_img_skipped += 1
            continue

        # Each pattern -> instance with its outer ring; holes attached below.
        inst_segs = [[p] for p, _ in patterns]
        pattern_names = [name for _, name in patterns]
        outer_polys = [_poly_xy(p) for p, _ in patterns]

        for rem in removes:
            cx, cy = _centroid(rem)
            # A remove region can sit inside overlapping material annotations;
            # subtract it from every mask that contains it.
            containing = [j for j, poly in enumerate(outer_polys)
                          if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0]
            if containing:
                parents = containing
                if len(parents) > 1:
                    n_multi_parent += 1
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
                parents = [best]
            for j in parents:
                inst_segs[j].append(rem)
                n_hole_attachments += 1
            n_hole += 1

        out_anns = [{"segmentation": segs, "category_name": name}
                    for segs, name in zip(inst_segs, pattern_names)]

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

    print(f"wrote {n_img} images, {n_inst} instances, {n_hole} remove polygons "
          f"attached {n_hole_attachments} times"
          f"{f', {n_multi_parent} attached to multiple masks' if n_multi_parent else ''}"
          f"{f', {n_hole_dropped} holes dropped' if n_hole_dropped else ''}"
          f"{f', {n_pattern_dropped} degenerate patterns dropped' if n_pattern_dropped else ''}"
          f"{f', {n_img_skipped} unlabeled images skipped' if n_img_skipped else ''}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
