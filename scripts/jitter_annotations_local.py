#!/usr/bin/env python3
"""Copy a local dataset while adding bounded polygon jitter to annotations."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path


def _chunks(poly):
    return list(zip(poly[0::2], poly[1::2]))


def _flat(points):
    out = []
    for x, y in points:
        out.extend([x, y])
    return out


def _area(poly):
    pts = _chunks(poly)
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _jitter_poly(poly, width, height, rng, sigma):
    pts = []
    for x, y in _chunks(poly):
        nx = min(max(x + rng.gauss(0.0, sigma), 0.0), width - 1.0)
        ny = min(max(y + rng.gauss(0.0, sigma), 0.0), height - 1.0)
        pts.append((nx, ny))
    return _flat(pts)


def _bbox(segmentation):
    xs, ys = [], []
    for poly in segmentation:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs:
        return [0, 0, 0, 0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [int(math.floor(x0)), int(math.floor(y0)),
            int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))]


def _jitter_annotation_file(src_ann, dst_ann, rng, sigma):
    with src_ann.open("r") as f:
        d = json.load(f)
    width = float(d["image"]["width"])
    height = float(d["image"]["height"])
    for ann in d.get("annotations", []):
        seg = ann.get("segmentation") or []
        new_seg = [_jitter_poly(poly, width, height, rng, sigma)
                   for poly in seg]
        ann["segmentation"] = new_seg
        holes = int(ann.get("num_holes") or max(0, len(new_seg) - 1))
        outer = _area(new_seg[0]) if new_seg else 0.0
        hole_area = sum(_area(p) for p in new_seg[1:1 + holes])
        ann["area"] = max(0.0, outer - hole_area)
        ann["bbox"] = _bbox(new_seg)
    with dst_ann.open("w") as f:
        json.dump(d, f, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    for img in (src / "images").iterdir():
        if img.is_file():
            shutil.copy2(img, out / "images" / img.name)
    rng = random.Random(args.seed)
    for ann in sorted((src / "annotations").glob("*.json")):
        _jitter_annotation_file(ann, out / "annotations" / ann.name, rng,
                                args.sigma)
    manifest = src / "selection_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, out / "selection_manifest.source.json")
    with (out / "jitter_manifest.json").open("w") as f:
        json.dump({
            "source": str(src),
            "sigma": args.sigma,
            "seed": args.seed,
        }, f, indent=2)
    print(f"wrote jittered dataset to {out}")


if __name__ == "__main__":
    main()
