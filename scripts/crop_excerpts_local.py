#!/usr/bin/env python3
"""Create real-like cropped excerpt views from a local synthetic dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_synthetic_v5 import _clip_annotations_to_crop


def _iter_ann_paths(src: Path):
    manifest = src / "selection_manifest.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text()).get("rows") or []
        out = []
        for row in rows:
            ann = src / "annotations" / (Path(row.get("source_ann", "")).stem + ".json")
            local = src / "annotations" / f"synth_{int(row['rank']):06d}.json" if "rank" in row else ann
            out.append(local if local.exists() else ann)
        return [p for p in out if p.exists()]
    return sorted((src / "annotations").glob("*.json"))


def _choose_crop(ann: dict, width: int, height: int, rng: random.Random,
                 min_keep: float, max_keep: float):
    keep_w = rng.uniform(min_keep, max_keep)
    keep_h = rng.uniform(min_keep, max_keep)
    cw = max(640, min(width, int(round(width * keep_w))))
    ch = max(640, min(height, int(round(height * keep_h))))
    if cw >= width and ch >= height:
        return (0, 0, width, height)

    anns = ann.get("annotations") or []
    target = max(anns, key=lambda a: float(a.get("area") or 0.0))
    bx, by, bw, bh = target.get("bbox", [width // 4, height // 4, width // 2, height // 2])
    cx = bx + bw * rng.uniform(0.25, 0.75)
    cy = by + bh * rng.uniform(0.25, 0.75)
    if rng.random() < 0.70:
        cx += rng.choice([-1, 1]) * cw * rng.uniform(0.08, 0.26)
    if rng.random() < 0.55:
        cy += rng.choice([-1, 1]) * ch * rng.uniform(0.06, 0.22)
    x0 = int(round(max(0, min(width - cw, cx - cw / 2))))
    y0 = int(round(max(0, min(height - ch, cy - ch / 2))))
    return (x0, y0, x0 + cw, y0 + ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keep", type=float, default=0.58)
    ap.add_argument("--max-keep", type=float, default=0.86)
    ap.add_argument("--attempts", type=int, default=8)
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

    rng = random.Random(args.seed)
    ann_paths = _iter_ann_paths(src)
    if len(ann_paths) < args.n:
        raise SystemExit(f"only found {len(ann_paths)} annotations, need {args.n}")

    rows = []
    total_anns = 0
    total_area_frac = 0.0
    for i, ann_path in enumerate(ann_paths[:args.n]):
        ann = json.loads(ann_path.read_text())
        img_name = ann.get("image", {}).get("file_name")
        img_path = src / "images" / img_name
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        best = None
        for _ in range(args.attempts):
            crop = _choose_crop(ann, width, height, rng, args.min_keep, args.max_keep)
            cw, ch = crop[2] - crop[0], crop[3] - crop[1]
            clipped = _clip_annotations_to_crop(ann.get("annotations") or [], crop, cw, ch)
            area = sum(float(a.get("area") or 0.0) for a in clipped)
            if clipped and (best is None or area / (cw * ch) > best[0]):
                best = (area / (cw * ch), crop, clipped)
        if best is None:
            crop = (0, 0, width, height)
            clipped = ann.get("annotations") or []
            best = (
                sum(float(a.get("area") or 0.0) for a in clipped) / max(width * height, 1),
                crop,
                clipped,
            )

        area_frac, crop, clipped = best
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        stem = f"synth_{i:06d}"
        out_img_name = f"{stem}.png"
        img.crop(crop).save(out / "images" / out_img_name)
        out_ann = dict(ann)
        out_ann["image"] = {"file_name": out_img_name, "width": cw, "height": ch}
        out_ann["annotations"] = clipped
        out_ann["source_dataset"] = str(src)
        out_ann["source_annotation"] = str(ann_path)
        out_ann["crop_box"] = list(crop)
        out_ann["crop_area_frac"] = area_frac
        (out / "annotations" / f"{stem}.json").write_text(json.dumps(out_ann, separators=(",", ":")))
        total_anns += len(clipped)
        total_area_frac += area_frac
        rows.append({
            "rank": i,
            "source_ann": str(ann_path),
            "mode": out_ann.get("mode", "unknown"),
            "crop_area_frac": area_frac,
            "ann_count": len(clipped),
        })

    modes = {}
    for row in rows:
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    manifest = {
        "n": len(rows),
        "source": str(src),
        "mean_crop_area_frac": total_area_frac / max(len(rows), 1),
        "annotations": total_anns,
        "modes": modes,
        "rows": rows,
    }
    (out / "crop_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n", "mean_crop_area_frac", "annotations", "modes")}, indent=2))


if __name__ == "__main__":
    main()
