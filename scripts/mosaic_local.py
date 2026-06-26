#!/usr/bin/env python3
"""Pack local synthetic samples into mosaic images with transformed annotations."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def _load_source_rows(src: Path) -> list[Path]:
    manifest = src / "selection_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        rows = data.get("rows") or []
        paths = []
        for row in rows:
            ann = src / "annotations" / (Path(row.get("source_ann", "")).stem + ".json")
            if ann.exists():
                paths.append(ann)
            else:
                # Selection outputs rename files to synth_000000.json etc.
                rank_ann = src / "annotations" / f"synth_{int(row['rank']):06d}.json"
                if rank_ann.exists():
                    paths.append(rank_ann)
        if paths:
            return paths
    return sorted((src / "annotations").glob("*.json"))


def _transform_segmentation(segmentation, scale: float, dx: float, dy: float):
    out = []
    for poly in segmentation or []:
        vals = []
        for i, v in enumerate(poly):
            if i % 2 == 0:
                vals.append(float(v) * scale + dx)
            else:
                vals.append(float(v) * scale + dy)
        out.append(vals)
    return out


def _bbox_from_segmentation(segmentation):
    xs, ys = [], []
    for poly in segmentation or []:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs or not ys:
        return [0, 0, 0, 0], 0.0
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0], max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _place_sample(canvas: Image.Image, src_root: Path, ann_path: Path, cell, next_id: int):
    ann = json.loads(ann_path.read_text())
    image = ann["image"]
    img = Image.open(src_root / "images" / image["file_name"]).convert("RGB")
    x, y, w, h = cell
    scale = min(w / img.width, h / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    dx = x + (w - nw) // 2
    dy = y + (h - nh) // 2
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas.paste(resized, (dx, dy))

    anns = []
    for obj in ann.get("annotations", []):
        seg = _transform_segmentation(obj.get("segmentation"), scale, dx, dy)
        bbox, area = _bbox_from_segmentation(seg)
        if area <= 4:
            continue
        new_obj = dict(obj)
        new_obj["id"] = next_id
        new_obj["segmentation"] = seg
        new_obj["bbox"] = bbox
        new_obj["area"] = area
        new_obj["mosaic_source"] = str(ann_path)
        anns.append(new_obj)
        next_id += 1
    return anns, next_id, ann.get("mode", "unknown")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--grid", choices=("1x2", "2x1", "2x2"), default="2x2")
    ap.add_argument("--cell", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-output", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; use --overwrite")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "annotations").mkdir(parents=True)

    ann_paths = _load_source_rows(src)
    if not ann_paths:
        raise SystemExit(f"no annotations found under {src}")
    rng = random.Random(args.seed)
    ordered = list(ann_paths)
    rng.shuffle(ordered)

    cols, rows = {"1x2": (2, 1), "2x1": (1, 2), "2x2": (2, 2)}[args.grid]
    per = cols * rows
    w, h = cols * args.cell, rows * args.cell
    cells = [
        (c * args.cell, r * args.cell, args.cell, args.cell)
        for r in range(rows)
        for c in range(cols)
    ]

    mode_counts = {}
    total_anns = 0
    for i in range(args.n):
        canvas = Image.new("RGB", (w, h), "white")
        out_anns = []
        next_id = 1
        sources = []
        for j in range(per):
            ann_path = ordered[(i * per + j) % len(ordered)]
            anns, next_id, mode = _place_sample(canvas, src, ann_path, cells[j], next_id)
            out_anns.extend(anns)
            sources.append(str(ann_path))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        stem = f"synth_{i:06d}"
        img_name = f"{stem}.png"
        if args.jpeg_output:
            canvas.save(out / "images" / img_name, format="JPEG", quality=88, optimize=False)
        else:
            canvas.save(out / "images" / img_name)
        record = {
            "image": {"file_name": img_name, "width": w, "height": h},
            "mode": "mosaic",
            "mosaic_grid": args.grid,
            "mosaic_sources": sources,
            "annotations": out_anns,
        }
        total_anns += len(out_anns)
        (out / "annotations" / f"{stem}.json").write_text(json.dumps(record, separators=(",", ":")))

    manifest = {
        "source": str(src),
        "n": args.n,
        "grid": args.grid,
        "cell": args.cell,
        "images": args.n,
        "annotations": total_anns,
        "source_slots": args.n * per,
        "source_mode_counts": mode_counts,
        "jpeg_output": args.jpeg_output,
        "seed": args.seed,
    }
    (out / "mosaic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
