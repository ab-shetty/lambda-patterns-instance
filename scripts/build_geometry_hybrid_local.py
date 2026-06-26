#!/usr/bin/env python3
"""Replace a small slice of a base local dataset with geometry-targeted examples."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    ann_path: Path
    img_path: Path
    mode: str
    key: tuple[str, str]
    geom_score: float
    score: float
    ann_count: int
    strip_area: float
    broad_area: float


def _geometry(ann: dict):
    image = ann.get("image") or {}
    width = float(image.get("width") or 0)
    height = float(image.get("height") or 0)
    image_area = max(width * height, 1.0)
    anns = ann.get("annotations") or []
    total = 0.0
    strip = 0.0
    broad = 0.0
    for obj in anns:
        area = float(obj.get("area") or 0.0)
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        bw = float(bbox[2] or 0.0)
        bh = float(bbox[3] or 0.0)
        if area <= 0 or bw <= 0 or bh <= 0:
            continue
        total += area
        area_frac = area / image_area
        thin = min(bw, bh) / max(bw, bh)
        if thin <= 0.20 and area_frac >= 0.0015:
            strip += area
        if area_frac >= 0.055 or (bw * bh) / image_area >= 0.18:
            broad += area
    score = total / image_area
    strip_area = strip / image_area
    broad_area = broad / image_area
    mode = str(ann.get("mode") or "unknown")
    if mode == "roof_plan":
        geom = 0.55 * score + 1.10 * strip_area + 0.45 * broad_area
    elif mode == "freeform":
        geom = 0.50 * score + 1.35 * strip_area + 0.25 * broad_area
    elif mode == "elevation":
        geom = 0.60 * score + 0.35 * strip_area + 0.85 * broad_area
    else:
        geom = 0.55 * score + 0.75 * strip_area + 0.75 * broad_area
    ann_penalty = abs(math.log1p(len(anns)) - math.log1p(7.5))
    return geom - 0.04 * ann_penalty, score, len(anns), strip_area, broad_area


def _key(root: Path, ann_path: Path, ann: dict):
    source = str(ann.get("source_dataset") or root)
    image = str(ann.get("source_image") or ann.get("image", {}).get("file_name") or ann_path.name)
    return source, image


def _rows(root: Path):
    rows = []
    for ann_path in sorted((root / "annotations").glob("*.json")):
        ann = json.loads(ann_path.read_text())
        file_name = ann.get("image", {}).get("file_name")
        if not file_name:
            continue
        img_path = root / "images" / file_name
        if not img_path.exists():
            continue
        geom, score, ann_count, strip_area, broad_area = _geometry(ann)
        rows.append(Row(
            ann_path=ann_path,
            img_path=img_path,
            mode=str(ann.get("mode") or "unknown"),
            key=_key(root, ann_path, ann),
            geom_score=geom,
            score=score,
            ann_count=ann_count,
            strip_area=strip_area,
            broad_area=broad_area,
        ))
    return rows


def _parse_counts(text: str):
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        mode, count = part.split("=", 1)
        out[mode.strip()] = int(count)
    return out


def _copy(row: Row, out: Path, index: int, origin: str):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{row.img_path.suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(row.img_path, out_img)
    ann = json.loads(row.ann_path.read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann.setdefault("source_dataset", str(row.ann_path.parents[1]))
    ann.setdefault("source_image", old_name)
    ann["hybrid_origin"] = origin
    ann["hybrid_geom_score"] = row.geom_score
    ann["hybrid_strip_area"] = row.strip_area
    ann["hybrid_broad_area"] = row.broad_area
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--targeted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace-counts", required=True,
                    help="mode=count replacements, e.g. elevation=70,roof_plan=60,freeform=20")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    targeted = Path(args.targeted)
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    replace_counts = _parse_counts(args.replace_counts)
    base_rows = _rows(base)
    targeted_rows = _rows(targeted)
    base_keys = {r.key for r in base_rows}
    keep = []
    removed = []
    added = []
    used_keys = set(base_keys)

    for mode, count in replace_counts.items():
        mode_base = sorted([r for r in base_rows if r.mode == mode], key=lambda r: r.geom_score)
        remove = mode_base[:count]
        removed.extend(remove)
        remove_keys = {r.key for r in remove}
        used_keys -= remove_keys
        candidates = sorted(
            [r for r in targeted_rows if r.mode == mode and r.key not in used_keys],
            key=lambda r: r.geom_score,
            reverse=True,
        )
        chosen = candidates[:count]
        if len(chosen) < count:
            raise SystemExit(f"only found {len(chosen)} replacements for {mode}")
        added.extend(chosen)
        used_keys.update(r.key for r in chosen)

    removed_keys = {r.key for r in removed}
    keep = [r for r in base_rows if r.key not in removed_keys]
    selected = keep + added
    selected.sort(key=lambda r: (r.mode, r.geom_score, r.score), reverse=True)
    for i, row in enumerate(selected):
        origin = "added" if row in added else "base"
        _copy(row, out, i, origin)

    modes = {}
    for row in selected:
        modes[row.mode] = modes.get(row.mode, 0) + 1
    manifest = {
        "n": len(selected),
        "base": str(base),
        "targeted": str(targeted),
        "replace_counts": replace_counts,
        "modes": modes,
        "removed_geom_mean": sum(r.geom_score for r in removed) / len(removed),
        "added_geom_mean": sum(r.geom_score for r in added) / len(added),
        "selected_geom_mean": sum(r.geom_score for r in selected) / len(selected),
        "selected_strip_mean": sum(r.strip_area for r in selected) / len(selected),
        "selected_broad_mean": sum(r.broad_area for r in selected) / len(selected),
        "selected_ann_count_mean": sum(r.ann_count for r in selected) / len(selected),
        "removed": [str(r.ann_path) for r in removed],
        "added": [str(r.ann_path) for r in added],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "replace_counts", "modes", "removed_geom_mean", "added_geom_mean",
        "selected_geom_mean", "selected_strip_mean", "selected_broad_mean",
        "selected_ann_count_mean",
    )}, indent=2))


if __name__ == "__main__":
    main()
