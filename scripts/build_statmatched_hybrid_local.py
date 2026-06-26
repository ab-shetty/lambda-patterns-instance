#!/usr/bin/env python3
"""Replace base examples with source examples while preserving coarse stats."""

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
    source_key: tuple[str, str]
    score: float
    ann_count: int
    aspect: float
    max_side: float


def _row(root: Path, ann_path: Path):
    ann = json.loads(ann_path.read_text())
    image = ann.get("image") or {}
    file_name = image.get("file_name")
    if not file_name:
        return None
    img_path = root / "images" / file_name
    if not img_path.exists():
        return None
    width = float(image.get("width") or 0)
    height = float(image.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    anns = ann.get("annotations") or []
    if not anns:
        return None
    score = sum(float(a.get("area") or 0.0) for a in anns) / (width * height)
    src = str(ann.get("source_dataset") or root)
    src_img = str(ann.get("source_image") or file_name)
    return Row(
        ann_path=ann_path,
        img_path=img_path,
        mode=str(ann.get("mode") or "unknown"),
        source_key=(src, src_img),
        score=score,
        ann_count=len(anns),
        aspect=width / max(height, 1.0),
        max_side=max(width, height),
    )


def _rows(root: Path):
    out = []
    for ann_path in sorted((root / "annotations").glob("*.json")):
        row = _row(root, ann_path)
        if row is not None:
            out.append(row)
    return out


def _parse_counts(text: str):
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        mode, value = part.split("=", 1)
        out[mode.strip()] = int(value)
    return out


def _feature(row: Row):
    return (
        row.score,
        math.log1p(row.ann_count) / math.log(32),
        min(row.aspect, 5.0) / 5.0,
        math.log(max(row.max_side, 1.0)) / math.log(6000),
    )


def _dist(a: Row, b: Row):
    af = _feature(a)
    bf = _feature(b)
    return sum((x - y) * (x - y) for x, y in zip(af, bf))


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
    ann["hybrid_score"] = row.score
    ann["hybrid_ann_count"] = row.ann_count
    ann["hybrid_aspect"] = row.aspect
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace-counts", required=True)
    ap.add_argument("--min-score", type=float, default=0.16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    target = Path(args.target)
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    replace_counts = _parse_counts(args.replace_counts)
    base_rows = _rows(base)
    target_rows = [r for r in _rows(target) if r.score >= args.min_score]
    base_keys = {r.source_key for r in base_rows}

    removed = []
    added = []
    used_base = set()
    used_target = set(base_keys)
    for mode, count in replace_counts.items():
        mode_base = [r for r in base_rows if r.mode == mode and r.source_key not in used_base]
        mode_target = [r for r in target_rows if r.mode == mode and r.source_key not in used_target]
        if len(mode_target) < count:
            raise SystemExit(f"only found {len(mode_target)} target rows for mode={mode}")

        mode_base_sorted = sorted(mode_base, key=lambda r: (r.score, r.ann_count, r.aspect))
        anchors = []
        if count == 1:
            anchors = [mode_base_sorted[len(mode_base_sorted) // 2]]
        else:
            for i in range(count):
                idx = round(i * (len(mode_base_sorted) - 1) / (count - 1))
                anchors.append(mode_base_sorted[idx])

        for anchor in anchors:
            candidates = [r for r in mode_target if r.source_key not in used_target]
            add = min(candidates, key=lambda r: _dist(anchor, r))
            removals = [r for r in mode_base if r.source_key not in used_base]
            remove = min(removals, key=lambda r: _dist(add, r))
            added.append(add)
            removed.append(remove)
            used_target.add(add.source_key)
            used_base.add(remove.source_key)

    removed_keys = {r.source_key for r in removed}
    selected = [r for r in base_rows if r.source_key not in removed_keys] + added
    selected.sort(key=lambda r: (r.mode, r.score, r.ann_count), reverse=True)
    for i, row in enumerate(selected):
        _copy(row, out, i, "added" if row in added else "base")

    modes = {}
    origins = {}
    for row in selected:
        modes[row.mode] = modes.get(row.mode, 0) + 1
    for ann_path in (out / "annotations").glob("*.json"):
        ann = json.loads(ann_path.read_text())
        origin = ann.get("hybrid_origin", "base")
        origins[origin] = origins.get(origin, 0) + 1
    manifest = {
        "n": len(selected),
        "base": str(base),
        "target": str(target),
        "replace_counts": replace_counts,
        "min_score": args.min_score,
        "modes": modes,
        "origins": origins,
        "selected_score_mean": sum(r.score for r in selected) / len(selected),
        "selected_ann_count_mean": sum(r.ann_count for r in selected) / len(selected),
        "added_score_mean": sum(r.score for r in added) / len(added),
        "removed_score_mean": sum(r.score for r in removed) / len(removed),
        "added_ann_count_mean": sum(r.ann_count for r in added) / len(added),
        "removed_ann_count_mean": sum(r.ann_count for r in removed) / len(removed),
        "added": [str(r.ann_path) for r in added],
        "removed": [str(r.ann_path) for r in removed],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "replace_counts", "min_score", "modes", "origins",
        "selected_score_mean", "selected_ann_count_mean",
        "added_score_mean", "removed_score_mean",
        "added_ann_count_mean", "removed_ann_count_mean",
    )}, indent=2))


if __name__ == "__main__":
    main()
