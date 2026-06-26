#!/usr/bin/env python3
"""Select local synthetic data by geometry tied to known real failure modes.

The older diverse selector only sees total area, annotation count, aspect, and
sheet size. This selector keeps those useful coarse signals, but adds object-level
features for:

- long thin material strips, matching perimeter/deck/slab failures;
- broad material fields, matching roof/elevation field failures.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    ann_path: Path
    img_path: Path
    source: Path
    mode: str
    score: float
    ann_count: int
    cat_count: int
    width: float
    height: float
    strip_area: float
    broad_area: float
    max_obj_area: float
    sheet_aspect: float


def _iter_candidates(source: Path):
    for ann_path in sorted((source / "annotations").glob("*.json")):
        try:
            ann = json.loads(ann_path.read_text())
        except Exception as exc:
            print(f"skip unreadable {ann_path}: {exc}")
            continue
        image = ann.get("image") or {}
        file_name = image.get("file_name")
        if not file_name:
            continue
        img_path = source / "images" / file_name
        if not img_path.exists():
            continue
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        anns = ann.get("annotations") or []
        if not anns:
            continue

        image_area = width * height
        total_area = 0.0
        strip_area = 0.0
        broad_area = 0.0
        max_obj_area = 0.0
        cats = set()
        for obj in anns:
            area = float(obj.get("area") or 0.0)
            bbox = obj.get("bbox") or [0, 0, 0, 0]
            bw = float(bbox[2] or 0.0)
            bh = float(bbox[3] or 0.0)
            if area <= 0 or bw <= 0 or bh <= 0:
                continue
            total_area += area
            max_obj_area = max(max_obj_area, area)
            cats.add(obj.get("category_name", "pattern"))
            long_side = max(bw, bh)
            short_side = max(min(bw, bh), 1.0)
            thinness = short_side / long_side
            obj_area_frac = area / image_area
            bbox_area_frac = (bw * bh) / image_area
            if thinness <= 0.20 and obj_area_frac >= 0.0015:
                strip_area += area
            if obj_area_frac >= 0.055 or bbox_area_frac >= 0.18:
                broad_area += area

        if total_area <= 0:
            continue
        yield Candidate(
            ann_path=ann_path,
            img_path=img_path,
            source=source,
            mode=str(ann.get("mode") or "unknown"),
            score=total_area / image_area,
            ann_count=len(anns),
            cat_count=len(cats),
            width=width,
            height=height,
            strip_area=strip_area / image_area,
            broad_area=broad_area / image_area,
            max_obj_area=max_obj_area / image_area,
            sheet_aspect=width / max(height, 1.0),
        )


def _parse_quotas(text: str, n: int):
    quotas = {}
    for part in text.split(","):
        if not part.strip():
            continue
        mode, count = part.split("=", 1)
        quotas[mode.strip()] = int(count)
    if sum(quotas.values()) != n:
        raise SystemExit("--mode-quotas must sum to --n")
    return quotas


def _feature(c: Candidate):
    aspect = min(c.sheet_aspect, 5.0) / 5.0
    return (
        c.score,
        c.strip_area,
        c.broad_area,
        math.log1p(c.ann_count) / math.log(32),
        math.log1p(c.cat_count) / math.log(8),
        aspect,
    )


def _dist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _rank(c: Candidate):
    ann_penalty = abs(math.log1p(c.ann_count) - math.log1p(7.5))
    cat_penalty = abs(math.log1p(c.cat_count) - math.log1p(2.5))
    if c.mode == "roof_plan":
        geom = 1.15 * c.strip_area + 0.55 * c.broad_area
    elif c.mode == "freeform":
        geom = 1.45 * c.strip_area + 0.25 * c.broad_area
    elif c.mode == "elevation":
        geom = 0.45 * c.strip_area + 0.95 * c.broad_area
    else:
        geom = 0.75 * c.strip_area + 0.75 * c.broad_area
    return (
        0.68 * c.score + geom - 0.055 * ann_penalty - 0.025 * cat_penalty,
        c.strip_area + c.broad_area,
        c.score,
    )


def _select(rows: list[Candidate], count: int, diversity: float):
    rows = sorted(rows, key=_rank, reverse=True)
    if len(rows) < count:
        raise SystemExit(f"need {count} rows, found {len(rows)}")
    if count <= 0:
        return []

    feats = [_feature(c) for c in rows]
    chosen = [0]
    selected = {0}
    min_d = [_dist(f, feats[0]) for f in feats]
    while len(chosen) < count:
        best_i = None
        best_key = None
        for i, c in enumerate(rows):
            if i in selected:
                continue
            rank = _rank(c)[0]
            key = (rank + diversity * min_d[i], _rank(c)[1], _rank(c)[2])
            if best_key is None or key > best_key:
                best_i = i
                best_key = key
        selected.add(best_i)
        chosen.append(best_i)
        bf = feats[best_i]
        for i, f in enumerate(feats):
            if i not in selected:
                min_d[i] = min(min_d[i], _dist(f, bf))
    return [rows[i] for i in chosen]


def _copy(c: Candidate, out: Path, index: int):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{c.img_path.suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(c.img_path, out_img)
    ann = json.loads(c.ann_path.read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann["source_dataset"] = str(c.source)
    ann["source_image"] = old_name
    ann["selection_score"] = c.score
    ann["selection_ann_count"] = c.ann_count
    ann["selection_cat_count"] = c.cat_count
    ann["selection_strip_area"] = c.strip_area
    ann["selection_broad_area"] = c.broad_area
    ann["selection_max_obj_area"] = c.max_obj_area
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--mode-quotas", required=True)
    ap.add_argument("--min-score", type=float, default=0.14)
    ap.add_argument("--diversity", type=float, default=0.08)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    quotas = _parse_quotas(args.mode_quotas, args.n)
    candidates = []
    for source_s in args.sources:
        source = Path(source_s)
        for c in _iter_candidates(source):
            if c.score >= args.min_score:
                candidates.append(c)

    selected = []
    used = set()
    for mode, count in quotas.items():
        rows = [c for c in candidates if c.mode == mode and (c.source, c.ann_path) not in used]
        chosen = _select(rows, count, args.diversity)
        selected.extend(chosen)
        used.update((c.source, c.ann_path) for c in chosen)

    selected.sort(key=lambda c: (c.mode, _rank(c), c.score), reverse=True)
    for i, c in enumerate(selected):
        _copy(c, out, i)

    modes = {}
    scores, anns, strips, broads, maxes = [], [], [], [], []
    for c in selected:
        modes[c.mode] = modes.get(c.mode, 0) + 1
        scores.append(c.score)
        anns.append(c.ann_count)
        strips.append(c.strip_area)
        broads.append(c.broad_area)
        maxes.append(c.max_obj_area)
    manifest = {
        "n": len(selected),
        "sources": args.sources,
        "quotas": quotas,
        "min_score": args.min_score,
        "diversity": args.diversity,
        "score_mean": sum(scores) / len(scores),
        "score_median": sorted(scores)[len(scores) // 2],
        "score_min": min(scores),
        "score_max": max(scores),
        "ann_count_mean": sum(anns) / len(anns),
        "ann_count_median": sorted(anns)[len(anns) // 2],
        "strip_area_mean": sum(strips) / len(strips),
        "broad_area_mean": sum(broads) / len(broads),
        "max_obj_area_mean": sum(maxes) / len(maxes),
        "modes": modes,
        "rows": [
            {
                "rank": i,
                "source": str(c.source),
                "source_ann": str(c.ann_path),
                "mode": c.mode,
                "score": c.score,
                "ann_count": c.ann_count,
                "cat_count": c.cat_count,
                "strip_area": c.strip_area,
                "broad_area": c.broad_area,
                "max_obj_area": c.max_obj_area,
                "width": c.width,
                "height": c.height,
            }
            for i, c in enumerate(selected)
        ],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "quotas", "score_min", "score_median", "score_mean", "score_max",
        "ann_count_mean", "ann_count_median", "strip_area_mean",
        "broad_area_mean", "max_obj_area_mean", "modes",
    )}, indent=2))


if __name__ == "__main__":
    main()
