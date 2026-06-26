#!/usr/bin/env python3
"""Select broad high-area synthetic examples with controlled annotation counts."""

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
        area = sum(float(a.get("area") or 0.0) for a in anns)
        cats = len({a.get("category_name", "pattern") for a in anns})
        yield Candidate(
            ann_path=ann_path,
            img_path=img_path,
            source=source,
            mode=str(ann.get("mode") or "unknown"),
            score=area / (width * height),
            ann_count=len(anns),
            cat_count=cats,
            width=width,
            height=height,
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


def _parse_targets(text: str | None):
    targets = {}
    if not text:
        return targets
    for part in text.split(","):
        if not part.strip():
            continue
        mode, value = part.split("=", 1)
        targets[mode.strip()] = float(value)
    return targets


def _rank_key(c: Candidate, target_ann: float, target_cat: float):
    ann_penalty = abs(math.log1p(c.ann_count) - math.log1p(target_ann))
    cat_penalty = abs(math.log1p(c.cat_count) - math.log1p(target_cat))
    aspect = c.width / max(c.height, 1.0)
    # High area stays primary, but avoid extremely fragmented examples. Mildly
    # prefer wide sheets because many hard held-out reals are wide excerpts.
    return (
        c.score - 0.09 * ann_penalty - 0.03 * cat_penalty,
        c.score,
        -abs(aspect - 2.2) * 0.01,
        -c.ann_count,
    )


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
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--mode-quotas", required=True)
    ap.add_argument("--target-ann", default=None,
                    help="mode=count targets, e.g. elevation=5,roof_plan=3,freeform=4")
    ap.add_argument("--target-cat", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-ann", type=int, default=12)
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
    target_ann_by_mode = _parse_targets(args.target_ann)
    candidates = []
    for source_s in args.sources:
        source = Path(source_s)
        for c in _iter_candidates(source):
            if c.score >= args.min_score and c.ann_count <= args.max_ann:
                candidates.append(c)

    selected = []
    used = set()
    for mode, count in quotas.items():
        rows = [c for c in candidates if c.mode == mode]
        target_ann = target_ann_by_mode.get(mode, 4.5)
        rows.sort(key=lambda c: _rank_key(c, target_ann, args.target_cat), reverse=True)
        chosen = []
        for c in rows:
            key = (c.source, c.ann_path)
            if key in used:
                continue
            chosen.append(c)
            used.add(key)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise SystemExit(f"only found {len(chosen)} candidates for mode={mode}")
        selected.extend(chosen)

    selected.sort(key=lambda c: (c.mode, c.score, -c.ann_count), reverse=True)
    for i, c in enumerate(selected):
        _copy(c, out, i)

    modes = {}
    scores = []
    anns = []
    cats = []
    for c in selected:
        modes[c.mode] = modes.get(c.mode, 0) + 1
        scores.append(c.score)
        anns.append(c.ann_count)
        cats.append(c.cat_count)
    manifest = {
        "n": len(selected),
        "sources": args.sources,
        "quotas": quotas,
        "target_ann": target_ann_by_mode,
        "target_cat": args.target_cat,
        "min_score": args.min_score,
        "max_ann": args.max_ann,
        "score_mean": sum(scores) / len(scores),
        "score_median": sorted(scores)[len(scores) // 2],
        "score_min": min(scores),
        "score_max": max(scores),
        "ann_count_mean": sum(anns) / len(anns),
        "ann_count_median": sorted(anns)[len(anns) // 2],
        "cat_count_mean": sum(cats) / len(cats),
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
                "width": c.width,
                "height": c.height,
            }
            for i, c in enumerate(selected)
        ],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "quotas", "score_min", "score_median", "score_mean", "score_max",
        "ann_count_mean", "ann_count_median", "cat_count_mean", "modes",
    )}, indent=2))


if __name__ == "__main__":
    main()
