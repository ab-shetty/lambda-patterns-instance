#!/usr/bin/env python3
"""Select a diverse local synthetic subset with optional mode quotas."""

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
    mode: str
    score: float
    ann_count: int
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
        yield Candidate(
            ann_path=ann_path,
            img_path=img_path,
            mode=str(ann.get("mode") or "unknown"),
            score=area / (width * height),
            ann_count=len(anns),
            width=width,
            height=height,
        )


def _parse_quotas(text: str | None, candidates: list[Candidate], n: int):
    if text:
        quotas = {}
        for part in text.split(","):
            if not part.strip():
                continue
            mode, count = part.split("=", 1)
            quotas[mode.strip()] = int(count)
        if sum(quotas.values()) != n:
            raise SystemExit("--mode-quotas must sum to --n")
        return quotas
    counts = {}
    for c in candidates:
        counts[c.mode] = counts.get(c.mode, 0) + 1
    raw = {m: n * k / len(candidates) for m, k in counts.items()}
    quotas = {m: int(math.floor(v)) for m, v in raw.items()}
    for m, _ in sorted(raw.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True):
        if sum(quotas.values()) >= n:
            break
        quotas[m] += 1
    return quotas


def _features(c: Candidate):
    aspect = c.width / max(c.height, 1.0)
    # Keep score dominant enough that selection covers the broad-mask continuum,
    # while ann_count/aspect/size break ties toward genuinely different sheets.
    return (
        c.score,
        math.log1p(c.ann_count) / math.log(32),
        min(aspect, 5.0) / 5.0,
        math.log(max(c.width, c.height)) / math.log(6000),
    )


def _dist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _select_diverse(rows: list[Candidate], count: int):
    rows = sorted(rows, key=lambda c: (c.score, c.ann_count), reverse=True)
    if len(rows) < count:
        raise SystemExit(f"need {count} rows, found {len(rows)}")
    if count <= 0:
        return []

    feats = [_features(c) for c in rows]
    chosen_idx = [0, len(rows) // 2, len(rows) - 1]
    chosen_idx = list(dict.fromkeys(i for i in chosen_idx if 0 <= i < len(rows)))[:count]
    min_d = [min(_dist(f, feats[i]) for i in chosen_idx) for f in feats]
    selected = set(chosen_idx)

    while len(chosen_idx) < count:
        best_i = None
        best_key = None
        for i, c in enumerate(rows):
            if i in selected:
                continue
            # Farthest-point sampling in feature space, with a mild preference
            # for higher area when diversity is otherwise similar.
            key = (min_d[i], c.score * 0.05, c.ann_count * 0.002)
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        selected.add(best_i)
        chosen_idx.append(best_i)
        bf = feats[best_i]
        for i, f in enumerate(feats):
            if i not in selected:
                min_d[i] = min(min_d[i], _dist(f, bf))
    return [rows[i] for i in chosen_idx]


def _copy(c: Candidate, out: Path, index: int, source: Path):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{c.img_path.suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(c.img_path, out_img)
    ann = json.loads(c.ann_path.read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann["source_dataset"] = str(source)
    ann["source_image"] = old_name
    ann["selection_score"] = c.score
    ann["selection_ann_count"] = c.ann_count
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", dest="sources")
    ap.add_argument("--sources", nargs="+", dest="sources_many")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--mode-quotas", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source_args = []
    if args.sources:
        source_args.extend(args.sources)
    if args.sources_many:
        source_args.extend(args.sources_many)
    if not source_args:
        raise SystemExit("pass --source DIR or --sources DIR ...")
    sources = [Path(p) for p in source_args]
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    candidates = []
    source_by_ann = {}
    for source in sources:
        for c in _iter_candidates(source):
            if c.score < args.min_score:
                continue
            candidates.append(c)
            source_by_ann[c.ann_path] = source
    quotas = _parse_quotas(args.mode_quotas, candidates, args.n)
    selected = []
    for mode, count in quotas.items():
        selected.extend(_select_diverse([c for c in candidates if c.mode == mode], count))
    selected.sort(key=lambda c: (c.mode, c.score, c.ann_count), reverse=True)
    for i, c in enumerate(selected):
        _copy(c, out, i, source_by_ann[c.ann_path])

    scores = [c.score for c in selected]
    anns = [c.ann_count for c in selected]
    modes = {}
    for c in selected:
        modes[c.mode] = modes.get(c.mode, 0) + 1
    manifest = {
        "n": len(selected),
        "sources": [str(s) for s in sources],
        "quotas": quotas,
        "score_mean": sum(scores) / len(scores),
        "score_median": sorted(scores)[len(scores) // 2],
        "score_min": min(scores),
        "score_max": max(scores),
        "ann_count_mean": sum(anns) / len(anns),
        "ann_count_median": sorted(anns)[len(anns) // 2],
        "modes": modes,
        "rows": [
            {
                "rank": i,
                "source_ann": str(c.ann_path),
                "mode": c.mode,
                "score": c.score,
                "ann_count": c.ann_count,
                "width": c.width,
                "height": c.height,
            }
            for i, c in enumerate(selected)
        ],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "quotas", "score_min", "score_median", "score_mean",
        "score_max", "ann_count_mean", "ann_count_median", "modes",
    )}, indent=2))


if __name__ == "__main__":
    main()
