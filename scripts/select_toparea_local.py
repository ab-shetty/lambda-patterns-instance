#!/usr/bin/env python3
"""Build a synthetic dataset by selecting high labelled-area examples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    score: float
    ann_area: float
    image_area: float
    ann_count: int
    source: Path
    ann_path: Path
    img_path: Path
    mode: str
    class_count: int
    max_class_instances: int


def _iter_candidates(source: Path):
    ann_dir = source / "annotations"
    img_dir = source / "images"
    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            with ann_path.open("r") as f:
                ann = json.load(f)
        except Exception as exc:
            print(f"skip unreadable annotation {ann_path}: {exc}")
            continue
        image = ann.get("image") or {}
        file_name = image.get("file_name")
        if not file_name:
            continue
        img_path = img_dir / file_name
        if not img_path.exists():
            print(f"skip missing image for {ann_path}: {img_path}")
            continue
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        image_area = width * height
        if image_area <= 0:
            continue
        anns = ann.get("annotations") or []
        class_counts = Counter(a.get("category_name", "pattern") for a in anns)
        ann_area = sum(float(a.get("area") or 0.0) for a in anns)
        score = ann_area / image_area
        yield Candidate(
            score=score,
            ann_area=ann_area,
            image_area=image_area,
            ann_count=len(anns),
            source=source,
            ann_path=ann_path,
            img_path=img_path,
            mode=str(ann.get("mode") or "unknown"),
            class_count=len(class_counts),
            max_class_instances=max(class_counts.values(), default=0),
        )


def _copy_candidate(cand: Candidate, out: Path, index: int):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{cand.img_path.suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(cand.img_path, out_img)
    with cand.ann_path.open("r") as f:
        ann = json.load(f)
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann["source_dataset"] = str(cand.source)
    ann["source_image"] = old_name
    ann["selection_score"] = cand.score
    ann["selection_ann_area"] = cand.ann_area
    ann["selection_image_area"] = cand.image_area
    with out_ann.open("w") as f:
        json.dump(ann, f, separators=(",", ":"))
    return out_img, out_ann


def _parse_mode_quotas(text: str | None):
    if not text:
        return None
    quotas = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit("--mode-quotas entries must look like mode=count")
        mode, count = part.split("=", 1)
        quotas[mode.strip()] = int(count.strip())
    return quotas


def _select(candidates: list[Candidate], n: int, quotas: dict[str, int] | None):
    if not quotas:
        return candidates[:n]
    if sum(quotas.values()) != n:
        raise SystemExit(f"--mode-quotas must sum to --n ({n})")
    selected = []
    used = set()
    for mode, count in quotas.items():
        mode_rows = [c for c in candidates if c.mode == mode]
        if len(mode_rows) < count:
            raise SystemExit(
                f"only found {len(mode_rows)} candidates for mode={mode}, "
                f"need {count}"
            )
        chosen = mode_rows[:count]
        selected.extend(chosen)
        used.update((c.source, c.ann_path) for c in chosen)
    selected.sort(key=lambda c: (c.score, c.ann_area, -c.ann_count), reverse=True)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="local synthetic dataset directories")
    ap.add_argument("--out", required=True,
                    help="output dataset directory")
    ap.add_argument("--n", type=int, default=1300,
                    help="number of examples to select")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="optional minimum labelled-area fraction")
    ap.add_argument("--max-classes", type=int, default=0,
                    help="Keep images with at most this many pattern classes (0=off)")
    ap.add_argument("--min-max-repeat", type=int, default=0,
                    help="Require at least one pattern class with this many instances")
    ap.add_argument("--mode-quotas", default=None,
                    help="optional comma-separated mode quotas, e.g. "
                         "elevation=776,roof_plan=417,freeform=107")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    sources = [Path(p) for p in args.sources]
    candidates = []
    for source in sources:
        candidates.extend(c for c in _iter_candidates(source)
                          if c.score >= args.min_score and c.ann_count > 0
                          and (not args.max_classes or c.class_count <= args.max_classes)
                          and c.max_class_instances >= args.min_max_repeat)
    candidates.sort(key=lambda c: (c.score, c.ann_area, -c.ann_count), reverse=True)
    selected = _select(candidates, args.n, _parse_mode_quotas(args.mode_quotas))
    if len(selected) < args.n:
        raise SystemExit(f"only found {len(selected)} candidates for n={args.n}")

    rows = []
    for i, cand in enumerate(selected):
        _copy_candidate(cand, out, i)
        rows.append({
            "rank": i,
            "score": cand.score,
            "ann_area": cand.ann_area,
            "image_area": cand.image_area,
            "ann_count": cand.ann_count,
            "mode": cand.mode,
            "class_count": cand.class_count,
            "max_class_instances": cand.max_class_instances,
            "source": str(cand.source),
            "source_ann": str(cand.ann_path),
            "source_image": str(cand.img_path),
        })

    scores = [r["score"] for r in rows]
    modes = {}
    srcs = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1
    manifest = {
        "n": len(rows),
        "sources": [str(s) for s in sources],
        "score_min": min(scores),
        "score_median": sorted(scores)[len(scores) // 2],
        "score_mean": sum(scores) / len(scores),
        "score_max": max(scores),
        "modes": modes,
        "source_counts": srcs,
        "rows": rows,
    }
    with (out / "selection_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in (
        "n", "score_min", "score_median", "score_mean", "score_max",
        "modes", "source_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
