#!/usr/bin/env python3
"""Copy a local dataset and append duplicated examples to a target size."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _records(root: Path):
    rows = []
    for ann_path in sorted((root / "annotations").glob("*.json")):
        ann = json.loads(ann_path.read_text())
        file_name = ann.get("image", {}).get("file_name")
        if not file_name:
            continue
        img_path = root / "images" / file_name
        if not img_path.exists():
            continue
        image = ann.get("image") or {}
        area = sum(float(a.get("area") or 0.0) for a in ann.get("annotations", []))
        denom = max(float(image.get("width") or 0) * float(image.get("height") or 0), 1.0)
        rows.append({
            "ann_path": ann_path,
            "img_path": img_path,
            "mode": str(ann.get("mode") or "unknown"),
            "score": area / denom,
            "ann_count": len(ann.get("annotations", [])),
        })
    return rows


def _copy(row, out: Path, index: int, origin: str, duplicate_of: str | None = None):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{row['img_path'].suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(row["img_path"], out_img)
    ann = json.loads(row["ann_path"].read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann.setdefault("source_dataset", str(row["ann_path"].parents[1]))
    ann.setdefault("source_image", old_name)
    ann["upsample_origin"] = origin
    if duplicate_of is not None:
        ann["duplicate_of"] = duplicate_of
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def _select_duplicates(rows, count: int, strategy: str, top_score_count: int | None = None):
    if top_score_count is not None:
        top_n = max(0, min(count, int(top_score_count)))
        top = _select_duplicates(rows, top_n, "top_score")
        used = {r["ann_path"] for r in top}
        remaining_rows = [r for r in rows if r["ann_path"] not in used] or rows
        rest = _select_duplicates(remaining_rows, count - top_n, strategy)
        return (top + rest)[:count]
    if strategy == "top_score":
        ordered = sorted(rows, key=lambda r: (r["score"], r["ann_count"]), reverse=True)
    elif strategy == "balanced":
        by_mode = {}
        for row in rows:
            by_mode.setdefault(row["mode"], []).append(row)
        for mode_rows in by_mode.values():
            mode_rows.sort(key=lambda r: (r["score"], r["ann_count"]), reverse=True)
        ordered = []
        cursors = {m: 0 for m in by_mode}
        modes = sorted(by_mode, key=lambda m: len(by_mode[m]), reverse=True)
        while len(ordered) < count:
            progressed = False
            for mode in modes:
                cur = cursors[mode]
                if cur < len(by_mode[mode]):
                    ordered.append(by_mode[mode][cur])
                    cursors[mode] += 1
                    progressed = True
                    if len(ordered) == count:
                        break
            if not progressed:
                break
    else:
        raise SystemExit(f"unknown strategy {strategy!r}")
    if len(ordered) < count:
        repeats = []
        i = 0
        while len(ordered) + len(repeats) < count:
            repeats.append(ordered[i % len(ordered)])
            i += 1
        ordered.extend(repeats)
    return ordered[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--strategy", choices=("top_score", "balanced"), default="balanced")
    ap.add_argument("--top-score-count", type=int, default=None,
                    help="duplicate this many top-score rows first, then fill "
                         "the remaining duplicate budget with --strategy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    rows = _records(src)
    if len(rows) > args.n:
        raise SystemExit(f"source has {len(rows)} rows, target n={args.n}")
    dup_count = args.n - len(rows)
    duplicates = _select_duplicates(rows, dup_count, args.strategy, args.top_score_count)
    selected = rows + duplicates

    for i, row in enumerate(rows):
        _copy(row, out, i, "base")
    for j, row in enumerate(duplicates, start=len(rows)):
        _copy(row, out, j, "duplicate", duplicate_of=row["ann_path"].name)

    modes = {}
    origins = {"base": len(rows), "duplicate": len(duplicates)}
    for row in selected:
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    manifest = {
        "n": len(selected),
        "src": str(src),
        "strategy": args.strategy,
        "top_score_count": args.top_score_count,
        "origins": origins,
        "modes": modes,
        "score_mean": sum(r["score"] for r in selected) / len(selected),
        "ann_count_mean": sum(r["ann_count"] for r in selected) / len(selected),
        "duplicates": [str(r["ann_path"]) for r in duplicates],
    }
    (out / "upsample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "strategy", "origins", "modes", "score_mean", "ann_count_mean",
    )}, indent=2))


if __name__ == "__main__":
    main()
