#!/usr/bin/env python3
"""Extend a local synthetic dataset with unique examples from source pools."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


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
    anns = ann.get("annotations") or []
    if width <= 0 or height <= 0 or not anns:
        return None
    score = sum(float(a.get("area") or 0.0) for a in anns) / (width * height)
    return {
        "root": root,
        "ann_path": ann_path,
        "img_path": img_path,
        "mode": str(ann.get("mode") or "unknown"),
        "source_key": (
            str(ann.get("source_dataset") or root),
            str(ann.get("source_image") or file_name),
        ),
        "score": score,
        "ann_count": len(anns),
    }


def _rows(root: Path):
    rows = []
    for ann_path in sorted((root / "annotations").glob("*.json")):
        row = _row(root, ann_path)
        if row is not None:
            rows.append(row)
    return rows


def _copy(row, out: Path, index: int, origin: str):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{row['img_path'].suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(row["img_path"], out_img)
    ann = json.loads(row["ann_path"].read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann.setdefault("source_dataset", str(row["root"]))
    ann.setdefault("source_image", old_name)
    ann["extend_origin"] = origin
    ann["extend_score"] = row["score"]
    ann["extend_ann_count"] = row["ann_count"]
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def _parse_counts(text: str | None):
    if not text:
        return None
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        mode, count = part.split("=", 1)
        out[mode.strip()] = int(count)
    return out


def _select(rows, count: int, quotas: dict[str, int] | None):
    selected = []
    used = set()
    if quotas is None:
        quotas = {}
        modes = sorted({r["mode"] for r in rows})
        for mode in modes:
            quotas[mode] = count // len(modes)
        for mode in modes[:count - sum(quotas.values())]:
            quotas[mode] += 1
    if sum(quotas.values()) != count:
        raise SystemExit("--add-mode-quotas must sum to the add count")
    for mode, mode_count in quotas.items():
        mode_rows = [r for r in rows if r["mode"] == mode]
        mode_rows.sort(key=lambda r: (r["score"], r["ann_count"]), reverse=True)
        chosen = []
        for row in mode_rows:
            if row["source_key"] in used:
                continue
            chosen.append(row)
            used.add(row["source_key"])
            if len(chosen) == mode_count:
                break
        if len(chosen) < mode_count:
            raise SystemExit(f"only found {len(chosen)} rows for mode={mode}")
        selected.extend(chosen)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--add-mode-quotas", default=None)
    ap.add_argument("--min-score", type=float, default=0.16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    base_rows = _rows(base)
    if len(base_rows) > args.n:
        raise SystemExit("base has more rows than target n")
    add_count = args.n - len(base_rows)
    base_keys = {r["source_key"] for r in base_rows}

    candidates = []
    for source_s in args.sources:
        source = Path(source_s)
        for row in _rows(source):
            if row["score"] >= args.min_score and row["source_key"] not in base_keys:
                candidates.append(row)
    added = _select(candidates, add_count, _parse_counts(args.add_mode_quotas))

    selected = base_rows + added
    for i, row in enumerate(base_rows):
        _copy(row, out, i, "base")
    for i, row in enumerate(added, start=len(base_rows)):
        _copy(row, out, i, "added")

    modes = {}
    origins = {"base": len(base_rows), "added": len(added)}
    for row in selected:
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    manifest = {
        "n": len(selected),
        "base": str(base),
        "sources": args.sources,
        "add_mode_quotas": _parse_counts(args.add_mode_quotas),
        "min_score": args.min_score,
        "origins": origins,
        "modes": modes,
        "score_mean": sum(r["score"] for r in selected) / len(selected),
        "ann_count_mean": sum(r["ann_count"] for r in selected) / len(selected),
        "added": [str(r["ann_path"]) for r in added],
    }
    (out / "extend_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n", "origins", "modes", "score_mean", "ann_count_mean",
    )}, indent=2))


if __name__ == "__main__":
    main()
