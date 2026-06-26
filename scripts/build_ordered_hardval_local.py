#!/usr/bin/env python3
"""Build a 1300 local dataset with hard examples in the deterministic val split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def _score_ann(path: Path):
    ann = json.loads(path.read_text())
    image = ann.get("image") or {}
    w = float(image.get("width") or 0)
    h = float(image.get("height") or 0)
    anns = ann.get("annotations") or []
    area = sum(float(a.get("area") or 0.0) for a in anns)
    cats = len({a.get("category_name", "pattern") for a in anns})
    score = area / (w * h) if w > 0 and h > 0 else 0.0
    aspect = w / h if h else 1.0
    return ann, {
        "score": score,
        "ann_count": len(anns),
        "cat_count": cats,
        "mode": ann.get("mode", "unknown"),
        "width": w,
        "height": h,
        "aspect": aspect,
    }


def _rows(root: Path):
    out = []
    for ann_path in sorted((root / "annotations").glob("*.json")):
        ann, meta = _score_ann(ann_path)
        image_name = ann.get("image", {}).get("file_name")
        img_path = root / "images" / image_name
        if image_name and img_path.exists() and meta["ann_count"] > 0:
            out.append((ann_path, img_path, meta))
    return out


def _parse_quotas(text: str | None):
    if not text:
        return None
    quotas = {}
    for part in text.split(","):
        if not part.strip():
            continue
        mode, count = part.split("=", 1)
        quotas[mode.strip()] = int(count)
    return quotas


def _copy(src_ann: Path, src_img: Path, out: Path, index: int, src_root: Path):
    stem = f"synth_{index:06d}"
    out_img = out / "images" / f"{stem}{src_img.suffix.lower()}"
    out_ann = out / "annotations" / f"{stem}.json"
    shutil.copy2(src_img, out_img)
    ann = json.loads(src_ann.read_text())
    old_name = ann.get("image", {}).get("file_name")
    ann.setdefault("image", {})["file_name"] = out_img.name
    ann["source_dataset"] = str(src_root)
    ann["source_image"] = old_name
    ann["source_annotation"] = str(src_ann)
    out_ann.write_text(json.dumps(ann, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", required=True,
                    help="preferred pool for deterministic train indices")
    ap.add_argument("--hard-src", action="append", default=[],
                    help="candidate pools for deterministic synth-val indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--train-split", type=float, default=0.98)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--preserve-train-positions", action="store_true",
                    help="copy train-src item i into output item i for all "
                         "deterministic train positions")
    ap.add_argument("--target-score", type=float, default=None,
                    help="prefer validation candidates near this area fraction")
    ap.add_argument("--target-ann-count", type=float, default=None,
                    help="prefer validation candidates near this annotation count")
    ap.add_argument("--target-cat-count", type=float, default=None,
                    help="prefer validation candidates near this category count")
    ap.add_argument("--val-mode-quotas", default=None,
                    help="optional mode=count quotas for deterministic val split; "
                         "counts must sum to the number of val positions")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-score", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    idx = list(range(args.n))
    random.Random(args.split_seed).shuffle(idx)
    n_train = int(args.n * args.train_split)
    train_positions = set(idx[:n_train])
    val_positions = set(idx[n_train:])

    train_root = Path(args.train_src)
    train_rows = _rows(train_root)
    if len(train_rows) < len(train_positions):
        raise SystemExit("not enough train rows")

    hard_rows = []
    for root_s in args.hard_src or [args.train_src]:
        root = Path(root_s)
        for ann_path, img_path, meta in _rows(root):
            if args.min_score <= meta["score"] <= args.max_score:
                hard_rows.append((root, ann_path, img_path, meta))
    def hard_key(r):
        meta = r[3]
        if args.target_score is not None:
            return (
                -abs(meta["score"] - args.target_score),
                -abs(meta["ann_count"] - (args.target_ann_count or meta["ann_count"])),
                -abs(meta["cat_count"] - (args.target_cat_count or meta["cat_count"])),
                meta["score"],
            )
        # Default: hardest available validation examples.
        return (
            min(meta["score"], 0.55),
            meta["cat_count"],
            meta["ann_count"],
            abs(meta["aspect"] - 1.8),
        )
    hard_rows.sort(key=hard_key, reverse=True)
    if len(hard_rows) < len(val_positions):
        raise SystemExit("not enough hard rows")

    used_ann = set()
    chosen = {}
    val_quotas = _parse_quotas(args.val_mode_quotas)
    if val_quotas is not None and sum(val_quotas.values()) != len(val_positions):
        raise SystemExit("--val-mode-quotas must sum to the number of val positions")
    if val_quotas is None:
        val_chosen = hard_rows[:len(val_positions)]
    else:
        val_chosen = []
        for mode, count in val_quotas.items():
            rows = [
                r for r in hard_rows
                if r[3]["mode"] == mode and (r[0], r[1]) not in used_ann
            ]
            if len(rows) < count:
                raise SystemExit(f"only found {len(rows)} hard val rows for mode={mode}")
            val_chosen.extend(rows[:count])
            used_ann.update((r[0], r[1]) for r in rows[:count])
        val_chosen.sort(key=hard_key, reverse=True)
    for pos, (root, ann_path, img_path, _meta) in zip(sorted(val_positions), val_chosen):
        chosen[pos] = (root, ann_path, img_path, "val")
        used_ann.add((root, ann_path))

    if args.preserve_train_positions:
        by_name = {p.stem: (p, img, meta) for p, img, meta in train_rows}
        for pos in sorted(train_positions):
            key_name = f"synth_{pos:06d}"
            if key_name not in by_name:
                raise SystemExit(f"missing preserved train row {key_name}")
            ann_path, img_path, _meta = by_name[key_name]
            chosen[pos] = (train_root, ann_path, img_path, "train")
            used_ann.add((train_root, ann_path))
    else:
        train_iter = iter(train_rows)
        for pos in sorted(train_positions):
            while True:
                ann_path, img_path, _meta = next(train_iter)
                key = (train_root, ann_path)
                if key not in used_ann:
                    break
            chosen[pos] = (train_root, ann_path, img_path, "train")
            used_ann.add(key)

    manifest_rows = []
    for pos in range(args.n):
        root, ann_path, img_path, split_role = chosen[pos]
        _copy(ann_path, img_path, out, pos, root)
        _ann, meta = _score_ann(out / "annotations" / f"synth_{pos:06d}.json")
        manifest_rows.append({
            "index": pos,
            "split_role": split_role,
            "source": str(root),
            "source_ann": str(ann_path),
            **meta,
        })

    summary = {}
    for role in ("train", "val"):
        rows = [r for r in manifest_rows if r["split_role"] == role]
        summary[role] = {
            "n": len(rows),
            "score_mean": sum(r["score"] for r in rows) / len(rows),
            "ann_count_mean": sum(r["ann_count"] for r in rows) / len(rows),
            "cat_count_mean": sum(r["cat_count"] for r in rows) / len(rows),
        }
    manifest = {
        "n": args.n,
        "train_src": args.train_src,
        "hard_src": args.hard_src,
        "val_mode_quotas": val_quotas,
        "train_positions": sorted(train_positions),
        "val_positions": sorted(val_positions),
        "summary": summary,
        "rows": manifest_rows,
    }
    (out / "ordered_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
