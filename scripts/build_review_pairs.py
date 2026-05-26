#!/usr/bin/env python3
"""Build a browsable real-vs-synth review folder for JupyterLab."""

import argparse
import csv
import io
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _open_bytes_image(buf):
    return Image.open(io.BytesIO(buf)).convert("RGB")


def _features(img):
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean = float(g.mean())
    std = float(g.std())
    white = float((g > 0.97).mean())
    edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    aspect = img.width / max(1, img.height)
    return {
        "mean": mean,
        "std": std,
        "white": white,
        "edge": edge,
        "aspect": aspect,
    }


def _distance(a, b):
    return (
        4.0 * abs(np.log(max(a["aspect"], 1e-6) / max(b["aspect"], 1e-6)))
        + 2.0 * abs(a["mean"] - b["mean"])
        + 2.0 * abs(a["std"] - b["std"])
        + 2.5 * abs(a["white"] - b["white"])
        + 3.0 * abs(a["edge"] - b["edge"])
    )


def _resize_contain(img, size):
    out = img.copy()
    out.thumbnail(size, Image.BILINEAR)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - out.width) // 2
    y = (size[1] - out.height) // 2
    canvas.paste(out, (x, y))
    return canvas


def _safe_name(s):
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "."):
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "pair"


def load_real_examples(cache_dir):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ds = load_dataset(
        "abshetty/floz-synth-v5",
        "real-world-test",
        split="test",
        cache_dir=cache_dir,
    )
    rows = []
    for rec in ds:
        filename = rec["filename"]
        rows.append(
            {
                "filename": filename,
                "mode": infer_real_mode(filename),
                "image": _open_bytes_image(rec["image"]),
            }
        )
    return rows


def infer_real_mode(filename):
    floor = {
        "2025.2.11- 3376 Las Huertas - 2.12.25_page2_excerpt1",
        "Ceilhunt Full Plan Set_page11_excerpt1",
        "Ceilhunt Full Plan Set_page12_excerpt2",
    }
    roof = {
        "2025.2.11- 3376 Las Huertas - 2.12.25_page3_excerpt2",
        "Construction Documents_page11_excerpt1",
        "Construction Documents_page14_excerpt2",
    }
    if filename in floor:
        return "freeform"
    if filename in roof:
        return "roof_plan"
    return "elevation"


def load_synth_examples(root):
    rows = []
    for img_path in sorted(glob(str(Path(root) / "images" / "*.png"))):
        stem = Path(img_path).stem
        ann_path = Path(root) / "annotations" / f"{stem}.json"
        mode = "unknown"
        if ann_path.exists():
            with ann_path.open() as f:
                mode = json.load(f).get("mode", "unknown")
        rows.append(
            {
                "filename": Path(img_path).name,
                "mode": mode,
                "image": Image.open(img_path).convert("RGB"),
            }
        )
    return rows


def choose_pairs(real_rows, synth_rows):
    synth_meta = []
    for idx, row in enumerate(synth_rows):
        synth_meta.append((idx, row["mode"], _features(row["image"])))

    used = set()
    pairs = []
    for real in real_rows:
        rf = _features(real["image"])
        target_mode = real.get("mode", "unknown")
        same_mode = [
            (_distance(rf, sf), idx)
            for idx, mode, sf in synth_meta
            if idx not in used and mode == target_mode
        ]
        candidates = same_mode if same_mode else [
            (_distance(rf, sf), idx)
            for idx, _mode, sf in synth_meta
            if idx not in used
        ]
        ranked = sorted(
            candidates,
            key=lambda x: x[0],
        )
        if not ranked:
            break
        _, best_idx = ranked[0]
        used.add(best_idx)
        pairs.append((real, synth_rows[best_idx]))
    return pairs


def render_pair(real_row, synth_row, out_path, cell_size=(1100, 850)):
    gap = 36
    pad = 24
    label_h = 54
    title_h = 34
    W = pad * 2 + cell_size[0] * 2 + gap
    H = pad * 2 + title_h + label_h + cell_size[1]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title_font = _font(24)
    label_font = _font(18)

    d.text((pad, pad), "Real | Synth", fill=(20, 20, 20), font=title_font)
    left_x = pad
    right_x = pad + cell_size[0] + gap
    top_y = pad + title_h + label_h

    d.text((left_x, pad + title_h), f"real:  {real_row['filename']}", fill=(60, 60, 60), font=label_font)
    d.text((right_x, pad + title_h), f"synth: {synth_row['filename']}", fill=(60, 60, 60), font=label_font)

    img.paste(_resize_contain(real_row["image"], cell_size), (left_x, top_y))
    img.paste(_resize_contain(synth_row["image"], cell_size), (right_x, top_y))

    d.rectangle([left_x, top_y, left_x + cell_size[0], top_y + cell_size[1]], outline=(180, 180, 180), width=1)
    d.rectangle([right_x, top_y, right_x + cell_size[0], top_y + cell_size[1]], outline=(180, 180, 180), width=1)
    img.save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", required=True, help="Synthetic dataset root with images/ + annotations/")
    ap.add_argument("--out", required=True, help="Output folder for paired review images")
    ap.add_argument("--cache-dir", default="./data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows = load_real_examples(args.cache_dir)
    synth_rows = load_synth_examples(args.synth_root)
    pairs = choose_pairs(real_rows, synth_rows)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "real_filename", "synth_filename", "pair_image"])
        for i, (real_row, synth_row) in enumerate(pairs):
            pair_name = f"{i:03d}_{_safe_name(real_row['filename'])}.png"
            render_pair(real_row, synth_row, out_dir / pair_name)
            w.writerow([i, real_row["filename"], synth_row["filename"], pair_name])

    print(f"wrote {len(pairs)} review pairs to {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
