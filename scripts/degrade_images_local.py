#!/usr/bin/env python3
"""Copy a local COCO-style dataset while applying mild image degradation."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def degrade_image(img: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    img = img.convert("RGB")

    # Keep this image-only: annotation geometry stays valid, but rendering becomes
    # closer to scanned/faint/low-quality real plans.
    contrast = rng.uniform(1.0 - 0.22 * strength, 1.0 + 0.10 * strength)
    brightness = rng.uniform(1.0 - 0.10 * strength, 1.0 + 0.06 * strength)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    if rng.random() < 0.75 * strength:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75 * strength)))

    if rng.random() < 0.70 * strength:
        arr = np.asarray(img).astype(np.int16)
        sigma = rng.uniform(2.0, 8.0 * strength)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

    if rng.random() < 0.60 * strength:
        w, h = img.size
        scale = rng.uniform(0.72, 0.92)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BILINEAR)

    if rng.random() < 0.80 * strength:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.uniform(42, 82)), optimize=False)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def process_one(task: tuple[str, str, str, float, int, bool]) -> str:
    name, src_images, out_images, strength, seed, jpeg_output = task
    rng = random.Random(seed)
    img = Image.open(Path(src_images) / name)
    degraded = degrade_image(img, rng, max(0.0, min(1.0, strength)))
    out_path = Path(out_images) / name
    if jpeg_output:
        degraded.save(out_path, format="JPEG", quality=86, optimize=False)
    else:
        degraded.save(out_path)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jpeg-output", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; use --overwrite")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "annotations").mkdir(parents=True)

    rng = random.Random(args.seed)
    ann_dir = src / "annotations"
    coco_ann = ann_dir / "instances.json"
    if coco_ann.exists():
        ann = json.loads(coco_ann.read_text())
        image_names = [image["file_name"] for image in ann["images"]]
        ann_count = len(ann["annotations"])
        shutil.copy2(coco_ann, out / "annotations" / "instances.json")
    else:
        ann_files = sorted(ann_dir.glob("*.json"))
        image_names = []
        ann_count = 0
        for ann_file in ann_files:
            sample = json.loads(ann_file.read_text())
            image_names.append(sample["image"]["file_name"])
            ann_count += len(sample.get("annotations", []))
            shutil.copy2(ann_file, out / "annotations" / ann_file.name)

    tasks = [
        (
            name,
            str(src / "images"),
            str(out / "images"),
            args.strength,
            rng.randrange(2**32),
            args.jpeg_output,
        )
        for name in image_names
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            process_one(task)
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(process_one, tasks, chunksize=8), 1):
                if i % 100 == 0:
                    print(f"{i}/{len(tasks)}", flush=True)

    manifest = {
        "source": str(src),
        "strength": args.strength,
        "seed": args.seed,
        "images": len(image_names),
        "annotations": ann_count,
        "workers": workers,
        "jpeg_output": args.jpeg_output,
    }
    (out / "degrade_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
