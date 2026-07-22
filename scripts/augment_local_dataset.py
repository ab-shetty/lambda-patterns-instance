#!/usr/bin/env python3
"""Build an augmented local dataset without introducing outside images.

The source directory must use the repository's ``images/`` + ``annotations/``
layout.  Every output image is either a copied source image or a transformed
version of one; polygon coordinates are transformed with geometric changes.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from build_mix import augment_scenes, extra_scenes, merge_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="input local-data directory")
    parser.add_argument("--out", required=True, help="output local-data directory")
    parser.add_argument("--aug-per-scene", type=int, required=True)
    parser.add_argument("--target-long", type=int, default=0,
                        help="resize augmented sources to this long side; 0 keeps size")
    parser.add_argument("--seed", type=int, default=5858)
    parser.add_argument("--strong", action="store_true",
                        help="use geometric and richer photometric augmentation")
    parser.add_argument("--omit-originals", action="store_true",
                        help="write only transformed views (useful when resizing all views)")
    args = parser.parse_args()

    if args.aug_per_scene < 1:
        parser.error("--aug-per-scene must be positive")

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if src == out:
        parser.error("--src and --out must differ")

    scenes = extra_scenes(str(src), args.target_long)
    if not scenes:
        parser.error(f"no annotations found under {src}")

    with tempfile.TemporaryDirectory(prefix="rf_aug_") as tmp_name:
        tmp = Path(tmp_name)
        n_aug = augment_scenes(scenes, str(tmp), args.aug_per_scene,
                               args.seed, strong=args.strong)
        inputs = [str(tmp)] if args.omit_originals else [str(src), str(tmp)]
        n_total = merge_dirs(inputs, str(out))

    manifest = {
        "source": str(src),
        "source_images": len(scenes),
        "augmented_images": n_aug,
        "total_images": n_total,
        "aug_per_scene": args.aug_per_scene,
        "target_long": args.target_long,
        "seed": args.seed,
        "strong": args.strong,
        "originals_included": not args.omit_originals,
        "outside_training_images": 0,
    }
    (out / "augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
