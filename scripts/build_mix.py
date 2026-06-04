#!/usr/bin/env python3
"""Build the "best-of-both" training mix: broad synth + augmented real.

Reproduces the mix that hits divergence ~0 with held-out real ~0.32 (the parity
setup referenced in synth_progress.md / codex_doc.md). The 28 real plans are
split deterministically (random.Random(1234)) into 14 TRAIN reals and 14
HELD-OUT reals; only the train reals are augmented into the mix, so the held-out
14 stay a clean eval set (pass them to train.py via --real-eval-indices).

Each train real gets K photometric augmentations (brightness/contrast jitter,
optional light blur, optional horizontal flip with mask flip, gaussian noise).
The given synth dir is then merged with the augmented reals into one local-data
folder (images/ + annotations/) ready for --local-data.

The real FRACTION depends on the synth-dir size (aug count is aug-per-scene x 14).
The established parity mix (~10% real, divergence ~0, held-out real ~0.32) used a
~500-image synth dir + 56 aug. The script prints the realized real_frac so you can
tune. Example reproducing that mix:
    python scripts/build_mix.py --synth-dir /tmp/synth_500 \
        --out /tmp/mix_best --aug-per-scene 4     # 56 aug / 556 = ~10% real
    # Train with:
    #   --local-data /tmp/mix_best --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24

The held-out 14 indices are printed at the end (and are stable for a given
--split-seed), so you can copy them straight into --real-eval-indices.
"""

import argparse
import glob
import io
import json
import os
import random
import shutil

import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refmask2former import load_parquet_records


def _norm_anns(anns):
    """Annotations may arrive as a JSON string or as a list possibly holding
    numpy arrays; return a plain JSON-safe list."""
    if isinstance(anns, str):
        anns = json.loads(anns)
    return json.loads(json.dumps(
        anns, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def augment_real(train_idx, recs, out_root, k_per_scene, seed):
    """Write k_per_scene photometric augmentations of each train real."""
    os.makedirs(f"{out_root}/images", exist_ok=True)
    os.makedirs(f"{out_root}/annotations", exist_ok=True)
    rng = np.random.RandomState(seed)
    n = 0
    for ti in train_idx:
        rec = recs[ti]
        base = Image.open(io.BytesIO(rec["image"])).convert("RGB")
        W, H = base.size
        anns = _norm_anns(rec["annotations"])
        for _ in range(k_per_scene):
            img = base
            img = ImageEnhance.Brightness(img).enhance(0.90 + 0.20 * rng.rand())
            img = ImageEnhance.Contrast(img).enhance(0.90 + 0.20 * rng.rand())
            if rng.rand() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.3 + 0.6 * rng.rand()))
            if rng.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                cur = [{**a, "segmentation": [
                    [(W - 1 - c) if (j % 2 == 0) else c for j, c in enumerate(p)]
                    for p in a["segmentation"]]} for a in anns]
            else:
                cur = anns
            arr = np.asarray(img).astype(np.float32) + \
                rng.normal(0, 2 + 5 * rng.rand(), (H, W, 3))
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            fn = f"aug_{n:06d}.png"
            img.save(f"{out_root}/images/{fn}")
            json.dump({"image": {"file_name": fn, "width": W, "height": H},
                       "mode": "freeform", "annotations": cur},
                      open(f"{out_root}/annotations/aug_{n:06d}.json", "w"))
            n += 1
    return n


def merge_dirs(src_dirs, dst):
    """Copy images+annotations from each src dir into dst with fresh names."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(f"{dst}/images")
    os.makedirs(f"{dst}/annotations")
    n = 0
    for src in src_dirs:
        for jf in sorted(glob.glob(f"{src}/annotations/*.json")):
            ann = json.load(open(jf))
            old = ann["image"]["file_name"]
            fn = f"item_{n:06d}.png"
            shutil.copy(f"{src}/images/{old}", f"{dst}/images/{fn}")
            ann["image"]["file_name"] = fn
            json.dump(ann, open(f"{dst}/annotations/{fn[:-4]}.json", "w"))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True,
                    help="broad synth local-data dir (images/ + annotations/)")
    ap.add_argument("--out", required=True, help="output mix dir")
    ap.add_argument("--aug-per-scene", type=int, default=4,
                    help="photometric augmentations per train real (total aug = "
                         "this x 14). Real fraction depends on synth-dir size; "
                         "~500 synth + 4/scene -> ~10%% real. Check printed real_frac.")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--split-seed", type=int, default=1234,
                    help="seed for the 14/14 real train/held split (keep at 1234 "
                         "to match the established setup)")
    ap.add_argument("--aug-seed", type=int, default=5858)
    ap.add_argument("--aug-tmp", default=None,
                    help="scratch dir for the augmented reals (default <out>_augtmp)")
    args = ap.parse_args()

    recs = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                config="real-world-test", split="test")
    perm = list(range(len(recs)))
    random.Random(args.split_seed).shuffle(perm)
    train_idx, held_idx = perm[:14], perm[14:]
    print(f"real plans: {len(recs)}  train={sorted(train_idx)}  held-out={sorted(held_idx)}")

    aug_tmp = args.aug_tmp or f"{args.out}_augtmp"
    if os.path.exists(aug_tmp):
        shutil.rmtree(aug_tmp)
    n_aug = augment_real(train_idx, recs, aug_tmp, args.aug_per_scene, args.aug_seed)
    n_synth = len(glob.glob(f"{args.synth_dir}/images/*.png"))
    print(f"augmented reals: {n_aug}  ({args.aug_per_scene}/scene x {len(train_idx)})")

    total = merge_dirs([args.synth_dir, aug_tmp], args.out)
    shutil.rmtree(aug_tmp)
    real_frac = n_aug / total if total else 0.0
    print(f"merged {total} items into {args.out}  "
          f"(synth={n_synth}, aug={n_aug}, real_frac={real_frac:.1%})")
    print("HELD-OUT indices for --real-eval-indices:")
    print("  " + ",".join(str(i) for i in held_idx))


if __name__ == "__main__":
    main()
