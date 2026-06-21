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

EXTRA REAL POOLS (e.g. Roboflow-labelled plans). Pass one or more local-data
dirs (images/ + annotations/, the format scripts/roboflow_to_local.py writes) via
--real-extra-dir. They are folded into the TRAIN side only (augmented like the HF
train reals); the HF held-out 14 stay a clean, untouched eval set. Because such
pools can be at a different (often smaller, e.g. 640x640) resolution than the HF
reals, each extra real is first resized so its long side == --real-extra-size
(default 1024, polygons scaled to match) BEFORE augmentation. That matters: the
photometric augmentation uses ABSOLUTE-pixel blur radius and noise sigma, so
applied at native 640 it would hit far harder (relatively) than on a multi-thousand
-px HF real — normalising to a common working resolution keeps augmentation
strength comparable and stops the loader from blindly upscaling tiny squares.
NOTE: more real shifts real_frac up; the established sweet spot is ~10% real
(see synth_progress.md ratio sweep), so watch the printed real_frac and tune
--aug-per-scene / --real-extra-aug-per-scene / synth size accordingly.
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


def _scale_anns(anns, s):
    """Scale every polygon coordinate (outer + holes) by s."""
    return [{**a, "segmentation": [[c * s for c in poly] for poly in a["segmentation"]]}
            for a in anns]


def hf_scenes(train_idx, recs):
    """Build augmentable scenes from the HF train reals (native resolution)."""
    scenes = []
    for ti in train_idx:
        rec = recs[ti]
        img = Image.open(io.BytesIO(rec["image"])).convert("RGB")
        scenes.append({"image": img, "anns": _norm_anns(rec["annotations"]),
                       "name": f"hf{ti}"})
    return scenes


def extra_scenes(extra_dir, target_long):
    """Load an extra real pool (local-data dir) as augmentable scenes, resized so
    each long side == target_long (polygons scaled with it) for resolution parity."""
    scenes = []
    for jf in sorted(glob.glob(f"{extra_dir}/annotations/*.json")):
        ann = json.load(open(jf))
        fn = ann["image"]["file_name"]
        img = Image.open(f"{extra_dir}/images/{fn}").convert("RGB")
        W, H = img.size
        anns = _norm_anns(ann["annotations"])
        if target_long and max(W, H) != target_long:
            s = target_long / max(W, H)
            img = img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
            anns = _scale_anns(anns, s)
        scenes.append({"image": img, "anns": anns,
                       "name": os.path.splitext(os.path.basename(jf))[0]})
    return scenes


def _rotate_img_and_polys(img, anns, deg):
    """Rotate image about its center by `deg` (CCW, white fill, no expand) and
    apply the MATCHING transform to every polygon coordinate so masks stay
    aligned. Verified by scripts overlay (see build_mix --real-aug-strong)."""
    W, H = img.size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rimg = img.rotate(deg, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255))
    th = np.deg2rad(deg)
    cos, sin = np.cos(th), np.sin(th)
    # image y is DOWN, so a visually-CCW PIL rotation maps a point as below.
    def _tx(poly):
        out = []
        for j in range(0, len(poly), 2):
            x, y = poly[j] - cx, poly[j + 1] - cy
            out.append(cx + x * cos + y * sin)
            out.append(cy - x * sin + y * cos)
        return out
    rot = [{**a, "segmentation": [_tx(p) for p in a["segmentation"]]} for a in anns]
    return rimg, rot


def augment_scenes(scenes, out_root, k_per_scene, seed, start_n=0, strong=False):
    """Write k_per_scene augmentations of each scene; returns count.
    start_n continues the aug_NNNNNN filename numbering across multiple calls.
    strong=False: the established mild photometric+flip recipe (byte-identical to
    before). strong=True: RICHER per-real augmentation (wider brightness/contrast,
    color+sharpness jitter, stronger blur/noise, occasional grayscale, and a small
    rotation with matching polygon transform) -- tests whether augmentation DIVERSITY
    (not copy count) can squeeze more from the fixed real pool."""
    os.makedirs(f"{out_root}/images", exist_ok=True)
    os.makedirs(f"{out_root}/annotations", exist_ok=True)
    rng = np.random.RandomState(seed)
    n = start_n
    for sc in scenes:
        base = sc["image"]
        W, H = base.size
        anns = sc["anns"]
        for _ in range(k_per_scene):
            img = base
            cur = anns
            if strong:
                # small rotation first (transforms polygons too)
                if rng.rand() < 0.7:
                    img, cur = _rotate_img_and_polys(img, cur,
                                                     float(rng.uniform(-8, 8)))
                img = ImageEnhance.Brightness(img).enhance(0.75 + 0.50 * rng.rand())
                img = ImageEnhance.Contrast(img).enhance(0.75 + 0.50 * rng.rand())
                img = ImageEnhance.Color(img).enhance(0.6 + 0.8 * rng.rand())
                img = ImageEnhance.Sharpness(img).enhance(0.5 + 1.5 * rng.rand())
                if rng.rand() < 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=0.3 + 1.2 * rng.rand()))
                if rng.rand() < 0.15:
                    img = img.convert("L").convert("RGB")
                if rng.rand() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    cur = [{**a, "segmentation": [
                        [(W - 1 - c) if (j % 2 == 0) else c for j, c in enumerate(p)]
                        for p in a["segmentation"]]} for a in cur]
                noise = rng.normal(0, 3 + 9 * rng.rand(), (H, W, 3))
            else:
                img = ImageEnhance.Brightness(img).enhance(0.90 + 0.20 * rng.rand())
                img = ImageEnhance.Contrast(img).enhance(0.90 + 0.20 * rng.rand())
                if rng.rand() < 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=0.3 + 0.6 * rng.rand()))
                if rng.rand() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    cur = [{**a, "segmentation": [
                        [(W - 1 - c) if (j % 2 == 0) else c for j, c in enumerate(p)]
                        for p in a["segmentation"]]} for a in anns]
                noise = rng.normal(0, 2 + 5 * rng.rand(), (H, W, 3))
            arr = np.asarray(img).astype(np.float32) + noise
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            fn = f"aug_{n:06d}.png"
            img.save(f"{out_root}/images/{fn}")
            json.dump({"image": {"file_name": fn, "width": W, "height": H},
                       "mode": "freeform", "annotations": cur},
                      open(f"{out_root}/annotations/aug_{n:06d}.json", "w"))
            n += 1
    return n - start_n


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
                    help="photometric augmentations per HF train real (total HF aug "
                         "= this x 14). Real fraction depends on synth-dir size; "
                         "~500 synth + 4/scene -> ~10%% real. Check printed real_frac.")
    ap.add_argument("--real-extra-dir", action="append", default=[],
                    help="extra real pool as a local-data dir (images/ + "
                         "annotations/, e.g. from scripts/roboflow_to_local.py). "
                         "Folded into the TRAIN side only; HF held-out 14 stay clean. "
                         "Repeatable.")
    ap.add_argument("--real-extra-aug-per-scene", type=int, default=None,
                    help="augmentations per extra real (default: same as "
                         "--aug-per-scene). Tune to keep real_frac near ~10%%.")
    ap.add_argument("--real-extra-size", type=int, default=1024,
                    help="resize each extra real so its long side == this (polygons "
                         "scaled to match) before augmentation, for resolution parity "
                         "with the HF reals. 0 = keep native resolution.")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--split-seed", type=int, default=1234,
                    help="seed for the 14/14 real train/held split (keep at 1234 "
                         "to match the established setup)")
    ap.add_argument("--aug-seed", type=int, default=5858)
    ap.add_argument("--real-aug-strong", action="store_true",
                    help="richer per-real augmentation (color/sharpness jitter, "
                         "wider brightness/contrast, stronger blur/noise, occasional "
                         "grayscale, small rotation w/ matching polygon transform) "
                         "instead of the mild default. Tests augmentation DIVERSITY "
                         "as a lever vs. the fixed real pool.")
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

    # HF train reals (native resolution) -> aug_000000.. (byte-identical to before
    # when no extra pools are given, so the established parity mix reproduces).
    n_hf = augment_scenes(hf_scenes(train_idx, recs), aug_tmp,
                          args.aug_per_scene, args.aug_seed, start_n=0,
                          strong=args.real_aug_strong)
    print(f"augmented HF reals: {n_hf}  ({args.aug_per_scene}/scene x {len(train_idx)})")

    # Extra real pools (resolution-normalised) -> continue the aug numbering.
    n_extra = 0
    k_extra = args.real_extra_aug_per_scene or args.aug_per_scene
    for ei, ed in enumerate(args.real_extra_dir):
        sc = extra_scenes(ed, args.real_extra_size)
        added = augment_scenes(sc, aug_tmp, k_extra, args.aug_seed + 1 + ei,
                               start_n=n_hf + n_extra, strong=args.real_aug_strong)
        n_extra += added
        sz = "native" if not args.real_extra_size else f"long-side {args.real_extra_size}"
        print(f"augmented extra reals [{ed}]: {added}  "
              f"({k_extra}/scene x {len(sc)}, {sz})")

    n_aug = n_hf + n_extra
    n_synth = len(glob.glob(f"{args.synth_dir}/images/*.png"))

    total = merge_dirs([args.synth_dir, aug_tmp], args.out)
    shutil.rmtree(aug_tmp)
    real_frac = n_aug / total if total else 0.0
    print(f"merged {total} items into {args.out}  "
          f"(synth={n_synth}, real_aug={n_aug} [HF {n_hf} + extra {n_extra}], "
          f"real_frac={real_frac:.1%})")
    print("HELD-OUT indices for --real-eval-indices:")
    print("  " + ",".join(str(i) for i in held_idx))


if __name__ == "__main__":
    main()
