#!/usr/bin/env python3
"""Offline hard-example mining: a training pool of base sheets plus semi-hard fresh sheets.

1. A miner model scores every question of a FRESH pool with the HF14 protocol
   (`evaluate_refunet_selection.py --local-pool <fresh> --indices all`, possibly sharded:
   pass every shard's metrics JSON to --scored).
2. Questions that no model could answer at the training resolution are dropped: the
   naive resolution ceiling (target union area-pooled onto the stride-4 decision grid
   of a `--image-max-size` input, upsampled, thresholded, back to native -- the
   `label_ceiling.py` "naive" bound) must reach --min-ceiling. Synthetic labels are
   exact, so these are the only "noisy" questions a loss-ranked miner would chase.
3. Sheet hardness = mean over its families of (1 - mean IoU of that family's kept
   questions), matching the trainer's one-family-per-sheet question sampler.
4. Semi-hard band: sheets ranked by hardness, skipping the hardest --exclude-top
   fraction and keeping down to the --hard-frac quantile; --hard-n are drawn uniformly
   from that band. Never hardest-only: `--confusable-prob 1.0` (-0.104) showed what an
   all-hard distribution does.
5. Output pool = the first --base-n sheets of --base-pool + the mined sheets, hardlinked.
   With base = the control's own pool, a paired run against that control differs only in
   the mined half.
"""
import argparse
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, ".")
from refmask2former.dataset import render_instance_mask


def naive_ceiling(target, size, thresh, stride=4):
    h0, w0 = target.shape
    scale = size / max(h0, w0)
    nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
    grid = cv2.resize(target.astype(np.float32), (max(1, nw // stride), max(1, nh // stride)),
                      interpolation=cv2.INTER_AREA)
    up = cv2.resize(grid, (nw, nh), interpolation=cv2.INTER_LINEAR) > thresh
    pred = cv2.resize(up.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    union = int((pred | target).sum())
    return int((pred & target).sum()) / max(union, 1)


def sheet_ceilings(job):
    ann_path, size, thresh = job
    d = json.loads(Path(ann_path).read_text())
    h0, w0 = d["image"]["height"], d["image"]["width"]
    cats = [a.get("category_name", "pattern") for a in d["annotations"]]
    masks = {}
    for a, c in zip(d["annotations"], cats):
        m = render_instance_mask(a["segmentation"], h0, w0).astype(bool)
        masks[c] = m if c not in masks else (masks[c] | m)
    return {c: naive_ceiling(m, size, thresh) for c, m in masks.items()}


def link_sheet(pool, ann_file, out):
    d = json.loads(ann_file.read_text())
    img = pool / "images" / d["image"]["file_name"]
    for src, dst in ((ann_file, out / "annotations" / ann_file.name),
                     (img, out / "images" / img.name)):
        if dst.exists():
            raise SystemExit(f"name clash: {dst} (base and fresh pools must use disjoint ids)")
        os.link(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh-pool", required=True)
    ap.add_argument("--scored", nargs="+", required=True,
                    help="metrics JSON(s) from evaluate_refunet_selection.py --local-pool <fresh>")
    ap.add_argument("--base-pool", required=True)
    ap.add_argument("--base-n", type=int, default=1000)
    ap.add_argument("--hard-n", type=int, default=1000)
    ap.add_argument("--hard-frac", type=float, default=0.30,
                    help="semi-hard band reaches down to this hardness quantile (0.30 = top 30%%)")
    ap.add_argument("--exclude-top", type=float, default=0.01,
                    help="skip this hardest fraction of sheets")
    ap.add_argument("--min-ceiling", type=float, default=0.85)
    ap.add_argument("--image-max-size", type=int, default=2048)
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fresh = Path(args.fresh_pool)
    fresh_anns = sorted((fresh / "annotations").glob("*.json"))
    rows = [r for f in args.scored for r in json.loads(Path(f).read_text())["selections"]]
    scored_sheets = sorted({r["image_index"] for r in rows})

    with Pool(args.workers) as pool:
        ceil = dict(zip(scored_sheets, pool.map(
            sheet_ceilings, [(fresh_anns[i], args.image_max_size, args.mask_thresh)
                             for i in scored_sheets], chunksize=4)))

    fam_ious, dropped = {}, 0
    for r in rows:
        if ceil[r["image_index"]][r["category"]] < args.min_ceiling:
            dropped += 1
            continue
        fam_ious.setdefault(r["image_index"], {}).setdefault(r["category"], []).append(r["iou"])
    hardness = {i: float(np.mean([1 - np.mean(v) for v in fams.values()]))
                for i, fams in fam_ious.items()}
    ranked = sorted(hardness, key=hardness.get, reverse=True)
    lo = int(round(len(ranked) * args.exclude_top))
    hi = int(round(len(ranked) * args.hard_frac))
    band = ranked[lo:hi]
    if len(band) < args.hard_n:
        raise SystemExit(f"band has {len(band)} sheets < --hard-n {args.hard_n}; score more fresh sheets")
    mined = sorted(random.Random(args.seed).sample(band, args.hard_n))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "annotations").mkdir()
    base = Path(args.base_pool)
    base_anns = sorted((base / "annotations").glob("*.json"))[:args.base_n]
    for f in base_anns:
        link_sheet(base, f, out)
    for i in mined:
        link_sheet(fresh, fresh_anns[i], out)

    h = np.array(list(hardness.values()))
    summary = {
        "fresh_pool": str(fresh), "scored": args.scored, "base_pool": str(base),
        "generator_version": json.loads((fresh / "generation_manifest.json").read_text()).get("generator_version", "none")
        if (fresh / "generation_manifest.json").exists() else "none",
        "base_n": len(base_anns), "hard_n": len(mined), "hard_frac": args.hard_frac,
        "exclude_top": args.exclude_top, "min_ceiling": args.min_ceiling,
        "image_max_size": args.image_max_size, "seed": args.seed,
        "questions": len(rows), "questions_dropped_ceiling": dropped,
        "sheets_scored": len(scored_sheets), "sheets_ranked": len(ranked),
        "fresh_mean_iou_kept": float(1 - h.mean()),
        "band_hardness": [hardness[band[-1]], hardness[band[0]]],
        "mined_mean_iou": float(1 - np.mean([hardness[i] for i in mined])),
        "mined_files": [fresh_anns[i].name for i in mined],
    }
    (out / "hard_pool_manifest.json").write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "mined_files"})


if __name__ == "__main__":
    main()
