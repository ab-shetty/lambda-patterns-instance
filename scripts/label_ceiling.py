#!/usr/bin/env python3
"""How much union IoU is reachable at all, before any architecture question.

`RefUNet` emits logits on a stride-4 grid of the RESIZED input
(`ref_unet.py:decode`, whose finest level is `layer1`), bilinearly upsamples
them to the input size, thresholds, and -- in the evaluator
(`evaluate_refunet_selection.py`) -- NEAREST-resizes that binary map back to
native resolution before scoring against native-resolution ground truth.

So the decision grid is `image_max_size / 4` cells on the longest side, however
large the plan natively is. At 2560 that is 640 cells across a plan whose
median native long side is ~3k px. This script asks what the BEST POSSIBLE
model with that output geometry would score, which is the ceiling on every
reported number and on the "train IoU 0.95" target.

Two bounds per stride:

* `naive` -- put the area-pooled fractional coverage of the target on the grid.
  The obvious choice, and what a well-trained soft model approximates.
* `opt` -- optimise the grid directly. Upsampling is linear and the constraint
  is one-sided per pixel (above the threshold inside the target, below it
  outside), so minimising a hinge on that constraint is convex and lands at or
  very near the true optimum over all grids. This is the honest ceiling: no
  network can beat it, and `naive` can understate it.

Reported IoU uses the evaluator's exact chain, so the numbers are directly
comparable to reported mIoU.
"""
import argparse
import io
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")
from refmask2former import load_local_records, load_parquet_records
from refmask2former.dataset import render_instance_mask, sample_reference_box

HOLDOUT = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]


def targets_hf(records, indices):
    """Every labelled instance is one user selection, as the evaluator does."""
    for image_index in indices:
        rec = records[image_index]
        image = np.asarray(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        h0, w0 = image.shape[:2]
        masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                 for a in anns]
        cats = [a.get("category_name", "pattern") for a in anns]
        for ref_idx, cat in enumerate(cats):
            # Reproduce the evaluator's RNG so selections line up 1:1, even
            # though the box itself does not affect the ceiling.
            rng = random.Random(image_index * 1_000_003 + ref_idx * 65_537 + 12_345)
            sample_reference_box(masks[ref_idx], 128, 512, rng=rng)
            union = np.logical_or.reduce([m for m, c in zip(masks, cats) if c == cat])
            yield f"{image_index}:{ref_idx}", cat, union


def targets_local(records, limit):
    for rec in records[:limit]:
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        if isinstance(anns, dict):
            anns = anns.get("annotations", [])
        image = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
        h0, w0 = image.shape[:2]
        cats = [a.get("category_name", "pattern") for a in anns]
        masks = [render_instance_mask(a["segmentation"], h0, w0).astype(bool)
                 for a in anns]
        if not masks:
            continue
        for ref_idx, cat in enumerate(cats):
            union = np.logical_or.reduce([m for m, c in zip(masks, cats) if c == cat])
            yield f"{os.path.basename(rec['image_path'])}:{ref_idx}", cat, union


def chain_iou(grid, native_target, input_hw, thresh):
    """grid -> bilinear to input size -> threshold -> NEAREST to native -> IoU."""
    up = F.interpolate(grid[None, None], size=input_hw, mode="bilinear",
                       align_corners=False)[0, 0]
    pred_small = (up > thresh).to(torch.uint8).cpu().numpy()
    h0, w0 = native_target.shape
    pred = cv2.resize(pred_small, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    inter = int((pred & native_target).sum())
    union = int((pred | native_target).sum())
    return inter / max(union, 1)


def optimise_grid(target_small, gh, gw, thresh, iters, margin, device):
    """Best grid under the hinge constraint. Convex in the grid values."""
    x = F.interpolate(target_small[None, None], size=(gh, gw), mode="area").clone()
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.15)
    pos = target_small > 0.5
    neg = ~pos
    n_pos = pos.sum().clamp(min=1)
    n_neg = neg.sum().clamp(min=1)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        up = F.interpolate(x, size=target_small.shape, mode="bilinear",
                           align_corners=False)[0, 0]
        loss = (F.relu(thresh + margin - up[pos]).sum() / n_pos
                + F.relu(up[neg] - thresh + margin).sum() / n_neg)
        loss.backward()
        opt.step()
    return x.detach()[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf14", "val14", "local"], default="hf14")
    ap.add_argument("--local-data", default="data/synthetic/v6d_train2000")
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--n", type=int, default=200, help="records, local source only")
    ap.add_argument("--image-max-size", type=int, default=2560)
    ap.add_argument("--strides", default="4,2,1")
    ap.add_argument("--mask-thresh", type=float, default=0.35)
    ap.add_argument("--opt-iters", type=int, default=300)
    ap.add_argument("--opt-margin", type=float, default=0.05)
    ap.add_argument("--no-opt", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strides = [int(s) for s in a.strides.split(",")]

    if a.source == "local":
        records = load_local_records(a.local_data)
        items = targets_local(records, a.n)
        label = f"{os.path.basename(a.local_data)} (first {a.n})"
    else:
        records = load_parquet_records(a.hf_repo, cache_dir=a.cache_dir,
                                       config="real-world-test", split="test")
        indices = HOLDOUT if a.source == "hf14" else \
            [i for i in range(len(records)) if i not in HOLDOUT]
        items = targets_hf(records, indices)
        label = f"{a.source} ({len(indices)} images)"

    rows = []
    for key, cat, native in items:
        h0, w0 = native.shape
        scale = a.image_max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        # The dataset resizes masks with INTER_NEAREST; match it.
        small = torch.from_numpy(
            cv2.resize(native.astype(np.uint8), (nw, nh),
                       interpolation=cv2.INTER_NEAREST)).float().to(device)
        row = {"key": key, "category": cat, "native_hw": [h0, w0],
               "input_hw": [nh, nw], "target_px": int(native.sum())}
        for s in strides:
            gh, gw = max(1, nh // s), max(1, nw // s)
            naive = F.interpolate(small[None, None], size=(gh, gw), mode="area")[0, 0]
            row[f"naive_s{s}"] = chain_iou(naive, native, (nh, nw), a.mask_thresh)
            if not a.no_opt:
                grid = optimise_grid(small, gh, gw, a.mask_thresh,
                                     a.opt_iters, a.opt_margin, device)
                row[f"opt_s{s}"] = chain_iou(grid, native, (nh, nw), a.mask_thresh)
        rows.append(row)
        print(f"  {key:22s} native={h0}x{w0} grid={nh//strides[0]}x{nw//strides[0]} "
              + " ".join(f"{k}={row[k]:.4f}" for k in row if k.startswith(("naive", "opt"))),
              flush=True)

    print(f"\n== label ceiling: {label}, input {a.image_max_size}, "
          f"threshold {a.mask_thresh}, {len(rows)} selections")
    for s in strides:
        for kind in ("naive", "opt"):
            k = f"{kind}_s{s}"
            if k in rows[0]:
                v = np.array([r[k] for r in rows])
                print(f"   stride {s:2d} {kind:5s}  mean {v.mean():.4f}   "
                      f"median {np.median(v):.4f}   min {v.min():.4f}   "
                      f"frac<0.95 {(v < 0.95).mean():.2f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
        print(f"   wrote {a.out}")


if __name__ == "__main__":
    main()
