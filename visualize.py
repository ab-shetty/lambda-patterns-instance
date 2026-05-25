#!/usr/bin/env python3
"""Visualize RefMask2Former predictions.

For each sampled validation image, shows four panels:
  1. input image
  2. reference patch
  3. ground-truth instances of the target pattern (colored)
  4. predicted instances matched to the reference (colored)
"""

import argparse
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from refmask2former import (
    RefMask2Former, build_datasets, collate_fn, load_parquet_records,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def denorm(t):
    img = t.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def overlay(image, masks, alpha=0.5):
    out = image.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        color = rng.random(3)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hf-repo", type=str, default="abshetty/floz-synth-v5")
    p.add_argument("--cache-dir", type=str, default="./data")
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.5)
    p.add_argument("--max-records", type=int, default=200)
    p.add_argument("--output", type=str, default="predictions.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    saved = ck.get("args", {})
    model = RefMask2Former(
        num_queries=saved.get("num_queries", 100),
        hidden_dim=saved.get("hidden_dim", 256),
        mask_dim=saved.get("mask_dim", 256),
        ref_dim=saved.get("ref_dim", 128),
        nheads=saved.get("nheads", 8),
        dec_layers=saved.get("dec_layers", 9),
        pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir)
    if args.max_records:
        records = records.select(range(min(args.max_records, len(records))))
    _, val_ds = build_datasets(records, image_max_size=args.image_max_size)
    loader = DataLoader(val_ds, batch_size=1, shuffle=True,
                        collate_fn=partial(collate_fn, size_divisible=32))

    n = args.num_images
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[None]

    it = iter(loader)
    for i in range(n):
        batch = next(it)
        images = batch["images"].to(device)
        pmask = batch["pixel_mask"].to(device)
        refs = batch["references"].to(device)
        tgt = batch["targets"][0]

        results = model.predict(images, pmask, refs,
                                score_thresh=args.score_thresh,
                                match_thresh=args.match_thresh)[0]

        valid_h = int(pmask[0].any(1).sum().item())
        valid_w = int(pmask[0].any(0).sum().item())
        img = denorm(images[0])[:valid_h, :valid_w]
        ref_img = denorm(refs[0])

        gt_match = tgt["ref_match"].bool()
        gt_masks = tgt["masks"].bool().numpy()[:, :valid_h, :valid_w]
        gt_target = gt_masks[gt_match.numpy()] if len(gt_masks) else gt_masks
        pred_masks = results["masks"].cpu().numpy()[:, :valid_h, :valid_w]

        axes[i, 0].imshow(img); axes[i, 0].set_title("image"); axes[i, 0].axis("off")
        axes[i, 1].imshow(ref_img); axes[i, 1].set_title("reference"); axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay(img, gt_target))
        axes[i, 2].set_title(f"GT target instances ({len(gt_target)})"); axes[i, 2].axis("off")
        axes[i, 3].imshow(overlay(img, pred_masks))
        axes[i, 3].set_title(f"predicted matches ({len(pred_masks)})"); axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
