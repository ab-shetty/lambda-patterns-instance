#!/usr/bin/env python3
"""Threshold-independent assessment of the reference-matching head on synth val.

For each GT instance that was *well detected* (best-IoU kept pred >= iou_thresh),
collect (cosine_sim, is_match). Then report:
  - ROC-AUC of sim vs match label   (separability, threshold-free)
  - accuracy at 0.5                 (reproduces the training metric)
  - best achievable accuracy + its threshold
  - sim distribution for match vs non-match
This separates "matching head can't distinguish" from "0.5 is the wrong cutoff".
"""
import argparse
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import load_model, masks_iou
from refmask2former import build_datasets, collate_fn, load_parquet_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    p.add_argument("--cache-dir", default="./data")
    p.add_argument("--image-max-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-val", type=int, default=300)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--score-thresh", type=float, default=0.5)
    return p.parse_args()


def roc_auc(scores, labels):
    """Mann-Whitney U formulation of ROC-AUC."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir)
    _, val_ds = build_datasets(records, image_max_size=args.image_max_size)
    n = min(args.max_val, len(val_ds))
    val_ds.indices = val_ds.indices[:n]
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=partial(collate_fn, size_divisible=32))
    print(f"Assessing matching on {n} synthetic val images...")

    sims, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            pmask = batch["pixel_mask"].to(device)
            refs = batch["references"].to(device)
            H, W = images.shape[-2:]
            out = model(images, pmask, refs)
            probs = out["pred_logits"].softmax(-1)[..., 1]
            sim = torch.einsum("bqd,bd->bq", out["pred_ref"], out["reference_emb"])
            pred_masks = (F.interpolate(out["pred_masks"], size=(H, W),
                          mode="bilinear", align_corners=False).sigmoid() > 0.5)
            for b, tgt in enumerate(batch["targets"]):
                g = tgt["masks"].shape[0]
                if g == 0:
                    continue
                keep = probs[b] > args.score_thresh
                if keep.sum() == 0:
                    continue
                gt_masks = tgt["masks"].bool().to(device)
                iou = masks_iou(gt_masks, pred_masks[b][keep])      # [g, k]
                best_iou, best_k = iou.max(1)
                kept_sim = sim[b][keep][best_k]                     # [g]
                gt_match = tgt["ref_match"].to(device)
                # only assess matching where detection actually succeeded
                ok = best_iou >= args.iou_thresh
                sims.append(kept_sim[ok].cpu().numpy())
                labels.append(gt_match[ok].cpu().numpy())

    sims = np.concatenate(sims)
    labels = np.concatenate(labels).astype(int)
    pos, neg = sims[labels == 1], sims[labels == 0]

    print(f"\nWell-detected GT instances assessed: {len(sims)} "
          f"({labels.sum()} match / {(labels==0).sum()} non-match)")
    print(f"\nCosine sim distribution:")
    print(f"  match     : mean {pos.mean():.3f}  median {np.median(pos):.3f}  "
          f"p10 {np.percentile(pos,10):.3f}  p90 {np.percentile(pos,90):.3f}")
    print(f"  non-match : mean {neg.mean():.3f}  median {np.median(neg):.3f}  "
          f"p10 {np.percentile(neg,10):.3f}  p90 {np.percentile(neg,90):.3f}")

    auc = roc_auc(sims, labels)
    print(f"\nROC-AUC (separability, threshold-free): {auc:.4f}")

    acc05 = ((sims > 0.5).astype(int) == labels).mean()
    print(f"Accuracy @ thresh=0.5 (training metric):  {acc05:.4f}")

    ts = np.linspace(sims.min(), sims.max(), 200)
    accs = [((sims > t).astype(int) == labels).mean() for t in ts]
    bi = int(np.argmax(accs))
    print(f"Best accuracy:                            {accs[bi]:.4f} @ thresh={ts[bi]:.3f}")


if __name__ == "__main__":
    main()
