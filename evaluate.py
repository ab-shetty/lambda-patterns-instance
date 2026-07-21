#!/usr/bin/env python3
"""Evaluate RefMask2Former on the validation split.

Reports, for the reference-conditioned task (given a reference patch, return the
instances of that pattern):
  - mean IoU of matched instances
  - precision / recall / F1 at IoU >= 0.5
  - reference-match accuracy of kept predictions
plus class-agnostic detection recall (how many of ALL instances are found).
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from refmask2former import (
    RefMask2Former, build_datasets, collate_fn, load_parquet_records,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hf-repo", type=str, default="abshetty/floz-synth-v5")
    p.add_argument("--cache-dir", type=str, default="./data")
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.0)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    return p.parse_args()


def masks_iou(a, b):
    """a: [n, H, W] bool, b: [m, H, W] bool -> [n, m] IoU."""
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    inter = a @ b.t()
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / union.clamp(min=1)


def load_model(checkpoint, device):
    """Rebuild RefMask2Former from a checkpoint's saved args and load weights."""
    ck = torch.load(checkpoint, map_location=device)
    saved = ck.get("args", {})
    model = RefMask2Former(
        num_queries=saved.get("num_queries", 100),
        hidden_dim=saved.get("hidden_dim", 256),
        mask_dim=saved.get("mask_dim", 256),
        ref_dim=saved.get("ref_dim", 128),
        nheads=saved.get("nheads", 8),
        dec_layers=saved.get("dec_layers", 9),
        pretrained=False,
        stem_pool=saved.get("backbone_stem_pool", "max"),
        ref_pool_features=saved.get("ref_pool_features", False),
        ref_siamese_backbone=saved.get("ref_siamese_backbone", False),
        ref_siamese_level=saved.get("ref_siamese_level", "res5"),
        ref_siamese_stats=saved.get("ref_siamese_stats", False),
        ref_texture_backbone=saved.get("ref_texture_backbone", False)).to(device)
    if saved.get("ref_pairwise_head", False):
        model.enable_pairwise_match_head(saved.get("ref_dim", 128))
        model.to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def run_eval(model, loader, device, score_thresh=0.5, match_thresh=0.5,
             iou_thresh=0.5, desc="eval"):
    """Run the reference-conditioned eval loop; returns a metrics dict."""
    ious, tp, fp, fn = [], 0, 0, 0
    match_correct, match_total = 0, 0
    det_found, det_total = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
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
                gt_masks = tgt["masks"].bool().to(device)
                gt_match = tgt["ref_match"].to(device).bool()

                # Class-agnostic detection recall (all instances).
                fg = probs[b] > score_thresh
                if fg.any():
                    iou_all = masks_iou(gt_masks, pred_masks[b][fg])
                    det_found += (iou_all.max(1).values >= iou_thresh).sum().item()
                det_total += g

                # Reference-conditioned instances.
                keep = (probs[b] > score_thresh) & (sim[b] > match_thresh)
                kept = pred_masks[b][keep]
                gt_pos = gt_masks[gt_match]
                ng = gt_pos.shape[0]

                if kept.shape[0] == 0:
                    fn += ng
                    continue
                if ng == 0:
                    fp += kept.shape[0]
                    continue

                iou = masks_iou(gt_pos, kept)                  # [ng, k]
                best_iou, best_k = iou.max(1)
                matched_k = set()
                for gi in range(ng):
                    if best_iou[gi] >= iou_thresh and best_k[gi].item() not in matched_k:
                        tp += 1
                        matched_k.add(best_k[gi].item())
                        ious.append(best_iou[gi].item())
                    else:
                        fn += 1
                fp += kept.shape[0] - len(matched_k)

                # Reference-match accuracy: kept predictions vs their best GT label.
                iou_full = masks_iou(kept, gt_masks)
                best_gt = iou_full.max(1).indices
                pred_is_match = torch.ones(kept.shape[0], device=device)  # kept => predicted match
                gt_label = gt_match[best_gt].float()
                match_correct += (pred_is_match == gt_label).sum().item()
                match_total += kept.shape[0]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ref_acc": match_correct / max(match_total, 1),
        "det_recall": det_found / max(det_total, 1),
        "iou_thresh": iou_thresh,
    }


def print_metrics(m, title="Reference-conditioned instance segmentation"):
    it = m["iou_thresh"]
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Mean matched IoU:        {m['mean_iou']:.4f}")
    print(f"Precision @IoU>={it}:   {m['precision']:.4f}")
    print(f"Recall    @IoU>={it}:   {m['recall']:.4f}")
    print(f"F1        @IoU>={it}:   {m['f1']:.4f}")
    print(f"Ref-match accuracy:      {m['ref_acc']:.4f}")
    print(f"Class-agnostic recall:   {m['det_recall']:.4f} "
          f"(all instances found, ignoring reference)")
    print("=" * 60)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ck = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    grayscale = ck.get("args", {}).get("grayscale", False)
    if grayscale:
        print("Checkpoint trained in grayscale mode -> evaluating in grayscale.")

    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir)
    if args.max_records:
        records = records.select(range(min(args.max_records, len(records))))
    _, val_ds = build_datasets(records, image_max_size=args.image_max_size,
                               ref_size=args.ref_size, train_split=args.train_split,
                               grayscale=grayscale)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=partial(collate_fn, size_divisible=32))

    metrics = run_eval(model, loader, device, score_thresh=args.score_thresh,
                       match_thresh=args.match_thresh, iou_thresh=args.iou_thresh)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
