#!/usr/bin/env python3
"""Threshold sweep on the real-world test set.

Runs the model ONCE over the 28 images, caching per-query (prob, sim) and the
[Q, g] IoU-vs-GT matrix per image. Then re-applies a grid of (score, match)
thresholds against the cache — faithfully reproducing evaluate.run_eval's
matching logic — so we can separate the detection gap from the threshold gap.
"""
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import load_model, masks_iou
from refmask2former import InstanceSegDataset, collate_fn, load_parquet_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    p.add_argument("--config", default="real-world-test")
    p.add_argument("--split", default="test")
    p.add_argument("--cache-dir", default="./data")
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cache_outputs(model, loader, device):
    """One forward pass; return list of per-image dicts with cached tensors."""
    cache = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            pmask = batch["pixel_mask"].to(device)
            refs = batch["references"].to(device)
            H, W = images.shape[-2:]
            out = model(images, pmask, refs)
            probs = out["pred_logits"].softmax(-1)[..., 1]            # [B, Q]
            sim = torch.einsum("bqd,bd->bq", out["pred_ref"], out["reference_emb"])
            pred_masks = (F.interpolate(out["pred_masks"], size=(H, W),
                          mode="bilinear", align_corners=False).sigmoid() > 0.5)
            for b, tgt in enumerate(batch["targets"]):
                g = tgt["masks"].shape[0]
                if g == 0:
                    continue
                gt_masks = tgt["masks"].bool().to(device)
                iou_qg = masks_iou(pred_masks[b], gt_masks)           # [Q, g]
                cache.append({
                    "probs": probs[b].cpu(),
                    "sim": sim[b].cpu(),
                    "gt_match": tgt["ref_match"].bool().cpu(),
                    "iou_qg": iou_qg.cpu(),
                })
    return cache


def eval_thresholds(cache, score_t, match_t, iou_t):
    ious, tp, fp, fn = [], 0, 0, 0
    mc, mt, det_found, det_total = 0, 0, 0, 0
    for c in cache:
        probs, sim, gt_match, iou_qg = c["probs"], c["sim"], c["gt_match"], c["iou_qg"]
        g = gt_match.shape[0]

        # Class-agnostic detection recall (depends only on score_t).
        fg = probs > score_t
        if fg.any():
            iou_fg = iou_qg[fg]                                       # [n_fg, g]
            det_found += (iou_fg.max(0).values >= iou_t).sum().item()
        det_total += g

        # Reference-conditioned.
        keep = (probs > score_t) & (sim > match_t)
        k = int(keep.sum().item())
        gt_pos_idx = gt_match.nonzero(as_tuple=True)[0]
        ng = gt_pos_idx.numel()
        if k == 0:
            fn += ng
            continue
        if ng == 0:
            fp += k
            mt += k                                # all kept are false matches
            continue
        iou_kept_pos = iou_qg[keep][:, gt_pos_idx].t()               # [ng, k]
        best_iou, best_k = iou_kept_pos.max(1)
        matched = set()
        for gi in range(ng):
            if best_iou[gi] >= iou_t and best_k[gi].item() not in matched:
                tp += 1
                matched.add(best_k[gi].item())
                ious.append(best_iou[gi].item())
            else:
                fn += 1
        fp += k - len(matched)

        # Ref-match accuracy of kept preds.
        best_gt = iou_qg[keep].max(1).indices
        gt_label = gt_match[best_gt].float()
        mc += (gt_label == 1.0).sum().item()
        mt += k

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "ref_acc": mc / max(mt, 1), "det_recall": det_found / max(det_total, 1),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                   config=args.config, split=args.split)
    ds = InstanceSegDataset(records, range(len(records)),
                            image_max_size=args.image_max_size,
                            ref_size=args.ref_size, augment=False)
    g = torch.Generator(); g.manual_seed(args.seed)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=partial(collate_fn, size_divisible=32),
                        generator=g)
    print(f"Caching model outputs over {len(ds)} images...")
    cache = cache_outputs(model, loader, device)

    it = args.iou_thresh
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    matches = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # 1) Detection-recall curve vs score_thresh (reference-independent).
    print("\n=== Class-agnostic detection recall vs score_thresh ===")
    print(f"{'score':>6} | {'det_recall':>10}")
    for s in scores:
        m = eval_thresholds(cache, s, -1.0, it)   # match=-1 keeps everything for det
        print(f"{s:>6.2f} | {m['det_recall']:>10.4f}")

    # 2) Reference-conditioned recall/precision/F1 over the (score, match) grid.
    print("\n=== Reference-conditioned recall  (rows=score, cols=match) ===")
    print("score\\match " + " ".join(f"{mm:>7.2f}" for mm in matches))
    for s in scores:
        row = [eval_thresholds(cache, s, mm, it)["recall"] for mm in matches]
        print(f"{s:>10.2f} " + " ".join(f"{r:>7.4f}" for r in row))

    print("\n=== Reference-conditioned F1  (rows=score, cols=match) ===")
    print("score\\match " + " ".join(f"{mm:>7.2f}" for mm in matches))
    for s in scores:
        row = [eval_thresholds(cache, s, mm, it)["f1"] for mm in matches]
        print(f"{s:>10.2f} " + " ".join(f"{r:>7.4f}" for r in row))

    print("\n=== Reference-conditioned precision  (rows=score, cols=match) ===")
    print("score\\match " + " ".join(f"{mm:>7.2f}" for mm in matches))
    for s in scores:
        row = [eval_thresholds(cache, s, mm, it)["precision"] for mm in matches]
        print(f"{s:>10.2f} " + " ".join(f"{r:>7.4f}" for r in row))


if __name__ == "__main__":
    main()
