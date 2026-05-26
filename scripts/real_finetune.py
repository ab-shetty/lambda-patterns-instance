#!/usr/bin/env python3
"""Synth->real transfer test: warm-start from a synth-trained checkpoint,
fine-tune on a REAL train split, evaluate mean_gt_iou on HELD-OUT real.

Decisive question: can the model fit real plans at all when shown real data?
 - held-out real IoU jumps  -> gap is bridgeable; get more real annotations.
 - held-out real IoU flat    -> deeper issue (resolution / task formulation).
"""
import argparse, math, random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import load_model
from refmask2former import (RefMask2Former, HungarianMatcher, SetCriterion,
                            InstanceSegDataset, collate_fn, load_parquet_records)


def mean_gt_iou(model, loader, device, score_thresh=0.5):
    model.eval()
    iou_sum, iou_cnt, found = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device); pmask = batch["pixel_mask"].to(device)
            refs = batch["references"].to(device); H, W = images.shape[-2:]
            out = model(images, pmask, refs)
            probs = out["pred_logits"].softmax(-1)[..., 1]
            masks = (F.interpolate(out["pred_masks"], size=(H, W), mode="bilinear",
                     align_corners=False).sigmoid() > 0.5)
            for b, tgt in enumerate(batch["targets"]):
                g = tgt["masks"].shape[0]
                if g == 0:
                    continue
                keep = probs[b] > score_thresh
                gt = tgt["masks"].bool().to(device)
                if keep.sum() == 0:
                    iou_cnt += g; continue
                pm = masks[b][keep].flatten(1).float(); gm = gt.flatten(1).float()
                inter = gm @ pm.t()
                union = gm.sum(1)[:, None] + pm.sum(1)[None, :] - inter
                best = (inter / union.clamp(min=1)).max(1).values
                iou_sum += best.sum().item(); iou_cnt += g
                found += (best >= 0.5).sum().item()
    return iou_sum / max(iou_cnt, 1), found / max(iou_cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/best.pth")
    ap.add_argument("--image-max-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-test", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    model, ck = load_model(a.init, device)   # rebuilds arch from ckpt args
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                config="real-world-test", split="test")
    n = len(recs)
    idx = list(range(n)); random.Random(a.seed).shuffle(idx)
    test_idx, train_idx = idx[:a.n_test], idx[a.n_test:]

    mk = lambda ids, aug: DataLoader(
        InstanceSegDataset(recs, ids, image_max_size=a.image_max_size, augment=aug),
        batch_size=(a.batch_size if aug else 1), shuffle=aug, num_workers=4,
        collate_fn=partial(collate_fn, size_divisible=32), drop_last=aug)
    train_loader = mk(train_idx, True)
    test_loader = mk(test_idx, False)
    print(f"real split: {len(train_idx)} train / {len(test_idx)} test (seed {a.seed})")

    pre_iou, pre_rec = mean_gt_iou(model, test_loader, device)
    print(f"  held-out real BEFORE finetune: mean_gt_iou {pre_iou:.4f}  recall@.5 {pre_rec:.4f}")

    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=12544)
    criterion = SetCriterion(matcher, {"loss_ce": 2.0, "loss_mask": 5.0,
                             "loss_dice": 5.0, "loss_ref": 2.0},
                             eos_coef=0.1, num_points=12544).to(device)
    bb = [p for n_, p in model.named_parameters() if p.requires_grad and
          (n_.startswith("backbone.") or n_.startswith("reference_encoder."))]
    oth = [p for n_, p in model.named_parameters() if p.requires_grad and not
           (n_.startswith("backbone.") or n_.startswith("reference_encoder."))]
    opt = torch.optim.AdamW([{"params": oth, "lr": a.lr},
                             {"params": bb, "lr": a.lr * 0.1}], weight_decay=0.05)
    total = len(train_loader) * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    for ep in range(a.epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device); pmask = batch["pixel_mask"].to(device)
            refs = batch["references"].to(device)
            tgts = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(images, pmask, refs)
                loss, _ = criterion(out, tgts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            opt.step(); sched.step()
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            iou, rec = mean_gt_iou(model, test_loader, device)
            print(f"  epoch {ep+1:3d}: held-out real mean_gt_iou {iou:.4f}  recall@.5 {rec:.4f}")


if __name__ == "__main__":
    main()
