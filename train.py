#!/usr/bin/env python3
"""Train RefMask2Former: reference-conditioned instance segmentation.

Detects every distinct pattern instance (class-agnostic) and learns a
reference-matching embedding so a user-selected reference patch picks out the
instances of that pattern at inference time.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from refmask2former import (
    RefMask2Former, HungarianMatcher, SetCriterion,
    build_datasets, collate_fn, load_parquet_records,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train RefMask2Former instance segmentation")
    # Data
    p.add_argument("--hf-repo", type=str, default="abshetty/floz-synth-v5")
    p.add_argument("--cache-dir", type=str, default="./data")
    p.add_argument("--image-max-size", type=int, default=1024,
                   help="Longest image side after aspect-preserving resize")
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--max-records", type=int, default=0,
                   help="Limit number of records (0 = all). Useful for quick runs.")
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--step-size", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-grad", type=float, default=0.01)
    # Model
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--mask-dim", type=int, default=256)
    p.add_argument("--ref-dim", type=int, default=128)
    p.add_argument("--dec-layers", type=int, default=9)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--data-parallel", action="store_true",
                   help="Wrap model in nn.DataParallel across all visible GPUs")
    # Loss weights
    p.add_argument("--class-weight", type=float, default=2.0)
    p.add_argument("--mask-weight", type=float, default=5.0)
    p.add_argument("--dice-weight", type=float, default=5.0)
    p.add_argument("--ref-weight", type=float, default=2.0)
    p.add_argument("--eos-coef", type=float, default=0.1)
    p.add_argument("--num-points", type=int, default=12544)
    # IO
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def move_batch(batch, device):
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["pixel_mask"] = batch["pixel_mask"].to(device, non_blocking=True)
    batch["references"] = batch["references"].to(device, non_blocking=True)
    for t in batch["targets"]:
        t["masks"] = t["masks"].to(device, non_blocking=True)
        t["labels"] = t["labels"].to(device, non_blocking=True)
        t["ref_match"] = t["ref_match"].to(device, non_blocking=True)
    return batch


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    iou_sum, iou_cnt = 0.0, 0
    match_correct, match_total = 0, 0

    base = model.module if isinstance(model, nn.DataParallel) else model
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch["images"], batch["pixel_mask"], batch["references"])
        loss, _ = criterion(outputs, batch["targets"])
        total_loss += loss.item()
        n += 1

        H, W = batch["images"].shape[-2:]
        probs = outputs["pred_logits"].softmax(-1)[..., 1]          # [B, Q]
        masks = torch.nn.functional.interpolate(
            outputs["pred_masks"], size=(H, W), mode="bilinear",
            align_corners=False).sigmoid() > 0.5                    # [B, Q, H, W]
        ref_q = outputs["pred_ref"]
        ref_g = outputs["reference_emb"]
        sim = torch.einsum("bqd,bd->bq", ref_q, ref_g)              # [B, Q]

        for b, tgt in enumerate(batch["targets"]):
            g = tgt["masks"].shape[0]
            if g == 0:
                continue
            keep = probs[b] > 0.5
            pred_masks = masks[b][keep]                             # [k, H, W]
            gt_masks = tgt["masks"].bool()                          # [g, H, W]
            if pred_masks.shape[0] == 0:
                iou_cnt += g
                continue
            # IoU between each GT and each kept prediction.
            pm = pred_masks.flatten(1).float()                     # [k, HW]
            gm = gt_masks.flatten(1).float()                       # [g, HW]
            inter = gm @ pm.t()                                    # [g, k]
            union = gm.sum(1)[:, None] + pm.sum(1)[None, :] - inter
            iou = inter / union.clamp(min=1)
            best_iou, best_pred = iou.max(dim=1)                    # per GT
            iou_sum += best_iou.sum().item()
            iou_cnt += g

            # Reference-match accuracy on the best-matched prediction per GT.
            kept_sim = sim[b][keep][best_pred]                     # [g]
            pred_match = (kept_sim > 0.5).float()
            match_correct += (pred_match == tgt["ref_match"]).sum().item()
            match_total += g

    return {
        "loss": total_loss / max(n, 1),
        "mean_gt_iou": iou_sum / max(iou_cnt, 1),
        "ref_match_acc": match_correct / max(match_total, 1),
    }


def build_optimizer(model, args):
    base = model
    backbone_params, ref_params, other_params = [], [], []
    for name, prm in base.named_parameters():
        if not prm.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(prm)
        elif name.startswith("reference_encoder."):
            ref_params.append(prm)
        else:
            other_params.append(prm)
    groups = [
        {"params": other_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": ref_params, "lr": args.lr * args.backbone_lr_mult},
    ]
    return optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")

    print("\nLoading dataset records (parquet)...")
    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir)
    if args.max_records:
        records = records.select(range(min(args.max_records, len(records))))
    print(f"Total records: {len(records)}")

    train_ds, val_ds = build_datasets(
        records, image_max_size=args.image_max_size, ref_size=args.ref_size,
        train_split=args.train_split)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collate = partial(collate_fn, size_divisible=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)

    print("\nBuilding model...")
    model = RefMask2Former(
        num_queries=args.num_queries, hidden_dim=args.hidden_dim,
        mask_dim=args.mask_dim, ref_dim=args.ref_dim, nheads=args.nheads,
        dec_layers=args.dec_layers, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    matcher = HungarianMatcher(
        cost_class=args.class_weight, cost_mask=args.mask_weight,
        cost_dice=args.dice_weight, num_points=args.num_points)
    weight_dict = {"loss_ce": args.class_weight, "loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight, "loss_ref": args.ref_weight}
    criterion = SetCriterion(matcher, weight_dict, eos_coef=args.eos_coef,
                             num_points=args.num_points).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                          gamma=args.gamma)

    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    start_epoch, best_iou = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        base = model.module if isinstance(model, nn.DataParallel) else model
        base.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_iou = ck.get("best_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    def save(name, epoch):
        base = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"epoch": epoch, "model": base.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_iou": best_iou,
                    "args": vars(args)}, ckpt_dir / name)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for step, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            outputs = model(batch["images"], batch["pixel_mask"], batch["references"])
            loss, parts = criterion(outputs, batch["targets"])

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             ce=f"{parts['loss_ce'].item():.3f}",
                             mask=f"{parts['loss_mask'].item():.3f}",
                             ref=f"{parts['loss_ref'].item():.3f}")
            gstep = epoch * len(train_loader) + step
            writer.add_scalar("train/loss", loss.item(), gstep)

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)

        val = evaluate(model, val_loader, criterion, device)
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val["loss"], epoch)
        writer.add_scalar("epoch/val_mean_gt_iou", val["mean_gt_iou"], epoch)
        writer.add_scalar("epoch/val_ref_match_acc", val["ref_match_acc"], epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"\nEpoch {epoch}: time {time.time()-t0:.0f}s | "
              f"train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"mean_gt_iou {val['mean_gt_iou']:.4f} | "
              f"ref_match_acc {val['ref_match_acc']:.4f}")

        save("last.pth", epoch)
        if val["mean_gt_iou"] > best_iou:
            best_iou = val["mean_gt_iou"]
            save("best.pth", epoch)
            print(f"  -> saved best (mean_gt_iou={best_iou:.4f})")

    writer.close()
    print(f"\nDone. Best mean_gt_iou: {best_iou:.4f}")


if __name__ == "__main__":
    main()
