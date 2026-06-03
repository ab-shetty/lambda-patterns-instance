#!/usr/bin/env python3
"""Train RefMask2Former: reference-conditioned instance segmentation.

Detects every distinct pattern instance (class-agnostic) and learns a
reference-matching embedding so a user-selected reference patch picks out the
instances of that pattern at inference time.
"""

import argparse
import math
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
    build_datasets, collate_fn, load_parquet_records, load_local_records,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train RefMask2Former instance segmentation")
    # Data
    p.add_argument("--hf-repo", type=str, default="abshetty/floz-synth-v5")
    p.add_argument("--local-data", type=str, default=None,
                   help="Train on a local generator output folder (images/ + "
                        "annotations/) instead of the HF parquet. For fast "
                        "generator-iteration on a small batch.")
    p.add_argument("--cache-dir", type=str, default="./data")
    p.add_argument("--real-eval", action="store_true",
                   help="Also evaluate on the real-world-test config every epoch "
                        "and log the synth-val<->real divergence (the metric that "
                        "tells us whether synthetic is a faithful proxy for real).")
    p.add_argument("--real-config", type=str, default="real-world-test")
    p.add_argument("--real-split", type=str, default="test")
    p.add_argument("--image-max-size", type=int, default=1024,
                   help="Longest image side after aspect-preserving resize")
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--grayscale", action="store_true",
                   help="Feed luminance-only (3-channel grayscale) images to the "
                        "model for synth-train, synth-val AND real-eval. Removes "
                        "colour as a synth<->real domain-discriminating signal so "
                        "both sides share the same appearance.")
    p.add_argument("--realism-aug", action="store_true",
                   help="Train-time degradation (blur/noise/JPEG/contrast) on synth "
                        "scene+reference to mimic real PDF-export rasterization, "
                        "which is softer/noisier than crisp synth renders.")
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--max-records", type=int, default=0,
                   help="Limit number of records (0 = all). Useful for quick runs.")
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="accumulate grads over N micro-batches before stepping; "
                        "effective batch = batch-size * grad-accum. Lets a 12GB "
                        "GPU train at 2048px/bs1 with an effective batch of 8.")
    p.add_argument("--freeze-backbone-bn", action="store_true",
                   help="freeze backbone BatchNorm (eval running stats, no grad). "
                        "Standard for detection; required for valid bs1/grad-accum "
                        "since per-forward BN over 1 image is noise.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05,
                   help="Fraction of total steps for linear LR warmup")
    p.add_argument("--min-lr-frac", type=float, default=0.0,
                   help="Floor of the cosine decay as a fraction of base LR")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches each worker prefetches (num-workers > 0 only)")
    p.add_argument("--clip-grad", type=float, default=0.01)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable bf16 autocast (train in full fp32)")
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
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start MODEL weights from a checkpoint but start a "
                        "fresh run (new optimizer/scheduler, epoch 0). Use to "
                        "fine-tune the current model on a new data batch.")
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


def _roc_auc(scores, labels):
    """ROC-AUC via the Mann-Whitney U statistic (ties get averaged ranks)."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # average ranks within tied groups
    uniq, inv = torch.unique(scores, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(0, inv, ranks)
    cnts = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(
        0, inv, torch.ones_like(ranks))
    ranks = (sums / cnts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


@torch.no_grad()
def freeze_backbone_bn(model):
    """Put every BatchNorm in the backbone into eval mode and freeze its affine
    params, so it uses fixed pretrained running stats regardless of batch size."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    for m in base.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def set_backbone_bn_eval(model):
    """Re-assert eval() on backbone BN after a model.train() call (which would
    otherwise flip them back to training mode)."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    for m in base.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, n = 0.0, 0
    iou_sum, iou_cnt = 0.0, 0
    match_sims, match_labels = [], []

    base = model.module if isinstance(model, nn.DataParallel) else model
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
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

            # Matching quality, threshold-free: collect (sim, label) for the
            # best-matched prediction of each *well-detected* GT (IoU >= 0.5), so
            # the metric measures separability, not where a hard cutoff happens
            # to fall and not detection misses. Reported as ROC-AUC below.
            ok = best_iou >= 0.5
            if ok.any():
                match_sims.append(sim[b][keep][best_pred][ok].float().cpu())
                match_labels.append(tgt["ref_match"][ok].cpu())

    sims = torch.cat(match_sims) if match_sims else torch.zeros(0)
    labels = torch.cat(match_labels) if match_labels else torch.zeros(0)
    return {
        "loss": total_loss / max(n, 1),
        "mean_gt_iou": iou_sum / max(iou_cnt, 1),
        "ref_match_auc": _roc_auc(sims, labels),
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


def build_lr_scheduler(optimizer, total_steps, warmup_frac, min_lr_frac):
    """Per-iteration linear warmup then cosine decay, scheduled over the run's
    total optimizer steps. Returns a multiplier on each param group's base LR,
    so warmup/decay apply proportionally to the decoder and the (×0.1) backbone.
    Schedules by total steps, so it adapts to any epochs/batch/dataset size."""
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")
    print(f"Mixed precision (bf16 autocast): {use_amp}")

    if args.local_data:
        print(f"\nLoading local generator output from {args.local_data} ...")
        records = load_local_records(args.local_data)
    else:
        print("\nLoading dataset records (parquet)...")
        records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir)
    if args.max_records:
        sel = range(min(args.max_records, len(records)))
        records = records[:args.max_records] if isinstance(records, list) \
            else records.select(sel)
    print(f"Total records: {len(records)}")

    train_ds, val_ds = build_datasets(
        records, image_max_size=args.image_max_size, ref_size=args.ref_size,
        train_split=args.train_split, grayscale=args.grayscale,
        realism_aug=args.realism_aug)
    if args.grayscale:
        print("Grayscale mode: feeding luminance-only 3-channel images "
              "(synth-train, synth-val, real-eval).")
    if args.realism_aug:
        print("Realism aug: degrading synth train images toward real "
              "PDF-export appearance (blur/noise/JPEG/contrast).")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collate = partial(collate_fn, size_divisible=32)
    # persistent_workers keeps the (expensive to spawn) workers alive across epochs;
    # prefetch_factor deepens each worker's queue so the GPU is less likely to wait
    # on the 2048px decode+resize. Both require num_workers > 0.
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs = {"persistent_workers": True,
                         "prefetch_factor": args.prefetch_factor}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate, **loader_kwargs)

    real_loader = None
    if args.real_eval:
        from refmask2former import InstanceSegDataset
        real_recs = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                         config=args.real_config, split=args.real_split)
        real_ds = InstanceSegDataset(real_recs, range(len(real_recs)),
                                     image_max_size=args.image_max_size,
                                     ref_size=args.ref_size, augment=False,
                                     grayscale=args.grayscale)
        real_loader = DataLoader(real_ds, batch_size=1, shuffle=False,
                                 num_workers=2, collate_fn=collate)
        print(f"Real-eval set: {len(real_ds)} held-out real plans "
              f"({args.real_config}/{args.real_split})")

    print("\nBuilding model...")
    model = RefMask2Former(
        num_queries=args.num_queries, hidden_dim=args.hidden_dim,
        mask_dim=args.mask_dim, ref_dim=args.ref_dim, nheads=args.nheads,
        dec_layers=args.dec_layers, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"Warm-started model weights from {args.init_from} "
              f"(epoch {ck.get('epoch','?')}); fresh optimizer + epoch 0")

    if args.freeze_backbone_bn:
        # Standard DETR/Mask2Former choice: keep the pretrained ImageNet BN
        # running stats fixed instead of re-estimating them from tiny detection
        # batches. Essential when grad-accum forces bs1 forwards (per-forward BN
        # over 1 image is pure noise). Params frozen here so build_optimizer
        # (which filters requires_grad) excludes them.
        freeze_backbone_bn(model)
        print("Froze backbone BatchNorm (eval mode + no grad) — batch-size-"
              "independent backbone, valid under grad-accum / bs1.")

    matcher = HungarianMatcher(
        cost_class=args.class_weight, cost_mask=args.mask_weight,
        cost_dice=args.dice_weight, num_points=args.num_points)
    weight_dict = {"loss_ce": args.class_weight, "loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight, "loss_ref": args.ref_weight}
    criterion = SetCriterion(matcher, weight_dict, eos_coef=args.eos_coef,
                             num_points=args.num_points).to(device)

    optimizer = build_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_lr_scheduler(optimizer, total_steps, args.warmup_frac,
                                   args.min_lr_frac)

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
        if args.freeze_backbone_bn:
            set_backbone_bn_eval(model)
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        accum = max(1, args.grad_accum)
        optimizer.zero_grad()
        n_steps = len(train_loader)
        for step, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(batch["images"], batch["pixel_mask"], batch["references"])
                loss, parts = criterion(outputs, batch["targets"])

            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == n_steps:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             ce=f"{parts['loss_ce'].item():.3f}",
                             mask=f"{parts['loss_mask'].item():.3f}",
                             ref=f"{parts['loss_ref'].item():.3f}")
            gstep = epoch * len(train_loader) + step
            writer.add_scalar("train/loss", loss.item(), gstep)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], gstep)

        train_loss = running / max(len(train_loader), 1)

        val = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val["loss"], epoch)
        writer.add_scalar("epoch/val_mean_gt_iou", val["mean_gt_iou"], epoch)
        writer.add_scalar("epoch/val_ref_match_auc", val["ref_match_auc"], epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        msg = (f"\nEpoch {epoch}: time {time.time()-t0:.0f}s | "
               f"train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} | "
               f"synth_iou {val['mean_gt_iou']:.4f} | "
               f"ref_match_auc {val['ref_match_auc']:.4f}")
        if real_loader is not None:
            real = evaluate(model, real_loader, criterion, device, use_amp=use_amp)
            div = val["mean_gt_iou"] - real["mean_gt_iou"]
            writer.add_scalar("epoch/real_mean_gt_iou", real["mean_gt_iou"], epoch)
            writer.add_scalar("epoch/synth_real_divergence", div, epoch)
            msg += (f" || real_iou {real['mean_gt_iou']:.4f} | "
                    f"DIVERGENCE {div:+.4f}")
        print(msg)

        save("last.pth", epoch)
        if val["mean_gt_iou"] > best_iou:
            best_iou = val["mean_gt_iou"]
            save("best.pth", epoch)
            print(f"  -> saved best (mean_gt_iou={best_iou:.4f})")

    writer.close()
    print(f"\nDone. Best mean_gt_iou: {best_iou:.4f}")


if __name__ == "__main__":
    main()
