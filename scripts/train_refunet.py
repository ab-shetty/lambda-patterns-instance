#!/usr/bin/env python3
"""Train one extensible, directly reference-conditioned segmentation model."""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from refmask2former import build_datasets, collate_fn, load_local_records, load_parquet_records
from refmask2former.ref_unet import RefUNet
from scripts.evaluate_refunet_selection import HOLDOUT, evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--schedule-epochs", type=int, default=None,
                   help="Cosine schedule length in epochs; defaults to --epochs. "
                        "Allows an exactly reproducible planned schedule when a "
                        "run is deliberately stopped before it ends.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--image-max-size", type=int, default=1280)
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="Auxiliary hard-pair ranking loss on image-local pattern "
                        "identity. 0 = off, giving the exact baseline model.")
    p.add_argument("--rank-margin", type=float, default=1.0,
                   help="Margin the hardest positive must beat the hardest "
                        "negative by. The query lineage preferred 1.0 over 2.0/4.0.")
    p.add_argument("--metric-dim", type=int, default=128)
    p.add_argument("--anchor", action="store_true",
                   help="Feed WHERE the user drew as a 4th input plane. The "
                        "rectangle always lies inside an instance of the target "
                        "pattern, so it is a guaranteed-positive anchor; only the "
                        "crop was previously passed in.")
    p.add_argument("--anchor-dropout", type=float, default=0.0,
                   help="Fraction of TRAINING samples whose anchor plane is "
                        "hidden, forcing the model to stay able to solve the "
                        "task from the reference crop alone instead of leaning "
                        "on the anchor. Requires --anchor. 0 = off.")
    p.add_argument("--anchor-ref-plane", type=float, default=0.0,
                   help="Value of the anchor channel on the REFERENCE branch. "
                        "0.0 (default) matches the mostly-zero plane the image "
                        "branch sees, so the shared siamese filters get "
                        "consistent statistics. 1.0 was the original and is a "
                        "bug: a constant plane on one branch against a sparse "
                        "box on the other corrupts the reference features and "
                        "costs 0.069 HF14. Only set 1.0 to reproduce that.")
    p.add_argument("--corr-grid", type=int, default=0,
                   help="dense reference correlation: keep the reference as a "
                        "GxG token grid and cosine-match every image location "
                        "against all tokens. 0 = the old globally-averaged "
                        "reference vector only.")
    p.add_argument("--self-support", type=float, default=0.0,
                   help="Self-support prototype refinement (FSS review sec. 3.2, "
                        "Fan et al.): weight of the prototype pooled from the "
                        "first pass's own confident query pixels, mixed into the "
                        "reference prototype for a second decode. 0 = off.")
    p.add_argument("--self-support-thresh", type=float, default=0.7,
                   help="Confidence above which a first-pass pixel is pooled "
                        "into the self-support prototype.")
    p.add_argument("--self-support-aux", type=float, default=0.5,
                   help="Loss weight on the FIRST pass when --self-support is on. "
                        "The second pass depends on the first being roughly "
                        "right, so the first is supervised too.")
    p.add_argument("--dynamic-filter", action="store_true",
                   help="Generate the final 1x1 classifier from the reference "
                        "vector (hypernetwork conditioning, CS231n 2017 / FSS "
                        "conditional networks). Zero-initialised.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-split", type=float, default=0.99)
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=2.0)
    p.add_argument("--domain-random", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop when the monitored metric has not improved for N "
                        "epochs (0 = off). Saves the tail of a long schedule "
                        "once a run has plateaued.")
    p.add_argument("--early-stop-monitor", choices=("val_loss", "hf14"),
                   default="val_loss",
                   help="what patience watches. val_loss is held out from the "
                        "TRAINING pool and touches nothing in the acceptance "
                        "set. hf14 stops on the acceptance metric itself, which "
                        "is selection against it -- screening only, never for a "
                        "reported number.")
    p.add_argument("--compile", action="store_true",
                   help="channels_last + torch.compile. Needs --pad-grid >= 256: "
                        "domain-random gives every sample a unique shape, and "
                        "recompiling per shape is slower than eager (measured "
                        "0.79x with dynamic=True). Bucketed padding is ignored by "
                        "the losses via pixel_mask, so it is loss-neutral, but it "
                        "does enter GroupNorm statistics -- verify parity before "
                        "using a compiled run for a reported number.")
    p.add_argument("--pad-grid", type=int, default=32,
                   help="pad batches up to a multiple of this. 32 = tightest; "
                        "256 gives 16 distinct shapes at 1.31x padding waste.")
    p.add_argument("--realism-aug", action="store_true")
    p.add_argument("--scale-matched-ref", action="store_true",
                   help="crop the reference at the IMAGE's pixel scale instead "
                        "of resizing it to ref-size (median 5.6x magnification).")
    p.add_argument("--init-from")
    p.add_argument("--reset-optimizer", action="store_true",
                   help="On continuation, load model weights and epoch only, then "
                        "start a fresh optimizer and cosine schedule.")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--real-indices", default=HOLDOUT)
    return p.parse_args()


def union_targets(batch, device):
    # Precomputed in the dataloader worker when collate ran with union_only.
    if "union" in batch:
        return batch["union"].to(device, non_blocking=True).float()
    outputs = []
    for target in batch["targets"]:
        positive = target["ref_match"] > 0.5
        if positive.any():
            outputs.append(target["masks"][positive].any(0))
        else:
            outputs.append(torch.zeros_like(target["masks"][0], dtype=torch.bool))
    return torch.stack(outputs).float().unsqueeze(1).to(device)


def ranking_loss(image_embedding, reference_embedding, batch, device, margin):
    """Hard-pair ranking on image-local pattern identity.

    The union BCE+Dice objective never asks whether a region is the same
    MATERIAL as the reference -- only whether the final mask comes out right. The
    query-model lineage found this question to be its single biggest lever
    (ranking margin 2.0 -> 0.5368, margin 1.0 -> 0.5506, the passing result) and
    it was dropped, untested, in the move to RefUNet.

    Each ground-truth instance is pooled in the metric space and scored against
    the reference. Only the hardest pair in each image is penalised: the
    worst-matching instance that SHOULD match must still beat the best-matching
    instance that should NOT, by `margin`. Images with no negative (or no
    positive) carry no signal here and are skipped.
    """
    _, _, h, w = image_embedding.shape
    losses = []
    for i, target in enumerate(batch["targets"]):
        match = target["ref_match"].to(device) > 0.5
        if not match.any() or match.all():
            continue
        masks = target["masks"].to(device).float()[None]
        pooled_masks = F.interpolate(masks, size=(h, w), mode="area")[0]
        area = pooled_masks.flatten(1).sum(1)
        keep = area > 1e-3
        if not (match & keep).any() or not ((~match) & keep).any():
            continue
        pooled = (pooled_masks[:, None] * image_embedding[i][None]).flatten(2).sum(2)
        pooled = F.normalize(pooled / area[:, None].clamp(min=1e-3), dim=1)
        similarity = pooled @ reference_embedding[i]
        hardest_positive = similarity[match & keep].min()
        hardest_negative = similarity[(~match) & keep].max()
        losses.append(F.relu(margin - (hardest_positive - hardest_negative)))
    if not losses:
        return image_embedding.sum() * 0.0
    return torch.stack(losses).mean()


def mask_loss(logits, targets, valid, bce_weight, dice_weight):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    probabilities = logits.sigmoid() * valid
    targets = targets * valid
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice = 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce_weight * bce + dice_weight * dice, bce, dice


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    records = load_local_records(args.local_data)
    train_ds, val_ds = build_datasets(
        records, image_max_size=args.image_max_size, ref_size=args.ref_size,
        train_split=args.train_split, seed=args.seed,
        domain_random=args.domain_random, realism_aug=args.realism_aug,
        scale_matched_ref=args.scale_matched_ref)
    collate = partial(collate_fn, size_divisible=args.pad_grid,
                      union_only=args.rank_weight <= 0)
    loader_options = ({"persistent_workers": True,
                       "prefetch_factor": args.prefetch_factor}
                      if args.num_workers else {})
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
        generator=generator, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
    model = RefUNet(args.width, pretrained=args.init_from is None,
                    corr_grid=args.corr_grid,
                    metric_dim=(args.metric_dim if args.rank_weight > 0 else 0),
                    anchor=args.anchor,
                    anchor_dropout=args.anchor_dropout,
                    anchor_ref_plane=args.anchor_ref_plane,
                    self_support=args.self_support,
                    self_support_thresh=args.self_support_thresh,
                    dynamic_filter=args.dynamic_filter).to(device)
    start_epoch = 0
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("actual_epoch", checkpoint.get("epoch", -1))) + 1
    # `base_model` stays uncompiled: it owns the parameters, so it is what the
    # optimizer, the checkpoint and the per-epoch evaluator use. The evaluator
    # resizes differently from training, so running it through the compiled
    # wrapper would recompile every epoch for no gain.
    base_model = model
    if args.compile:
        if args.pad_grid < 256:
            raise SystemExit("--compile needs --pad-grid >= 256 (see --help)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        base_model = base_model.to(memory_format=torch.channels_last)
        model = torch.compile(base_model)
    # Fused AdamW is one multi-tensor kernel instead of ~160 small launches:
    # measured 23.0 ms -> 2.2 ms per step for clip+step, ~9.5 s/epoch. Same
    # algorithm, but the reduction order differs, so it rides with --compile
    # rather than silently changing the default recipe's last decimals.
    optimizer = torch.optim.AdamW(
        base_model.parameter_groups(args.lr, args.backbone_lr_mult),
        fused=args.compile,
        weight_decay=args.weight_decay)
    schedule_epochs = args.schedule_epochs or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(schedule_epochs * len(train_loader), 1),
        eta_min=args.lr * 0.05)
    if args.init_from and not args.reset_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    real_records = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                       config="real-world-test", split="test")
    real_indices = [int(x) for x in args.real_indices.split(",") if x.strip()]
    checkpoint_dir = Path(args.checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_real = -1.0
    best_monitor, stale = float('-inf'), 0
    print(f"Device: {device}; records={len(records)} train={len(train_ds)} "
          f"val={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,}")
    for local_epoch in range(args.epochs):
        actual_epoch = start_epoch + local_epoch
        started = time.time(); base_model.train(); running = []
        progress = tqdm(train_loader, desc=f"Epoch {actual_epoch} [train]")
        for batch in progress:
            images = batch["images"].to(device, non_blocking=True)
            references = batch["references"].to(device, non_blocking=True)
            if args.compile:
                images = images.to(memory_format=torch.channels_last)
                references = references.to(memory_format=torch.channels_last)
            valid = batch["pixel_mask"].to(device, non_blocking=True)[:, None].float()
            targets = union_targets(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref_boxes = (batch["ref_boxes"].to(device, non_blocking=True)
                             if args.anchor else None)
                auxiliary = None
                if args.rank_weight > 0:
                    logits, image_embedding, reference_embedding = model(
                        images, references, ref_box=ref_boxes,
                        return_embeddings=True)
                elif args.self_support > 0 and args.self_support_aux > 0:
                    logits, auxiliary = model(images, references,
                                              ref_box=ref_boxes, return_aux=True)
                else:
                    logits = model(images, references, ref_box=ref_boxes)
                loss, bce, dice = mask_loss(logits, targets, valid,
                                            args.bce_weight, args.dice_weight)
                if auxiliary is not None:
                    aux_loss, _, _ = mask_loss(auxiliary, targets, valid,
                                               args.bce_weight, args.dice_weight)
                    loss = loss + args.self_support_aux * aux_loss
                if args.rank_weight > 0:
                    rank = ranking_loss(image_embedding, reference_embedding,
                                        batch, device, args.rank_margin)
                    loss = loss + args.rank_weight * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            # Keep the running loss on the GPU. float(loss) forces a device sync,
            # and three of them per iteration serialises CPU and GPU -- which is
            # invisible in eager but becomes the bottleneck once the step itself
            # is compiled. Sync only to refresh the progress bar.
            running.append(loss.detach())
            if len(running) % 20 == 0:
                progress.set_postfix(loss=f"{float(loss):.3f}",
                                     bce=f"{float(bce):.3f}",
                                     dice=f"{float(dice):.3f}")
        base_model.eval(); val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["images"].to(device)
                references = batch["references"].to(device)
                valid = batch["pixel_mask"].to(device)[:, None].float()
                targets = union_targets(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images, references,
                                   ref_box=(batch["ref_boxes"].to(device)
                                            if args.anchor else None))
                    loss, _, _ = mask_loss(logits, targets, valid,
                                           args.bce_weight, args.dice_weight)
                val_losses.append(float(loss))
        rows = evaluate_model(base_model, real_records, real_indices,
                              args.image_max_size, args.ref_size,
                              args.mask_thresh, device,
                              scale_matched_ref=args.scale_matched_ref)
        real_iou = float(np.mean([row["iou"] for row in rows]))
        state = {"model": base_model.state_dict(), "epoch": local_epoch,
                 "actual_epoch": actual_epoch, "args": vars(args),
                 "real_mean_iou": real_iou,
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict()}
        torch.save(state, checkpoint_dir / f"epoch_{actual_epoch}.pth")
        torch.save(state, checkpoint_dir / "last.pth")
        if real_iou > best_real:
            best_real = real_iou; torch.save(state, checkpoint_dir / "best_real.pth")
        metrics = {"actual_epoch": actual_epoch, "real_mean_iou": real_iou,
                   "train_loss": float(torch.stack(running).mean()),
                   "val_loss": float(np.mean(val_losses)), "selections": rows}
        (checkpoint_dir / f"metrics_epoch_{actual_epoch}.json").write_text(
            json.dumps(metrics, indent=2,
                       default=lambda value: (int(value) if isinstance(value, np.integer)
                                              else float(value))) + "\n")
        if args.early_stop_patience:
            score = (-float(np.mean(val_losses))
                     if args.early_stop_monitor == "val_loss" else real_iou)
            if score > best_monitor + 1e-6:
                best_monitor, stale = score, 0
            else:
                stale += 1
        print(f"Epoch {actual_epoch}: train={float(torch.stack(running).mean()):.4f} "
              f"val={np.mean(val_losses):.4f} real_mIoU={real_iou:.6f} "
              f"time={time.time()-started:.0f}s", flush=True)
        if args.early_stop_patience and stale >= args.early_stop_patience:
            print(f"Early stop: {args.early_stop_monitor} has not improved for "
                  f"{stale} epochs (patience {args.early_stop_patience}).",
                  flush=True)
            break


if __name__ == "__main__":
    main()
