#!/usr/bin/env python3
"""Train a class-agnostic pretrained Mask R-CNN on local Floz records.

This is an architecture-level control for the lightweight query decoder in
``train.py``.  It consumes the same local ``images/`` + ``annotations/`` data,
trains every pattern instance as one foreground class, and evaluates the same
per-GT best-mask IoU on the fixed Hugging Face real holdout.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from refmask2former import load_local_records, load_parquet_records
from refmask2former.dataset import render_instance_mask


HOLDOUT = "12,16,27,7,11,25,23,1,18,2,0,3,14,24"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--log", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--real-eval-indices", default=HOLDOUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint to load")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tta-flips", action="store_true",
                   help="evaluate original + horizontal + vertical flip views")
    p.add_argument("--mask-nms-thresh", type=float, default=0.0,
                   help="deduplicate kept predictions by binary-mask IoU; 0 disables")
    return p.parse_args()


class FlozDetectionDataset(Dataset):
    def __init__(self, records, indices, max_size=1024, augment=False):
        self.records = records
        self.indices = list(indices)
        self.max_size = max_size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _load_record(rec):
        if "image_path" in rec:
            image = np.array(Image.open(rec["image_path"]).convert("RGB"))
        else:
            image = np.array(Image.open(io.BytesIO(rec["image"])).convert("RGB"))
        anns = rec["annotations"]
        if isinstance(anns, str):
            anns = json.loads(anns)
        return image, anns

    def __getitem__(self, pos):
        image, anns = self._load_record(self.records[self.indices[pos]])
        h0, w0 = image.shape[:2]
        masks = [render_instance_mask(a["segmentation"], h0, w0) for a in anns]

        scale = self.max_size / max(h0, w0)
        nh, nw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if masks:
            masks = np.stack([cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                              for m in masks])
        else:
            masks = np.zeros((0, nh, nw), dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1]
                masks = masks[:, :, ::-1]
            if random.random() < 0.25:
                image = image[::-1]
                masks = masks[:, ::-1]
            if random.random() < 0.8:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-12, 12)
                image = np.clip(image.astype(np.float32) * gain + bias,
                                0, 255).astype(np.uint8)

        image = np.ascontiguousarray(image)
        masks = np.ascontiguousarray(masks)
        keep_masks, boxes = [], []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if len(xs) < 4:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if x1 <= x0 or y1 <= y0:
                continue
            keep_masks.append(mask)
            boxes.append([x0, y0, x1 + 1, y1 + 1])

        if keep_masks:
            mask_t = torch.from_numpy(np.stack(keep_masks)).to(torch.uint8)
            box_t = torch.tensor(boxes, dtype=torch.float32)
        else:
            mask_t = torch.zeros((0, nh, nw), dtype=torch.uint8)
            box_t = torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": box_t,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": mask_t,
            "image_id": torch.tensor(self.indices[pos]),
        }
        image_t = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return image_t, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained, image_size):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights, min_size=image_size, max_size=image_size,
        box_detections_per_img=100, box_score_thresh=0.05,
    )
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, score_thresh, mask_thresh=0.5,
             return_per_image=False, tta_flips=False, mask_nms_thresh=0.0):
    model.eval()
    total_iou = 0.0
    total_gt = 0
    found = 0
    oracle_iou = 0.0
    per_image = []
    strict_iou_sum = 0.0
    strict_tp = strict_fp = strict_fn = 0
    pq_iou_sum = 0.0
    sem_fg_inter = sem_fg_union = 0
    sem_bg_inter = sem_bg_union = 0
    for images, targets in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        if tta_flips:
            outputs = []
            for image in images:
                views = [image, image.flip(-1), image.flip(-2)]
                view_outputs = model(views)
                masks = [view_outputs[0]["masks"],
                         view_outputs[1]["masks"].flip(-1),
                         view_outputs[2]["masks"].flip(-2)]
                outputs.append({
                    "masks": torch.cat(masks),
                    "scores": torch.cat([o["scores"] for o in view_outputs]),
                })
        else:
            outputs = model(images)
        for out, target in zip(outputs, targets):
            gt = target["masks"].to(device).bool()
            g = gt.shape[0]
            if not g:
                continue
            pred = out["masks"][:, 0] > mask_thresh
            scores = out["scores"]
            total_gt += g

            selected = torch.where(scores > score_thresh)[0]
            if mask_nms_thresh > 0 and len(selected):
                order = selected[scores[selected].argsort(descending=True)]
                retained = []
                flat = pred.flatten(1).float()
                areas = flat.sum(1)
                for candidate in order:
                    if retained:
                        prior = torch.stack(retained)
                        inter = flat[prior] @ flat[candidate]
                        union = areas[prior] + areas[candidate] - inter
                        if (inter / union.clamp(min=1)).max() > mask_nms_thresh:
                            continue
                    retained.append(candidate)
                selected = torch.stack(retained) if retained else selected[:0]

            def best_for(candidate_masks):
                if candidate_masks.shape[0] == 0:
                    return torch.zeros(g, device=device)
                pm = candidate_masks.flatten(1).float()
                gm = gt.flatten(1).float()
                inter = gm @ pm.t()
                union = gm.sum(1)[:, None] + pm.sum(1)[None] - inter
                return (inter / union.clamp(min=1)).max(1).values

            kept_pred = pred[selected]
            best = best_for(kept_pred)
            oracle = best_for(pred)
            total_iou += best.sum().item()
            oracle_iou += oracle.sum().item()
            found += (best >= 0.5).sum().item()

            if len(kept_pred):
                gm = gt.flatten(1).float()
                pm = kept_pred.flatten(1).float()
                inter = gm @ pm.t()
                union = gm.sum(1)[:, None] + pm.sum(1)[None] - inter
                matrix = inter / union.clamp(min=1)
                rows, cols = linear_sum_assignment(-matrix.cpu().numpy())
                paired = matrix[rows, cols]
                strict_iou_sum += paired.sum().item()
                tp = int((paired >= 0.5).sum())
            else:
                tp = 0
            strict_tp += tp
            strict_fp += len(kept_pred) - tp
            strict_fn += g - tp
            if len(kept_pred):
                pq_iou_sum += paired[paired >= 0.5].sum().item()

            gt_union = gt.any(0)
            pred_union = (kept_pred.any(0) if len(kept_pred)
                          else torch.zeros_like(gt_union))
            sem_fg_inter += int((gt_union & pred_union).sum())
            sem_fg_union += int((gt_union | pred_union).sum())
            sem_bg_inter += int((~gt_union & ~pred_union).sum())
            sem_bg_union += int((~gt_union | ~pred_union).sum())
            per_image.append({
                "image_id": int(target["image_id"]), "n_gt": g,
                "mean_gt_iou": best.mean().item(),
                "oracle_iou": oracle.mean().item(),
                "n_kept": len(kept_pred),
            })
    result = {
        "mean_gt_iou": total_iou / max(total_gt, 1),
        "oracle_iou": oracle_iou / max(total_gt, 1),
        "recall50": found / max(total_gt, 1),
        "n_gt": total_gt,
        "one_to_one_miou": strict_iou_sum / max(total_gt, 1),
        "precision50": strict_tp / max(strict_tp + strict_fp, 1),
        "one_to_one_recall50": strict_tp / max(strict_tp + strict_fn, 1),
        "f1_50": (2 * strict_tp /
                  max(2 * strict_tp + strict_fp + strict_fn, 1)),
        "n_predictions": strict_tp + strict_fp,
        "panoptic_quality50": (pq_iou_sum /
                               max(strict_tp + 0.5 * strict_fp +
                                   0.5 * strict_fn, 1)),
        "semantic_fg_iou": sem_fg_inter / max(sem_fg_union, 1),
        "semantic_bg_iou": sem_bg_inter / max(sem_bg_union, 1),
        "semantic_binary_miou": 0.5 * (
            sem_fg_inter / max(sem_fg_union, 1) +
            sem_bg_inter / max(sem_bg_union, 1)),
    }
    if return_per_image:
        result["per_image"] = per_image
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    train_records = load_local_records(args.local_data)
    heldout_records = load_parquet_records(
        "abshetty/floz-synth-v5", cache_dir="./data",
        config="real-world-test", split="test")
    heldout_idx = [int(x) for x in args.real_eval_indices.split(",") if x.strip()]
    train_ds = FlozDetectionDataset(train_records, range(len(train_records)),
                                    args.image_max_size, augment=True)
    eval_ds = FlozDetectionDataset(heldout_records, heldout_idx,
                                   args.image_max_size, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=2, persistent_workers=True,
        pin_memory=True, collate_fn=collate, generator=generator)
    eval_loader = DataLoader(
        eval_ds, batch_size=1, shuffle=False, num_workers=min(8, args.num_workers),
        persistent_workers=True, pin_memory=True, collate_fn=collate)

    model = build_model(not args.no_pretrained, args.image_max_size).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded {args.resume} (epoch {checkpoint.get('epoch', '?')})", flush=True)
    if args.eval_only:
        metrics = evaluate(model, eval_loader, device, args.score_thresh,
                           args.mask_thresh, return_per_image=True,
                           tta_flips=args.tta_flips,
                           mask_nms_thresh=args.mask_nms_thresh)
        print(json.dumps(metrics, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    global_step = 0
    print(f"device={device} train={len(train_ds)} heldout={len(eval_ds)} "
          f"batch={args.batch_size} workers={args.num_workers}", flush=True)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
                loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item()
            if step % 50 == 0:
                pieces = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.3f} {pieces}", flush=True)

        metrics = evaluate(model, eval_loader, device, args.score_thresh,
                           args.mask_thresh)
        elapsed = time.time() - start
        print(f"Epoch {epoch}: time {elapsed:.0f}s train_loss "
              f"{running/max(len(train_loader),1):.4f} real_iou "
              f"{metrics['mean_gt_iou']:.4f} oracle {metrics['oracle_iou']:.4f} "
              f"recall50 {metrics['recall50']:.4f}", flush=True)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "metrics": metrics, "args": vars(args),
        }
        torch.save(state, out_dir / "last.pth")
        if metrics["mean_gt_iou"] > best:
            best = metrics["mean_gt_iou"]
            torch.save(state, out_dir / "best_real.pth")
            print(f"  -> saved best_real ({best:.4f})", flush=True)
    print(f"Done. Best real mean_gt_iou: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
