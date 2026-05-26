#!/usr/bin/env python3
"""Compute the EXACT training-val metric (mean_gt_iou: per-GT best IoU vs kept
prob>0.5 preds, missed GTs counting 0, reference-independent) on the real set,
so synth-val IoU and real IoU are the same yardstick."""
import argparse
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import load_model
from refmask2former import InstanceSegDataset, collate_fn, load_parquet_records


def real_mean_gt_iou(model, loader, device, score_thresh=0.5):
    iou_sum, iou_cnt, det_found = 0.0, 0, 0
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
                    iou_cnt += g
                    continue
                pm = masks[b][keep].flatten(1).float(); gm = gt.flatten(1).float()
                inter = gm @ pm.t()
                union = gm.sum(1)[:, None] + pm.sum(1)[None, :] - inter
                iou = inter / union.clamp(min=1)
                best = iou.max(1).values
                iou_sum += best.sum().item(); iou_cnt += g
                det_found += (best >= 0.5).sum().item()
    return iou_sum / max(iou_cnt, 1), det_found / max(iou_cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-max-size", type=int, default=2048)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, ck = load_model(a.checkpoint, device)
    recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                                config="real-world-test", split="test")
    ds = InstanceSegDataset(recs, range(len(recs)), image_max_size=a.image_max_size,
                            augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=partial(collate_fn, size_divisible=32))
    miou, drec = real_mean_gt_iou(model, loader, device)
    print(f"{a.checkpoint} (epoch {ck.get('epoch','?')})")
    print(f"  REAL mean_gt_iou (same metric as synth val): {miou:.4f}")
    print(f"  REAL class-agnostic recall@0.5:              {drec:.4f}")


if __name__ == "__main__":
    main()
