"""Per-image diagnostic on the 28 held-out real plans.

The training metric `mean_gt_iou` is recall-of-best-prediction per GT, so it is
BLIND to over-prediction. This script also reports the predicted-instance count
per image (preds with class-prob > thr), exposing the real failure mode
(fragmentation / false-positive spray: model fires ~31 preds for ~5 GT).

Usage:
  PYTHONPATH=. python scripts/per_image_real.py --ckpt /tmp/ck_v60/best.pth \
      [--image-max-size 1024] [--score-thr 0.5]
"""
import argparse
import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader

from refmask2former import (RefMask2Former, collate_fn, load_parquet_records,
                            InstanceSegDataset)
from train import move_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf-repo", default="abshetty/floz-synth-v5")
    ap.add_argument("--cache-dir", default="./data")
    ap.add_argument("--real-config", default="real-world-test")
    ap.add_argument("--real-split", default="test")
    ap.add_argument("--image-max-size", type=int, default=1024)
    ap.add_argument("--ref-size", type=int, default=224)
    ap.add_argument("--score-thr", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                config=args.real_config, split=args.real_split)
    ds = InstanceSegDataset(recs, range(len(recs)),
                            image_max_size=args.image_max_size,
                            ref_size=args.ref_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=partial(collate_fn, size_divisible=32))

    model = RefMask2Former().to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = []
    tot_iou = tot_gt = tot_pred = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["images"], batch["pixel_mask"],
                            batch["references"])
            H, W = batch["images"].shape[-2:]
            probs = out["pred_logits"].softmax(-1)[..., 1][0]          # [Q]
            keep = probs > args.score_thr
            masks = (torch.nn.functional.interpolate(
                out["pred_masks"], size=(H, W), mode="bilinear",
                align_corners=False).sigmoid()[0] > 0.5)[keep]          # [k,H,W]
            tgt = batch["targets"][0]
            g = tgt["masks"].shape[0]
            pred_n = int(keep.sum())
            if g == 0:
                continue
            if pred_n == 0:
                iou = 0.0
            else:
                pm = masks.flatten(1).float()
                gm = tgt["masks"].bool().flatten(1).float()
                inter = gm @ pm.t()
                union = gm.sum(1)[:, None] + pm.sum(1)[None, :] - inter
                iou = (inter / union.clamp(min=1)).max(dim=1)[0].mean().item()
            rows.append((i, iou, g, pred_n))
            tot_iou += iou; tot_gt += g; tot_pred += pred_n

    n = len(rows)
    print(f"\nckpt={args.ckpt}  size={args.image_max_size}  thr={args.score_thr}")
    print(f"{'idx':>3} {'iou':>6} {'GT':>3} {'pred':>5}")
    for i, iou, g, p in sorted(rows, key=lambda r: r[1]):
        print(f"{i:>3} {iou:>6.3f} {g:>3} {p:>5}")
    print(f"\nMEAN mean_gt_iou={tot_iou/n:.4f}  mean_GT={tot_gt/n:.2f}  "
          f"mean_pred={tot_pred/n:.2f}  over-pred={tot_pred/max(tot_gt,1):.2f}x  "
          f"(n={n})")


if __name__ == "__main__":
    main()
