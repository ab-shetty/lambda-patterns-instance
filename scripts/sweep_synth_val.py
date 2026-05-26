#!/usr/bin/env python3
"""Reference-conditioned metrics on synthetic val across match thresholds."""
import argparse
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_model
from refmask2former import build_datasets, collate_fn, load_parquet_records
from sweep_thresh import cache_outputs, eval_thresholds


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
    return p.parse_args()


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
    print(f"Caching synth-val outputs over {n} images...")
    cache = cache_outputs(model, loader, device)

    it = args.iou_thresh
    score = 0.5
    print(f"\n=== Synthetic val, reference-conditioned (score_thresh={score}) ===")
    print(f"{'match':>6} | {'matchedIoU':>10} {'precision':>9} {'recall':>7} {'F1':>7}")
    for mt in [0.5, 0.2, 0.0, -0.1, -0.3]:
        m = eval_thresholds(cache, score, mt, it)
        print(f"{mt:>6.2f} | {m['mean_iou']:>10.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f}")

    # Class-agnostic detection recall ceiling for reference.
    m = eval_thresholds(cache, score, -10.0, it)
    print(f"\nClass-agnostic detection recall (ceiling): {m['det_recall']:.4f}")


if __name__ == "__main__":
    main()
