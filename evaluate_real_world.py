#!/usr/bin/env python3
"""Evaluate RefMask2Former on the hand-annotated real-world test set.

The synthetic training data (`floz-synth-v5`, default config) is generated; this
script measures generalization to *real* architectural plans. Those 28 excerpts
live in a separate dataset config (`real-world-test`, `test` split) that is
schema-compatible with training, so they flow through the same dataset pipeline.
The whole config is the eval set — there is no train/val split here.

Reports the same reference-conditioned metrics as `evaluate.py`:
  - mean IoU of matched instances
  - precision / recall / F1 at IoU >= 0.5
  - reference-match accuracy of kept predictions
  - class-agnostic detection recall (all instances, ignoring the reference)

Usage:
  python evaluate_real_world.py --checkpoint checkpoints/best.pth
"""

import argparse
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_model, run_eval, print_metrics
from refmask2former import InstanceSegDataset, collate_fn, load_parquet_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hf-repo", type=str, default="abshetty/floz-synth-v5")
    p.add_argument("--config", type=str, default="real-world-test")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--cache-dir", type=str, default="./data")
    p.add_argument("--image-max-size", type=int, default=1024)
    p.add_argument("--ref-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.0)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _seed_worker(_):
    # Reference patches are sampled with the global RNG; reseed each worker so the
    # 28-sample numbers are reproducible run to run.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ck = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    records = load_parquet_records(args.hf_repo, cache_dir=args.cache_dir,
                                   config=args.config, split=args.split)
    print(f"Real-world test set: {len(records)} samples "
          f"({args.config}/{args.split})")

    grayscale = ck.get("args", {}).get("grayscale", False)
    if grayscale:
        print("Checkpoint trained in grayscale mode -> evaluating in grayscale.")

    # Whole config is the eval set; no augmentation, deterministic reference crops.
    eval_ds = InstanceSegDataset(records, range(len(records)),
                                 image_max_size=args.image_max_size,
                                 ref_size=args.ref_size, augment=False,
                                 grayscale=grayscale)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=partial(collate_fn, size_divisible=32),
                        worker_init_fn=_seed_worker, generator=g)

    random.seed(args.seed)
    np.random.seed(args.seed)
    metrics = run_eval(model, loader, device, score_thresh=args.score_thresh,
                       match_thresh=args.match_thresh, iou_thresh=args.iou_thresh,
                       desc="real-world eval")
    print_metrics(metrics, title="Real-world generalization (held-out plans)")


if __name__ == "__main__":
    main()
