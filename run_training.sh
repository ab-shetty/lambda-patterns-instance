#!/usr/bin/env bash
# Train RefMask2Former on the floz-synth-v5 dataset (parquet auto-downloaded).
set -e

# Confirmed-best regime (bs1/grad-accum-4/freeze-backbone-bn) is train.py's
# default; it lifts held-out real_iou +0.03-0.08 vs batch-8 (see synth_progress.md).
# expandable_segments curbs fragmentation; bf16 autocast is on by default.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --hf-repo abshetty/floz-synth-v5 \
    --image-max-size 2048 \
    --epochs 15 \
    --lr 1e-4 \
    --num-workers 16 \
    --prefetch-factor 4 \
    --dec-layers 9 \
    --num-queries 100 \
    "$@"
