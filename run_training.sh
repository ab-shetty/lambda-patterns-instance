#!/usr/bin/env bash
# Train RefMask2Former on the floz-synth-v5 dataset (parquet auto-downloaded).
set -e

# bf16 autocast roughly halves activation memory, so batch 8 fits at 2048px
# (fp32 batch 8 OOMs on a 94.5GB GH200). expandable_segments curbs fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --hf-repo abshetty/floz-synth-v5 \
    --image-max-size 2048 \
    --batch-size 8 \
    --epochs 15 \
    --lr 1e-4 \
    --num-workers 16 \
    --prefetch-factor 4 \
    --dec-layers 9 \
    --num-queries 100 \
    "$@"
