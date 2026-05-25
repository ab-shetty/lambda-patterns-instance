#!/usr/bin/env bash
# Train RefMask2Former on the floz-synth-v5 dataset (parquet auto-downloaded).
set -e

python train.py \
    --hf-repo abshetty/floz-synth-v5 \
    --image-max-size 1024 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --num-workers 8 \
    --dec-layers 9 \
    --num-queries 100 \
    "$@"
