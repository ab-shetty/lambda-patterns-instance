#!/usr/bin/env bash
# Unfrozen backbone A/B: the one thing the frozen-feature probe cannot test.
#
# The 2026-09-20 probe ranked backbones on CACHED features, so every number it
# produced holds the backbone fixed. Production finetunes it at
# --backbone-lr-mult 0.1, and "a vision transformer needs more data" is a
# statement about finetuning, not about a frozen ImageNet representation. This
# runs the same pool, resolution, schedule and decoder against both backbones
# with the backbone TRAINING, which is the only way to settle it.
#
# swin_t (31.4M total) is deliberately the size-matched arm against RefUNet's
# 28.0M -- swin_b is 90.8M and would confound architecture with capacity.
#
# Reference point: v6d-only 1,600 @2048 scored 0.6400 (1 seed, startup.md).
set -euo pipefail
POOL=${1:-data/synthetic/v6d_train2000}
SEED=${2:-7}
SIZE=${3:-2048}
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for M in unet swin_t; do
  CK=data/runs/ck_ab_${M}_res${SIZE}_seed${SEED}
  TAG=$(basename "$CK")
  COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
    --image-max-size $SIZE --ref-size 224 --width 128 --model $M \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
  # Documented two-phase schedule: one epoch of a planned ten, then reset the
  # optimizer and run eight of a planned nine.
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON \
    > "logs/${TAG}_p1.log" 2>&1
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth" \
    > "logs/${TAG}_p2.log" 2>&1
  echo "== $TAG trained"
done
