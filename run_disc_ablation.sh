#!/usr/bin/env bash
# Synth-only: does DISCONNECTED synthetic elevation data lift real IoU?
# Arms differ only in the connectivity of their 889 elevations (share_local
# 0.396 vs 0.857); roof_plan/freeform records and the labelled-area
# distribution are identical. Two-phase schedule per startup.md.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35"

for SEED in 7 31 99; do
  for ARM in disc conn; do
    CK=data/runs/ck_synthonly_${ARM}_s${SEED}
    DATA=data/synthetic/mix1600_${ARM}
    [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; continue; }
    echo "=== $ARM seed $SEED ==="
    PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
      --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 --seed "$SEED" $COMMON
    PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
      --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 --seed "$SEED" $COMMON \
      --reset-optimizer --init-from "$CK/epoch_0.pth"
  done
done
echo ALLDONE
