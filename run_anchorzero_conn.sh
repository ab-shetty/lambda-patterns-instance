#!/usr/bin/env bash
# Second half of the fixed-anchor 2x2: does a WORKING anchor (reference plane
# 0.0) add anything over no anchor, on the connected pool as well as disc?
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --anchor --anchor-ref-plane 0.0"
for SEED in 7 31 99; do
  CK=data/runs/ck_synthonly_conn_anchorzero_s${SEED}
  [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; continue; }
  echo "=== conn anchor-refplane0 seed $SEED ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data data/synthetic/mix1600_conn \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 --seed "$SEED" $COMMON
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data data/synthetic/mix1600_conn \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 --seed "$SEED" $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
done
echo CONNZERO_DONE
