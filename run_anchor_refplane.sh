#!/usr/bin/env bash
# Is the anchor's damage caused by the all-ones REFERENCE plane, not a shortcut?
#
# The probe showed the anchored model is NOT riding the anchor (pred_local 0.282
# vs gt_local 0.266, follow_anchor 0.028) and its reference pathway is still
# fully causal (ref_sens 0.918) -- it is simply much worse (0.611 -> 0.276).
# That rules out feature suppression and points at the siamese input mismatch:
# the shared stem sees a sparse box on the image branch and a constant 1.0 on
# the reference branch. This arm sets the reference plane to 0.0 instead.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --anchor --anchor-ref-plane 0.0"

for SEED in 7 31 99; do
  CK=data/runs/ck_synthonly_disc_anchorzero_s${SEED}
  DATA=data/synthetic/mix1600_disc
  [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; continue; }
  echo "=== disc anchor-refplane0 seed $SEED ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 --seed "$SEED" $COMMON
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 --seed "$SEED" $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
done
echo REFPLANE_DONE
