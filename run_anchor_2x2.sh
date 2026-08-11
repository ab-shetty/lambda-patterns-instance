#!/usr/bin/env bash
# Does disconnected synthetic data make the ANCHOR usable?
#
# Adding the reference rectangle as a 4th input plane previously COST 0.247 on
# HF14 (0.680 -> 0.433) even though the plane is zero-initialised and could have
# been ignored. Suspected cause: the anchor settles ~84% of the training pool by
# itself, so loss falls down that path and the reference-matching pathway is
# starved of gradient. Two independent attacks on that:
#   - disc data: the anchor's component holds only 45% of the target, so leaning
#     on it is penalised rather than rewarded;
#   - anchor dropout: hide the anchor on half of training samples so the model
#     must stay able to work without it.
# The no-anchor halves of the 2x2 are already done (ck_synthonly_{disc,conn}_s*).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35"

for SEED in 7 31 99; do
  for ARM in disc conn; do
    for COND in anchor anchordrop; do
      EXTRA="--anchor"
      [ "$COND" = "anchordrop" ] && EXTRA="--anchor --anchor-dropout 0.5"
      CK=data/runs/ck_synthonly_${ARM}_${COND}_s${SEED}
      DATA=data/synthetic/mix1600_${ARM}
      [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; continue; }
      echo "=== $ARM $COND seed $SEED ==="
      PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
        --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 --seed "$SEED" $COMMON $EXTRA
      PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
        --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 --seed "$SEED" $COMMON $EXTRA \
        --reset-optimizer --init-from "$CK/epoch_0.pth"
    done
  done
done
echo ALLDONE
