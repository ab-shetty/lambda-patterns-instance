#!/usr/bin/env bash
# Can ANY of these architectures hold a high hard train IoU on 100k v6d plans?
# Three arms, one matched step budget, ~1 h each instead of 20.
#
# WHY THIS IS THE RIGHT SHAPE (2026-09-20, learned the hard way).
#
# Train IoU here is NOT memorisation. `--domain-random` gives every sample a
# fresh view, so train ~= fresh (0.8022 vs 0.7980 at 100k on record) and the
# number is really "IoU on the generator's distribution". That has two
# consequences that make the cheap version valid:
#
#   * More unique data at a MATCHED step budget RAISES train IoU rather than
#     lowering it (measured: 2,000 plans -> 0.5400, 10,000 plans -> 0.6954, both
#     at 7,500 steps). The model is learning a function, not memorising items.
#   * So the architecture's plateau is set by STEPS, not by how much of the pool
#     it has seen. A single pass over 100k at a fixed step budget is enough to
#     rank architectures.
#
# The budget is 25,000 steps because that is what produced the RefUNet number
# already on record (100k, 2048, train 0.8022) -- so `unet` here is a control
# that should land near it, and an arm that does not is a bug, not a finding.
#
# DO NOT substitute a few hundred images with augmentation off. That is how
# this repo briefly convinced itself of "0.9866 train IoU, capacity is not the
# limit", which was a CNN memorising a static pool and is retracted.
#
# Run at 1024 for the RANKING; resolution moves the absolute number a lot
# (0.6811 at 2048 vs 0.5400 at 1024 on the same pool at different budgets).
#
#   ./run_fit_100k.sh            # all three arms
#   ./run_fit_100k.sh swin_t     # one arm
set -euo pipefail
ARMS=${*:-unet swin_t swin_b}
SEED=${SEED:-7}
SIZE=${SIZE:-1024}
STEPS=${STEPS:-25000}
POOL=data/synthetic/v6d_100000
mkdir -p logs data/evaluations

if [ ! -f "$POOL/generation_manifest.json" ]; then
  echo "== generating 100,000 v6d plans (~13 min at 0.01 s/img on 48 cores)"
  python3 generate_synthetic_v6.py --n 100000 --out "$POOL" \
    --seed 6 --start 0 --workers 48 > logs/gen_100k.log 2>&1
fi
N=$(ls "$POOL/annotations" | wc -l)
# batch 8, so one pass is N/8 steps; pick epochs to hit the step budget.
EPOCHS=$(python3 -c "import math;print(max(1,round($STEPS/($N/8))))")
echo "== pool $N plans, batch 8 -> $((N/8)) steps/epoch, $EPOCHS epoch(s) ~= $STEPS steps"

for M in $ARMS; do
  CK=data/runs/ck_100k_${M}_res${SIZE}_seed${SEED}
  if [ ! -f "$CK/epoch_$((EPOCHS-1)).pth" ]; then
    echo "== training $M"
    PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
      --checkpoint-dir "$CK" --epochs "$EPOCHS" --schedule-epochs "$EPOCHS" \
      --batch-size 8 --num-workers 32 --prefetch-factor 6 \
      --image-max-size $SIZE --ref-size 224 --width 128 --model "$M" \
      --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.995 \
      --domain-random --mask-thresh 0.35 --seed "$SEED" --pad-grid 512 \
      > "logs/fit100k_${M}.log" 2>&1
  fi
  echo "== $M hard train IoU (threshold 0.35, NOT the soft-dice training log):"
  PYTHONPATH=. python3 scripts/fresh_synth_iou.py \
    --checkpoint "$CK/epoch_$((EPOCHS-1)).pth" --seen \
    --all-pool "$POOL" --trained-pool "$POOL" \
    --n 300 --image-max-size $SIZE 2>&1 | tail -1
done

cat <<'NOTE'

== reading this
  unet should land near 0.80 (the number on record). If it does not, fix that
  before believing either transformer arm.
  Prior expectation from smaller matched budgets: both transformers out-fit
  unet by 0.06-0.10, and swin_b ties swin_t despite 3x the parameters. If
  swin_b still ties at this budget, scale swin_t -- it is ~1.7x faster per step.
NOTE
