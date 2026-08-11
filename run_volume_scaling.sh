#!/usr/bin/env bash
# Does MORE DATA lift real IoU? Baseline is 3,652 records (0.6916 HF14, seed 31).
#
# Two different senses of "more images", because they are not equivalent:
#   synth 3200 / 6400  - more synthetic records, real side fixed. `startup.md`
#                        already screened 3200 as negative (val 0.7415 vs
#                        0.7672); this makes it a curve instead of one point.
#   rf3096_gen1008     - same 114 unique real sources, twice the offline
#                        variants. Separates "more records" from "more sources",
#                        which the docs name as the binding constraint.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"

run () {
  local TAG="$1" DATA="$2"
  local CK="data/runs/ck_vol_${TAG}_s31"
  [ -d "$DATA" ] || { echo "MISSING $DATA, skipping $TAG"; return; }
  [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; return; }
  echo "=== volume $TAG ($(ls $DATA/images | wc -l) records) ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
}

run synth3200 data/mixed/toparea3200_rf1548_gen504
run synth6400 data/mixed/toparea6400_rf1548_gen504
run realaug2x data/mixed/toparea1600_rf3096_gen1008
echo VOLDONE
