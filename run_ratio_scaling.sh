#!/usr/bin/env bash
# Separate MIX RATIO from TOTAL VOLUME.
#
# The volume sweep confounded them: adding synthetic also dilutes the real half
# (56% real -> 39% -> 24%). These two arms cut synthetic instead, raising the
# real fraction while REDUCING volume:
#
#   arm         records  synth  real-ish  real%
#   synth6400     8452    6400    2052     24%
#   synth3200     5252    3200    2052     39%
#   baseline      3652    1600    2052     56%
#   realaug2x     5704    1600    4104     72%   <- same ratio as synth800...
#   synth800      2852     800    2052     72%   <- ...at half the volume
#   synth400      2452     400    2052     84%
#
# synth800 vs realaug2x is the key pair: identical real fraction, 2x volume
# apart. Same score => ratio is the lever and volume is irrelevant.
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
  [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; return; }
  echo "=== ratio $TAG ($(ls $DATA/images | wc -l) records) ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
}
run synth800 data/mixed/toparea800_rf1548_gen504
run synth400 data/mixed/toparea400_rf1548_gen504
echo RATIODONE
