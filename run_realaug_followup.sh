#!/usr/bin/env bash
# Follow-ups to the volume/ratio sweep, in priority order.
#
# Finding: the number of REAL-DERIVED records is what moves real IoU. At a fixed
# 72% real fraction, realaug2x (4,104 real-ish) beat synth800 (2,052) by +0.042
# HF14, while synthetic volume showed no trend at all. realaug2x is the best
# result so far (HF14 0.7069) but is one seed and selected its LAST epoch.
#
#   1. confirm  - realaug2x at seeds 7 and 99, so the win has 3 seeds. The repo
#                 has retracted a 3-seed win before; this is the minimum bar.
#   2. long     - realaug2x on a 15-epoch schedule. It picked epoch 8 of 9, so
#                 the schedule may simply have truncated it.
#   3. realaug4x- 8,208 real-ish records. Does the axis keep paying, or does
#                 re-augmenting 114 sources 72x each turn over into memorisation?
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
BASE="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35"
D2=data/mixed/toparea1600_rf3096_gen1008
D4=data/mixed/toparea1600_rf6192_gen2016

run () {  # tag data seed total_epochs
  local TAG="$1" DATA="$2" SEED="$3" EP="$4"
  local CK="data/runs/ck_${TAG}_s${SEED}"
  [ -f "$CK/epoch_$((EP-1)).pth" ] && { echo "skip $CK"; return; }
  echo "=== $TAG seed $SEED ($(ls $DATA/images | wc -l) records, ${EP}ep) ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs $((EP+1)) --seed "$SEED" $BASE
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs $((EP-1)) --schedule-epochs $EP --seed "$SEED" $BASE \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
}

run realaug2x "$D2" 7  9
run realaug2x "$D2" 99 9
run realaug2xlong "$D2" 31 15
until grep -q AUG72DONE /tmp/aug72.log 2>/dev/null; do sleep 20; done
run realaug4x "$D4" 31 9
echo FOLLOWUPDONE
