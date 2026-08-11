#!/usr/bin/env bash
# Does the anchor plane fix hold on the PROJECT'S HEADLINE recipe?
#
# The retraction so far is measured synth-only at 1,600 records, but the
# documented 0.680 -> 0.433 anchor collapse was on this 3,652-record mix. Three
# arms at the documented seed 31:
#   noanchor    should reproduce ~0.680 -> validates the rebuilt data pipeline
#   anchor      should reproduce ~0.433 -> confirms the bug reproduces here
#   anchorzero  the question: same anchor, reference plane 0.0 instead of 1.0
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
DATA=data/mixed/toparea1600_rf1548_gen504
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"

run () {  # $1 = tag, $2.. = extra flags
  local TAG="$1"; shift
  local CK="data/runs/ck_mix3652_${TAG}_s31"
  [ -f "$CK/epoch_8.pth" ] && { echo "skip $CK"; return; }
  echo "=== mix3652 $TAG seed 31 ==="
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON "$@"
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON "$@" \
    --reset-optimizer --init-from "$CK/epoch_0.pth"
}

run anchorzero --anchor --anchor-ref-plane 0.0
run anchor     --anchor
run noanchor
echo MIXFIX_DONE
