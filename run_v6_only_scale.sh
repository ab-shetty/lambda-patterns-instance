#!/usr/bin/env bash
# v6d-only pools at scale, 2048, extended (warm-restart) schedule.
#
# The question: the v6 generator is visually far closer to real plans than v5,
# and drawing a plan costs 0.01 s -- so what does v6d ALONE reach if the pool
# is made large? Recorded v6d-only points are all small and all at 1280:
# 1,600 plans -> 0.5348 / 0.5125 (seeds 7/31, 9 epochs, synth_progress.md
# 2026-09-07), and a v6b volume test at 4,000 was null (0.520).
#
# Nothing real, nothing v5: the pool IS the training set, so this measures the
# generator on its own rather than its marginal value in a mix.
#
# Arms (nested -- the generator seeds per image id, so the 1,600 pool is a
# strict subset of the 12,000 one, same draws):
#   1600    the recorded pool, re-measured at 2048 on this machine (the control)
#   12000   7.5x the pool
#
# Read the volume contrast at EQUAL EPOCHS, and note what that means: 9 epochs
# is 1,800 optimizer steps on 1,600 records and 13,500 on 12,000, so the large
# arm gets more steps as well as more images. That is inherent to "more data at
# the same schedule" (startup.md, "schedule length is a step budget"), and the
# control's extended schedule is what keeps it from being merely step-starved.
#
# Usage: ./run_v6_only_scale.sh <1600|12000> [seed]
set -euo pipefail

N=${1:?usage: run_v6_only_scale.sh <1600|12000> [seed]}
SEED=${2:-7}
SIZE=2048
POOL=data/synthetic/v6d_${N}
CK=data/runs/ck_v6donly_${N}_res${SIZE}_seed${SEED}
TAG=$(basename "$CK")
mkdir -p logs data/evaluations

[ "$(ls "$POOL/annotations" 2>/dev/null | wc -l)" -eq "$N" ] || {
  echo "pool $POOL is not $N records"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 32 --prefetch-factor 6 \
  --image-max-size $SIZE --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --compile --pad-grid 512"

run_phase() {   # epochs schedule_epochs init_from_epoch last_epoch
  local EP=$1 SCH=$2 FROM=$3 LAST=$4
  [ -f "$CK/epoch_${LAST}.pth" ] && { echo "== phase to epoch $LAST done"; return; }
  local INIT=()
  [ -n "$FROM" ] && INIT=(--reset-optimizer --init-from "$CK/epoch_${FROM}.pth")
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs "$EP" --schedule-epochs "$SCH" \
    $COMMON "${INIT[@]}" >> "logs/$TAG.log" 2>&1
}

run_phase 1  10 ""  0     # documented phase 1
run_phase 8   9 0   8     # documented phase 2 -> the 9-epoch recipe's result
run_phase 10 10 8   18    # warm restart 1
run_phase 10 10 18  28    # warm restart 2

PYTHONPATH=. python3 scripts/select_epoch_on_val.py --runs "$CK" \
  --image-max-size "$SIZE" --out "data/evaluations/val_$TAG.json" \
  > "logs/val_$TAG.log" 2>&1
echo "== $TAG"; cat "data/evaluations/val_$TAG.json"
