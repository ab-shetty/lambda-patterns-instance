#!/usr/bin/env bash
# v6 synthetic volume x extended (warm-restart) training, at 2048.
#
# Two open threads from 2026-09-17 meet here:
#   - `mixv6d` (documented mix + 1,600 v6d) peaked at its FINAL epoch at both
#     seeds (synth_progress.md 2026-09-08) -- that pool never converged in the
#     documented 9 epochs, so its +0.028 was measured under-trained.
#   - warm restarts (fresh cosine from a converged checkpoint, repeatedly) took
#     `mix6676_with_r4` from 0.7924 to 0.8127 val-complement (codex_doc.md,
#     one seed).
# So: does more v6 data plus a schedule long enough to consume it beat the
# documented recipe, and is the volume or the schedule doing the work?
#
# Arms (both = documented mix5092 + N drawn v6d plans; the 1,600 subset is a
# strict subset of the 3,200 one, same generator seed, so volume is the only
# difference between them):
#   v6d1600   6,692 records -- the 2026-09-08 pool, now trained past 9 epochs
#   v6d3200   8,292 records -- new: twice the v6d
#
# Schedule: documented two-phase (epochs 0-8) then two warm restarts of 10
# epochs each (9-18, 19-28), each a fresh cosine with the optimizer reset.
# Selection is on the validation complement only -- HF14 is never used to pick
# an epoch. Usage: ./run_v6_volume_extended.sh <v6d1600|v6d3200> [seed]
set -euo pipefail

ARM=${1:?usage: run_v6_volume_extended.sh <v6d1600|v6d3200> [seed]}
SEED=${2:-7}
SIZE=${SIZE_OVERRIDE:-2048}
N=${ARM#v6d}

DATA=data/mixed/v5_1600_v6d_${N}_rf1548_gen504_gem1440
CK=data/runs/ck_mixattn_${ARM}_res${SIZE}_seed${SEED}
TAG=$(basename "$CK")
mkdir -p logs data/evaluations

[ -d "$DATA" ] || python3 scripts/merge_local_datasets.py --sources \
  data/synthetic/toparea1600_balanced \
  data/synthetic/v6d_${N} \
  data/roboflow/floz-real-pool-v2-strong18 \
  data/roboflow/floz-genreal-v1-strong18 \
  data/roboflow/floz-gen-gemini-r23-strong18 \
  --out "$DATA" > "logs/merge_$(basename "$DATA").log" 2>&1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 32 --prefetch-factor 6 \
  --image-max-size $SIZE --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --early-stop-patience 4 --early-stop-monitor val_loss --model crossattn --domain-random --mask-thresh 0.35 --seed $SEED --compile --pad-grid 512"

run_phase() {   # epochs schedule_epochs init_from_epoch last_epoch
  local EP=$1 SCH=$2 FROM=$3 LAST=$4
  [ -f "$CK/epoch_${LAST}.pth" ] && { echo "== phase to epoch $LAST done"; return; }
  local INIT=()
  [ -n "$FROM" ] && INIT=(--reset-optimizer --init-from "$CK/epoch_${FROM}.pth")
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$DATA" \
    --checkpoint-dir "$CK" --epochs "$EP" --schedule-epochs "$SCH" \
    $COMMON "${INIT[@]}" >> "logs/$TAG.log" 2>&1
}

run_phase 1  10 ""  0     # documented phase 1
run_phase 8   9 0   8     # documented phase 2 -> the 9-epoch recipe's result
run_phase 10 10 8   18    # warm restart 1
[ "${MAXEP:-28}" -ge 28 ] && run_phase 10 10 18  28

PYTHONPATH=. python3 scripts/select_epoch_on_val.py --runs "$CK" \
  --image-max-size "$SIZE" --out "data/evaluations/val_$TAG.json" \
  > "logs/val_$TAG.log" 2>&1
echo "== $TAG"; cat "data/evaluations/val_$TAG.json"
