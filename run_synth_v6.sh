#!/usr/bin/env bash
# Reproduce any synthetic-v6 arm from the 2026-09-07 session.
#
#   ./run_synth_v6.sh <arm> [seed] [size]
#
# Arms (see synth_progress.md for what each measured):
#   headline    v5 + v6d union 3,200, synth-only        -- 0.686 +- 0.009 at 2048, 2 seeds
#   union       v5 + v6d union 3,200 at 1280            -- 0.610, 2 seeds
#   unionb      v5 + v6b union 3,200 at 1280            -- 0.613, 2 seeds
#   v5only      v5 toparea1600 alone                    -- 0.573, 2 seeds
#   v6only      v6d 1,600 alone                         -- the pool in isolation
#   mixv6b      documented mix + v6b 1,600 at 2048      -- 0.7754, 1 seed
#
# Every arm runs the documented two-phase schedule, then val-selection and
# checkpoint averaging. Pools are built on demand and are deterministic given
# --seed 6; nothing here reads HF14 to make a choice.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . "$HOME/.env"; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
mkdir -p logs

ARM=${1:?usage: run_synth_v6.sh <arm> [seed] [size]}
SEED=${2:-7}
SIZE=${3:-}

# ---- pools -----------------------------------------------------------------
# 4,000 drawn plans; ~0.2% of draws produce no labelled region and are skipped,
# so take the first 1,600 that exist rather than ids 0..1599.
build_pool() {   # variant(v6b|v6c|v6d|v6e) out_dir n
  local VAR=$1 OUT=$2 N=$3
  [ -f "data/synthetic/${VAR}_4000/generation_manifest.json" ] || \
    python3 generate_synthetic_v6.py --n 4000 --out "data/synthetic/${VAR}_4000" \
      --seed 6 --workers 40 > "logs/gen_${VAR}_4000.log" 2>&1
  [ "$(ls "$OUT/annotations" 2>/dev/null | wc -l)" -eq "$N" ] && return
  mkdir -p "$OUT/images" "$OUT/annotations"
  local i=0
  while [ "$(ls "$OUT/annotations" | wc -l)" -lt "$N" ]; do
    local f; f=$(printf "%06d" $i)
    if [ -f "data/synthetic/${VAR}_4000/annotations/synth6_$f.json" ]; then
      ln -sf "$PWD/data/synthetic/${VAR}_4000/images/synth6_$f.png"      "$OUT/images/"
      ln -sf "$PWD/data/synthetic/${VAR}_4000/annotations/synth6_$f.json" "$OUT/annotations/"
    fi
    i=$((i+1))
  done
}

merge() {   # out_dir sources...
  local OUT=$1; shift
  [ -d "$OUT" ] || python3 scripts/merge_local_datasets.py --sources "$@" --out "$OUT" \
    > "logs/merge_$(basename "$OUT").log" 2>&1
}

# ---- the documented two-phase run, then selection ---------------------------
two_phase() {   # data ck seed size workers
  local DATA=$1 CK=$2 SD=$3 SZ=$4 NW=$5
  local COMMON="--batch-size 8 --num-workers $NW --prefetch-factor 4 \
    --image-max-size $SZ --ref-size 224 --width 128 \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed $SD --compile --pad-grid 512"
  local TAG; TAG=$(basename "$CK")
  if [ ! -f "$CK/epoch_8.pth" ]; then
    python3 scripts/train_refunet.py --local-data "$DATA" --checkpoint-dir "$CK" \
      --epochs 1 --schedule-epochs 10 $COMMON > "logs/$TAG.log" 2>&1
    python3 scripts/train_refunet.py --local-data "$DATA" --checkpoint-dir "$CK" \
      --epochs 8 --schedule-epochs 9 $COMMON \
      --reset-optimizer --init-from "$CK/epoch_0.pth" >> "logs/$TAG.log" 2>&1
  fi
  python3 scripts/select_epoch_on_val.py --runs "$CK" --image-max-size "$SZ" \
    --out "data/evaluations/val_$TAG.json" > "logs/val_$TAG.log" 2>&1
  python3 scripts/average_checkpoints.py --run "$CK" --windows 3-8,4-8,5-8,6-8 \
    --image-max-size "$SZ" --out "data/evaluations/avg_$TAG.json" > "logs/avg_$TAG.log" 2>&1
  echo "== $TAG"; cat "data/evaluations/val_$TAG.json"; grep -a picks "logs/avg_$TAG.log"
}

V5=data/synthetic/toparea1600_balanced
case "$ARM" in
  headline|union)
    build_pool v6d data/synthetic/v6d_1600 1600
    merge data/synthetic/union_v5_v6d_3200 "$V5" data/synthetic/v6d_1600
    [ "$ARM" = headline ] && SIZE=${SIZE:-2048} || SIZE=${SIZE:-1280}
    two_phase data/synthetic/union_v5_v6d_3200 \
      "data/runs/ck_synthonly_uniond_${SIZE}_s${SEED}" "$SEED" "$SIZE" 16 ;;
  unionb)
    build_pool v6b data/synthetic/v6b_1600 1600
    merge data/synthetic/union_v5_v6b_3200 "$V5" data/synthetic/v6b_1600
    SIZE=${SIZE:-1280}
    two_phase data/synthetic/union_v5_v6b_3200 \
      "data/runs/ck_synthonly_unionb_${SIZE}_s${SEED}" "$SEED" "$SIZE" 12 ;;
  v5only)
    SIZE=${SIZE:-1280}
    two_phase "$V5" "data/runs/ck_synthonly_v5_${SIZE}_s${SEED}" "$SEED" "$SIZE" 12 ;;
  v6only)
    build_pool v6d data/synthetic/v6d_1600 1600
    SIZE=${SIZE:-1280}
    two_phase data/synthetic/v6d_1600 \
      "data/runs/ck_synthonly_v6d_${SIZE}_s${SEED}" "$SEED" "$SIZE" 12 ;;
  mixv6b)
    build_pool v6b data/synthetic/v6b_1600 1600
    merge data/mixed/v5_1600_v6b_1600_rf1548_gen504_gem1440 \
      "$V5" data/synthetic/v6b_1600 \
      data/roboflow/floz-real-pool-v2-strong18 \
      data/roboflow/floz-genreal-v1-strong18 \
      data/roboflow/floz-gen-gemini-r23-strong18
    SIZE=${SIZE:-2048}
    two_phase data/mixed/v5_1600_v6b_1600_rf1548_gen504_gem1440 \
      "data/runs/ck_mixv5v6b_${SIZE}_seed${SEED}" "$SEED" "$SIZE" 16 ;;
  *) echo "unknown arm: $ARM"; sed -n '3,20p' "$0"; exit 1 ;;
esac
