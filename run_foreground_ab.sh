#!/usr/bin/env bash
# Foreground-objects A/B (2026-11-14): does training with entourage in front of the house (--foreground, revit_render
# draw_foreground) help the real sheets? Two one-epoch restarts of gem2s-x (run_single_pass.sh: DR + no-DR arms, blend50)
# on pools that differ ONLY in --foreground: same ids, seed and generator flags, each merged with the same Gemini holdout
# records (the gem recipe's mix). Scored as the product output (evaluate_refunet_selection.py --product) on valfix4 /
# hf14fix4 (label fixes v4) and Gemini30.
#   N (default 9000) sheets per pool; FG (default 0.4) the --foreground probability of the treatment pool.
set -euo pipefail
cd "$(dirname "$0")"
export ALLOW_STALE_POOLS=1 PYTHONPATH=.
N=${N:-9000}; FG=${FG:-0.4}; START=${START:-700000}; SEED=${SEED:-7}
INIT=${INIT:-data/runs/ck_best/gem2s-x.pth}
GEM=data/roboflow/floz-gen-gemini-r234-no30-strong18          # run_gemini_holdout.sh steps 1-2 build it
[ -d $GEM ] || { echo "missing $GEM: run run_gemini_holdout.sh steps 1-2 first"; exit 1; }
GEN="--plan-source data/reference/swiss_dwellings/layouts.pkl.gz"
mkdir -p logs data/mixed
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24

for ARM in ctl fg; do
  POOL=data/synthetic/fgab_${ARM}_s$START
  X=$([ $ARM = fg ] && echo "--foreground $FG" || echo "--foreground 0")
  [ -f $POOL/generation_manifest.json ] || python3 scripts/generate_synthetic_fc.py --out $POOL --n $N --start $START --seed 6 \
    --revit --revit-plans --shaped --workers 64 --mode-weights 66,16,18 $GEN $X > logs/gen_fgab_$ARM.log 2>&1
  echo "== $ARM pool: $(ls $POOL/annotations | wc -l) sheets"
  MIX=data/mixed/fgab_${ARM}_mix
  [ -d $MIX ] || python3 scripts/merge_local_datasets.py --out $MIX --sources $POOL $GEM > /dev/null
  TAG=fgab_${ARM}_s$SEED POOL=$MIX SEED=$SEED INIT=$INIT ./run_single_pass.sh
  CK=data/runs/ck_fgab_${ARM}_s${SEED}_blend50.pth
  O=data/evaluations/fgab_${ARM}_s$SEED
  for job in "valfix4:--indices $VAL --label-fixes eval_labels/real_fixes_v4.json" \
             "hf14fix4:--indices $HF14 --label-fixes eval_labels/real_fixes_v4.json" \
             "gemini30:--local-pool data/eval_pools/gemini30_s20261006 --indices all --boxes eval_boxes/gemini30_v1.json"; do
    OUT=$O/product_${job%%:*}.json
    [ -f $OUT ] || python3 scripts/evaluate_refunet_selection.py --checkpoint $CK ${job#*:} --product --ref-size 224 \
      --workers 8 --save-probs data/floz-diag/fgab_${ARM}_s$SEED/${job%%:*} --metrics-out $OUT > $O/product_${job%%:*}.log 2>&1
    python3 -c "import json; m = json.load(open('$OUT')); print('$ARM ${job%%:*} product', round(m['mean_iou'], 4))"
  done
done
echo "FOREGROUND AB DONE $(date '+%H:%M:%S')"
