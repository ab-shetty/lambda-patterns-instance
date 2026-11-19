#!/usr/bin/env bash
# One training change vs fgavg78 (2026-11-19): the fgavg78 recipe (run_single_pass.sh: one-epoch restarts, DR + no-DR blend50)
# at SEEDS (default 7 8) with TRAINX on POOL, the two-seed 50/50 blend, product-scored at HEAD on valfix4 / hf14fix4 / gemini30
# (+ EXTRA_SPLITS "name:args" entries), and the pre-registered gate (as v2 / v3) against fgavg78 at HEAD:
#   hf14fix4 >= +0.002, valfix4 not < -0.002, gemini30 not < -0.003.   Nothing is published.
#   TAG=refunf POOL=data/mixed/fgab_fg_mix INIT=data/runs/ck_best/gem2s-x.pth TRAINX="--refine-head 32" ./run_ab_vs_best.sh
set -euo pipefail
cd "$(dirname "$0")"
export ALLOW_STALE_POOLS=1 PYTHONPATH=.:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:?}; POOL=${POOL:?}; TRAINX=${TRAINX:-}; SEEDS=${SEEDS:-7 8}
INIT=${INIT:-data/runs/ck_best/gem2s-x.pth}      # fgavg78's own restart point (its components restart gem2s-x)
BEST=data/runs/ck_fgavg78.pth
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
G=data/eval_pools/gemini30_s20261006
SPLITS=("valfix4:--indices $VAL --label-fixes eval_labels/real_fixes_v4.json"
        "hf14fix4:--indices $HF14 --label-fixes eval_labels/real_fixes_v4.json"
        "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json")
for e in ${EXTRA_SPLITS:-}; do SPLITS+=("${e//+/ }"); done

for S in $SEEDS; do
  TAG=${TAG}_s$S POOL=$POOL SEED=$S INIT=$INIT TRAINX="$TRAINX" BATCH=${BATCH:-8} ./run_single_pass.sh
done
set -- $SEEDS
B=data/runs/ck_${TAG}_avg.pth
[ -f $B ] || python3 scripts/blend_ckpts.py $B data/runs/ck_${TAG}_s$1_blend50.pth:0.5 data/runs/ck_${TAG}_s$2_blend50.pth:0.5

# product scores at HEAD, two jobs at a time; fgavg78's from the v3 run when present (same HEAD product)
declare -A CK=([best]=$BEST [ab]=$B [ab_s$1]=data/runs/ck_${TAG}_s$1_blend50.pth [ab_s$2]=data/runs/ck_${TAG}_s$2_blend50.pth)
dir() { case $1 in best) echo data/evaluations/fgavg78_head ;; *) echo data/evaluations/${TAG}_${1#ab} ;; esac; }
for sp in valfix4 hf14fix4 gemini30; do
  [ -f data/evaluations/selftrain_ctl/product_$sp.json ] && [ ! -f $(dir best)/product_$sp.json ] && \
    { mkdir -p $(dir best); cp data/evaluations/selftrain_ctl/product_$sp.json $(dir best)/; }
done
jobs=()
for arm in best ab ab_s$1 ab_s$2; do
  for sp in "${SPLITS[@]}"; do
    [ -f $(dir $arm)/product_${sp%%:*}.json ] || jobs+=("$arm|${sp%%:*}|${sp#*:}")
  done
done
run_job() {
  IFS='|' read -r arm split args <<< "$1"
  local O=$(dir $arm); mkdir -p $O
  python3 scripts/evaluate_refunet_selection.py --checkpoint ${CK[$arm]} $args --product --ref-size 224 --workers 8 \
    --metrics-out $O/product_$split.json > $O/product_$split.log 2>&1
}
for ((i = 0; i < ${#jobs[@]}; i += 2)); do
  run_job "${jobs[$i]}" &
  [ $((i + 1)) -lt ${#jobs[@]} ] && run_job "${jobs[$((i + 1))]}" &
  wait
done
for sp in "${SPLITS[@]}"; do
  s=${sp%%:*}; echo "== $TAG vs fgavg78 $s"
  python3 scripts/paired_compare.py --a $(dir best)/product_$s.json --b $(dir ab)/product_$s.json --label-a fgavg78 --label-b $TAG | head -6
  for x in $1 $2; do
    python3 -c "import json; a=json.load(open('$(dir best)/product_$s.json'))['mean_iou']; b=json.load(open('$(dir ab_s$x)/product_$s.json'))['mean_iou']; print('   seed $x alone: %.4f (%+.4f)' % (b, b - a))"
  done
done | tee logs/${TAG}_compare.txt
python3 - $(dir best) $(dir ab) <<'PY' | tee -a logs/${TAG}_compare.txt
import json, sys
m = lambda d, s: json.load(open(f"{d}/product_{s}.json"))["mean_iou"]
d = {s: m(sys.argv[2], s) - m(sys.argv[1], s) for s in ("valfix4", "hf14fix4", "gemini30")}
ok = d["hf14fix4"] >= 0.002 and d["valfix4"] >= -0.002 and d["gemini30"] >= -0.003
print("GATE (blend vs fgavg78):", "PASS" if ok else "FAIL", {k: round(v, 4) for k, v in d.items()})
PY
echo "AB $TAG DONE $(date '+%H:%M:%S')"
