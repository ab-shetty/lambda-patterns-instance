#!/usr/bin/env bash
# --foreground A/B at three seeds (2026-11-19), all scored with the HEAD product: per seed fg vs ctl (blend50 each) and the
# three-seed blends (1/3 each) fg vs ctl, paired.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
G=data/eval_pools/gemini30_s20261006
for A in ctl fg; do
  B=data/runs/ck_fgab_${A}_3seed.pth
  [ -f $B ] || python3 scripts/blend_ckpts.py $B data/runs/ck_fgab_${A}_s7_blend50.pth:0.3333333 \
    data/runs/ck_fgab_${A}_s8_blend50.pth:0.3333333 data/runs/ck_fgab_${A}_s9_blend50.pth:0.3333334
done
# HEAD product scores, one dir per arm (s9 from run_foreground_ab.sh, already HEAD)
declare -A DIR=([ctl_s7]=data/evaluations/fg3_ctl_s7 [ctl_s8]=data/evaluations/fg3_ctl_s8 [ctl_s9]=data/evaluations/fgab_ctl_s9
                [fg_s7]=data/evaluations/fg3_fg_s7 [fg_s8]=data/evaluations/fg3_fg_s8 [fg_s9]=data/evaluations/fgab_fg_s9
                [ctl_3]=data/evaluations/fg3_ctl_3seed [fg_3]=data/evaluations/fg3_fg_3seed)
declare -A CK=([ctl_s7]=data/runs/ck_fgab_ctl_s7_blend50.pth [ctl_s8]=data/runs/ck_fgab_ctl_s8_blend50.pth
               [fg_s7]=data/runs/ck_fgab_fg_s7_blend50.pth [fg_s8]=data/runs/ck_fgab_fg_s8_blend50.pth
               [ctl_3]=data/runs/ck_fgab_ctl_3seed.pth [fg_3]=data/runs/ck_fgab_fg_3seed.pth)
jobs=()
for arm in ctl_s7 ctl_s8 fg_s7 fg_s8 ctl_3 fg_3; do
  for sp in "valfix4:--indices $VAL --label-fixes eval_labels/real_fixes_v4.json" \
            "hf14fix4:--indices $HF14 --label-fixes eval_labels/real_fixes_v4.json" \
            "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
    [ -f ${DIR[$arm]}/product_${sp%%:*}.json ] || jobs+=("$arm|${sp%%:*}|${sp#*:}")
  done
done
run_job() {
  IFS='|' read -r arm split args <<< "$1"
  local O=${DIR[$arm]}; mkdir -p $O
  python3 scripts/evaluate_refunet_selection.py --checkpoint ${CK[$arm]} $args --product --ref-size 224 --workers 8 \
    --metrics-out $O/product_$split.json > $O/product_$split.log 2>&1
}
for ((i = 0; i < ${#jobs[@]}; i += 2)); do
  run_job "${jobs[$i]}" &
  [ $((i + 1)) -lt ${#jobs[@]} ] && run_job "${jobs[$((i + 1))]}" &
  wait
done
for pair in "ctl_s7 fg_s7" "ctl_s8 fg_s8" "ctl_s9 fg_s9" "ctl_3 fg_3"; do
  set -- $pair
  for sp in valfix4 hf14fix4 gemini30; do
    echo "== $2 vs $1 $sp"
    python3 scripts/paired_compare.py --a ${DIR[$1]}/product_$sp.json --b ${DIR[$2]}/product_$sp.json --label-a $1 --label-b $2 | head -6
  done
done | tee logs/fg3seed_compare.txt
echo "FG3SEED DONE $(date '+%H:%M:%S')"
