#!/usr/bin/env bash
# Stress test (2026-11-14): the product output on degraded copies of the real eval sheets (evaluate_refunet_selection.py
# --degrade: jpeg50 / jpeg25 / lowres2000 / lowres1400 / scan / gray, all at native size so labels hold). No new labels.
#   CKS="tag:path ..." (default gem2s-x and the fgab fg seed-7 checkpoint if present)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
CKS=${CKS:-"gem2s_x:data/runs/ck_best/gem2s-x.pth fgab_fg_s7:data/runs/ck_fgab_fg_s7_blend50.pth"}
for ck in $CKS; do
  TAG=${ck%%:*}; CK=${ck#*:}; [ -f $CK ] || { echo "skip $TAG (no $CK)"; continue; }
  O=data/evaluations/stress_$TAG; mkdir -p $O
  for D in none jpeg50 jpeg25 lowres2000 lowres1400 scan gray; do
    X=$([ $D = none ] && echo "" || echo "--degrade $D")
    for job in "valfix4:$VAL" "hf14fix4:$HF14"; do
      S=${job%%:*}; OUT=$O/${S}_$D.json
      [ -f $OUT ] || python3 scripts/evaluate_refunet_selection.py --checkpoint $CK --indices ${job#*:} \
        --label-fixes eval_labels/real_fixes_v4.json --product --ref-size 224 --workers 8 $X \
        --save-probs /tmp/stress_maps/$TAG/${S}_$D --metrics-out $OUT > $O/${S}_$D.log 2>&1
      rm -rf /tmp/stress_maps/$TAG/${S}_$D
      python3 -c "import json; m = json.load(open('$OUT')); print('$TAG $S $D', round(m['mean_iou'], 4))"
    done
  done
done
echo "STRESS DONE $(date '+%H:%M:%S')"
