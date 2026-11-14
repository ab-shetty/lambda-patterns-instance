#!/usr/bin/env bash
# Rescore checkpoints with the current product (evaluate_refunet_selection.py --product) on valfix4 / hf14fix4 / gemini30,
# two GPU jobs at a time (three OOM the 94.5 GB GH200 at 5120); maps kept under data/floz-diag/rescore_<tag>/<split>/<scale> for vertsweep.py.
#   CKS="tag:path ..." ./scripts/rescore_product.sh      (2026-11-14, fgab seed average)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
for ck in $CKS; do
  TAG=${ck%%:*}; CK=${ck#*:}
  for job in "valfix4:--indices $VAL --label-fixes eval_labels/real_fixes_v4.json" \
             "hf14fix4:--indices $HF14 --label-fixes eval_labels/real_fixes_v4.json" \
             "gemini30:--local-pool data/eval_pools/gemini30_s20261006 --indices all --boxes eval_boxes/gemini30_v1.json"; do
    S=${job%%:*}; O=data/evaluations/rescore_$TAG; mkdir -p $O
    echo "$CK|$S|${job#*:}|$O/product_$S.json|data/floz-diag/rescore_$TAG/$S"
  done
done | xargs -P ${PAR:-2} -I{} bash -c 'IFS="|" read CK S ARGS OUT MAPS <<< "{}"; [ -f $OUT ] || python3 scripts/evaluate_refunet_selection.py --checkpoint $CK $ARGS --product --ref-size 224 --workers 8 --save-probs $MAPS --metrics-out $OUT > ${OUT%.json}.log 2>&1 || echo "FAILED $OUT"; python3 -c "import json; print(\"$OUT\", round(json.load(open(\"$OUT\"))[\"mean_iou\"], 4))"'
echo "RESCORE DONE $(date +%H:%M:%S)"
