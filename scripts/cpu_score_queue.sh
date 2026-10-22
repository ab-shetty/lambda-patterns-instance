#!/usr/bin/env bash
# CPU scoring queue, at most 2 jobs at a time (2026-10-22: 6 parallel CPU evals beside training ran RAM to 24 GB free).
# Lines of logs/cpu_queue.txt: CKPT OUTDIR SPLIT  (split: hf14 hf14fix2 val gemini30). Done jobs (OUTDIR/SPLIT_4096.json) are skipped.
cd "$(dirname "$0")/.."
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24; VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; G=data/eval_pools/gemini30_s20261006
run() {
  ck=$1; o=$2; sp=$3; mkdir -p $o; [ -f $o/${sp}_4096.json ] && return
  case $sp in hf14) a="--indices $HF14";; hf14fix2) a="--indices $HF14 --label-fixes eval_labels/hf14_fixes_v2.json";;
              val) a="--indices $VAL";; gemini30) a="--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json";; esac
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $ck $a \
    --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --workers 0 --metrics-out $o/${sp}_4096.json > $o/${sp}_cpu.log 2>&1
  echo "$(date +%H:%M) $o $sp $(python3 -c "import json;print(round(json.load(open('$o/${sp}_4096.json'))['mean_iou'],4))" 2>/dev/null || echo FAILED)"
}
while true; do
  mapfile -t todo < <(while read ck o sp; do [ -n "$ck" ] && [ ! -f $o/${sp}_4096.json ] && [ ! -f $o/${sp}.claimed ] && echo "$ck $o $sp"; done < logs/cpu_queue.txt)
  [ ${#todo[@]} = 0 ] && break
  for j in "${todo[@]:0:2}"; do set -- $j; mkdir -p $2; touch $2/$3.claimed; run $j & done
  wait
done
echo "QUEUE EMPTY $(date +%H:%M)"
