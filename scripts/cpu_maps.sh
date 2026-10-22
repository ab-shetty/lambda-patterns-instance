#!/usr/bin/env bash
# Save probability maps on CPU, 2 jobs at a time, for scripts/pp_probe/stack.py (2026-10-22):
#   scripts/cpu_maps.sh CKPT MAPS_DIR   -> MAPS_DIR/{hf14,val,gemini30}_{4096,2048}/probs + metrics.json
cd "$(dirname "$0")/.."
CK=$1; M=$2
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24; VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; G=data/eval_pools/gemini30_s20261006
one() {
  sp=$1; sz=$2; d=$M/${sp}_$sz; [ -f $d/metrics.json ] && return; mkdir -p $d/probs
  case $sp in hf14) a="--indices $HF14";; val) a="--indices $VAL";; gemini30) a="--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json";; esac
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK $a \
    --image-max-size $sz --ref-size 224 --mask-thresh 0.35 --workers 0 --save-probs $d/probs --metrics-out $d/metrics.json > $d/log.txt 2>&1
  echo "$(date +%H:%M) $d $(python3 -c "import json;print(round(json.load(open('$d/metrics.json'))['mean_iou'],4))" 2>/dev/null || echo FAILED)"
}
for pair in "gemini30 4096:gemini30 2048" "val 4096:val 2048" "hf14 4096:hf14 2048"; do
  one ${pair%%:*} & one ${pair#*:} & wait
done
echo "MAPS DONE $M $(date +%H:%M)"
