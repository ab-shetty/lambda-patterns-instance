#!/usr/bin/env bash
# TTA probability maps for the equal-compute comparison with extra scales (2026-11-11):
#   scripts/gpu_maps_tta.sh CKPT MAPS_DIR "4096:2 4096:4 2048:2" "val hf14 gemini30"  -> MAPS_DIR/{split}_{size}t{n}/probs
cd "$(dirname "$0")/.."
CK=$1; M=$2; JOBS=${3:-"4096:2 4096:4 2048:2"}; SPLITS=${4:-"val hf14 gemini30"}
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24; VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; G=data/eval_pools/gemini30_s20261006
run() { sp=$1; sz=$2; n=$3; d=$M/${sp}_${sz}t$n; [ -f $d/metrics.json ] && return; mkdir -p $d/probs
  case $sp in hf14) a="--indices $HF14";; val) a="--indices $VAL";; gemini30) a="--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json";; esac
  PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK $a --image-max-size $sz --ref-size 224 --tta $n \
    --mask-thresh 0.35 --workers 4 --save-probs $d/probs --metrics-out $d/metrics.json > $d/log.txt 2>&1; }
for j in $JOBS; do for sp in $SPLITS; do run $sp ${j%%:*} ${j##*:} & done; wait; done
for d in $M/*t[0-9]/; do echo "$d $(python3 -c "import json;print(round(json.load(open('$d/metrics.json'))['mean_iou'],4))" 2>/dev/null)"; done
echo "TTA MAPS DONE $M $(date +%H:%M)"
