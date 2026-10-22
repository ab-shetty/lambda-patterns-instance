#!/usr/bin/env bash
# Score a checkpoint @4096 with hand boxes on HF14, HF14 fix v2, val, Gemini30 -- four jobs side by side (2026-10-22).
#   scripts/score4.sh CKPT OUTDIR [SIZE]      -> OUTDIR/{hf14,hf14fix2,val,gemini30}_SIZE.json ; prints the four means
set -euo pipefail
CK=$1; O=$2; SZ=${3:-4096}; mkdir -p $O
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24; VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26; G=data/eval_pools/gemini30_s20261006
for job in "hf14:--indices $HF14" "hf14fix2:--indices $HF14 --label-fixes eval_labels/hf14_fixes_v2.json" \
           "val:--indices $VAL" "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
  OUT=$O/${job%%:*}_$SZ.json
  [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK ${job#*:} \
    --image-max-size $SZ --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT > $O/${job%%:*}_$SZ.log 2>&1 &
done
wait
python3 -c "
import json,sys
print('$O', ' '.join(f\"{s} {json.load(open('$O/'+s+'_$SZ.json'))['mean_iou']:.4f}\" for s in ('hf14','hf14fix2','val','gemini30')))"
