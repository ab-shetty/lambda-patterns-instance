#!/usr/bin/env bash
# Fair, fast screen of synthetic-data changes: synth-only swin_t at 1024, all
# arms concurrently on one GPU (~20 min on a GH200), same 1,600 ids / recipe /
# seed, differing only in the generator knob. Scored at epoch 8 with 2048
# inference (2x the training size; see startup.md on --domain-random), paired
# over the 52 HF14 selections and on validation. Replaces run_synth_ft_ab.sh,
# whose gentle fine-tune barely moves predictions (48/52 ties).
#   ./run_synth_only_screen.sh [seed]
set -euo pipefail
SEED=${1:-7}
declare -A POOL=([v6d]=data/synthetic/v6d_1600 [mottle]=data/synthetic/v6dMO_1600 [hardscape]=data/synthetic/v6dHS2_1600)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations; mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 12 --prefetch-factor 4 --image-max-size 1024 --ref-size 224 \
  --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 --domain-random \
  --mask-thresh 0.35 --seed $SEED --pad-grid 512"
train() { local CK=data/runs/ck_so_$1_s$SEED
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$1]} \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/so_$1_p1.log 2>&1
  [ -f $CK/epoch_8.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$1]} \
    --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/so_$1_p2.log 2>&1
  for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
    PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_8.pth \
      --indices $IDX --image-max-size 2048 --ref-size 224 --mask-thresh 0.35 \
      --metrics-out $E/so_$1_s${SEED}_$split.json > /dev/null 2>&1; done; }
for A in v6d mottle hardscape; do train $A & done; wait
for A in v6d mottle hardscape; do python3 -c "
import json,numpy as np
v=np.mean([r['iou'] for r in json.load(open('$E/so_${A}_s${SEED}_val.json'))['selections']])
h=np.mean([r['iou'] for r in json.load(open('$E/so_${A}_s${SEED}_hf14.json'))['selections']])
print(f'$A: val {v:.4f}  HF14 {h:.4f}')"; done
for A in mottle hardscape; do python3 scripts/paired_compare.py --a $E/so_v6d_s${SEED}_hf14.json \
  --b $E/so_${A}_s${SEED}_hf14.json --label-a v6d --label-b $A | sed -n '/paired mean/,/sign-test/p'; done
