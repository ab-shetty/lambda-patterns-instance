#!/usr/bin/env bash
# How many plans would synthetic need if it were exactly like real? Measure the
# data curve on held-out plans of the SAME distribution (fresh v6d, ids 100000+,
# scored with the exact HF14 protocol by question_difficulty.py), plus HF14/val.
#
# swin_t @2048, v6d-only, nested pools (2k in 8k in 16k), one seed. Passes per
# plan fall as the pool grows (single-pass is the right regime at 100k): 2k = 9
# epochs (the published run, abshetty/floz-refunet-swint-v6d-e8, not retrained),
# 8k = 5, 16k = 4. Documented two-phase shape: epoch 0 of a planned E+1 cosine,
# then reset and E-1 epochs of a planned E. Final epoch scored, no selection.
#   ./run_v6d_scaling.sh [seed]        ~1h45 on a GH200
set -euo pipefail
SEED=${1:-7}
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations/v6dscale; mkdir -p logs $E
FRESH=data/synthetic/v6d_fresh600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 --image-max-size 2048 \
  --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
  --train-split 0.99 --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"

score() {   # tag checkpoint
  for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
    [ -f $E/$1_$split.json ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py \
      --checkpoint $2 --indices $IDX --image-max-size 2048 --ref-size 224 --mask-thresh 0.35 \
      --metrics-out $E/$1_$split.json > /dev/null 2>&1
  done
  [ -f $E/$1_fresh.json ] || PYTHONPATH=. python3 scripts/question_difficulty.py --checkpoint $2 \
    --pool $FRESH --n-images 150 --image-max-size 2048 --out $E/$1_fresh.json > logs/v6dscale_$1_qd.log 2>&1
}

CTRL=data/runs/ck_swint_v6d2k_pub/epoch_8.pth
[ -f $CTRL ] || PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-swint-v6d-e8 --out $CTRL
score n2000 $CTRL

for NE in 8000:5 16000:4; do
  N=${NE%:*} EP=${NE#*:} CK=data/runs/ck_v6dscale_${NE%:*}_s$SEED
  POOL=data/synthetic/v6d_$N LAST=$CK/epoch_$((EP - 1)).pth
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs $((EP + 1)) $COMMON > logs/v6dscale_${N}_p1.log 2>&1
  [ -f $LAST ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs $((EP - 1)) --schedule-epochs $EP $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/v6dscale_${N}_p2.log 2>&1
  score n$N $LAST
  echo "== done $N"
done

echo; echo "plans   fresh-v6d(HF14 protocol)   val     HF14"
for T in n2000 n8000 n16000; do python3 -c "
import json, numpy as np
f = [r['iou'] for r in json.load(open('$E/${T}_fresh.json'))['rows'] if r['is_real'] == 0]
v = json.load(open('$E/${T}_val.json'))['mean_iou']; h = json.load(open('$E/${T}_hf14.json'))['mean_iou']
print(f'${T#n}'.ljust(8), f'{np.mean(f):.4f} (n={len(f)})'.ljust(26), f'{v:.4f}  {h:.4f}')"; done
