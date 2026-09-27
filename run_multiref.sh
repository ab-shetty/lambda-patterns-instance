#!/usr/bin/env bash
# Is the one-question-per-plan sampler why swin_t underfits family grouping?
# The 16k v6d model scores 0.736 on its OWN training plans under the HF14
# protocol (every family asked) against a 0.904 resolution ceiling, and most of
# the loss is picking regions on 4-6 family sheets. Training asks one random
# family per plan per epoch. --refs-per-image 6 asks every family (<= 6) of a
# plan in the same step, backbone run once.
#
# v6d 2k @2048, same recipe/seed as the published one-question run
# (abshetty/floz-refunet-swint-v6d-e8: fresh 0.660, HF14 0.7066), batch 3 plans
# x 6 questions (~66 GiB worst case). ~40 min train + ~5 min scoring.
#   ./run_multiref.sh [seed] [v6d|revit]
# Env: DEC=selfattn swaps the decoder (--swin-decoder); RESTART_FROM=<ckpt>
# instead continues that checkpoint with one fresh 9-epoch cosine (step-budget
# test); TAG names the run; BS overrides the batch (plans per step).
# revit: same six-question recipe on revit_train2000, compared with the
# one-question revit 2k run (run_revit_2k.sh) on fresh revit plans.
set -euo pipefail
SEED=${1:-7}
ARM=${2:-v6d}
S=data/evaluations/v6dscale
if [ $ARM = revit ]; then
  POOL=data/synthetic/revit_train2000 FRESH=data/synthetic/revit_fresh400
  CK=data/runs/ck_multiref6_revit2k_s$SEED E=data/evaluations/multiref_revit
  B_FRESH=$S/revit2k_fresh_revit.json B_VAL=data/evaluations/revit2k/revit_s7_i2048_val.json
  B_HF14=data/evaluations/revit2k/revit_s7_i2048_hf14.json
else
  POOL=data/synthetic/v6d_train2000 FRESH=data/synthetic/v6d_fresh600
  CK=data/runs/ck_multiref6_v6d2k_s$SEED E=data/evaluations/multiref
  B_FRESH=$S/n2000_fresh.json B_VAL=$S/n2000_val.json B_HF14=$S/n2000_hf14.json
fi
DEC=${DEC:-baseline}; BS=${BS:-3}; TAG=${TAG:-}
if [ -n "$TAG" ]; then CK=${CK}_$TAG; E=${E}_$TAG; ARM=${ARM}_$TAG; fi
mkdir -p logs $E
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size $BS --swin-decoder $DEC --refs-per-image 6 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 2048 --ref-size 224 --width 128 --model swin_t --lr 2e-4 \
  --backbone-lr-mult 0.1 --train-split 0.99 --domain-random --mask-thresh 0.35 \
  --seed $SEED --pad-grid 512"
if [ -n "${RESTART_FROM:-}" ]; then
  LAST=$CK/epoch_17.pth
  [ -f $LAST ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs 9 --schedule-epochs 9 $COMMON --reset-optimizer \
    --init-from $RESTART_FROM > logs/multiref_${ARM}_s${SEED}_restart.log 2>&1
else
  LAST=$CK/epoch_8.pth
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/multiref_${ARM}_s${SEED}_p1.log 2>&1
  [ -f $LAST ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $POOL \
    --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/multiref_${ARM}_s${SEED}_p2.log 2>&1
fi
for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
  [ -f $E/${split}.json ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py \
    --checkpoint $LAST --indices $IDX --image-max-size 2048 --ref-size 224 \
    --mask-thresh 0.35 --metrics-out $E/${split}.json > /dev/null 2>&1
done
for p in fresh:$FRESH seen:$POOL; do
  [ -f $E/${p%%:*}.json ] || PYTHONPATH=. python3 scripts/question_difficulty.py --checkpoint $LAST \
    --pool ${p#*:} --n-images 150 --image-max-size 2048 --out $E/${p%%:*}.json > logs/multiref_${ARM}_qd_${p%%:*}.log 2>&1
done
python3 - <<EOF
import json, numpy as np
syn = lambda f: np.mean([r['iou'] for r in json.load(open(f))['rows'] if r['is_real'] == 0])
mi = lambda f: json.load(open(f))['mean_iou']
print('run                         fresh    seen     val      HF14')
print('$ARM 2k one-question        %.4f   -        %.4f   %.4f' % (syn('$B_FRESH'), mi('$B_VAL'), mi('$B_HF14')))
if '$ARM' == 'v6d':
    print('16k one-question            %.4f   %.4f   %.4f   %.4f' % (syn('$S/n16000_fresh.json'), syn('$S/n16000_seen.json'), mi('$S/n16000_val.json'), mi('$S/n16000_hf14.json')))
print('$ARM 2k six-question (this) %.4f   %.4f   %.4f   %.4f' % (syn('$E/fresh.json'), syn('$E/seen.json'), mi('$E/val.json'), mi('$E/hf14.json')))
EOF
for split in val hf14; do B=$B_VAL; [ $split = hf14 ] && B=$B_HF14
  python3 scripts/paired_compare.py --a $B --b $E/$split.json \
  --label-a one-q --label-b six-q | grep -m2 "paired mean\|better on" | sed "s/^/  $split vs 2k one-q: /"; done
