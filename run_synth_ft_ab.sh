#!/usr/bin/env bash
# Does a synthetic-data change help the best real-data model? Paired fine-tune A/B.
#
# Both arms start from the current best checkpoint (restart e13, HF14 0.8170
# @4096) and get an identical gentle fine-tune (lr 5e-5, 3 epochs, reset
# optimizer); they differ ONLY in the synthetic quarter of the mix:
#   control    data/mixed/v6dmix_plus_r4        (v6d_1600)
#   treatment  $TREAT_DATA                       (default v6dHS: --hardscape-plan
#              0.5 --same-fill-subtle 0.5, from the 2026-09-24 visual audit)
# Each epoch is scored on VALIDATION at 4096. Pre-declared HF14 comparison of
# the two final checkpoints (one read each), paired, with the audit's targets
# called out: HF14 images 0 and 12 (floor plans, exterior hardscape) for the
# hardscape change -- validation has no floor plans, so it cannot judge it.
#
#   ./run_synth_ft_ab.sh            # ~1.2 h on a GH200
set -euo pipefail
BASE=data/runs/ck_swint_mixr4_restart_full/epoch_13.pth
TREAT_DATA=${TREAT_DATA:-data/mixed/v6dHSmix_plus_r4}
TREAT_TAG=${TREAT_TAG:-v6dHS}
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations
mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARM in control $TREAT_TAG; do
  DATA=data/mixed/v6dmix_plus_r4; [ $ARM = control ] || DATA=$TREAT_DATA
  CK=data/runs/ck_ftab_${ARM}
  [ -f $CK/epoch_16.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
    --checkpoint-dir $CK --epochs 3 --schedule-epochs 3 \
    --batch-size 8 --num-workers 16 --prefetch-factor 4 \
    --image-max-size 2048 --ref-size 224 --width 128 --model swin_t \
    --lr 5e-5 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed 7 --pad-grid 512 \
    --reset-optimizer --init-from $BASE > logs/ftab_${ARM}.log 2>&1
  for e in 14 15 16; do
    out=$E/ftab_${ARM}_e${e}_val4096.json
    [ -f $out ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $CK/epoch_$e.pth \
      --indices $VAL --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --metrics-out $out > /dev/null 2>&1
  done
  echo "== $ARM trained"
done

echo "== validation @4096 (targets: thin bucket, img17)"
PYTHONPATH=. python3 scripts/val_failures.py $E/ck_swint_mixr4_restart_full_restart_e13_val4096.json \
  $E/ftab_control_e1{4,5,6}_val4096.json $E/ftab_${TREAT_TAG}_e1{4,5,6}_val4096.json 2>/dev/null

echo "== pre-declared HF14 comparison, final epochs @4096 (one read per arm)"
for ARM in control $TREAT_TAG; do
  out=$E/ftab_${ARM}_e16_hf14_4096.json
  [ -f $out ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py \
    --checkpoint data/runs/ck_ftab_${ARM}/epoch_16.pth --indices $HF14 --image-max-size 4096 \
    --ref-size 224 --mask-thresh 0.35 --metrics-out $out > /dev/null 2>&1
done
python3 scripts/paired_compare.py --a $E/ftab_control_e16_hf14_4096.json \
  --b $E/ftab_${TREAT_TAG}_e16_hf14_4096.json --label-a control --label-b $TREAT_TAG
