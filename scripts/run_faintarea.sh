#!/usr/bin/env bash
# faintarea1600: ink <= 9% then ranked by labelled area (ink median 1.55%,
# area median 0.360). Pure ink quantile-matching lost on validation because it
# also halved labelled area; this keeps the density match AND the supervision.
# Also screens a longer schedule: validation peaked at epoch 7 of 9 on all three
# baseline seeds, suggesting the cosine ends too early.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
BASE="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"

ck=data/runs/ck_faintarea_seed31
D=data/mixed/faintarea1600_rf1548_gen504
python3 scripts/train_refunet.py --local-data $D --checkpoint-dir $ck \
  --epochs 1 --schedule-epochs 10 $BASE > logs_faintarea_A.log 2>&1
python3 scripts/train_refunet.py --local-data $D --checkpoint-dir $ck \
  --epochs 8 --schedule-epochs 9 $BASE --reset-optimizer \
  --init-from $ck/epoch_0.pth > logs_faintarea_B.log 2>&1

# longer schedule on the unchanged baseline mix
ck2=data/runs/ck_long15_seed31
D2=data/mixed/toparea1600_rf1548_gen504
python3 scripts/train_refunet.py --local-data $D2 --checkpoint-dir $ck2 \
  --epochs 1 --schedule-epochs 16 $BASE > logs_long15_A.log 2>&1
python3 scripts/train_refunet.py --local-data $D2 --checkpoint-dir $ck2 \
  --epochs 14 --schedule-epochs 15 $BASE --reset-optimizer \
  --init-from $ck2/epoch_0.pth > logs_long15_B.log 2>&1
echo "FAINTAREA+LONG DONE"
