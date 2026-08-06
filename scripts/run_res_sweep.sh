#!/usr/bin/env bash
# Resolution sweep, decided on the 14-image validation split (never HF14).
# HF eval plans run to 3506px; at --image-max-size 1280 they are downscaled 2.7x
# and thin wall poche (the largest remaining failure, image 12) is destroyed.
# Single seed for screening; the winner gets replicated at 3 seeds.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
DATA=data/mixed/toparea1600_rf1548_gen504
for res in 1792 2048; do
  ck="data/runs/ck_res${res}_seed31"
  BS=8; [ "$res" -ge 2048 ] && BS=4
  COMMON="--batch-size $BS --num-workers 24 --prefetch-factor 4 \
    --image-max-size $res --ref-size 224 --width 128 \
    --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    --domain-random --mask-thresh 0.35 --seed 31"
  python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir "$ck" \
    --epochs 1 --schedule-epochs 10 $COMMON > "logs_res${res}_A.log" 2>&1
  python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir "$ck" \
    --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$ck/epoch_0.pth" > "logs_res${res}_B.log" 2>&1
done
echo "RES SWEEP DONE"
