#!/usr/bin/env bash
# Separate run-to-run (training seed) noise from which-28-you-drew noise.
# real28_s1 and the generated 28 are each retrained at two extra seeds; the
# spread within a fixed dataset is run noise, so the real28 draw spread
# (sd 0.030 over three draws) can be attributed properly.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
COMMON="--epochs 10 --schedule-epochs 10 --batch-size 8 --num-workers 16 \
  --prefetch-factor 4 --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 --domain-random --mask-thresh 0.35"
for seed in 7 99; do
  for tag in real28_s1 floz-genreal-v1; do
    python3 scripts/train_refunet.py --local-data "data/roboflow/${tag}-strong18" \
      --checkpoint-dir "data/runs/ck_noise_${tag}_seed${seed}" --seed $seed $COMMON \
      > "logs_noise_${tag}_seed${seed}.log" 2>&1
  done
done
echo "SEED NOISE DONE"
