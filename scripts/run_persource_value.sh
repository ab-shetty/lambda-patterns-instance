#!/usr/bin/env bash
# Per-source value: is one generated plan worth more or less than one real plan?
#
# The 86 real sources are all 640x640 web-scraped drawings (~0.41 MP, 2 instances,
# 15% labelled area); the 28 generated are 1.5 MP, 3 instances, 28% labelled area.
# Comparing 86-vs-28 confounds count with content, so every run here uses exactly
# 28 unique sources expanded to 504 strong18 records:
#
#   real28_s1/s2/s3   three random 28-draws from the real pool (draw noise)
#   genreal28_640     the 28 generated, pre-downscaled to a 640 long side
#
# genreal28_640 vs the native-resolution gen504 run (0.2650) isolates resolution;
# genreal28_640 vs the real28 mean isolates content at matched resolution.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

COMMON="--epochs 10 --schedule-epochs 10 \
  --batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31 --mask-thresh 0.35"

for tag in real28_s1 real28_s2 real28_s3 genreal28_640; do
  python3 scripts/train_refunet.py \
    --local-data "data/roboflow/${tag}_strong18" \
    --checkpoint-dir "data/runs/ck_refunet_${tag}_s31" \
    $COMMON > "logs_${tag}.log" 2>&1
done

echo "PERSOURCE RUNS DONE"
