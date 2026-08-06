#!/usr/bin/env bash
# Replicate the headline mix3148-vs-mix3652 comparison at two more seeds.
# The +0.036 gain from adding 504 generated-plan records is the main claim of
# this session; a single seed cannot distinguish it from run-to-run variance.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35"
for seed in 7 99; do
  for pair in "mix3148:data/mixed/toparea1600_rfstrong1548" \
              "mix3652:data/mixed/toparea1600_rf1548_gen504"; do
    tag="${pair%%:*}"; data="${pair##*:}"; ck="data/runs/ck_rep_${tag}_seed${seed}"
    python3 scripts/train_refunet.py --local-data "$data" --checkpoint-dir "$ck" \
      --epochs 1 --schedule-epochs 10 --seed $seed $COMMON \
      > "logs_rep_${tag}_seed${seed}_A.log" 2>&1
    python3 scripts/train_refunet.py --local-data "$data" --checkpoint-dir "$ck" \
      --epochs 8 --schedule-epochs 9 --seed $seed $COMMON \
      --reset-optimizer --init-from "$ck/epoch_0.pth" \
      > "logs_rep_${tag}_seed${seed}_B.log" 2>&1
  done
done
echo "REPLICATE DONE"
