#!/usr/bin/env bash
# Retrain the winning mix with the fixed reference-box sampler.
# Old sampler left 57% of real-plan reference crops partially/entirely outside
# their own mask (32x32 centroid fallback); the fix guarantees containment.
# Three seeds so the result is comparable to the 0.6331 +- 0.0132 old-sampler mean.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35"
for seed in 31 7 99; do
  ck="data/runs/ck_fix_mix3652_seed${seed}"
  python3 scripts/train_refunet.py --local-data data/mixed/toparea1600_rf1548_gen504 \
    --checkpoint-dir "$ck" --epochs 1 --schedule-epochs 10 --seed $seed $COMMON \
    > "logs_fix_mix3652_seed${seed}_A.log" 2>&1
  python3 scripts/train_refunet.py --local-data data/mixed/toparea1600_rf1548_gen504 \
    --checkpoint-dir "$ck" --epochs 8 --schedule-epochs 9 --seed $seed $COMMON \
    --reset-optimizer --init-from "$ck/epoch_0.pth" \
    > "logs_fix_mix3652_seed${seed}_B.log" 2>&1
done
echo "FIXED SAMPLER RUNS DONE"
