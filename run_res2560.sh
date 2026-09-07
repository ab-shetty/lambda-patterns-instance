#!/usr/bin/env bash
# Resolution past 2048: the documented recipe at --image-max-size 2560.
#   ./run_res2560.sh <local-data> <tag> <seed> [batch]
# 1280 -> 2048 was worth +0.05-0.07 and the Gemini pool's median long side is
# 3168, so 2048 may still truncate it. Batch stays 8 unless memory forces 4.
set -uo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
DATA=$1; TAG=$2; SEED=$3; BS=${4:-8}
CK=data/runs/ck_${TAG}_res2560_seed$SEED
COMMON="--batch-size $BS --num-workers 16 --prefetch-factor 4 --image-max-size 2560 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 --domain-random --mask-thresh 0.35 --seed $SEED --compile --pad-grid 512"
python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/$(basename $CK).log 2>&1
python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from $CK/epoch_0.pth >> logs/$(basename $CK).log 2>&1
python3 scripts/select_epoch_on_val.py --runs $CK --image-max-size 2560 --out data/evaluations/val_$(basename $CK).json > logs/val_$(basename $CK).log 2>&1
echo "RES2560 $TAG s$SEED DONE $(date)"
