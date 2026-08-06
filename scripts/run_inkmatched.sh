#!/usr/bin/env bash
# Ink-matched synthetic selection vs the top-area default.
# toparea1600 has ~13.9% median ink coverage; real plans have ~5.5%. The model
# therefore trains on dense sheets and over-predicts on sparse faint ones --
# images 14/12/7 (1.7-4.7% ink) carry 14 of 52 selections at mean ~0.36.
# inkmatched1600 quantile-matches ink to the generated-real pool (4.68% median).
# Screening at one seed; decided on validation, never HF14.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
DATA=data/mixed/inkmatched1600_rf1548_gen504
ck=data/runs/ck_inkmatched_seed31
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"
python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir "$ck" \
  --epochs 1 --schedule-epochs 10 $COMMON > logs_inkmatched_A.log 2>&1
python3 scripts/train_refunet.py --local-data $DATA --checkpoint-dir "$ck" \
  --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from "$ck/epoch_0.pth" > logs_inkmatched_B.log 2>&1
echo "INKMATCHED DONE"
