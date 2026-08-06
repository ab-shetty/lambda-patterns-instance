#!/usr/bin/env bash
# Ablation: what do the 28 generated-realistic plans add on top of the documented mix?
#
#   rf1548   86 unique real sources only            (isolates the real pool)
#   mix3148  1600 synth + rf1548                    (documented composition)
#   mix3652  1600 synth + rf1548 + gen504           (documented composition + the 28)
#
# mix3148/mix3652 use the two-phase schedule that produced 0.6127448856: actual
# epoch 0 on a planned 10-epoch cosine, then a reset optimizer and actual epochs
# 1-8 on the first eight of a planned 9-epoch cosine. --mask-thresh 0.35 makes
# the per-epoch real_mIoU directly comparable to the accepted result.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

COMMON="--batch-size 8 --num-workers 48 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31 --mask-thresh 0.35"

# Single-phase control on the real pool alone.
python3 scripts/train_refunet.py --local-data data/roboflow/floz-real-pool-v2-strong18 \
  --checkpoint-dir data/runs/ck_refunet_rf1548_s31 \
  --epochs 10 --schedule-epochs 10 $COMMON > logs_rf1548.log 2>&1

for pair in "mix3148:data/mixed/toparea1600_rfstrong1548" \
            "mix3652:data/mixed/toparea1600_rf1548_gen504"; do
  tag="${pair%%:*}"; data="${pair##*:}"; ck="data/runs/ck_refunet_${tag}_s31"
  python3 scripts/train_refunet.py --local-data "$data" --checkpoint-dir "$ck" \
    --epochs 1 --schedule-epochs 10 $COMMON > "logs_${tag}_phaseA.log" 2>&1
  python3 scripts/train_refunet.py --local-data "$data" --checkpoint-dir "$ck" \
    --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$ck/epoch_0.pth" > "logs_${tag}_phaseB.log" 2>&1
done

echo "ALL RUNS DONE"
