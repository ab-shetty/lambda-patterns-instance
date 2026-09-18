#!/usr/bin/env bash
# Single-pass training on a large procedural pool.
#
# Motivated by the classic data-scaling crossover: a weak-prior model that
# loses at 100k samples can win at 10M. The v6 grammar is procedural (0.02 s
# per plan) and measured NOT to saturate -- at 2,000 drawn plans only 4% have a
# near-duplicate neighbour and the NN-distance scaling implies an intrinsic
# dimension of ~23 -- so unique data is effectively unlimited here.
#
# The point is that with enough UNIQUE data you do not need epochs, only steps:
# every step sees a fresh image, so there is nothing to memorise. Tonight's
# runs took ~19,700 steps over 8,292 records (19 repeats each). This takes a
# comparable step budget over 100,000 records (2 repeats each).
#
# Matched against the 12,000-plan run's selected point (e17 = 26,730 steps):
#   12,000 records x 18 epochs = 26,730 steps, 18 repeats/image -> HF14 0.6934
#   100,000 records x 2 epochs = 25,000 steps,  2 repeats/image -> this run
# Same compute, 8.3x the unique images. No warm restarts: those exist to
# recycle a small pool and have no purpose when every step sees new data.
#
# Usage: ./run_v6_singlepass.sh [records] [seed]
set -euo pipefail
N=${1:-100000}
SEED=${2:-7}
SIZE=2048
POOL=data/synthetic/v6d_${N}
CK=data/runs/ck_v6single_${N}_res${SIZE}_seed${SEED}
TAG=$(basename "$CK")
mkdir -p logs data/evaluations

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
  --checkpoint-dir "$CK" --epochs 2 --schedule-epochs 2 \
  --batch-size 8 --num-workers 32 --prefetch-factor 6 \
  --image-max-size $SIZE --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.995 \
  --domain-random --mask-thresh 0.35 --seed $SEED --compile --pad-grid 512 \
  > "logs/$TAG.log" 2>&1

PYTHONPATH=. python3 scripts/select_epoch_on_val.py --runs "$CK" \
  --image-max-size "$SIZE" --out "data/evaluations/val_$TAG.json" \
  > "logs/val_$TAG.log" 2>&1
echo "== $TAG"; cat "data/evaluations/val_$TAG.json"
