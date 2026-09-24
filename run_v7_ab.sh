#!/usr/bin/env bash
# v7 vs v6d, synthetic-only, swin_t, 2048, matched count (1,995) and recipe.
#
# The v6d arm is already on record: `run_backbone_ab.sh` swin_t seed 7 =
# 0.7066 val-selected, published as abshetty/floz-refunet-swint-v6d-e8. This
# trains ONLY the v7 arm with the identical recipe, re-evaluates the published
# v6d checkpoint on this machine (a reproduction check of the setup), and
# compares the two PAIRED over the 52 fixed HF14 selections.
#
#   ./run_v7_ab.sh [seed] [arm]      # ~25 min on a GH200
#   arm: v7 (default) | v7roof (elevation roofs labelled as v6d does)
set -euo pipefail
SEED=${1:-7}
ARM=${2:-v7}
SIZE=2048
case $ARM in
  v7) GEN_ARGS="" ;;
  v7roof) GEN_ARGS="--label-roofs" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
POOL=data/synthetic/${ARM}_train1995
CK=data/runs/ck_ab_swin_t_${ARM}_res${SIZE}_seed${SEED}
TAG=$(basename "$CK")
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "$POOL" ]; then
  # same ids and build seed as v6d_train2000 (seed 6, ids 0-1999), then trim to
  # the v6d pool's 1,995 so both arms see the same number of optimizer steps
  python3 generate_synthetic_v7.py --n 2000 --out "$POOL" --seed 6 --start 0 --workers 64 $GEN_ARGS
  ls "$POOL"/annotations | sort | tail -n +1996 | while read -r f; do
    rm "$POOL/annotations/$f" "$POOL/images/${f%.json}.png"
  done
fi
echo "pool: $(ls "$POOL"/images | wc -l) images"

COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
if [ ! -f "$CK/epoch_0.pth" ]; then
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON > "logs/${TAG}_p1.log" 2>&1
fi
if [ ! -f "$CK/epoch_8.pth" ]; then
  PYTHONPATH=. python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
    --reset-optimizer --init-from "$CK/epoch_0.pth" > "logs/${TAG}_p2.log" 2>&1
fi
echo "== $TAG trained"

PYTHONPATH=. python3 scripts/select_epoch_on_val.py --runs "$CK" --image-max-size $SIZE \
  --out "data/evaluations/${TAG}_valsel.json" | tee "logs/${TAG}_valsel.log"
