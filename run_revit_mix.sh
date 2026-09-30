#!/usr/bin/env bash
# The shipped 0.8440 chain with the Revit synthetic source (2026-09-30): the
# `v6dmix_plus_r4` recipe with v6d_1600 swapped for the first 1,600 sheets of
# data/synthetic/revitnew_train10000 (generate_synthetic_fc.py --revit --revit-plans
# --shaped --mode-weights 66,16,18, seed 6; the pool's first 1,990 ids are the
# screened revitnew 2k pool). Real / generated / Gemini r2-r4 sources unchanged.
# Stage 1 here (documented 1+8 epochs, no ROI, as mixr4-e8); stage 2 is
#   BASE=$CK/epoch_8.pth DATA=data/mixed/revitmix_plus_r4 EXTRA="--roi-ref --roi-ref-mode add" \
#     CK=data/runs/ck_revitmix_roiadd CMP=<shipped hf14 json> ./run_restart_swa.sh
# Trained after 728def3, so the reference also gets --domain-random's jitter.
#   ./run_revit_mix.sh [seed]
set -euo pipefail
SEED=${1:-7}
SIZE=2048
DATA=data/mixed/revitmix_plus_r4
CK=data/runs/ck_swint_revitmix_r4_res${SIZE}_seed${SEED}
TAG=$(basename $CK)
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
mkdir -p logs data/evaluations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

augment18() {  # SRC OUT: 17 strong augs per scene, split across 64 processes
  local src=$1 out=$2 tmp=$2.chunks
  [ -d "$out" ] && return
  rm -rf "$tmp"; mkdir -p "$tmp"
  python3 - "$src" "$tmp" <<'PY'
import os, sys
src, tmp = sys.argv[1:]
imgs = {os.path.splitext(f)[0]: f for f in os.listdir(f"{src}/images")}
for i, x in enumerate(sorted(os.listdir(f"{src}/annotations"))):
    c = f"{tmp}/in{i % 64:02d}"
    os.makedirs(f"{c}/images", exist_ok=True); os.makedirs(f"{c}/annotations", exist_ok=True)
    stem = os.path.splitext(x)[0]
    os.link(f"{src}/annotations/{x}", f"{c}/annotations/{x}")
    os.link(f"{src}/images/{imgs[stem]}", f"{c}/images/{imgs[stem]}")
PY
  for c in "$tmp"/in*; do
    python3 scripts/augment_local_dataset.py --src "$c" --out "$tmp/out${c##*/in}" \
      --aug-per-scene 17 --strong --seed 5858 > /dev/null &
  done
  wait
  python3 scripts/merge_local_datasets.py --out "$out" --sources "$tmp"/out* > /dev/null
  rm -rf "$tmp"
  echo "$out: $(ls "$out"/images | wc -l) records"
}

# ---- sources ------------------------------------------------------------------
RF=data/roboflow
if [ ! -d $RF/floz-real-pool-v2-clean ] || [ ! -d $RF/floz-genreal-v1-clean ]; then
  python3 - <<'PY'
import os
from roboflow import Roboflow
ws = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("perceive-ai")
ws.project("floz-real-pool").version(2).download(
    "coco-segmentation", location="data/roboflow/floz-real-pool-v2-raw", overwrite=True)
ws.project("floz-generated-realistic-label-pool").version(1).download(
    "coco-segmentation", location="data/roboflow/floz-genreal-v1-raw", overwrite=True)
PY
  for p in floz-real-pool-v2 floz-genreal-v1; do
    python3 scripts/roboflow_to_local.py --coco $RF/$p-raw/train/_annotations.coco.json \
      --img-dir $RF/$p-raw/train --out $RF/$p-clean | tail -1
  done
fi
for r in r2 r3 r4; do
  [ -d $RF/floz-gen-gemini-$r-clean ] || python3 scripts/roboflow_to_local.py \
    --coco $RF/floz-gen-gemini-$r-raw/train/_annotations.coco.json \
    --img-dir $RF/floz-gen-gemini-$r-raw/train --out $RF/floz-gen-gemini-$r-clean | tail -1
done
[ -d $RF/floz-gen-gemini-r234-clean ] || python3 scripts/merge_local_datasets.py \
  --out $RF/floz-gen-gemini-r234-clean --sources $RF/floz-gen-gemini-r2-clean \
  $RF/floz-gen-gemini-r3-clean $RF/floz-gen-gemini-r4-clean > /dev/null
# revit_1600 = the first 1,600 sheets (sorted ids) of the 10k Revit pool
V6=data/synthetic/revit_1600
if [ ! -d $V6 ]; then
  mkdir -p $V6/images $V6/annotations
  ls data/synthetic/revitnew_train10000/annotations | sort | head -1600 | while read -r f; do
    ln data/synthetic/revitnew_train10000/annotations/$f $V6/annotations/$f
    ln data/synthetic/revitnew_train10000/images/${f%.json}.png $V6/images/${f%.json}.png
  done
fi

# all three augmentations at once: 3 x 64 processes
augment18 $RF/floz-real-pool-v2-clean  $RF/floz-real-pool-v2-strong18 &
augment18 $RF/floz-genreal-v1-clean    $RF/floz-genreal-v1-strong18 &
augment18 $RF/floz-gen-gemini-r234-clean $RF/floz-gen-gemini-r234-strong18 &
wait
[ -d $DATA ] || python3 scripts/merge_local_datasets.py --out $DATA --sources $V6 \
  $RF/floz-real-pool-v2-strong18 $RF/floz-genreal-v1-strong18 \
  $RF/floz-gen-gemini-r234-strong18 > /dev/null
echo "mix: $(ls $DATA/images | wc -l) records (v6d version: 6,676 rebuilt)"

# ---- train ------------------------------------------------------------------------
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"
[ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/${TAG}_p1.log 2>&1
[ -f $CK/epoch_8.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from $CK/epoch_0.pth > logs/${TAG}_p2.log 2>&1
echo "== trained"

