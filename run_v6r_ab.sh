#!/usr/bin/env bash
# Does matching v6d's region scale to real plans raise HF14 on swin_t?
#
# The gap (measured 2026-09-21, scratchpad/region_scale.py): v6d regions occupy
# 0.00896 of the sheet against real's 0.02411 -- 2.7x too small -- and v6d puts
# 11.7 of them on a sheet against real's 3.3. A model trained on that learns a
# "find small things" prior that is wrong by ~2.7x on every real plan.
#
# WHY SWIN_T AND NOT REFUNET. The three realism levers tried in August
# (--realism-aug 0.7398, ink-matched 0.7218, faint+high-area 0.7227, all against
# a 0.7672 baseline) were ALL measured on RefUNet, and RefUNet is the backbone
# with the least to gain: its HF14 *rises* with more synthetic training
# (0.5993 -> 0.6513 over the 100k run) while swin_t's *falls* (0.7484 -> 0.6913).
# swin_t is the model visibly damaged by the synth-real gap, so it is the one
# where closing that gap has something to give back. Note also that the same
# August table lists "resolution 2048" as a loser at 0.7524 -- now the project's
# largest win at +0.05-0.07 -- so those one-seed screens have a proven false
# negative on the biggest lever in the repo.
#
# LABELLED-AREA TRAP. Ink-matching failed in August because density realism cost
# labelled area (0.387 -> 0.286). Matching real's union/sheet directly would be
# worse: 0.373 -> 0.137, a 63% cut. Moving region SIZE up and region COUNT down
# together roughly cancels -- v6r lands at 0.326, a 13% reduction. Checked
# before training, not after.
#
# BASELINE: swin_t on v6d 1,995 plans @2048, same two-phase 9-epoch recipe,
# val-selected = 0.7066 (codex_doc.md:16, one seed, +0.0707 over RefUNet,
# t(51)=2.20 p=0.032). Both pools regenerated at seed 6, both 1,995 plans, same
# mode split (1333/310/352). Only --view-count-weights and --max-label-fams
# differ. One seed each, so read a move under ~0.024 (the repo's HF14 sd) as
# noise.
set -euo pipefail
cd /home/ubuntu/lambda-patterns-instance
set -a; . /home/ubuntu/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
POOL=${POOL:-data/synthetic/v6r_2000}
SEED=${SEED:-7}
SIZE=${SIZE:-2048}
CK=data/runs/ck_v6r_swint_res${SIZE}_seed${SEED}
TAG=$(basename "$CK")
mkdir -p logs data/runs data/evaluations

COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size $SIZE --ref-size 224 --width 128 --model swin_t \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512"

if [ ! -f "$CK/epoch_0.pth" ]; then
  echo "== phase 1: epoch 0 (schedule 10)"
  python3 scripts/train_refunet.py --local-data "$POOL" \
    --checkpoint-dir "$CK" --epochs 1 --schedule-epochs 10 $COMMON \
    > "logs/${TAG}_p1.log" 2>&1
fi
echo "== phase 2: epochs 1-8 (schedule 9, optimizer reset)"
python3 scripts/train_refunet.py --local-data "$POOL" \
  --checkpoint-dir "$CK" --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from "$CK/epoch_0.pth" \
  > "logs/${TAG}_p2.log" 2>&1

echo "== val-selected epoch (real validation complement, never HF14)"
python3 scripts/select_epoch_on_val.py --runs "$CK" \
  --image-max-size "$SIZE" --mask-thresh 0.35 \
  --out "data/evaluations/val_${TAG}.json" > "logs/val_${TAG}.log" 2>&1
echo "== $TAG   baseline swin_t/v6d = 0.7066"
cat "data/evaluations/val_${TAG}.json"
