#!/usr/bin/env bash
# swin_t synth-only at 2048, ~2k plans, documented two-phase 9-epoch recipe,
# ONE seed: r8 (v6 + R8 flags), revit (FreeCAD massing, --revit), r8+revit.
# Control is the published v6d 2k run (abshetty/floz-refunet-swint-v6d-e8,
# seed 7, same pool ids/recipe: HF14 0.7066), evaluated, not retrained.
# Pools are seed 6, ids 0-1999; the FreeCAD pools use v6's 66/16/18 view mix.
# Scored at epoch 8 (no selection), 2048 and 4096 inference, val + HF14.
# Later arms: revitRL (revit + --real-labelling only: identical images, fewer
# labels) and fcplain (FreeCAD massing, v6 drawing, no renderer, no r8).
# revitnew (2026-09-29): --revit --revit-plans --shaped with the 2026-09-29 label
# defaults (casing holes, trim cut, distinct looks), same ids/seed/view mix. The
# revit control is the published abshetty/floz-refunet-swint-revit2k-e8 when
# the revit arm is not trained here.
# Ablation of revitnew: revitplans = --revit --revit-plans with the pre-09-29 label
# flags (elevations byte-identical to revit); revitshaped = --revit --shaped with
# the 09-29 label defaults, no --revit-plans. revitlocal = the revit pool rebuilt
# and retrained on this machine (same-machine control; val 17 swings 0.2-0.5/epoch).
# revitnewfix = revitnew retrained after 728def3 (reference gets the sheet jitter);
# revitnew itself was trained with the old sheet-only jitter.
# EXTRA="..." appends train flags to every arm of the call. revitnewroi / revitcp50roi
# (run with EXTRA="--roi-ref --roi-ref-mode add", the shipped conditioning): the fixed-
# jitter new pool, and the same pool with --colour-pairs 0.5.
# revit10kroi: 10,000 sheets (ids 0-9999, same recipe as revitnew), ROI, P2=5.
# revitmk50roi / revit10kmk50roi (2026-09-30, built here if missing): revitnew / the 10k pool
# with --markup 0.5 (colour markup painted over half the line-only elevations, labels
# unchanged). Same drawings and labels as revitnew; paired control revitnewroi / revit10kroi
# (published abshetty/floz-refunet-swint-revit2k-roi-e8 / -revit10k-roi-e5). Run with
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 revitmk50roi
#   EXTRA="--roi-ref --roi-ref-mode add" P2=5 ./run_revit_2k.sh 7 revit10kmk50roi
#   ./run_revit_2k.sh [seed] [arms]      e.g. ./run_revit_2k.sh 7 "revitRL fcplain"
set -euo pipefail
SEED=${1:-7}
P2=${P2:-8}          # phase-2 epochs (the documented recipe: 8 of a planned 9)
ARMS=${2:-"r8 revit r8revit"}
declare -A POOL=([r8]=data/synthetic/r8_train2000 [revit]=data/synthetic/revit_train2000
                 [r8revit]=data/synthetic/r8revit_train2000
                 [revitRL]=data/synthetic/revitRL_train2000 [fcplain]=data/synthetic/fcplain_train2000
                 [revitnew]=data/synthetic/revitnew_train2000
                 [revitplans]=data/synthetic/revitplans_train2000
                 [revitshaped]=data/synthetic/revitshaped_train2000
                 [revitlocal]=data/synthetic/revit_train2000
                 [revitnewfix]=data/synthetic/revitnew_train2000
                 [revitnewroi]=data/synthetic/revitnew_train2000
                 [revitcp50roi]=data/synthetic/revitcp50_train2000
                 [revitcp100roi]=data/synthetic/revitcp100_train2000
                 [revit10kroi]=data/synthetic/revitnew_train10000
                 [revitmk50roi]=data/synthetic/revitmk50_train2000
                 [revit10kmk50roi]=data/synthetic/revitmk50_train10000)
declare -A BUILD=([revitmk50roi]="2000 --markup 0.5" [revit10kmk50roi]="10000 --markup 0.5")
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations/revit2k; mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 --image-max-size 2048 \
  --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
  --train-split 0.99 --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512 ${EXTRA:-}"

score() {   # tag checkpoint
  for I in 2048 4096; do for split in val hf14; do IDX=$VAL; [ $split = hf14 ] && IDX=$HF14
    local OUT=$E/$1_i${I}_$split.json
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $2 \
      --indices $IDX --image-max-size $I --ref-size 224 --mask-thresh 0.35 \
      --metrics-out $OUT > /dev/null 2>&1
  done; done
}

CTRL=data/runs/ck_swint_v6d2k_pub/epoch_8.pth
[ -f $CTRL ] || PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py \
  --repo abshetty/floz-refunet-swint-v6d-e8 --out $CTRL
score v6d_s7 $CTRL
RCTRL=data/runs/ck_pub/revit2k-e8.pth
if [[ " $ARMS " != *" revit "* ]]; then
  [ -f $RCTRL ] || PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py \
    --repo abshetty/floz-refunet-swint-revit2k-e8 --out $RCTRL
  score revit_s$SEED $RCTRL
fi

for A in $ARMS; do
  CK=data/runs/ck_2k_${A}_r2048_s$SEED
  if [ -n "${BUILD[$A]:-}" ] && [ ! -f ${POOL[$A]}/generation_manifest.json ]; then
    set -- ${BUILD[$A]}; N=$1; shift
    python3 scripts/generate_synthetic_fc.py --out ${POOL[$A]} --n $N --seed 6 --revit --revit-plans \
      --shaped --workers 64 --mode-weights 66,16,18 "$@" > logs/gen_$(basename ${POOL[$A]}).log 2>&1
  fi
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$A]} \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/2k_${A}_s${SEED}_p1.log 2>&1
  [ -f $CK/epoch_$P2.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data ${POOL[$A]} \
    --checkpoint-dir $CK --epochs $P2 --schedule-epochs $((P2 + 1)) $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/2k_${A}_s${SEED}_p2.log 2>&1
  score ${A}_s$SEED $CK/epoch_$P2.pth
  echo "== done $A"
done

for I in 2048 4096; do for split in val hf14; do
  for A in v6d_s7 revit_s$SEED $(for a in $ARMS; do [ $a = revit ] || echo ${a}_s$SEED; done); do
    python3 -c "import json,numpy as np;d=json.load(open('$E/${A}_i${I}_$split.json'));print(f'infer $I $split  $A  {np.mean([r[\"iou\"] for r in d[\"selections\"]]):.4f}')"
  done
  for a in $ARMS; do A=${a}_s$SEED
    for C in v6d_s7 revit_s$SEED; do [ $C = $A ] && continue
      python3 scripts/paired_compare.py --a $E/${C}_i${I}_$split.json --b $E/${A}_i${I}_$split.json \
        --label-a ${C%_s*} --label-b $a | grep -m2 "paired mean\|better on" | sed "s/^/  $a vs ${C%_s*}: /"
    done
  done
done; done
