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
# NOT RECOMMENDED (2026-09-30 correction: HF14's markup-sheet failures are look-alike
# textures, not paint; see synth_progress.md). Kept for reference:
# revitmk50roi / revit10kmk50roi (2026-09-30, built here if missing): revitnew / the 10k pool
# with --markup 0.5 (colour markup painted over half the line-only elevations, labels
# unchanged). Same drawings and labels as revitnew; paired control revitnewroi / revit10kroi
# (published abshetty/floz-refunet-swint-revit2k-roi-e8 / -revit10k-roi-e5). Run with
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 revitmk50roi
#   EXTRA="--roi-ref --roi-ref-mode add" P2=5 ./run_revit_2k.sh 7 revit10kmk50roi
# revit10kcurroi / revit10kfailroi (2026-10-02, built here if missing): the 10k recipe on pools
# rebuilt with TODAY's defaults (plan-v2 1, cedar 0.5, masonry 0.35, openings-per-face 1), without
# and with the four options aimed at the synthetic-only HF14 failures (synth_progress.md "Synthetic-
# only failures on HF14"): --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2.
# failroi vs curroi isolates the four options; curroi vs revit10k-roi-e5 the default changes.
#   EXTRA="--roi-ref --roi-ref-mode add" P2=5 ./run_revit_2k.sh 7 "revit10kcurroi revit10kfailroi"
# revitcurroi / revitfailroi (2026-10-02): the same two pools at 2k (ids 0-1999), the 2k ROI screen
# recipe (P2=8). 10k vs 2k ROI was flat on HF14 (+0.001 / +0.016 n.s.), so screen here first;
# curroi vs the published revit2k-roi-e8 measures the default changes since 09-30.
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 "revitcurroi revitfailroi"
# revithardroi (2026-10-02): 1,000 revitcurroi sheets + 1,000 mined semi-hard fresh sheets
# (built by run_hard_mine.sh / build_hard_pool.py, not here); paired control revitcurroi.
# revitfailswissroi (2026-10-06): revitfailroi's pool (same seed / ids / flags) with floor plans from
# real Swiss Dwellings layouts (--plan-source, p 1.0; build the file with scripts/swiss_plans.py).
# Only floor-plan sheets differ, so it pairs with revitfailroi.
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 revitfailswissroi
# revitgemroi (2026-10-06): revitfailswissroi's pool + Gemini-like sheets (--tight-crop --pale-ink 0.25
# --revit-view-weights 50,45,5: framing, piece size, pale line work; --paper left out on purpose).
# Pairs with revitfailswissroi.
# revitnearroi (2026-10-08): revitfailswissroi's flags, ids and seed with the 2026-10-07 generator defaults
# (near-miss family pairs on elevations and finish floor plans, balcony doors) AND the 2026-10-08 training
# default --ref-min-side 32. revitfailswissr32roi: the control pool rebuilt with the new defaults switched off
# (byte-identical to revitfailswiss_train2000) and trained with --ref-min-side 32: isolates the pool change
# from the reference floor. Published control (ref floor off): abshetty/floz-refunet-swint-revitfailswiss2k-e8.
# Every arm also gets Gemini30 hand boxes and the invariance probe at 2048 (extra_scores below).
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 "revitnearroi revitfailswissr32roi"
#   ./run_revit_2k.sh [seed] [arms]      e.g. ./run_revit_2k.sh 7 "revitRL fcplain"
set -euo pipefail
. scripts/pool_guard.sh     # need_pool: never silently reuse a pool from older generator defaults
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
                 [revit10kmk50roi]=data/synthetic/revitmk50_train10000
                 [revit10kcurroi]=data/synthetic/revitcur_train10000
                 [revit10kfailroi]=data/synthetic/revitfail_train10000
                 [revitcurroi]=data/synthetic/revitcur_train2000
                 [revitfailroi]=data/synthetic/revitfail_train2000
                 [revitfailswissroi]=data/synthetic/revitfailswiss_train2000
                 [revitgemroi]=data/synthetic/revitgem_train2000
                 [revitnearroi]=data/synthetic/revitnear_train2000
                 [revitfailswissr32roi]=data/synthetic/revitfailswissR_train2000
                 [revithardroi]=data/synthetic/revitcur_hard2000
                 [revithardfailroi]=data/synthetic/revitfail_hard2000)
declare -A BUILD=([revitmk50roi]="2000 --markup 0.5" [revit10kmk50roi]="10000 --markup 0.5"
                  [revit10kcurroi]="10000"
                  [revit10kfailroi]="10000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2"
                  [revitcurroi]="2000"
                  [revitfailroi]="2000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2"
                  [revitfailswissroi]="2000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz"
                  [revitnearroi]="2000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz"
                  [revitfailswissr32roi]="2000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz --near-pairs 0 --near-pairs-plan 0 --balcony-doors 0"
                  [revitgemroi]="2000 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz --tight-crop --pale-ink 0.25 --revit-view-weights 50,45,5")
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
E=data/evaluations/revit2k; mkdir -p logs $E
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 --image-max-size 2048 \
  --ref-size 224 --width 128 --model swin_t --lr 2e-4 --backbone-lr-mult 0.1 \
  --train-split 0.99 --domain-random --mask-thresh 0.35 --seed $SEED --pad-grid 512 ${EXTRA:-}"

score() {   # tag checkpoint   (hf14fix = HF14 with eval_labels/hf14_fixes_v1.json, 2026-10-02)
  for I in 2048 4096; do for split in val hf14 hf14fix; do IDX=$VAL; [ $split != val ] && IDX=$HF14
    local OUT=$E/$1_i${I}_$split.json FIX=""
    [ $split = hf14fix ] && FIX="--label-fixes eval_labels/hf14_fixes_v1.json"
    [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $2 \
      --indices $IDX --image-max-size $I --ref-size 224 --mask-thresh 0.35 $FIX \
      --metrics-out $OUT > /dev/null 2>&1
  done; done
}

extra_scores() {   # tag checkpoint: Gemini30 (user's boxes) + invariance probe, 2048 (2026-10-08)
  local G=data/eval_pools/gemini30_s20261006 P=data/eval_pools/probe_invariance_v1
  [ -d $G ] || python3 scripts/build_gemini30_pool.py
  [ -d $P ] || python3 scripts/probes/make_invariance_probe.py $P > /dev/null
  [ -f $E/$1_i2048_gemini30.json ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $2 \
    --local-pool $G --boxes eval_boxes/gemini30_v1.json --image-max-size 2048 --mask-thresh 0.35 \
    --metrics-out $E/$1_i2048_gemini30.json > /dev/null 2>&1
  [ -f $E/$1_i2048_probe.json ] || { PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $2 \
    --local-pool $P --boxes $P/boxes.json --image-max-size 2048 --mask-thresh 0.35 \
    --metrics-out $E/$1_i2048_probe.json --save-probs $E/$1_i2048_probe_probs > /dev/null 2>&1 && \
    python3 scripts/probes/invariance_leak.py --metrics $E/$1_i2048_probe.json --probs $E/$1_i2048_probe_probs \
    --json $E/$1_i2048_probe_leak.json > $E/$1_i2048_probe_leak.txt; }
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
  if [ -n "${BUILD[$A]:-}" ] && need_pool ${POOL[$A]}; then
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
  extra_scores ${A}_s$SEED $CK/epoch_$P2.pth
  echo "== done $A"
done

# 10k pair (2026-10-02): also score the published 10k control, so the runbook's two comparisons print
R10=data/runs/ck_pub/revit10k-roi-e5.pth
if [[ " $ARMS " == *" revit10kcurroi "* ]]; then
  [ -f $R10 ] || PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py \
    --repo abshetty/floz-refunet-swint-revit10k-roi-e5 --out $R10
  score revit10k_pub $R10
fi

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

# the 10k pair: failroi vs curroi (the four options), curroi vs the published 10k (default changes),
# plus per-sheet means on the sheets the options target (HF14 25, 27, 14, 02)
if [[ " $ARMS " == *" revit10kcurroi "* && " $ARMS " == *" revit10kfailroi "* ]]; then
  CU=revit10kcurroi_s$SEED; FA=revit10kfailroi_s$SEED
  for I in 2048 4096; do for split in val hf14 hf14fix; do
    python3 scripts/paired_compare.py --a $E/${CU}_i${I}_$split.json --b $E/${FA}_i${I}_$split.json \
      --label-a curroi --label-b failroi | grep -m2 "paired mean\|better on" | sed "s/^/  [$I $split] failroi vs curroi: /"
    python3 scripts/paired_compare.py --a $E/revit10k_pub_i${I}_$split.json --b $E/${CU}_i${I}_$split.json \
      --label-a revit10k-roi-e5 --label-b curroi | grep -m2 "paired mean\|better on" | sed "s/^/  [$I $split] curroi vs revit10k-roi-e5: /"
  done; done
  python3 - "$E" "$CU" "$FA" <<'PY'
import json, sys, collections
E, cu, fa = sys.argv[1:]
for tag in ("revit10k_pub", cu, fa):
    d = json.load(open(f"{E}/{tag}_i4096_hf14.json"))["selections"]
    by = collections.defaultdict(list)
    for r in d:
        by[r["image_index"]].append(r["iou"])
    print(f"  4096 hf14 {tag:28s}", "  ".join(f"{i}: {sum(by[i]) / len(by[i]):.3f}" for i in (25, 27, 14, 2) if by[i]))
PY
fi
