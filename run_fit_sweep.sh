#!/usr/bin/env bash
# Is the synthetic-only ceiling a capacity limit? (2026-09-30)
#
# The 10k Revit ROI model scores its own training sheets no better than fresh ones
# (0.807 vs 0.868; 3+ families 0.76-0.79 seen): it underfits multi-family sheets. Before any
# 10k run, answer the capacity question at small scale:
#
#   Stage A (fit, ~1-2 h): swin_t / swin_s / swin_b trained on the SAME 500 sheets with >= 3
#     labelled families (revitnew recipe), 1 + P2A epochs, ROI add, domain-random on (the
#     recipe that underfits). Scored on 40 of those training sheets ("seen") and 40 fresh
#     >= 3-family sheets, by family count. Capacity-bound = swin_t's seen score stalls where a
#     bigger backbone's keeps climbing. NODR=1 repeats it without domain-random (separates
#     the shrink from capacity; the 64-plan overfit reached 0.90 that way).
#   Stage B (size curves, ~3-4 h): BACKBONES_B on 1k / 2k / 4k sheets of the standard pool
#     (ids 0-3999, all family counts, the 10k pool's first sheets), 1 + P2B epochs, scored
#     on 60 fresh sheets (ids 200000+). A steeper curve for the bigger backbone = worth a 10k
#     run; parallel curves = not.
#
# Only after that: one 10k run with the chosen backbone, HF14 once vs revit10k-roi-e5.
#
#   ./run_fit_sweep.sh A          # or B, or AB
#   env: SEED=7 P2A=15 P2B=8 BACKBONES_A="swin_t swin_s swin_b" BACKBONES_B="swin_t swin_s"
#        NODR=1 (no --domain-random) BS=8 (lower for all arms together if swin_b OOMs)
#        SMOKE=1 (tiny CPU-sized run of the whole chain, for checking the script only)
set -euo pipefail
cd "$(dirname "$0")"
STAGES=${1:-A}
SEED=${SEED:-7}
P2A=${P2A:-15}
P2B=${P2B:-8}
BACKBONES_A=${BACKBONES_A:-"swin_t swin_s swin_b"}
BACKBONES_B=${BACKBONES_B:-"swin_t swin_s"}
BS=${BS:-8}
SIZE=2048; WORKERS=64; NW=16; NA=3000; NSEL=500; NSEEN=40; NFRESH=40; NB="1000 2000 4000"; NFB=60
S=data/synthetic; E=data/evaluations/fit_sweep; R=data/runs
if [ "${SMOKE:-0}" = 1 ]; then
  SIZE=512; WORKERS=4; NW=2; NA=60; NSEL=12; NSEEN=3; NFRESH=3; NB="12 24"; NFB=4
  P2A=1; P2B=1; BS=2; BACKBONES_A=${BACKBONES_A_SMOKE:-"swin_t swin_s"}; BACKBONES_B="swin_t"
  S=data/smoke_fit/synthetic; E=data/smoke_fit/eval; R=data/smoke_fit/runs
  SMOKEX="--no-pretrained"
fi
DR="--domain-random"; TAGX=""
[ "${NODR:-0}" = 1 ] && { DR=""; TAGX="_nodr"; }
mkdir -p logs $E $S $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REVIT="--seed 6 --revit --revit-plans --shaped --mode-weights 66,16,18 --workers $WORKERS"

gen() {   # out start n
  [ -f $1/generation_manifest.json ] || python3 scripts/generate_synthetic_fc.py --out $1 \
    --start $2 --n $3 $REVIT > logs/gen_$(basename $1).log 2>&1
}
first_n() {   # src out n: the first n sheets (sorted ids) as hard links
  [ -d $2/annotations ] && [ "$(ls $2/annotations | wc -l)" -ge $3 ] && return
  mkdir -p $2/images $2/annotations
  ls $1/annotations | sort | head -$3 | while read -r f; do
    ln -f $1/annotations/$f $2/annotations/$f; ln -f $1/images/${f%.json}.png $2/images/${f%.json}.png
  done
}
train() {   # tag pool model p2 -> $R/ck_fit_<tag>_s$SEED/epoch_<p2>.pth
  local CK=$R/ck_fit_$1_s$SEED
  local COMMON="--batch-size $BS --num-workers $NW --prefetch-factor 4 --image-max-size $SIZE \
    --ref-size 224 --width 128 --model $3 --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
    $DR --mask-thresh 0.35 --seed $SEED --pad-grid 512 --roi-ref --roi-ref-mode add ${SMOKEX:-}"
  [ -f $CK/epoch_0.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $2 \
    --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON > logs/fit_$1_p1.log 2>&1
  [ -f $CK/epoch_$4.pth ] || PYTHONPATH=. python3 scripts/train_refunet.py --local-data $2 \
    --checkpoint-dir $CK --epochs $4 --schedule-epochs $(($4 + 1)) $COMMON --reset-optimizer \
    --init-from $CK/epoch_0.pth > logs/fit_$1_p2.log 2>&1
  # every epoch is saved (~0.5-1 GB each for swin_s/b with optimizer state): keep the final one
  [ -f $CK/epoch_$4.pth ] && find "$CK" -maxdepth 1 -name '*.pth' ! -name "epoch_$4.pth" -delete
  return 0
}
score() {   # tag ckpt pool nsheets
  local OUT=$E/$1.json
  [ -f $OUT ] || PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $2 \
    --local-pool $3 --indices $(python3 -c "print(','.join(map(str, range($4))))") \
    --image-max-size $SIZE --mask-thresh 0.35 --metrics-out $OUT > logs/fit_score_$1.log 2>&1
  python3 scripts/family_breakdown.py --pool $3 --metrics $OUT --label $1
}

if [[ $STAGES == *A* ]]; then
  gen $S/fitA_gen 700000 $NA
  [ -d $S/fitA_train ] || python3 scripts/select_by_families.py --src $S/fitA_gen --out $S/fitA_train \
    --min-families 3 --max $NSEL
  gen $S/fitA_freshgen 800000 $((NFRESH * 4))
  [ -d $S/fitA_fresh ] || python3 scripts/select_by_families.py --src $S/fitA_freshgen --out $S/fitA_fresh \
    --min-families 3 --max $NFRESH
  for M in $BACKBONES_A; do
    T=A_${M}${TAGX}
    train $T $S/fitA_train $M $P2A
    echo "== stage A $M (1+$P2A epochs, ${NSEL} sheets >= 3 families${TAGX:+, no DR})"
    score ${T}_seen $R/ck_fit_${T}_s$SEED/epoch_$P2A.pth $S/fitA_train $NSEEN
    score ${T}_fresh $R/ck_fit_${T}_s$SEED/epoch_$P2A.pth $S/fitA_fresh $NFRESH
  done
fi

if [[ $STAGES == *B* ]]; then
  NMAX=$(echo $NB | tr ' ' '\n' | sort -n | tail -1)
  gen $S/fitB_gen 0 $NMAX
  gen $S/fitB_fresh 200000 $NFB
  for N in $NB; do
    first_n $S/fitB_gen $S/fitB_$N $N
    for M in $BACKBONES_B; do
      T=B_${M}_n${N}${TAGX}
      train $T $S/fitB_$N $M $P2B
      echo "== stage B $M, $N sheets (1+$P2B epochs)"
      score ${T}_fresh $R/ck_fit_${T}_s$SEED/epoch_$P2B.pth $S/fitB_fresh $NFB
    done
  done
fi
