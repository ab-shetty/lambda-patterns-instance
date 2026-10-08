#!/usr/bin/env bash
# Offline hard-example mining on the 2k Revit ROI screen (2026-10-02, one seed).
# Miner = the revitcurroi screen model (run_revit_2k.sh, today's generator defaults, epoch 8).
# Fresh pool = the same generator settings at ids 100000+ (disjoint from the 0-1999 pool),
# scored with the HF14 protocol at 2048; build_hard_pool.py keeps the revitcurroi pool's first
# 1,000 sheets and adds 1,000 semi-hard fresh sheets (hardness top 30%, hardest 1% skipped).
# Trained with the identical recipe, so revithardroi vs revitcurroi is paired and differs
# only in the mined half.
# SRC=fail (2026-10-02): the same on the revitfailroi settings (+ the four failure options),
# miner and paired control revitfailroi, 4,000 fresh sheets on 8 shards -> revithardfailroi.
#   EXTRA="--roi-ref --roi-ref-mode add" ./run_hard_mine.sh [seed]          (SRC=cur, as run)
#   EXTRA="--roi-ref --roi-ref-mode add" SRC=fail NFRESH=4000 SHARDS=8 ./run_hard_mine.sh 7
# SRC=failswiss (2026-10-06): the fail settings + Swiss Dwellings floor plans (--plan-source); miner
# revitfailswissroi (run_revit_2k.sh), mined pool revitfailswiss_hard2000.
# SRC=near (2026-10-08): the failswiss flags with the 2026-10-07 generator defaults (near-miss pairs,
# balcony doors); miner revitnearroi (run_revit_2k.sh); sliver questions (< 32 px) are left out of
# hardness (build_hard_pool.py default). Mined pool revitnear_hard2000.
#   EXTRA="--roi-ref --roi-ref-mode add" SRC=near NFRESH=4000 SHARDS=8 NOTRAIN=1 ./run_hard_mine.sh 7
# NOTRAIN=1 (2026-10-06): stop after building the mined pool (no 2k arm on it), e.g. when the next
# step restarts an existing long-schedule model on the mined pool instead.
set -euo pipefail
. scripts/pool_guard.sh     # need_pool: never silently reuse a pool from older generator defaults
SEED=${1:-7}
SRC=${SRC:-cur}
NFRESH=${NFRESH:-6000}
SHARDS=${SHARDS:-4}
declare -A GENFLAGS=([cur]="" [fail]="--roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2"
  [failswiss]="--roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz"
  [near]="--roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2 --plan-source data/reference/swiss_dwellings/layouts.pkl.gz")
declare -A ARM=([cur]=revithardroi [fail]=revithardfailroi [failswiss]=revithardfailswissroi [near]=revithardnearroi)
MINER=data/runs/ck_2k_revit${SRC}roi_r2048_s$SEED/epoch_8.pth
BASE=data/synthetic/revit${SRC}_train2000
FRESH=data/synthetic/revit${SRC}_fresh$NFRESH
HARD=data/synthetic/revit${SRC}_hard2000
M=data/evaluations/hardmine; [ $SRC = cur ] || M=$M/$SRC; mkdir -p logs $M
[ -f $MINER ] || { echo "missing miner $MINER (run run_revit_2k.sh revit${SRC}roi first)"; exit 1; }

need_pool $FRESH && python3 scripts/generate_synthetic_fc.py --out $FRESH \
  --n $NFRESH --start 100000 --seed 6 --revit --revit-plans --shaped --workers 64 \
  --mode-weights 66,16,18 ${GENFLAGS[$SRC]} > logs/gen_$(basename $FRESH).log 2>&1

N=$(ls $FRESH/annotations | wc -l)
for S in $(seq 0 $((SHARDS - 1))); do
  OUT=$M/fresh_shard$S.json
  [ -f $OUT ] && continue
  IDX=$(python3 -c "print(','.join(str(i) for i in range($S, $N, $SHARDS)))")
  PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint $MINER \
    --local-pool $FRESH --indices $IDX --image-max-size 2048 --ref-size 224 \
    --mask-thresh 0.35 --metrics-out $OUT > logs/hardmine_${SRC}_score$S.log 2>&1 &
done
wait

need_pool $BASE && { echo "STOP: base pool $BASE missing (run_revit_2k.sh revit${SRC}roi first)" >&2; exit 3; }
need_pool $HARD && PYTHONPATH=. python3 scripts/build_hard_pool.py \
  --fresh-pool $FRESH --scored $M/fresh_shard*.json --base-pool $BASE \
  --base-n 1000 --hard-n 1000 --out $HARD | tee logs/hardmine_${SRC}_build.log

[ "${NOTRAIN:-0}" = 1 ] && { echo "NOTRAIN=1: mined pool $HARD built, no 2k arm"; exit 0; }
./run_revit_2k.sh $SEED ${ARM[$SRC]}

E=data/evaluations/revit2k
for I in 2048 4096; do for split in val hf14; do
  for C in revit${SRC}roi revitcurroi revithardroi; do [ $C = ${ARM[$SRC]} ] && continue
    python3 scripts/paired_compare.py --a $E/${C}_s${SEED}_i${I}_$split.json \
      --b $E/${ARM[$SRC]}_s${SEED}_i${I}_$split.json --label-a $C --label-b ${ARM[$SRC]} \
      | grep -m2 "paired mean\|better on" | sed "s/^/  infer $I $split ${ARM[$SRC]} vs $C: /"
  done
done; done
