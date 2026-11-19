#!/usr/bin/env bash
# Self-training A/B v3 (2026-11-19): as v2 (same teacher discovery disc2_r5, recipe, seeds, control and pre-registered gate),
# but every pseudo-label family is first reviewed by a Gemini vision model (scripts/selftrain/gemini_refine.py; MODEL, default
# gemini-4-hydrogen) and pseudolabel.py --refine applies the verdicts: families on non-material (windows, blank paper) dropped,
# wrong areas of the target and missed areas of the background ignored, same-material families merged. The refinement rules
# were checked against human labels on Gemini30 / val first (run_gref_validate.sh), never on HF14.
set -euo pipefail
cd "$(dirname "$0")"
export ALLOW_STALE_POOLS=1 PYTHONPATH=.:scripts PATH=$HOME/tess/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TEACHER=data/runs/ck_fgavg78.pth
MODEL=${MODEL:-gemini-4-hydrogen}; M=${MODEL//[^a-z0-9.]/_}
GREF=data/selftrain/gref_${M}_r5
REFX=${REFX:-}            # e.g. --no-merge
INIT=data/runs/ck_best/gem2s-x.pth
R5=data/image_generation/gemini_r5
EL=data/selftrain/r5_elev_images
DISC=data/selftrain/disc2_r5
V=v3${VTAG:-}
PL=data/selftrain/pool_r5_$V
PL18=data/selftrain/pool_r5_${V}_strong18
BASE=data/mixed/fgab_fg_mix                 # fg pool + Gemini strong18: exactly the control's training mix
MIX=data/mixed/selftrain_r5${V}_mix
SEEDS=${SEEDS:-7 8}
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
G=data/eval_pools/gemini30_s20261006
mkdir -p logs data/selftrain

# 1. discovery on the round-5 elevations (teacher)
if [ ! -d $EL ]; then
  mkdir -p $EL
  while read -r n; do ln -s $PWD/$R5/$n.png $EL/$n.png; done < data/selftrain/r5_elevations.txt
fi
# discovery is reused from v2 (data/selftrain/disc2_r5, restored from abshetty/floz-runs-backup)
set -a; . ~/.env; set +a
python3 scripts/selftrain/gemini_refine.py --disc $DISC --images $EL --out $GREF --model $MODEL --only data/selftrain/r5_elevations.txt \
  > logs/gref_${M}_r5.log 2>&1
echo "== Gemini verdicts: $(ls $GREF | wc -l) sheets ($(grep -c FAILED logs/gref_${M}_r5.log || true) failed)"
echo "== discovery: $(ls $DISC | wc -l) sheets"

# 2. pseudo-labelled pool, text rings (reference boxes stay off text), x18 strong augmentation (as the Gemini pool)
[ -f $PL/selftrain_manifest.json ] || python3 scripts/selftrain/pseudolabel.py --disc $DISC --images $EL --write-pool $PL \
  --refine $GREF $REFX --workers 32 | tee logs/selftrain_pool_$V.log
[ -f $PL/.ref_exclude ] || { python3 scripts/add_ref_exclude.py --pool $PL > /dev/null && touch $PL/.ref_exclude; }
if [ ! -d $PL18 ]; then
  python3 scripts/augment_local_dataset.py --src $PL --out $PL18 --aug-per-scene 17 --strong --seed 5858 > /dev/null
fi
echo "== pseudo pool: $(ls $PL/annotations | wc -l) sheets, $(ls $PL18/annotations | wc -l) records" \
     "(hand-labelled Gemini: $(ls data/roboflow/floz-gen-gemini-r234-no30-strong18/annotations | wc -l))"
# the control's mix (fgab_fg_mix) is built by the seed-9 run; one training job on the GPU at a time
until grep -q "FG SEED9 DONE" logs/fg_seed9.log 2>/dev/null; do sleep 60; done
[ -d $MIX ] || python3 scripts/merge_local_datasets.py --out $MIX --sources $BASE $PL18 > /dev/null
echo "== mix: $(ls $MIX/annotations | wc -l) records (control mix: $(ls $BASE/annotations | wc -l))"

# 3. treatment training, same recipe and seeds as the control
for S in $SEEDS; do
  TAG=st${V}_r5_s$S POOL=$MIX SEED=$S INIT=$INIT ./run_single_pass.sh
done
set -- $SEEDS
B=data/runs/ck_st${V}_r5_avg.pth
[ -f $B ] || python3 scripts/blend_ckpts.py $B data/runs/ck_st${V}_r5_s$1_blend50.pth:0.5 data/runs/ck_st${V}_r5_s$2_blend50.pth:0.5

# 4. product scoring at HEAD, two jobs at a time (three OOM at 5120): control and treatment, blends and single seeds
jobs=()
for arm in "ctl:$TEACHER" "st:$B" "ctl_s$1:data/runs/ck_fgab_fg_s$1_blend50.pth" "ctl_s$2:data/runs/ck_fgab_fg_s$2_blend50.pth" \
           "st_s$1:data/runs/ck_st${V}_r5_s$1_blend50.pth" "st_s$2:data/runs/ck_st${V}_r5_s$2_blend50.pth"; do
  for sp in "valfix4:--indices $VAL --label-fixes eval_labels/real_fixes_v4.json" \
            "hf14fix4:--indices $HF14 --label-fixes eval_labels/real_fixes_v4.json" \
            "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
    jobs+=("${arm%%:*}|${arm#*:}|${sp%%:*}|${sp#*:}")
  done
done
run_job() {
  IFS='|' read -r tag ck split args <<< "$1"
  local O=data/evaluations/selftrain${V}_$tag; [[ $tag == ctl* ]] && O=data/evaluations/selftrain_$tag   # controls: v1's scores
  mkdir -p $O
  [ -f $O/product_$split.json ] && return 0
  python3 scripts/evaluate_refunet_selection.py --checkpoint $ck $args --product --ref-size 224 --workers 8 \
    --save-probs /tmp/st${V}_maps/${tag}_$split --metrics-out $O/product_$split.json > $O/product_$split.log 2>&1
  rm -rf /tmp/st${V}_maps/${tag}_$split
}
for ((i = 0; i < ${#jobs[@]}; i += 2)); do
  run_job "${jobs[$i]}" &
  [ $((i + 1)) -lt ${#jobs[@]} ] && run_job "${jobs[$((i + 1))]}" &
  wait
done
python3 - $V <<'PY'
import json, sys
V = sys.argv[1]
def m(t, s): return json.load(open(f"data/evaluations/selftrain{'' if t.startswith('ctl') else V}_{t}/product_{s}.json"))["mean_iou"]
for pair in [("ctl", "st"), ("ctl_s7", "st_s7"), ("ctl_s8", "st_s8")]:
    print(pair[1], "vs", pair[0], "  ".join(f"{s} {m(pair[0], s):.4f} -> {m(pair[1], s):.4f} ({m(pair[1], s) - m(pair[0], s):+.4f})"
                                         for s in ("valfix4", "hf14fix4", "gemini30")))
d = {s: m("st", s) - m("ctl", s) for s in ("valfix4", "hf14fix4", "gemini30")}
ok = d["hf14fix4"] >= 0.002 and d["valfix4"] >= -0.002 and d["gemini30"] >= -0.003
print("GATE (blend vs fgavg78):", "PASS" if ok else "FAIL", {k: round(v, 4) for k, v in d.items()})
PY
echo "SELFTRAIN AB $V DONE $(date '+%H:%M:%S')"
