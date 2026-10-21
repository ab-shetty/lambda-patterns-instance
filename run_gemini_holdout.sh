#!/usr/bin/env bash
# Does Gemini's own drawing of materials, with exact hand labels, move Gemini30? (2026-10-21)
#
# The twin test (synth_progress.md 2026-10-20) found Gemini30's worst sheets fail on how Gemini DRAWS the materials
# (sheet 1 0.654 -> 0.980, 26 0.478 -> 0.821 when repainted with our textures; no question failed on layout). The 139
# hand-labelled Gemini r2-r4 sheets that are NOT in Gemini30 are exactly that appearance with exact labels, and no run
# has trained on them and read Gemini30 cleanly (earlier mixes held all 169 r2-r4 sheets, Gemini30 included).
#
# Same synthetic pool for both conditions -- the current best's own kq pool (GEN 2026-10-15b, 18k sheets, ids 600000+):
#   ctl  synthetic only                                  -> data/runs/ck_gemho_ctl_s{7,8}_*
#   gem  synthetic + Gemini r2-r4 minus Gemini30 x18      -> data/runs/ck_gemho_gem_s{7,8}_*
# each the standard one-epoch restart of longswiss-swa37-41 (run_single_pass.sh: DR + no-DR arms, 50/50 blend), at
# seeds 7 and 8 (seed noise ~0.016 on Gemini30), then the two-seed 0.5 / 0.5 blend per condition, scored @4096 with
# hand boxes on HF14, HF14 fix v2, val and Gemini30, and compared ctl vs gem (scripts/paired_compare.py).
# ~4.5-5 h on a GH200 (4 runs x ~45 min train + scoring). Resumable: finished steps are skipped.
#   ./run_gemini_holdout.sh                 # SEEDS="7" for a one-seed screen (~2.5 h)
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ~/.env; set +a      # ROBOFLOW_API_KEY (downloads), HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
. scripts/pool_guard.sh
SEEDS=${SEEDS:-7 8}
SYN=data/synthetic/gemho_s600000
GEM=data/roboflow/floz-gen-gemini-r234-no30
MIX=data/mixed/gemho_mix
RF=data/roboflow; mkdir -p $RF logs data/mixed

# 1. Gemini r2-r4 hand labels from Roboflow (same versions as Gemini30: r2 v1, r3 v3, r4 v2), minus Gemini30
python3 - <<'PY'
import os
from roboflow import Roboflow
ws = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("perceive-ai")
for proj, ver, name in [("floz-gen-gemini-r2", 1, "floz-gen-gemini-r2"), ("floz-gen-gemini-r3", 3, "floz-gen-gemini-r3v3"),
                        ("floz-gen-gemini-r4", 2, "floz-gen-gemini-r4v2")]:
    loc = f"data/roboflow/{name}-raw"
    if not os.path.exists(f"{loc}/train/_annotations.coco.json"):
        ws.project(proj).version(ver).download("coco-segmentation", location=loc, overwrite=True)
PY
for p in floz-gen-gemini-r2 floz-gen-gemini-r3v3 floz-gen-gemini-r4v2; do
  [ -d $RF/$p-clean ] && continue
  parts=""
  for sp in train valid test; do
    [ -f $RF/$p-raw/$sp/_annotations.coco.json ] || continue
    python3 scripts/roboflow_to_local.py --coco $RF/$p-raw/$sp/_annotations.coco.json \
      --img-dir $RF/$p-raw/$sp --out $RF/$p-clean-$sp | tail -1 | sed "s/^/$p $sp: /"
    parts="$parts $RF/$p-clean-$sp"
  done
  if [ $(echo $parts | wc -w) = 1 ]; then mv $parts $RF/$p-clean
  else python3 scripts/merge_local_datasets.py --out $RF/$p-clean --sources $parts > /dev/null && rm -rf $parts; fi
done
[ -d $GEM-clean ] || python3 scripts/build_gemini_holdout_pool.py --out $GEM-clean

# 2. x18 strong augmentation (as run_build_realmix.sh's augment18: 1 original + 17 augmented per sheet)
if [ ! -d $GEM-strong18 ]; then
  tmp=$GEM-strong18.chunks; rm -rf $tmp; mkdir -p $tmp
  python3 - $GEM-clean $tmp <<'PY'
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
  for c in $tmp/in*; do
    python3 scripts/augment_local_dataset.py --src $c --out $tmp/out${c##*/in} --aug-per-scene 17 --strong --seed 5858 > /dev/null &
  done
  wait
  python3 scripts/merge_local_datasets.py --out $GEM-strong18 --sources $tmp/out* > /dev/null
  rm -rf $tmp
fi
echo "== Gemini holdout: $(ls $GEM-clean/annotations | wc -l) sheets, $(ls $GEM-strong18/annotations | wc -l) records"

# 3. the synthetic pool: EXACTLY the current best's kq pool (generator at GEN 2026-10-15b, the last commit before
#    6b25130; ids 600000+, seed 6, only --plan-source), so ctl at seed 7 reproduces single18kq-blend50 (a check on the
#    machine) and the only difference between ctl and gem is the Gemini data. The pool is "stale" to pool_guard on
#    purpose: ALLOW_STALE_POOLS=1 below.
export ALLOW_STALE_POOLS=1
if [ ! -f $SYN/generation_manifest.json ]; then
  WT=/tmp/gen_2026-10-15b; [ -d $WT ] || git worktree add -f $WT 6b25130^
  (cd $WT && python3 scripts/generate_synthetic_fc.py --out $OLDPWD/$SYN --n 18000 --start 600000 --seed 6 \
    --revit --revit-plans --shaped --workers 64 --mode-weights 66,16,18 \
    --plan-source $OLDPWD/data/reference/swiss_dwellings/layouts.pkl.gz) > logs/gen_gemho.log 2>&1
fi
python3 -c "import json; print('== synthetic:', json.load(open('$SYN/generation_manifest.json')).get('generator_version'))"
echo "== synthetic: $(ls $SYN/annotations | wc -l) sheets (kq pool: 17,906)"
[ -d $MIX ] || python3 scripts/merge_local_datasets.py --out $MIX --sources $SYN $GEM-strong18 > /dev/null
echo "== mix: $(ls $MIX/annotations | wc -l) records"

# 4. train + score: ctl and gem at each seed (run_single_pass.sh: two arms, 50/50 blend, four splits @4096)
for S in $SEEDS; do
  TAG=gemho_ctl_s$S POOL=$SYN SEED=$S ./run_single_pass.sh
  TAG=gemho_gem_s$S POOL=$MIX SEED=$S ./run_single_pass.sh
done

# 5. two-seed blends per condition, scored the same way
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
HF14=12,16,27,7,11,25,23,1,18,2,0,3,14,24
G=data/eval_pools/gemini30_s20261006
set -- $SEEDS
if [ $# -ge 2 ]; then
  for C in ctl gem; do
    B=data/runs/ck_gemho_${C}_2seed.pth; O=data/evaluations/gemho_${C}_2seed; mkdir -p $O
    [ -f $B ] || python3 - data/runs/ck_gemho_${C}_s$1_blend50.pth data/runs/ck_gemho_${C}_s$2_blend50.pth $B <<'PY'
import sys, torch
a, b = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
avg = {k: ((a["model"][k].float() + b["model"][k].float()) / 2).to(a["model"][k].dtype)
       if a["model"][k].is_floating_point() else a["model"][k] for k in a["model"]}
torch.save({**a, "model": avg, "epoch": None, "actual_epoch": None}, sys.argv[3])
PY
    for job in "hf14:--indices $HF14" "hf14fix2:--indices $HF14 --label-fixes eval_labels/hf14_fixes_v2.json" \
               "val:--indices $VAL" "gemini30:--local-pool $G --indices all --boxes eval_boxes/gemini30_v1.json"; do
      OUT=$O/blend50_${job%%:*}_4096.json
      [ -f $OUT ] || python3 scripts/evaluate_refunet_selection.py --checkpoint $B ${job#*:} \
        --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --metrics-out $OUT > $O/${job%%:*}.log 2>&1 &
    done
    wait
  done
fi

# 6. ctl vs gem, paired, per seed and for the two-seed blends
RUNS="$(for S in $SEEDS; do echo s$S; done) $(if [ $# -ge 2 ]; then echo 2seed; fi)"
for R in $RUNS; do
  for sp in hf14 hf14fix2 val gemini30; do
    A=data/evaluations/gemho_ctl_$R/blend50_${sp}_4096.json; B=data/evaluations/gemho_gem_$R/blend50_${sp}_4096.json
    echo "== $R $sp"; python3 scripts/paired_compare.py --a $A --b $B --label-a ctl --label-b gem \
      --out data/evaluations/gemho_compare_${R}_$sp.json | head -14
  done
done | tee logs/gemho_compare.txt
# reproduction check: ctl at seed 7 IS the kq recipe on the kq pool -- it should match single18kq-blend50 (same-seed sd <= 0.012)
for sp in hf14 gemini30; do
  echo "== reproduction ctl_s7 vs single18kq-blend50 $sp"
  python3 scripts/paired_compare.py --a eval_baselines/single18kq-blend50_${sp}_4096.json \
    --b data/evaluations/gemho_ctl_s7/blend50_${sp}_4096.json --label-a kq --label-b ctl_s7 | head -6
done | tee -a logs/gemho_compare.txt
echo "GEMINI HOLDOUT DONE $(date '+%H:%M:%S')  (logs/gemho_compare.txt)"
