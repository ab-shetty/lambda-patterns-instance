#!/usr/bin/env bash
# Hard-mined Gemini holdout (2026-10-22): the gem condition of run_gemini_holdout.sh, plus extra augmented copies of the
# holdout sheets the current best fails on. Aimed at Gemini30's failure modes (look-alike over-selection, trim bleed)
# WITHOUT touching Gemini30: the miner scores the 139 non-Gemini30 sheets only.
#   1. xblend-kq30-kms30 scores every holdout question (automatic boxes, 2048; data/evaluations/mine_gemho/shard*.json)
#   2. sheet hardness = mean over its families of (1 - family mean IoU); the hardest HARD_FRAC of sheets get HARD_AUG
#      extra strong-augmented copies (own seed) on top of the x18 already in data/mixed/gemho_mix
#   3. standard one-epoch restart (run_single_pass.sh: DR + no-DR arms, 50/50 blend), seed 7, scored @4096
# Compare with data/evaluations/gemho_gem_s7 (same seed, same pool minus the extras).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=. ALLOW_STALE_POOLS=1
HARD_FRAC=${HARD_FRAC:-0.4}; HARD_AUG=${HARD_AUG:-32}; SEED=${SEED:-7}
SRC=data/roboflow/floz-gen-gemini-r234-no30-clean
X=data/roboflow/floz-gen-gemini-r234-no30-hard
MIX=data/mixed/gemhard_mix

# the GPU is shared with run_gemini_holdout.sh: wait for its seed-7 gem scores, then stop it (seed 8 is confirmation only)
until [ -f data/evaluations/gemho_gem_s7/blend50_gemini30_4096.json ] && [ -f data/evaluations/gemho_gem_s7/blend50_val_4096.json ] \
   && [ -f data/evaluations/gemho_gem_s7/blend50_hf14_4096.json ] && [ -f data/evaluations/gemho_gem_s7/blend50_hf14fix2_4096.json ]; do
  sleep 20
done
if tmux has-session -t gemho 2>/dev/null; then
  tmux kill-session -t gemho; pkill -f "ck_gemho_ctl_s8" || true; sleep 10
  echo "== stopped run_gemini_holdout.sh after seed 7"
fi
M=data/evaluations/mine_gemho
if [ $(ls $M/shard*.json 2>/dev/null | wc -l) -lt 6 ] && [ ! -f $M/gpu.json ]; then
  pkill -f "mine_gemho/shard" || true      # CPU shards not finished: mine on the GPU instead (~3 min)
  python3 scripts/evaluate_refunet_selection.py --checkpoint data/runs/ck_best/xblend-kq30-kms30.pth --local-pool $SRC \
    --indices all --image-max-size 2048 --ref-size 224 --mask-thresh 0.35 --metrics-out $M/gpu.json > logs/mine_gemho_gpu.log 2>&1
fi
SCORES=$( [ -f $M/gpu.json ] && echo $M/gpu.json || echo $M/shard*.json )

if [ ! -d $X ]; then
  python3 - $SRC $X-sel $HARD_FRAC $SCORES <<'PY'
import collections, json, os, statistics as st, sys
src, out, frac, shards = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
names = sorted(os.listdir(f"{src}/annotations"))
fam = collections.defaultdict(list)
for s in shards:
    for r in json.load(open(s))["selections"]:
        fam[(r["image_index"], r["category"])].append(r["iou"])
hard = collections.defaultdict(list)
for (i, c), v in fam.items():
    hard[i].append(1 - st.mean(v))
rank = sorted(hard, key=lambda i: -st.mean(hard[i]))
assert len(rank) == len(names), (len(rank), len(names))
keep = rank[:round(frac * len(rank))]
print(f"mined {len(keep)} / {len(rank)} sheets; hardness {st.mean(hard[keep[0]]):.3f} .. {st.mean(hard[keep[-1]]):.3f}")
os.makedirs(f"{out}/images"); os.makedirs(f"{out}/annotations")
for i in keep:
    A = json.load(open(f"{src}/annotations/{names[i]}"))
    os.link(f"{src}/annotations/{names[i]}", f"{out}/annotations/{names[i]}")
    os.link(f"{src}/images/{A['image']['file_name']}", f"{out}/images/{A['image']['file_name']}")
json.dump([names[i] for i in keep], open(f"{out}/mined.json", "w"), indent=1)
PY
  tmp=$X.chunks; rm -rf $tmp; mkdir -p $tmp
  python3 - $X-sel $tmp <<'PY'
import os, sys
src, tmp = sys.argv[1:]
imgs = {os.path.splitext(f)[0]: f for f in os.listdir(f"{src}/images")}
for i, x in enumerate(sorted(os.listdir(f"{src}/annotations"))):
    c = f"{tmp}/in{i % 56:02d}"
    os.makedirs(f"{c}/images", exist_ok=True); os.makedirs(f"{c}/annotations", exist_ok=True)
    stem = os.path.splitext(x)[0]
    os.link(f"{src}/annotations/{x}", f"{c}/annotations/{x}")
    os.link(f"{src}/images/{imgs[stem]}", f"{c}/images/{imgs[stem]}")
PY
  for c in $tmp/in*; do
    python3 scripts/augment_local_dataset.py --src $c --out $tmp/out${c##*/in} --aug-per-scene $HARD_AUG --strong \
      --omit-originals --seed 9191 > /dev/null &
  done
  wait
  python3 scripts/merge_local_datasets.py --out $X --sources $tmp/out* > /dev/null
  cp $X-sel/mined.json $X/
  rm -rf $tmp
fi
echo "== hard extras: $(ls $X/annotations | wc -l) records"
[ -d $MIX ] || python3 scripts/merge_local_datasets.py --out $MIX --sources data/mixed/gemho_mix $X > /dev/null
echo "== hard mix: $(ls $MIX/annotations | wc -l) records"

TAG=gemhard_s$SEED POOL=$MIX SEED=$SEED ./run_single_pass.sh
echo "GEMINI HARD DONE $(date '+%H:%M:%S')"
