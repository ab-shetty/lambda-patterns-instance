#!/usr/bin/env bash
# Third seed of the --foreground A/B (2026-11-19): the Gemini holdout pool (run_gemini_holdout.sh steps 1-2), then
# run_foreground_ab.sh at SEED=9 (ctl and fg arms, same pools as seeds 7 / 8: only the generator default changed since).
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
  { [ -d $RF/$p-clean ] || [ -d $RF/$p-clean-train ]; } && continue
  parts=""
  for sp in train valid test; do
    [ -f $RF/$p-raw/$sp/_annotations.coco.json ] || continue
    python3 scripts/roboflow_to_local.py --coco $RF/$p-raw/$sp/_annotations.coco.json \
      --img-dir $RF/$p-raw/$sp --out $RF/$p-clean-$sp | tail -1 | sed "s/^/$p $sp: /"
    parts="$parts $RF/$p-clean-$sp"
  done
  # a multi-split export stays in its parts: merging renames files to item_NNNNNN, and the holdout builder needs the
  # floz_gen_NNN names to drop the Gemini30 sheets (it reads the parts directly)
  if [ $(echo $parts | wc -w) = 1 ]; then mv $parts $RF/$p-clean; fi
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
while ! grep -q SETUP_DONE logs/setup.log; do sleep 30; done    # checkpoints / Gemini30 pool from the VM setup
SEED=9 ./run_foreground_ab.sh
echo "FG SEED9 DONE $(date '+%H:%M:%S')"
