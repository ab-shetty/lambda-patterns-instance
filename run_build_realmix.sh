#!/usr/bin/env bash
# Real + Gemini mix for fine-tuning a synthetic-pretrained model (2026-10-03):
# real 86 x18 + generated 28 x18 + Gemini r2-r4 168 x18 (as v6dmix_plus_r4) + a synthetic
# pool given as $1 (default the mined Revit pool the long synthetic model trained on).
set -euo pipefail
SYN=${1:-data/synthetic/revitfail_hard2000}
OUT=${2:-data/mixed/realmix_revithard}
RF=data/roboflow; mkdir -p $RF logs
augment18() {
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
python3 - <<'PY'
import os
from roboflow import Roboflow
ws = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("perceive-ai")
for proj, ver, name in [("floz-real-pool", 2, "floz-real-pool-v2"), ("floz-generated-realistic-label-pool", 1, "floz-genreal-v1"),
                        ("floz-gen-gemini-r2", 1, "floz-gen-gemini-r2"), ("floz-gen-gemini-r3", 1, "floz-gen-gemini-r3"),
                        ("floz-gen-gemini-r4", 1, "floz-gen-gemini-r4")]:
    loc = f"data/roboflow/{name}-raw"
    if not os.path.exists(f"{loc}/train/_annotations.coco.json"):
        ws.project(proj).version(ver).download("coco-segmentation", location=loc, overwrite=True)
PY
for p in floz-real-pool-v2 floz-genreal-v1 floz-gen-gemini-r2 floz-gen-gemini-r3 floz-gen-gemini-r4; do
  [ -d $RF/$p-clean ] || python3 scripts/roboflow_to_local.py --coco $RF/$p-raw/train/_annotations.coco.json \
    --img-dir $RF/$p-raw/train --out $RF/$p-clean | tail -1 | sed "s/^/$p: /"
done
[ -d $RF/floz-gen-gemini-r234-clean ] || python3 scripts/merge_local_datasets.py \
  --out $RF/floz-gen-gemini-r234-clean --sources $RF/floz-gen-gemini-r2-clean \
  $RF/floz-gen-gemini-r3-clean $RF/floz-gen-gemini-r4-clean > /dev/null
augment18 $RF/floz-real-pool-v2-clean $RF/floz-real-pool-v2-strong18 &
augment18 $RF/floz-genreal-v1-clean $RF/floz-genreal-v1-strong18 &
augment18 $RF/floz-gen-gemini-r234-clean $RF/floz-gen-gemini-r234-strong18 &
wait
[ -d $OUT ] || python3 scripts/merge_local_datasets.py --out $OUT --sources $SYN \
  $RF/floz-real-pool-v2-strong18 $RF/floz-genreal-v1-strong18 $RF/floz-gen-gemini-r234-strong18 > /dev/null
echo "$OUT: $(ls $OUT/images | wc -l) images, $(ls $OUT/annotations | wc -l) annotations"
