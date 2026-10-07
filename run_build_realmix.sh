#!/usr/bin/env bash
# Real + Gemini mix for fine-tuning a synthetic-pretrained model (2026-10-03):
# real 86 x18 + generated 28 x18 + Gemini r2-r4 168 x18 (as v6dmix_plus_r4) + a synthetic
# pool given as $1 (default the mined Revit pool the long synthetic model trained on).
# TX=1 (2026-10-06): real / generated / Gemini sources get text regions as ref_exclude
# (scripts/add_ref_exclude.py on copies of the clean dirs, *-cleantx), so training's reference boxes stay
# off room tags / callouts / dimension text; the augmentation carries the rings. Augmented pools and the
# default OUT get a "tx" suffix; images and labels are identical to TX=0.
set -euo pipefail
. scripts/pool_guard.sh     # need_pool: never silently reuse a pool from older generator defaults
SYN=${1:-data/synthetic/revitfail_hard2000}
TX=${TX:-0}; S=""; [ "$TX" = 1 ] && S=tx
OUT=${2:-data/mixed/realmix_revithard$S}
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
                        ("floz-gen-gemini-r2", 1, "floz-gen-gemini-r2"), ("floz-gen-gemini-r3", 3, "floz-gen-gemini-r3v3"),
                        ("floz-gen-gemini-r4", 2, "floz-gen-gemini-r4v2")]:
    loc = f"data/roboflow/{name}-raw"
    if not os.path.exists(f"{loc}/train/_annotations.coco.json"):
        ws.project(proj).version(ver).download("coco-segmentation", location=loc, overwrite=True)
PY
# every split the version has (r3 v3 splits train / valid / test; the others are train only)
for p in floz-real-pool-v2 floz-genreal-v1 floz-gen-gemini-r2 floz-gen-gemini-r3v3 floz-gen-gemini-r4v2; do
  [ -d $RF/$p-clean ] && continue
  parts=""
  for sp in train valid test; do
    [ -f $RF/$p-raw/$sp/_annotations.coco.json ] || continue
    python3 scripts/roboflow_to_local.py --coco $RF/$p-raw/$sp/_annotations.coco.json \
      --img-dir $RF/$p-raw/$sp --out $RF/$p-clean-$sp | tail -1 | sed "s/^/$p $sp: /"
    parts="$parts $RF/$p-clean-$sp"
  done
  if [ $(echo $parts | wc -w) = 1 ]; then mv $parts $RF/$p-clean   # train only: same files as before
  else python3 scripts/merge_local_datasets.py --out $RF/$p-clean --sources $parts > /dev/null && rm -rf $parts; fi
done
# r4 version 2 (2026-10-06): the user's label fix on r4 #091 (roof and walls were one family). r3 version 3
# (2026-10-07): the user's fix on r3 #048 (a missed pattern3 piece) + newly labelled r3 #016. New names
# (r4v2, r3v3, r2r3v3r4v2) so an older build on disk is never reused.
[ -d $RF/floz-gen-gemini-r2r3v3r4v2-clean ] || python3 scripts/merge_local_datasets.py \
  --out $RF/floz-gen-gemini-r2r3v3r4v2-clean --sources $RF/floz-gen-gemini-r2-clean \
  $RF/floz-gen-gemini-r3v3-clean $RF/floz-gen-gemini-r4v2-clean > /dev/null
if [ "$TX" = 1 ]; then
  for p in floz-real-pool-v2 floz-genreal-v1 floz-gen-gemini-r2r3v3r4v2; do
    [ -d $RF/$p-cleantx ] && continue
    cp -r $RF/$p-clean $RF/$p-cleantx.tmp
    PYTHONPATH=. python3 scripts/add_ref_exclude.py --pool $RF/$p-cleantx.tmp && mv $RF/$p-cleantx.tmp $RF/$p-cleantx
  done
fi
augment18 $RF/floz-real-pool-v2-clean$S $RF/floz-real-pool-v2-strong18$S &
augment18 $RF/floz-genreal-v1-clean$S $RF/floz-genreal-v1-strong18$S &
augment18 $RF/floz-gen-gemini-r2r3v3r4v2-clean$S $RF/floz-gen-gemini-r2r3v3r4v2-strong18$S &
wait
need_pool $SYN && { echo "STOP: synthetic pool $SYN missing" >&2; exit 3; }
need_pool $OUT && python3 scripts/merge_local_datasets.py --out $OUT --sources $SYN \
  $RF/floz-real-pool-v2-strong18$S $RF/floz-genreal-v1-strong18$S $RF/floz-gen-gemini-r2r3v3r4v2-strong18$S > /dev/null
echo "$OUT: $(ls $OUT/images | wc -l) images, $(ls $OUT/annotations | wc -l) annotations"
