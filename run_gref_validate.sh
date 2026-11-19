#!/usr/bin/env bash
# Gemini pseudo-label refinement, validation (2026-11-19): teacher discovery on Gemini30 and val (labelled, never trained on
# here), Gemini verdicts (scripts/selftrain/gemini_refine.py), and pseudolabel.py reports with and without --refine against the
# human labels. HF14 is never used. MODEL (default gemini-pro-latest).
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ~/.env; set +a
export PYTHONPATH=.:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=data/runs/ck_fgavg78.pth
MODEL=${MODEL:-gemini-pro-latest}; M=${MODEL//[^a-z0-9.]/_}
VAL=4,5,6,8,9,10,13,15,17,19,20,21,22,26
G=data/eval_pools/gemini30_s20261006
S=data/selftrain
while ! grep -q SETUP_DONE logs/setup.log; do sleep 30; done
if [ ! -f $S/disc_gemini30/.done ]; then
  python3 scripts/selftrain/discover.py --checkpoint $T --local-pool $G --out $S/disc_gemini30 --mem-frac ${MEMFRAC:-0.3} --q-batch 8 \
    > logs/gref_disc_gemini30.log 2>&1
  touch $S/disc_gemini30/.done
fi
if [ ! -f $S/disc_val/.done ]; then
  python3 scripts/selftrain/discover.py --checkpoint $T --real $VAL --out $S/disc_val --mem-frac ${MEMFRAC:-0.3} --q-batch 8 \
    > logs/gref_disc_val.log 2>&1
  touch $S/disc_val/.done
fi
for sp in gemini30 val; do
  src=$([ $sp = val ] && echo --real || echo "--local-pool $G")
  python3 scripts/selftrain/gemini_refine.py --disc $S/disc_$sp $src --out $S/gref_${M}_$sp --model $MODEL > logs/gref_${M}_$sp.log 2>&1
  [ -f $S/rep_base_$sp.json ] || python3 scripts/selftrain/pseudolabel.py --disc $S/disc_$sp $src --report $S/rep_base_$sp.json > /dev/null
  python3 scripts/selftrain/pseudolabel.py --disc $S/disc_$sp $src --refine $S/gref_${M}_$sp --report $S/rep_${M}_$sp.json > /dev/null
  python3 scripts/selftrain/pseudolabel.py --disc $S/disc_$sp $src --refine $S/gref_${M}_$sp --no-merge --report $S/rep_${M}_nomerge_$sp.json > /dev/null
done
python3 - $M <<'PY'
import json, sys
M = sys.argv[1]
keys = ["pseudo_families", "pl_covered_families", "pl_pos_precision", "pl_pos_recall", "pl_neg_error", "pl_ignored",
        "pl_own_material_as_background", "unmatched_pl_families", "unmatched_pl_on_labels"]
for sp in ("gemini30", "val"):
    print(f"== {sp}")
    for t in ("base", M, f"{M}_nomerge"):
        s = json.load(open(f"data/selftrain/rep_{t}_{sp}.json"))["summary"]
        print(f"  {t:34s}", "  ".join(f"{k.replace('pl_', '')} {s[k]}" for k in keys))
PY
echo "GREF VALIDATE DONE $(date '+%H:%M:%S')"
