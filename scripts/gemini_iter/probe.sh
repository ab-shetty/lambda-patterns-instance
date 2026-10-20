#!/usr/bin/env bash
# One generator iteration: render 100 elevation sheets with the given flags (same ids every time, so variants are paired),
# then the per-statistic gap to Gemini round-5 elevations.   scripts/gemini_iter/probe.sh NAME [generator flags...]
set -euo pipefail
NAME=$1; shift
OUT=data/synthetic/gi_$NAME
rm -rf $OUT
python3 scripts/generate_synthetic_fc.py --out $OUT --n 100 --start 960000 --seed 6 --revit --revit-plans --shaped \
  --workers 4 --mode-weights 100,0,0 "$@" 2>&1 | grep -E "ok=|Error|Traceback" | tail -1
python3 scripts/gemini_gap_stats.py --gemini data/image_generation/gemini_r5 \
  --gemini-modes data/image_generation/gemini_r5/specs_described.jsonl --gen $OUT --out data/diag/gap_$NAME.json | tail -16
