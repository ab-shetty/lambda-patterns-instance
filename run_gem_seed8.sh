#!/usr/bin/env bash
# Seed 8 of the better Gemini-holdout condition (2026-10-22): waits for run_gemini_hard.sh, picks gem or gemhard by
# Gemini30 at seed 7, trains the same one-epoch restart at seed 8, for the two-seed blend.
set -euo pipefail
cd "$(dirname "$0")"
export ALLOW_STALE_POOLS=1
until grep -q "GEMINI HARD DONE" logs/gemhard_run.log 2>/dev/null; do sleep 20; done
pick=$(python3 -c "
import json
g=json.load(open('data/evaluations/gemho_gem_s7/blend50_gemini30_4096.json'))['mean_iou']
h=json.load(open('data/evaluations/gemhard_s7/blend50_gemini30_4096.json'))['mean_iou']
print('gemhard' if h > g else 'gem')")
echo "== seed 8 on $pick"
if [ $pick = gemhard ]; then TAG=gemhard_s8 POOL=data/mixed/gemhard_mix SEED=8 ./run_single_pass.sh
else TAG=gemho_gem_s8 POOL=data/mixed/gemho_mix SEED=8 ./run_single_pass.sh; fi
echo "SEED8 DONE $(date '+%H:%M:%S')"
