#!/usr/bin/env bash
# Close-out training queue (2026-10-22): after gem seed 8, (A) the multi-resolution arm on the Gemini mix, then
# (C) a second epoch from the gem_s7 blend. Each is run_single_pass.sh (DR + no-DR arms, blend50, four splits @4096).
set -euo pipefail
cd "$(dirname "$0")"
export ALLOW_STALE_POOLS=1
until grep -q "SEED8 DONE" logs/seed8_run.log 2>/dev/null; do sleep 15; done
TAG=gemkms_s7 POOL=data/mixed/gemho_mix TRAINX="--image-max-size-max 3072" BATCH=4 ./run_single_pass.sh
echo "== A done $(date +%H:%M:%S)"
TAG=geme2_s7 POOL=data/mixed/gemho_mix INIT=data/runs/ck_gemho_gem_s7_blend50.pth ./run_single_pass.sh
echo "CLOSEOUT TRAIN DONE $(date +%H:%M:%S)"
