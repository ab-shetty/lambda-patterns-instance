#!/usr/bin/env bash
# The "train for vertices" test (2026-11-09): --edge-weight W --edge-px 3 on the exact gem_s7 recipe (gemho_mix pool, seed 7,
# longswiss-swa37-41 init), compared one-to-one with gemho_gem_s7. Per weight: train + score (run_single_pass.sh), maps
# 4096 + 2048 (gpu_maps.sh), then stack / vertcount / boundary_band (scripts/pp_probe/run_probes.sh). Resumable.
#   W=4 ./run_edge_test.sh        # also maps + probes gemho_gem_s7 (fetched from the Hub) if not yet done
set -euo pipefail
cd "$(dirname "$0")"
W=${W:-4}; TAG=gemedge${W}_s7
TAG=$TAG POOL=data/mixed/gemho_mix SEED=7 TRAINX="--edge-weight $W --edge-px 3" ./run_single_pass.sh
echo "TRAINED+SCORED $TAG $(date +%H:%M)"
for m in $TAG gemho_gem_s7; do
  [ -f data/runs/ck_${m}_blend50.pth ] || { echo "missing data/runs/ck_${m}_blend50.pth"; exit 1; }
  scripts/gpu_maps.sh data/runs/ck_${m}_blend50.pth data/floz-diag/$m
done
for m in $TAG gemho_gem_s7; do
  grep -q "PROBES DONE" logs/probes_$m.txt 2>/dev/null || scripts/pp_probe/run_probes.sh data/floz-diag/$m
done
echo "EDGE TEST DONE $TAG $(date +%H:%M)"
