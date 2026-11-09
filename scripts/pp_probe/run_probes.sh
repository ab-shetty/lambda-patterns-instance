#!/usr/bin/env bash
# stack.py + vertcount.py + boundary_band.py for one maps folder, on gemini30 / val / hf14 (2026-11-09, the edge-weight test).
#   scripts/pp_probe/run_probes.sh data/floz-diag/<model>   -> logs/probes_<model>.txt, rows_{stack,vert,band}_<model>_<split>.json
# vertcount / boundary_band write untagged rows files; they are renamed per model here so two models do not overwrite.
set -uo pipefail
cd "$(dirname "$0")/../.."
M=$1; T=$(basename $M); OUT=logs/probes_$T.txt; : > $OUT
export PYTHONPATH=. OMP_NUM_THREADS=8
for s in gemini30 val hf14; do
  (python3 scripts/pp_probe/stack.py $s $M > logs/probe_stack_${T}_$s.log 2>&1) &
done; wait
for s in gemini30 val hf14; do
  (python3 scripts/pp_probe/vertcount.py $s $M > logs/probe_vert_${T}_$s.log 2>&1 && mv scripts/pp_probe/rows_vert_$s.json scripts/pp_probe/rows_vert_${T}_$s.json) &
  (python3 scripts/pp_probe/boundary_band.py $s $M > logs/probe_band_${T}_$s.log 2>&1 && mv scripts/pp_probe/rows_band_$s.json scripts/pp_probe/rows_band_${T}_$s.json) &
done; wait
for s in gemini30 val hf14; do
  for k in stack vert band; do echo "== $T $k $s"; grep -v -iE "warn|^\s*$" logs/probe_${k}_${T}_$s.log | tail -12; done
done > $OUT
echo "PROBES DONE $T $(date +%H:%M)" >> $OUT
