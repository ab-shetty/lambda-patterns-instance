#!/usr/bin/env bash
# GPU queue (2026-11-19): one job at a time, each on an empty GPU; progress in logs/gpu_queue.log (read by the heartbeat).
#   1 v3 Gemini-refined self-training (running before this queue)   2 frozen-candidate benchmark (~/frozen_ab, queued)
#   3 refinement head unfrozen (run_ab_vs_best.sh)                   4 one edge convention (run_edgeconv.sh)
cd "$(dirname "$0")"
L=logs/gpu_queue.log
say() { echo "$(date -u '+%F %T') $*" >> $L; }
gpu_free() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done; }
say "queue started"
until grep -q "SELFTRAIN AB v3 DONE" logs/selftrain_ab_v3.log 2>/dev/null; do sleep 30; done; say "1 v3 done"
until grep -q "FROZEN BENCH DONE" ~/frozen_ab/benchmark.log 2>/dev/null; do sleep 30; done; say "2 frozen benchmark done"
gpu_free; say "3 refunf start"
TAG=refunf POOL=data/mixed/fgab_fg_mix BATCH=4 TRAINX="--refine-head 32 --image-max-size-max 3072" \
  ./run_ab_vs_best.sh > logs/ab_refunf.log 2>&1 && say "3 refunf done: $(grep GATE logs/refunf_compare.txt)" || say "3 refunf FAILED (logs/ab_refunf.log)"
gpu_free
if [ -x run_edgeconv.sh ]; then say "4 edgeconv start"
  ./run_edgeconv.sh > logs/ab_edgeconv.log 2>&1 && say "4 edgeconv done: $(grep GATE logs/edgeconv_compare.txt)" || say "4 edgeconv FAILED (logs/ab_edgeconv.log)"
else say "4 edgeconv SKIPPED: run_edgeconv.sh not ready"; fi
say "QUEUE DONE"
