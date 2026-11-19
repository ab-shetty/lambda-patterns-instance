#!/usr/bin/env bash
# Heartbeat (2026-11-19): is the GPU queue alive and making progress? Read-only: never kills or changes anything.
cd "$(dirname "$0")/.."
echo "== $(date -u '+%F %T') UTC"
echo "-- queue log (last 6):"; tail -6 logs/gpu_queue.log 2>/dev/null
q=$(pgrep -f 'bash ./gpu_queue.sh' | head -1); echo "-- queue process: ${q:-NOT RUNNING}"
echo "-- GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r p m; do
  echo "   pid $p $m: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"; done
echo "-- newest training log:"; f=$(ls -t logs/*_dr.log logs/*_nodr.log 2>/dev/null | head -1)
[ -n "$f" ] && echo "   $f ($(( $(date +%s) - $(stat -c %Y $f) ))s since last write): $(tail -c 160 $f | tr '\r' '\n' | tail -1)"
echo "-- errors in the last 30 min of logs:"
find logs ~/frozen_ab -name '*.log' -mmin -30 2>/dev/null | xargs -r grep -l -E 'Traceback|OutOfMemory|FAILED|Error' 2>/dev/null | sed 's/^/   /'
echo "-- results so far:"; grep -h 'GATE' logs/selftrain_ab_v3.log logs/*_compare.txt 2>/dev/null | sed 's/^/   /'
grep -q 'FROZEN BENCH DONE' ~/frozen_ab/benchmark.log 2>/dev/null && echo "   frozen benchmark: $(tail -1 ~/frozen_ab/benchmark.log)"
true
