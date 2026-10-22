#!/usr/bin/env bash
# Kill CPU-side evaluations (evaluate_refunet_selection.py ... --workers 0) when available RAM drops under 50 GB:
# a host OOM kills the whole job unit (tmux, training, the session) -- 2026-10-22 01:10.
while true; do
  a=$(free -g | awk 'NR==2{print $7}')
  if [ "$a" -lt 50 ]; then
    pkill -f "evaluate_refunet_selection[.]py.*--workers 0" && echo "$(date +%H:%M:%S) killed CPU evals, avail ${a}G" >> logs/memguard.log
  fi
  sleep 5
done
