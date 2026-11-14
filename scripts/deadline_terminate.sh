#!/usr/bin/env bash
# Hard deadline (the user, 2026-11-14): terminate THIS Lambda instance at the scheduled time no matter what else is running.
# Fired by cron AND by a detached sleep (two independent triggers); idempotent (lock + status check). Only curl + bash.
#   1. best effort, time-limited: commit + push whatever results / logs exist (never blocks termination)
#   2. terminate INSTANCE_ID via the Lambda Cloud API, only if the API lists it with this machine's public IP;
#      retried every 60 s for up to 30 min until the API reports it terminating / terminated.
INSTANCE_ID=5764a1b50bda44b6a72254984e199176
API=https://cloud.lambdalabs.com/api/v1
REPO=/home/ubuntu/lambda-patterns-instance
cd $REPO || exit 1
exec 9> /tmp/deadline_terminate.lock
flock -n 9 || { echo "$(date +%T) another deadline run holds the lock"; exit 0; }
KEY=$(grep -E '^(export )?LAMBDA_API_KEY=' ~/.env | head -1 | cut -d= -f2- | tr -d "\"' ")
[ -n "$KEY" ] || { echo "$(date +%T) no LAMBDA_API_KEY"; exit 1; }
echo "$(date +%T) deadline fired"

# 1. best-effort git push (3 min cap)
if [ -n "$DRY" ]; then echo "$(date +%T) DRY: git push skipped"; else timeout 180 bash -c '
  git add -f logs/selftrain_ab_v2.log logs/finalize.log logs/deadline.log data/selftrain/drafts_uploaded 2>/dev/null
  git add -f data/evaluations/selftrain2_* data/evaluations/st2_r5_s* 2>/dev/null
  GIT_AUTHOR_NAME=Claude GIT_AUTHOR_EMAIL=noreply@anthropic.com GIT_COMMITTER_NAME=Claude GIT_COMMITTER_EMAIL=noreply@anthropic.com \
    git commit -qm "Deadline: results / logs as they stand before instance termination

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>/dev/null
  git pull -q --rebase && git push -q' && echo "$(date +%T) git pushed" || echo "$(date +%T) git push failed (not blocking)"; fi

# 2. terminate, verified and retried
MYIP=$(curl -s -m 20 https://api.ipify.org)
for i in $(seq 1 30); do
  INFO=$(curl -s -m 30 -u "$KEY:" $API/instances/$INSTANCE_ID)
  STATUS=$(echo "$INFO" | grep -o '"status": *"[a-z]*"' | head -1 | grep -o '[a-z]*"$' | tr -d '"')
  IP=$(echo "$INFO" | grep -o '"ip": *"[0-9.]*"' | head -1 | grep -o '[0-9.]*')
  echo "$(date +%T) try $i: status=$STATUS ip=$IP myip=$MYIP"
  case "$STATUS" in terminating|terminated) echo "$(date +%T) done"; exit 0;; esac
  if [ -n "$DRY" ]; then echo "$(date +%T) DRY: would terminate (ip match: $([ "$IP" = "$MYIP" ] && echo yes || echo no))"; exit 0; fi
  if [ -n "$IP" ] && [ "$IP" = "$MYIP" ]; then
    curl -s -m 60 -u "$KEY:" -X POST -H "Content-Type: application/json" \
      -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" $API/instance-operations/terminate; echo
  elif [ -n "$IP" ]; then
    echo "$(date +%T) refusing: $INSTANCE_ID has ip $IP, this machine is $MYIP"; exit 1
  fi
  sleep 60
done
