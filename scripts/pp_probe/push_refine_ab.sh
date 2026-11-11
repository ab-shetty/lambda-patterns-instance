#!/usr/bin/env bash
# Copy the refine A/B probes' printed lines next to their rows (data/ is not tracked), commit, push.  push_refine_ab.sh "message"
cd "$(dirname "$0")/../.."
mkdir -p scripts/pp_probe/refine_ab
for d in data/evaluations/refine_ab/*/; do t=$(basename $d)
  for f in $d*.txt; do [ -s $f ] && cp $f scripts/pp_probe/refine_ab/${t}_$(basename $f); done; done
git add -f scripts/pp_probe/rows_*.json; git add scripts/pp_probe/refine_ab scripts/pp_probe/push_refine_ab.sh run_refine_ab.sh
git -c user.name="$(git log -1 --format=%an)" -c user.email="$(git log -1 --format=%ae)" commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD && git log -1 --format='%h %s' | head -1
