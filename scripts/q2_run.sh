#!/bin/sh
# Q2: smoke-test every variant first, then run them for real, then compare against the base models.
# Stops before any full run if a single smoke test fails.
#
#   PY=.venv/Scripts/python.exe sh scripts/q2_run.sh "<variants>" [category] [extra score.py args]
#
# H2-b (input normalisation, PatchCore, ~5 min each):
#   sh scripts/q2_run.sh "patchcore_clahe patchcore_histeq patchcore_standardize patchcore_gray patchcore_retinex"
# H2-c (local lighting augmentation, PatchCore; add the best H2-b norm as patchcore_light_<norm>):
#   sh scripts/q2_run.sh "patchcore_light patchcore_light_clahe"
# H2-d (winners on Dinomaly-L, ~45 min each at full GPU; add --game-mode when the user plays):
#   sh scripts/q2_run.sh "dinomaly_l_clahe dinomaly_l_light" walnuts
# Final report, once, on the held-out half:
#   $PY scripts/q2_compare.py --category walnuts --half eval --variants patchcore dinomaly_l <winners>
set -u
PY=${PY:-.venv/Scripts/python.exe}
VARIANTS=$1
CAT=${2:-walnuts}
EXTRA=${3:-}
LOG=results/logs
STATUS=$LOG/q2_status.txt
mkdir -p "$LOG"

for m in $VARIANTS; do
  echo "$(date +%T) smoke $m $CAT" >> "$STATUS"
  if ! $PY scripts/score.py --model "$m" --category "$CAT" --smoke --num-workers 0 $EXTRA > "$LOG/smoke.$m.$CAT.log" 2>&1; then
    echo "$(date +%T) smoke FAILED $m (see $LOG/smoke.$m.$CAT.log); no full run started" >> "$STATUS"
    exit 1
  fi
done

for m in $VARIANTS; do
  if [ -f "results/scores/${m}__${CAT}__seed0.csv" ]; then
    echo "$(date +%T) skip $m (scores exist)" >> "$STATUS"
    continue
  fi
  echo "$(date +%T) start $m $CAT" >> "$STATUS"
  $PY scripts/score.py --model "$m" --category "$CAT" $EXTRA > "$LOG/q2.$m.$CAT.log" 2>&1
  echo "$(date +%T) $m exit $?" >> "$STATUS"
done

$PY scripts/q2_compare.py --category "$CAT" --half dev --variants patchcore dinomaly_l $VARIANTS > "$LOG/q2_compare.$CAT.log" 2>&1
echo "$(date +%T) q2 done" >> "$STATUS"
