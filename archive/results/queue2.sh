#!/bin/sh
# GPU queue 2: starts when queue 1 is done. Latency first (on a quiet GPU), then the EfficientAD sweep.
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/queue_status.txt; }
until grep -q "queue1 done" results/logs/queue_status.txt; do sleep 60; done
log "latency start"
$PY scripts/latency.py --category sheet_metal > results/logs/latency.log 2>&1
log "latency exit $?"
log "efficientad sweep start"
PY=$PY SCORE_ARGS="--game-mode" sh scripts/sweep.sh "efficientad" "can fabric fruit_jelly rice vial wallplugs walnuts"
log "queue2 done"
