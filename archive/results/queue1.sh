#!/bin/sh
# GPU queue 1 (game mode): tiled MH-PatchCore on can/wallplugs, then brightness-augmented PatchCore.
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/queue_status.txt; }
for spec in "can 4x2" "wallplugs 3x3"; do
  set -- $spec
  log "mhpatchcore_tile $1 start"
  $PY scripts/score.py --model mhpatchcore_tile --category $1 --root datasets/tiles/$1_$2 \
    --out results/scores_tiles --game-mode > results/logs/mhpatchcore_tile.$1.log 2>&1
  rc=$?; log "mhpatchcore_tile $1 exit $rc"
  [ $rc -eq 0 ] && $PY scripts/aggregate_tiles.py results/scores_tiles/mhpatchcore_tile__$1__seed0.csv >> results/logs/queue_status.txt 2>&1
done
log "patchcore_aug sweep start"
PY=$PY SCORE_ARGS="--game-mode" sh scripts/sweep.sh "patchcore_aug" "can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts"
log "queue1 done"
