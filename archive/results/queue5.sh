#!/bin/sh
# Queue 5 replaces the rest of queue 4: PatchCore at 392 is ~20 min per category (coreset over 2.3x more
# patches), so only the three categories the resolution question is about, plus can (already running).
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/queue4_status.txt; }
log "queue4 shortened: patchcore_392 only can (running) + rice sheet_metal fabric"
while powershell -NoProfile -Command "exit [int](-not (Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'scripts.score[.]py' -and \$_.Name -match 'python' }))"; do sleep 30; done
log "2 p392 can finished: $(ls results/scores/patchcore_392__can__seed0.csv 2>/dev/null)"
$PY scripts/export_maps.py --model patchcore_392 --category can > results/logs/maps.patchcore_392.can.log 2>&1; log "2 p392 maps can exit $?"
for c in rice sheet_metal fabric; do
  $PY scripts/score.py --model patchcore_392 --category $c > results/logs/patchcore_392.$c.log 2>&1; log "2 p392 $c exit $?"
  $PY scripts/export_maps.py --model patchcore_392 --category $c > results/logs/maps.patchcore_392.$c.log 2>&1; log "2 p392 maps $c exit $?"
done
log "3 validation scores start"
for m in patchcore mhpatchcore efficientad dinomaly_l; do for c in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
  $PY scripts/export_maps.py --model $m --category $c --split validation > results/logs/val.$m.$c.log 2>&1; log "3 val $m $c exit $?"
done; done
log "4 tiles start"
for spec in "can 4x2" "wallplugs 3x3"; do
  set -- $spec
  $PY scripts/score.py --model mhpatchcore_tilebig --category $1 --root datasets/tiles/$1_$2 --out results/scores_tiles > results/logs/mhpatchcore_tilebig.$1.log 2>&1
  log "4 tilebig $1 exit $?"
  $PY scripts/aggregate_tiles.py results/scores_tiles/mhpatchcore_tilebig__$1__seed0.csv >> results/logs/queue4_status.txt 2>&1
  for m in mhpatchcore_tile mhpatchcore_tilebig; do
    $PY scripts/export_maps.py --model $m --category $1 --root datasets/tiles/$1_$2 --out results/tile_study > results/logs/maps.$m.$1.log 2>&1
    log "4 maps $m $1 exit $?"
  done
done
log "queue4 done"
