#!/bin/sh
# Queue 6 replaces the rest of queue 5 after choosing PatchCore + Dinomaly-L as the research models:
# finish PatchCore-392 fabric, then validation scores for those two models only. The MH-PatchCore tile
# rerun is dropped (MH-PatchCore is no longer a research model).
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/queue4_status.txt; }
log "queue changed: research models are patchcore + dinomaly_l; mh tile rerun dropped"
while powershell -NoProfile -Command "exit [int](-not (Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'scripts.score[.]py' -and \$_.Name -match 'python' }))"; do sleep 30; done
log "2 p392 fabric finished: $(ls results/scores/patchcore_392__fabric__seed0.csv 2>/dev/null)"
$PY scripts/export_maps.py --model patchcore_392 --category fabric > results/logs/maps.patchcore_392.fabric.log 2>&1; log "2 p392 maps fabric exit $?"
log "3 validation scores start"
for m in patchcore dinomaly_l; do for c in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
  $PY scripts/export_maps.py --model $m --category $c --split validation > results/logs/val.$m.$c.log 2>&1; log "3 val $m $c exit $?"
done; done
log "queue4 done"
