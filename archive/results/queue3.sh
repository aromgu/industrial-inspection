#!/bin/sh
# Queue 3: after the EfficientAD sweep, (1) PatchCore on the original MVTec AD to check the pipeline
# against published numbers, (2) anomaly maps for every AD 2 model/category (no retraining).
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/queue3_status.txt; }
until [ "$(grep -c 'lane \[efficientad\] done' results/logs/sweep_status.txt)" -ge 2 ]; do sleep 60; done
log "ad1 patchcore start"
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  $PY scripts/score.py --model patchcore --category $c --dataset mvtecad --root datasets/MVTecAD \
    --out results/scores_ad1 > results/logs/ad1_patchcore.$c.log 2>&1
  log "ad1 patchcore $c exit $?"
done
log "maps start"
for m in patchcore mhpatchcore efficientad dinomaly_l; do
  for c in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
    $PY scripts/export_maps.py --model $m --category $c > results/logs/maps.$m.$c.log 2>&1
    log "maps $m $c exit $?"
  done
done
log "queue3 done"
