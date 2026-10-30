#!/bin/sh
# Run score.py for every (model, category) pair that has no score file yet.
# Resumable: finished pairs are skipped. A failing pair is logged and the sweep goes on.
#
#   sh scripts/sweep.sh "patchcore mhpatchcore" "can fabric fruit_jelly rice vial wallplugs walnuts"
#   SCORE_ARGS="--game-mode" sh scripts/sweep.sh ...   # extra score.py flags, e.g. leave GPU room for a game
#
# Each run writes results/logs/<model>.<category>.log and appends a line to results/logs/sweep_status.txt.
# Run scripts/score.py --smoke on one category per model before starting a long sweep.

models="$1"
categories="$2"
root="${3:-datasets/MVTec_AD_2}"
mkdir -p results/logs

for category in $categories; do
  for model in $models; do
    out="results/scores/${model}__${category}__seed0.csv"
    if [ -f "$out" ]; then
      echo "$(date +%H:%M:%S) $model $category skipped (exists)" >> results/logs/sweep_status.txt
      continue
    fi
    echo "$(date +%H:%M:%S) $model $category start" >> results/logs/sweep_status.txt
    uv run python scripts/score.py --model "$model" --category "$category" --root "$root" $SCORE_ARGS \
      > "results/logs/${model}.${category}.log" 2>&1
    echo "$(date +%H:%M:%S) $model $category exit $?" >> results/logs/sweep_status.txt
  done
done
echo "$(date +%H:%M:%S) lane [$models] done" >> results/logs/sweep_status.txt
