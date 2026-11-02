"""How should the threshold be set when only a few defective parts are known?

On a real line a handful of defective parts is available at commissioning. For each score file,
repeatedly draw k defective parts (all their capture conditions) plus half of the good parts as
calibration data, set a threshold, and measure miss/overkill on the parts that were NOT used.
Splits are by physical part, so a part is never on both sides.

Threshold rules:
  min_defect  lowest calibration-defect score (miss 0 on what you have seen)
  margin      min_defect lowered by half the inter-quartile range of the calibration-good scores
              (a safety margin in the scale of normal variation; always below min_defect)
  normal_only split-conformal quantile of calibration-good scores for <= ALPHA overkill; uses no
              defects at all (k = 0)

    uv run python scripts/few_shot_calibration.py results/scores/{patchcore,mhpatchcore,dinomaly_l}__*.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_inspection.plots import fewshot_figure

KS = [1, 2, 3, 5, 7]
ALPHA = 0.10


def part_ids(image_paths: pd.Series) -> np.ndarray:
    p = image_paths.str.replace("\\", "/", regex=False).str.rsplit("/", n=2)
    return (p.str[-2] + "/" + p.str[-1].str.split("_").str[0]).to_numpy()


def thresholds(s_good: np.ndarray, s_def: np.ndarray | None) -> dict[str, float]:
    n = s_good.size
    rank = min(n, math.ceil((n + 1) * (1 - ALPHA)))  # split-conformal: rank-th smallest good score
    out = {"normal_only": float(np.sort(s_good)[rank - 1])}
    if s_def is not None:
        m = float(s_def.min())
        out["min_defect"] = m
        q1, q3 = np.quantile(s_good, [0.25, 0.75])
        out["margin"] = m - 0.5 * float(q3 - q1)
    return out


def run(csv: Path, draws: int, seed: int) -> list[dict]:
    model, category, _ = csv.stem.split("__")
    df = pd.read_csv(csv)
    s, y, g = df.score.to_numpy(), df.label.to_numpy(), part_ids(df.image_path)
    good_parts, def_parts = np.unique(g[y == 0]), np.unique(g[y == 1])
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(draws):
        gp = rng.permutation(good_parts)
        g_cal, g_eval = gp[: len(gp) // 2], gp[len(gp) // 2 :]
        dp = rng.permutation(def_parts)
        good_cal = np.isin(g, g_cal) & (y == 0)
        good_eval = np.isin(g, g_eval) & (y == 0)
        for k in [0, *KS]:
            d_cal = np.isin(g, dp[:k]) & (y == 1)
            d_eval = np.isin(g, dp[k:]) & (y == 1)
            for rule, thr in thresholds(s[good_cal], s[d_cal] if k else None).items():
                if rule == "normal_only" and k:
                    continue  # identical for every k; report it once at k = 0
                rows.append({
                    "model": model, "category": category, "draw": d, "k": k, "rule": rule,
                    "miss": float((s[d_eval] < thr).mean()),
                    "overkill": float((s[good_eval] >= thr).mean()),
                })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("results/fewshot_report.csv"))
    args = p.parse_args()
    raw = pd.DataFrame([r for i, csv in enumerate(sorted(args.csvs)) for r in run(csv, args.draws, seed=i)])
    summary = (
        raw.groupby(["model", "category", "rule", "k"])
        .agg(miss=("miss", "mean"), miss_p95=("miss", lambda v: v.quantile(0.95)),
             overkill=("overkill", "mean"), overkill_p95=("overkill", lambda v: v.quantile(0.95)))
        .reset_index()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    overall = summary.groupby(["rule", "k"])[["miss", "miss_p95", "overkill", "overkill_p95"]].mean().reset_index()
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(overall.to_string(index=False))
        print(summary.groupby(["model", "rule", "k"])[["miss", "overkill"]].mean().unstack("k").round(3).to_string())
    print("figure:", fewshot_figure(overall))


if __name__ == "__main__":
    main()
