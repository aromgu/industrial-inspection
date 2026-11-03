"""How much do the headline numbers move if a different set of physical parts had been photographed?

Part-level bootstrap: within each class, resample the physical parts with replacement (a part
brings all its capture conditions), then recompute AUROC and the held-out overkill at a 0 % miss
budget (mean of SPLITS part-level splits). No retraining: it shows the uncertainty that comes
from having only ~15 defective and 4-12 good parts per category, not seed-to-seed variation.

    uv run python scripts/bootstrap_ci.py results/scores/{patchcore,mhpatchcore,dinomaly_l,efficientad}__*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import held_out_operating_point
from industrial_inspection.plots import bootstrap_figure

SPLITS = 10


def part_ids(image_paths: pd.Series) -> np.ndarray:
    p = image_paths.str.replace("\\", "/", regex=False).str.rsplit("/", n=2)
    return (p.str[-2] + "/" + p.str[-1].str.split("_").str[0]).to_numpy()


def stats(s: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int) -> tuple[float, float]:
    ok = [held_out_operating_point(s, y, 0.0, seed=seed * 100 + k, groups=g)[1].overkill_rate for k in range(SPLITS)]
    return roc_auc_score(y, s), float(np.mean(ok))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("results/bootstrap_ci.csv"))
    args = p.parse_args()

    draws = []
    for csv in sorted(args.csvs):
        model, category, _ = csv.stem.split("__")
        df = pd.read_csv(csv)
        s, y, g = df.score.to_numpy(), df.label.to_numpy(), part_ids(df.image_path)
        point = stats(s, y, g, seed=0)
        rng = np.random.default_rng(0)
        parts = {c: np.unique(g[y == c]) for c in (0, 1)}
        index = {pid: np.flatnonzero(g == pid) for pid in np.unique(g)}
        for d in range(args.draws):
            idx, groups = [], []
            for c in (0, 1):
                for k, pid in enumerate(rng.choice(parts[c], size=parts[c].size, replace=True)):
                    idx.append(index[pid])
                    groups.append(np.full(index[pid].size, f"{pid}#{k}"))  # a repeated part counts as a new part
            idx = np.concatenate(idx)
            auroc, ok = stats(s[idx], y[idx], np.concatenate(groups), seed=d + 1)
            draws.append({"model": model, "category": category, "draw": d, "auroc": auroc, "overkill": ok,
                          "auroc_point": point[0], "overkill_point": point[1]})
        print(f"{model} {category}: done", flush=True)

    raw = pd.DataFrame(draws)
    lo, hi = (lambda v: v.quantile(0.025)), (lambda v: v.quantile(0.975))
    per_cat = raw.groupby(["model", "category"]).agg(
        auroc=("auroc_point", "first"), auroc_lo=("auroc", lo), auroc_hi=("auroc", hi),
        overkill=("overkill_point", "first"), overkill_lo=("overkill", lo), overkill_hi=("overkill", hi)).reset_index()
    # Model means: average the categories inside each draw, then take that distribution's interval.
    per_draw = raw.groupby(["model", "draw"])[["auroc", "overkill"]].mean().reset_index()
    points = per_cat.groupby("model")[["auroc", "overkill"]].mean()
    per_model = per_draw.groupby("model").agg(auroc_lo=("auroc", lo), auroc_hi=("auroc", hi),
                                              overkill_lo=("overkill", lo), overkill_hi=("overkill", hi))
    per_model = points.join(per_model).reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_cat.to_csv(args.out, index=False)
    per_model.to_csv(args.out.with_name(args.out.stem + "_models.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(per_model.to_string(index=False))
        print(per_cat.to_string(index=False))
    print("figure:", bootstrap_figure(per_cat, per_model))


if __name__ == "__main__":
    main()
