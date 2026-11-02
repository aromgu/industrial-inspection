"""Turn score CSVs into a line-inspection report.

For each (model, category) and each miss budget, the threshold is calibrated
on one stratified half of test_public and reported on the other half,
repeated over several splits. Halves are drawn by physical part, not by image:
test_public shows each part under 4-7 capture conditions, so an image-level split
puts the same part on both sides and flatters the held-out numbers. The same-set ("calib") numbers are kept beside
the held-out ones so the optimism of tuning on the report set is visible.

    uv run python scripts/evaluate.py results/scores/*.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import held_out_operating_point
from industrial_inspection.plots import comparison_figure, summary_figure, test_figure, training_figure


def part_ids(image_paths: pd.Series) -> np.ndarray:
    """Physical part id per image: folder (good/bad) plus the leading number, e.g. bad/007."""
    p = image_paths.str.replace("\\", "/", regex=False).str.rsplit("/", n=2)
    return (p.str[-2] + "/" + p.str[-1].str.split("_").str[0]).to_numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--miss-budgets", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("results/report.csv"))
    p.add_argument("--split-unit", choices=["part", "image"], default="part",
                   help="split test_public by physical part (default) or by image (the old, optimistic way)")
    p.add_argument("--no-figures", action="store_true", help="table only (e.g. for a comparison run)")
    args = p.parse_args()

    rows = []
    for csv in args.csvs:
        model, category, _ = csv.stem.split("__")
        df = pd.read_csv(csv)
        s, y = df.score.to_numpy(), df.label.to_numpy()
        groups = part_ids(df.image_path) if args.split_unit == "part" else None
        auroc = roc_auc_score(y, s)
        for budget in args.miss_budgets:
            pts = [held_out_operating_point(s, y, budget, seed=k, groups=groups) for k in range(args.splits)]
            calib = np.array([[c.miss_rate, c.overkill_rate] for c, _ in pts])
            held = np.array([[e.miss_rate, e.overkill_rate] for _, e in pts])
            rows.append({
                "model": model,
                "category": category,
                "auroc": auroc,
                "split_unit": args.split_unit,
                "miss_budget": budget,
                "calib_miss": calib[:, 0].mean(),
                "calib_overkill": calib[:, 1].mean(),
                "heldout_miss": held[:, 0].mean(),
                "heldout_miss_p95": np.quantile(held[:, 0], 0.95),
                "heldout_overkill": held[:, 1].mean(),
                "heldout_overkill_p95": np.quantile(held[:, 1], 0.95),
            })

    report = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    if args.no_figures:
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(report.to_string(index=False))
        return

    # Test-time figures are mandatory: one per score file, plus a per-category model comparison.
    by_category: dict[str, list[Path]] = defaultdict(list)
    for csv in args.csvs:
        by_category[csv.stem.split("__")[1]].append(csv)
        print("figure:", test_figure(csv))
        fig = training_figure(Path("results/curves") / csv.stem)
        if fig:
            print("figure:", fig)
    for category, csvs in by_category.items():
        if len(csvs) > 1:
            print("figure:", comparison_figure(csvs, category))

    if report.category.nunique() > 1:
        print("figure:", summary_figure(report))
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
