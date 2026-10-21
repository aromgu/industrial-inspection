"""Turn score CSVs into a line-inspection report.

For each (model, category) and each miss budget, the threshold is calibrated
on one stratified half of test_public and reported on the other half,
repeated over several splits. The same-set ("calib") numbers are kept beside
the held-out ones so the optimism of tuning on the report set is visible.

    uv run python scripts/evaluate.py results/scores/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import held_out_operating_point


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--miss-budgets", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("results/report.csv"))
    args = p.parse_args()

    rows = []
    for csv in args.csvs:
        model, category, _ = csv.stem.split("__")
        df = pd.read_csv(csv)
        s, y = df.score.to_numpy(), df.label.to_numpy()
        auroc = roc_auc_score(y, s)
        for budget in args.miss_budgets:
            pts = [held_out_operating_point(s, y, budget, seed=k) for k in range(args.splits)]
            calib = np.array([[c.miss_rate, c.overkill_rate] for c, _ in pts])
            held = np.array([[e.miss_rate, e.overkill_rate] for _, e in pts])
            rows.append({
                "model": model,
                "category": category,
                "auroc": auroc,
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
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
