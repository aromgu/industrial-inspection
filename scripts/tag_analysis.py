"""Break score files down by the capture condition encoded in test_public file names.

Names look like 000_overexposed-7.png / 000_regular-14.png / 000_shift_1-7.png. Per condition:
  - AUROC (4 good vs 15 defective per condition on sheet_metal: indicative, not precise)
  - miss / overkill when the threshold is calibrated on the "regular" images only and then applied
    unchanged to every condition, i.e. calibrate once under normal conditions and let the lighting
    change. The evaluated conditions are disjoint from the calibration images, so no split is needed;
    the "regular" row itself is the calibration set and therefore optimistic.
  - median score of the good images, relative to regular: how far the condition moves normal parts.

    uv run python scripts/tag_analysis.py results/scores/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import calibrate_threshold, evaluate_at_threshold
from industrial_inspection.plots import tag_figure

TAG_ORDER = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4"]
# The "-<n>" suffix is optional: can uses both 011_overexposed.png and 006_underexposed-2.png.
TAG_RE = r"^\d+_(.+?)(?:-\d+)?\.png$"


def analyse(csv: Path, miss_budget: float) -> list[dict]:
    model, category, _ = csv.stem.split("__")
    df = pd.read_csv(csv)
    name = df.image_path.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    df["tag"] = name.str.extract(TAG_RE)[0]
    if df.tag.isna().any():
        raise ValueError(f"{csv}: {int(df.tag.isna().sum())} file names without a condition tag")

    ref = df[df.tag == "regular"]
    thr = calibrate_threshold(ref.score, ref.label, miss_budget)
    ref_good = ref[ref.label == 0].score.median()
    rows = []
    for tag in [t for t in TAG_ORDER if t in set(df.tag)]:
        g = df[df.tag == tag]
        p = evaluate_at_threshold(g.score, g.label, thr)
        rows.append({
            "model": model,
            "category": category,
            "tag": tag,
            "n_good": int((g.label == 0).sum()),
            "n_defective": int((g.label == 1).sum()),
            "auroc": roc_auc_score(g.label, g.score),
            "miss": p.miss_rate,
            "overkill": p.overkill_rate,
            "good_score_vs_regular": g[g.label == 0].score.median() / ref_good,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--miss-budget", type=float, default=0.0, help="miss budget used to calibrate on regular")
    p.add_argument("--out", type=Path, default=Path("results/tag_report.csv"))
    args = p.parse_args()

    report = pd.DataFrame([r for csv in args.csvs for r in analyse(csv, args.miss_budget)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(report.to_string(index=False))
    for category, rep in report.groupby("category"):
        print("figure:", tag_figure(rep, category, args.miss_budget))


if __name__ == "__main__":
    main()
