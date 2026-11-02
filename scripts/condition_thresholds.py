"""Does knowing the capture condition help set the threshold?

Part-level split of test_public (half of the good and half of the defective parts calibrate, the
rest is reported), repeated over many draws, at a 0 % miss budget. Rules:
  global         one threshold from all calibration images
  regular_only   one threshold from the calibration images taken under "regular" lighting
  per_condition  one threshold per condition, each from that condition's calibration images
  normalized     scores standardised per condition by the calibration-good median and IQR,
                 then one global threshold

    uv run python scripts/condition_thresholds.py results/scores/{patchcore,mhpatchcore,dinomaly_l}__*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_inspection.metrics import calibrate_threshold
from industrial_inspection.plots import condition_threshold_figure

TAG_RE = r"^\d+_(.+?)(?:-\d+)?\.png$"


def run(csv: Path, draws: int, seed: int) -> list[dict]:
    model, category, _ = csv.stem.split("__")
    df = pd.read_csv(csv)
    path = df.image_path.str.replace("\\", "/", regex=False).str.rsplit("/", n=2)
    df["part"] = path.str[-2] + "/" + path.str[-1].str.split("_").str[0]
    df["tag"] = path.str[-1].str.extract(TAG_RE)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(draws):
        cal_parts = set()
        for lab in (0, 1):
            parts = rng.permutation(df.part[df.label == lab].unique())
            cal_parts |= set(parts[: len(parts) // 2])
        cal = df.part.isin(cal_parts)
        c, e = df[cal], df[~cal]

        def score(thr_of_row: np.ndarray) -> tuple[float, float]:
            rej = e.score.to_numpy() >= thr_of_row
            y = e.label.to_numpy()
            return float((~rej & (y == 1)).sum() / max((y == 1).sum(), 1)), float((rej & (y == 0)).sum() / max((y == 0).sum(), 1))

        out = {}
        out["global"] = score(np.full(len(e), calibrate_threshold(c.score, c.label, 0.0)))
        r = c[c.tag == "regular"]
        out["regular_only"] = score(np.full(len(e), calibrate_threshold(r.score, r.label, 0.0)))
        per = {t: calibrate_threshold(g.score, g.label, 0.0) for t, g in c.groupby("tag") if g.label.any()}
        out["per_condition"] = score(e.tag.map(per).to_numpy(dtype=float))
        stats = {t: (g.score.median(), max(g.score.quantile(0.75) - g.score.quantile(0.25), 1e-12))
                 for t, g in c[c.label == 0].groupby("tag")}
        norm = lambda x: (x.score - x.tag.map(lambda t: stats[t][0])) / x.tag.map(lambda t: stats[t][1])  # noqa: E731
        thr = calibrate_threshold(norm(c).to_numpy(), c.label.to_numpy(), 0.0)
        rej_norm = norm(e).to_numpy() >= thr
        y = e.label.to_numpy()
        out["normalized"] = (float((~rej_norm & (y == 1)).sum() / (y == 1).sum()),
                             float((rej_norm & (y == 0)).sum() / (y == 0).sum()))
        for rule, (miss, ok) in out.items():
            rows.append({"model": model, "category": category, "draw": d, "rule": rule, "miss": miss, "overkill": ok})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("results/condition_thresholds.csv"))
    args = p.parse_args()
    raw = pd.DataFrame([r for i, csv in enumerate(sorted(args.csvs)) for r in run(csv, args.draws, seed=i)])
    summary = raw.groupby(["model", "category", "rule"])[["miss", "overkill"]].mean().reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.groupby("rule")[["miss", "overkill"]].mean().to_string())
        print(summary.groupby(["model", "rule"])[["miss", "overkill"]].mean().unstack("rule").round(3).to_string())
    print("figure:", condition_threshold_figure(summary))


if __name__ == "__main__":
    main()
