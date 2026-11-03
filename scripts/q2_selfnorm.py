"""Q2 step 1: can the lighting-induced score drift be removed inside each image's own anomaly map?

Hypothesis: a lighting change raises the whole anomaly map (normal regions included), while a
defect raises only a few places. Subtracting the image's own typical level (median) or dividing by
its spread should then cancel most of the drift and keep the defect signal. No retraining: the
maps saved by export_maps.py are reused.

Protocol (the research rules agreed on 2026-11-03):
  * one category at a time (walnuts for Q2), physical parts split once into a DEV half (used to
    compare and choose methods) and an EVAL half (touched only for the final report);
  * main metric: threshold set on "regular" images at a 0 % miss budget, then applied unchanged
    to the shift_* conditions -> overkill and miss there; plus AUROC and how far the median good
    score moves under shift (drift, 1.0 = no drift).

    uv run python scripts/q2_selfnorm.py --category walnuts --models patchcore dinomaly_l
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_inspection.plots import q2_selfnorm_figure
from industrial_inspection.q2 import evaluate, half_mask, parts_and_tags


def variants(maps: np.ndarray) -> dict[str, np.ndarray]:
    flat = maps.reshape(len(maps), -1).astype(np.float64)
    mx, med = flat.max(axis=1), np.median(flat, axis=1)
    q1, q3 = np.quantile(flat, [0.25, 0.75], axis=1)
    top1 = np.sort(flat, axis=1)[:, -max(1, flat.shape[1] // 100):].mean(axis=1)
    return {
        "max": mx,
        "max - median": mx - med,
        "(max - median) / IQR": (mx - med) / np.maximum(q3 - q1, 1e-12),
        "max / median": mx / np.maximum(med, 1e-12),
        "top1% - median": top1 - med,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--models", nargs="+", default=["patchcore", "dinomaly_l"])
    p.add_argument("--half", choices=["dev", "eval"], default="dev")
    p.add_argument("--out", type=Path, default=Path("results/q2"))
    args = p.parse_args()

    rows = []
    for model in args.models:
        z = np.load(f"results/maps/{model}__{args.category}.npz")
        maps, y = z["maps"], z["labels"]
        parts, tag = parts_and_tags(z["paths"])
        keep = half_mask(parts, y, args.half)
        for vname, s in variants(maps).items():
            rows.append({"model": model, "category": args.category, "half": args.half, "score": vname,
                         "n_images": int(keep.sum()), "n_good_parts": len(np.unique(parts[keep & (y == 0)])),
                         **evaluate(s[keep], y[keep], tag[keep])})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"selfnorm__{args.category}__{args.half}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print("figure:", q2_selfnorm_figure(df, args.category, args.half))


if __name__ == "__main__":
    main()
