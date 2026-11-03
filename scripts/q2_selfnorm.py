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
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import calibrate_threshold, evaluate_at_threshold
from industrial_inspection.plots import q2_selfnorm_figure

TAG_RE = r"^\d+_(.+?)(?:-\d+)?\.png$"


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


def split_parts(parts: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Boolean mask of the DEV half, drawn per class over physical parts."""
    rng = np.random.default_rng(seed)
    dev = np.zeros(len(parts), dtype=bool)
    for c in (0, 1):
        u = rng.permutation(np.unique(parts[labels == c]))
        dev |= np.isin(parts, u[: len(u) // 2])
    return dev


def evaluate(s: np.ndarray, y: np.ndarray, tag: np.ndarray) -> dict[str, float]:
    reg, sh = tag == "regular", np.char.startswith(tag.astype(str), "shift")
    thr = calibrate_threshold(s[reg], y[reg], 0.0)
    op = evaluate_at_threshold(s[sh], y[sh], thr)
    good_reg, good_sh = np.median(s[reg & (y == 0)]), np.median(s[sh & (y == 0)])
    return {"auroc_all": roc_auc_score(y, s), "auroc_shift": roc_auc_score(y[sh], s[sh]),
            "overkill_shift": op.overkill_rate, "miss_shift": op.miss_rate,
            "drift": good_sh / good_reg if good_reg != 0 else np.nan}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--models", nargs="+", default=["patchcore", "dinomaly_l"])
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--half", choices=["dev", "eval"], default="dev")
    p.add_argument("--out", type=Path, default=Path("results/q2"))
    args = p.parse_args()

    rows = []
    for model in args.models:
        z = np.load(f"results/maps/{model}__{args.category}.npz")
        maps, y = z["maps"], z["labels"]
        paths = pd.Series(z["paths"]).str.replace("\\", "/", regex=False)
        name = paths.str.rsplit("/", n=1).str[-1]
        parts = (paths.str.rsplit("/", n=2).str[-2] + "/" + name.str.split("_").str[0]).to_numpy()
        tag = name.str.extract(TAG_RE)[0].to_numpy()
        keep = split_parts(parts, y, args.split_seed)
        if args.half == "eval":
            keep = ~keep
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
