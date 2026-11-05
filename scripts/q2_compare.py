"""Q2 steps 2-4: compare lighting variants of PatchCore / Dinomaly-L on one category.

Reads the score files written by score.py (e.g. patchcore, patchcore_clahe, patchcore_light, ...),
keeps one half of the physical parts (q2.split_parts) and reports, per variant: overkill and miss
under shift with the threshold set on "regular" at 0 % miss, AUROC under shift, drift of the median
good score, and part-bootstrap 95 % intervals. Also the change against the base model within each
bootstrap draw (paired), which is what decides whether a variant helped.

    uv run python scripts/q2_compare.py --category walnuts --half dev \
        --variants patchcore patchcore_clahe patchcore_histeq patchcore_standardize patchcore_gray patchcore_retinex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_inspection.plots import q2_compare_figure
from industrial_inspection.q2 import bootstrap_parts, evaluate, half_mask, parts_and_tags


def base_of(variant: str) -> str:
    return "dinomaly_l" if variant.startswith("dinomaly_l") else "patchcore"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--half", choices=["dev", "eval"], default="dev")
    p.add_argument("--scores", type=Path, default=Path("results/scores"))
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("results/q2"))
    args = p.parse_args()

    data = {}
    for v in args.variants:
        f = args.scores / f"{v}__{args.category}__seed0.csv"
        if not f.exists():
            print(f"missing: {f}")
            continue
        df = pd.read_csv(f).sort_values("image_path").reset_index(drop=True)
        parts, tag = parts_and_tags(df.image_path)
        keep = half_mask(parts, df.label.to_numpy(), args.half)
        data[v] = (df.score.to_numpy()[keep], df.label.to_numpy()[keep], tag[keep], parts[keep], df.image_path[keep].to_numpy())
    if not data:
        raise SystemExit("no score files found")
    ref = next(iter(data.values()))
    for v, d in data.items():  # same images in the same order, so the bootstrap can be paired
        if not np.array_equal(d[4], ref[4]):
            raise SystemExit(f"{v} was scored on different images")
    _, y, tag, parts, _ = ref

    point = {v: evaluate(d[0], y, tag) for v, d in data.items()}
    draws = {v: [] for v in data}
    for idx in bootstrap_parts(parts, y, args.draws):
        for v, d in data.items():
            try:
                draws[v].append(evaluate(d[0][idx], y[idx], tag[idx]))
            except ValueError:  # a draw without both classes in some condition
                draws[v].append({k: np.nan for k in point[v]})
    rows = []
    for v in data:
        b = pd.DataFrame(draws[v])
        base = base_of(v)
        row = {"variant": v, "base": base, "category": args.category, "half": args.half, "n_images": len(y),
               "n_good_parts": len(np.unique(parts[y == 0])), **point[v]}
        for k in ("overkill_shift", "auroc_shift", "drift", "fa_shift"):
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanquantile(b[k], [0.025, 0.975])
        if base in data and v != base:  # paired change against the base model
            ref_draws = pd.DataFrame(draws[base])
            for k, short in (("overkill_shift", "overkill"), ("fa_shift", "fa")):
                row[f"d_{short}_vs_base"] = point[v][k] - point[base][k]
                row[f"d_{short}_lo"], row[f"d_{short}_hi"] = np.nanquantile(b[k] - ref_draws[k], [0.025, 0.975])
        rows.append(row)
    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"compare__{args.category}__{args.half}.csv"
    out.to_csv(path, index=False)
    cols = ["variant", "overkill_shift", "overkill_shift_lo", "overkill_shift_hi", "miss_shift", "auroc_shift", "drift",
            "d_overkill_vs_base", "d_overkill_lo", "d_overkill_hi", "fa_shift", "fa_shift_lo", "fa_shift_hi",
            "miss_shift_n", "d_fa_vs_base", "d_fa_lo", "d_fa_hi"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(out[[c for c in cols if c in out]].to_string(index=False))
    print("table:", path)
    print("figure:", q2_compare_figure(out, args.category, args.half))


if __name__ == "__main__":
    main()
