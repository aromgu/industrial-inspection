"""Thresholds and post-processing for segmentation F1, on the saved anomaly maps (no GPU).

The test examples (seg_examples.py) showed three failure patterns: a large defect flagged only along
its outline (fabric), small scattered false alarms (rice), and thin scratches broken up or missed
(sheet_metal). This tries, on one model's maps:

  thresholds  official (validation mean + 3 std), validation 99th percentile, and the validation
              95th percentile times a gain (SuperADD's rule; it uses 1.3-1.5)
  post        none / fill holes / remove components smaller than A pixels / closing with a disk of
              radius r followed by hole filling, and their combinations

One configuration is chosen for all categories together (the mean F1 over categories), to keep
the choice from fitting test_public category by category. The final word is the private test.

    uv run python scripts/seg_post.py --model hires_dinov2l
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from industrial_inspection.plots import seg_post_figure
from industrial_inspection.segeval import eval_size, full_frame, ground_truth, view_of

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def postprocess(b: np.ndarray, close: int, fill: bool, min_area: int) -> np.ndarray:
    if close:
        b = ndimage.binary_closing(np.pad(b, close), structure=disk(close))[close:-close, close:-close]
    if fill:
        b = ndimage.binary_fill_holes(b)
    if min_area:
        lab, n = ndimage.label(b)
        if n:
            sizes = ndimage.sum(b, lab, index=np.arange(1, n + 1))
            b = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
    return b


THRESHOLDS = {"official": lambda v: v.mean() + 3 * v.std(), "p99": lambda v: np.quantile(v, 0.99),
              **{f"p95x{g}": (lambda g: lambda v: np.quantile(v, 0.95) * g)(g) for g in (1.0, 1.2, 1.4, 1.6)}}
POSTS = {"none": (0, False, 0), "fill": (0, True, 0), "small64": (0, False, 64), "fill+small64": (0, True, 64),
         "close3+fill": (3, True, 0), "close3+fill+small64": (3, True, 64), "close6+fill+small64": (6, True, 64),
         "fill+small256": (0, True, 256)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hires_dinov2l")
    p.add_argument("--categories", nargs="+", default=CATS)
    p.add_argument("--out", type=Path, default=Path("results/seg"))
    args = p.parse_args()

    rows = []
    for cat in args.categories:
        z = np.load(f"results/maps/{args.model}__{cat}.npz")
        v = np.load(f"results/maps/{args.model}__{cat}__val.npz")
        paths, labels = z["paths"].astype(str), z["labels"]
        hw, view = eval_size(paths[0]), view_of(args.model)
        maps = full_frame(z["maps"], hw, view)
        val = full_frame(v["maps"], hw, view).ravel()
        gt = np.stack([ground_truth(q, hw) for q in paths])
        for (tname, tfun), (pname, (close, fill, min_area)) in itertools.product(THRESHOLDS.items(), POSTS.items()):
            thr = float(tfun(val))
            pred = np.stack([postprocess(m > thr, close, fill, min_area) for m in maps])
            for proto, sel in (("defect-only", labels == 1), ("all", np.ones(len(labels), bool))):
                pr, t = pred[sel], gt[sel]
                tp = np.sum(pr & t)
                denom = 2 * tp + np.sum(pr & ~t) + np.sum(~pr & t)
                rows.append({"model": args.model, "category": cat, "threshold": tname, "post": pname, "protocol": proto,
                             "f1": float(2 * tp / denom) if denom else 0.0})
        print(f"{cat}: done", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"seg_post__{args.model}.csv", index=False)
    mean = df.groupby(["protocol", "threshold", "post"]).f1.mean().unstack("post") * 100
    with pd.option_context("display.width", 250, "display.float_format", "{:.1f}".format):
        for proto in ("defect-only", "all"):
            print(f"\n== mean F1 over categories, {proto}")
            print(mean.loc[proto][list(POSTS)])
    best = df[df.protocol == "defect-only"].groupby(["threshold", "post"]).f1.mean().idxmax()
    base = ("official", "none")
    d = df[df.protocol == "defect-only"]
    pick = lambda key: d[(d.threshold == key[0]) & (d.post == key[1])].set_index("category").f1  # noqa: E731
    cmp_ = pd.DataFrame({"baseline": pick(base), "best": pick(best)}) * 100
    print(f"\nbest for all categories together: threshold {best[0]}, post {best[1]}")
    print(cmp_.round(1).T.to_string())
    print("figure:", seg_post_figure(df, args.model, best))


if __name__ == "__main__":
    main()
