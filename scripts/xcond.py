"""Cross-condition consistency (see industrial_inspection.xcond): does scoring a part from several
aligned photos taken under different conditions separate good from defective parts better than
one photo under regular light?

No retraining: uses the anomaly maps saved by export_maps.py (results/maps/<model>__<category>.npz),
so every model is trained on regular-light images only, as before. A part is one sample.

Reported per model and category (all test_public parts; the halves are too small to split here):
  * part-level image AUROC with a part bootstrap 95 % interval, and the paired change against the
    one-photo baseline (regular light only);
  * held-out overkill at 0 % miss (part-level stratified splits, mean of 50; with ~7 defective parts
    on the calibration side a 5 % budget allows no miss either, so 0 % and 5 % coincide);
  * which defective parts each combiner rescues or loses at the threshold where no good part alarms.
The main configuration (pixel-min over every available condition) was fixed by the hypothesis; the
2- and 3-photo subsets, the other combiners and the alignment tolerance are secondary analyses.
Added after the first run (2026-11-05): "anchored" (anomalous in the regular photo and in at least
one other) and "hybrid" (mean of the label-free robust z-scores of the one-photo and pixel-min
scores), to keep defects that only one photo shows. Photo-to-photo alignment was measured instead of
added: the shifts are below one map cell in every category (results/xcond/shifts.csv).

    uv run python scripts/xcond.py --models patchcore dinomaly_l efficientad mhpatchcore
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import held_out_operating_point
from industrial_inspection.plots import xcond_figure, xcond_subset_figure
from industrial_inspection.q2 import parts_and_tags
from industrial_inspection.xcond import combine

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
COMBINERS = ["image-mean", "image-max", "pixel-median", "pixel-2nd", "pixel-min", "anchored"]


def robust_z(s: np.ndarray) -> np.ndarray:
    """Standardise a category's part scores by their median and IQR (no labels used)."""
    q1, med, q3 = np.quantile(s, [0.25, 0.5, 0.75])
    return (s - med) / max(q3 - q1, 1e-12)


def load(model: str, category: str):
    """Per part: label and {condition: map} (first photo of each condition)."""
    z = np.load(f"results/maps/{model}__{category}.npz")
    maps, y = z["maps"], z["labels"]
    parts, tags = parts_and_tags(z["paths"].astype(str))
    out = {}
    for i in np.argsort(z["paths"].astype(str)):
        lab, photos = out.setdefault(parts[i], (int(y[i]), {}))
        photos.setdefault(str(tags[i]), maps[i])
    return out


def part_scores(data: dict, conditions: list[str], how: str, radius: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names, s, y = [], [], []
    for part, (lab, photos) in sorted(data.items()):
        if not all(c in photos for c in conditions):
            continue
        names.append(part)
        y.append(lab)
        s.append(combine(np.stack([photos[c] for c in conditions]), how, radius))
    return np.array(s), np.array(y), names


def overkill(s: np.ndarray, y: np.ndarray, budget: float, splits: int = 50) -> float:
    return float(np.mean([held_out_operating_point(s, y, budget, seed=k)[1].overkill_rate for k in range(splits)]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["patchcore", "dinomaly_l", "efficientad", "mhpatchcore"])
    p.add_argument("--categories", nargs="+", default=CATS)
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--radius", type=int, default=2, help="alignment tolerance in map cells for the *_tol rows")
    p.add_argument("--out", type=Path, default=Path("results/xcond"))
    args = p.parse_args()
    rng = np.random.default_rng(0)

    main_rows, subset_rows, change_rows = [], [], []
    for model in args.models:
        for cat in args.categories:
            data = load(model, cat)
            conds = sorted({c for _, (_, ph) in data.items() for c in ph}, key=lambda c: (c != "regular", c))
            base_s, y, names = part_scores(data, ["regular"], "image-mean")
            # paired part bootstrap: the same resample of parts for every combiner
            idx = {c: np.flatnonzero(y == c) for c in (0, 1)}
            boots = [np.concatenate([rng.choice(idx[c], idx[c].size) for c in (0, 1)]) for _ in range(args.draws)]
            configs = [("one photo (regular)", ["regular"], "image-mean", 0)]
            configs += [(f"{how}, all {len(conds)}", conds, how, 0) for how in COMBINERS]
            configs += [(f"pixel-min, all {len(conds)}, tol {args.radius}", conds, "pixel-min", args.radius)]
            configs += [(f"hybrid (one photo + pixel-min), all {len(conds)}", conds, "hybrid", 0)]
            for label, cs, how, r in configs:
                if how == "hybrid":  # half the one-photo score, half the consistency score
                    s_min, y2, names2 = part_scores(data, cs, "pixel-min", r)
                    s = 0.5 * robust_z(base_s) + 0.5 * robust_z(s_min)
                else:
                    s, y2, names2 = part_scores(data, cs, how, r)
                assert names2 == names, "every part should have every condition"
                b = np.array([roc_auc_score(y[i], s[i]) for i in boots])
                b0 = np.array([roc_auc_score(y[i], base_s[i]) for i in boots])
                thr_good = s[y == 0].max()
                base_thr = base_s[y == 0].max()
                main_rows.append({
                    "model": model, "category": cat, "config": label, "n_good": int((y == 0).sum()), "n_bad": int(y.sum()),
                    "auroc": roc_auc_score(y, s), "auroc_lo": np.quantile(b, 0.025), "auroc_hi": np.quantile(b, 0.975),
                    "d_auroc": roc_auc_score(y, s) - roc_auc_score(y, base_s),
                    "d_lo": np.quantile(b - b0, 0.025), "d_hi": np.quantile(b - b0, 0.975),
                    "overkill_0": overkill(s, y, 0.0), "overkill_5": overkill(s, y, 0.05),
                    "caught_at_0_overkill": int((s[y == 1] > thr_good).sum()),
                    "caught_base": int((base_s[y == 1] > base_thr).sum())})
                if label.startswith("pixel-min, all") and r == 0:
                    for n, part in enumerate(names):
                        if y[n] == 1:
                            change_rows.append({"model": model, "category": cat, "part": part,
                                                "caught_base": bool(base_s[n] > base_thr), "caught_min": bool(s[n] > thr_good)})
            # how many photos are needed: every subset that contains regular, pixel-min vs image-mean
            for k in range(2, len(conds) + 1):
                for extra in itertools.combinations(conds[1:], k - 1):
                    cs = ["regular", *extra]
                    for how in ("image-mean", "pixel-min"):
                        s, y2, _ = part_scores(data, cs, how)
                        subset_rows.append({"model": model, "category": cat, "k": k, "conditions": "+".join(cs),
                                            "combiner": how, "auroc": roc_auc_score(y2, s)})
            print(f"{model} {cat}: done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    main_df, subset_df, change_df = pd.DataFrame(main_rows), pd.DataFrame(subset_rows), pd.DataFrame(change_rows)
    main_df.to_csv(args.out / "xcond_main.csv", index=False)
    subset_df.to_csv(args.out / "xcond_subsets.csv", index=False)
    change_df.to_csv(args.out / "xcond_parts.csv", index=False)
    key = main_df.config.str.replace(r", all \d+", ", all", regex=True)
    main_df = main_df.assign(key=key)
    summary = main_df.groupby(["model", "key"])[["auroc", "d_auroc", "overkill_0", "overkill_5"]].mean().round(3)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(summary)
        sig = main_df[(main_df.key == "pixel-min, all") & ((main_df.d_lo > 0) | (main_df.d_hi < 0))]
        print("\npixel-min, all: categories whose paired interval excludes 0:")
        print(sig[["model", "category", "d_auroc", "d_lo", "d_hi"]].round(3).to_string(index=False))
        ch = change_df.groupby("model")[["caught_base", "caught_min"]].sum()
        ch["rescued"] = change_df[~change_df.caught_base & change_df.caught_min].groupby("model").size()
        ch["lost"] = change_df[change_df.caught_base & ~change_df.caught_min].groupby("model").size()
        print("\ndefective parts caught with no good part alarming (one photo vs pixel-min):")
        print(ch.fillna(0).astype(int))
    print("figure:", xcond_figure(main_df))
    one = main_df[main_df.key == "one photo (regular)"].groupby("model").auroc.mean()
    print("figure:", xcond_subset_figure(subset_df, one))


if __name__ == "__main__":
    main()
