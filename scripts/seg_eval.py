"""Segmentation F1 on test_public (see industrial_inspection.segeval), from the maps of export_maps.py.

Needs results/maps/<model>__<category>.npz (test) and <model>__<category>__val.npz (good validation
images, `export_maps.py --split validation`). Per model and category:

  thresholds  official: mean + 3 std of the validation pixels (the MVTec AD 2 baseline rule)
              superadd: 95th percentile of the validation pixels x 1.421 (SuperADD's rule, which it
                        applies to held-out training images)
              oracle:   the best threshold on test_public itself. An upper bound for this map, not a
                        result: it uses the ground truth.
  protocols   defect-only (SuperADD's local protocol) and all (good images included).
  au_roc_005  pixel AU-ROC up to 5 % FPR on defect-only pixels, the "AU" column of SuperADD.

    uv run python scripts/seg_eval.py --models patchcore dinomaly_l efficientad mhpatchcore
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.plots import seg_eval_figure
from industrial_inspection.segeval import eval_size, f1_at, f1_curve, full_frame, ground_truth, view_of

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
# SuperADD, test_public, defect-only protocol (arXiv 2605.14808, Table: AU-ROC_0.05 / F1)
SUPERADD_PUBLIC_F1 = {"can": 0.0, "fabric": None, "fruit_jelly": None, "rice": None, "sheet_metal": 59.54,
                      "vial": 64.77, "wallplugs": 79.16, "walnuts": 75.69, "mean": 62.61}


def evaluate(model: str, category: str, max_au_pixels: int = 4_000_000, seed: int = 0) -> dict:
    z = np.load(f"results/maps/{model}__{category}.npz")
    v = np.load(f"results/maps/{model}__{category}__val.npz")
    paths, labels = z["paths"].astype(str), z["labels"]
    hw = eval_size(paths[0])
    view = view_of(model)
    maps = full_frame(z["maps"], hw, view)
    val = full_frame(v["maps"], hw, view).ravel()
    gt = np.stack([ground_truth(p, hw) for p in paths])

    thr = {"official": float(val.mean() + 3 * val.std()), "superadd": float(np.quantile(val, 0.95) * 1.421)}
    row = {"model": model, "category": category, "n_val": len(v["maps"]), "eval_h": hw[0], "eval_w": hw[1]}
    for proto, sel in (("defect-only", labels == 1), ("all", np.ones(len(labels), bool))):
        s, t = maps[sel].ravel(), gt[sel].ravel()
        for name, th in thr.items():
            row[f"f1_{proto}_{name}"] = f1_at(s, t, th)
        cand = np.quantile(s, np.linspace(0.5, 0.99999, 400))
        curve = f1_curve(s, t, cand)
        row[f"f1_{proto}_oracle"] = float(curve.max())
        row[f"thr_{proto}_oracle"] = float(cand[curve.argmax()])
    s, t = maps[labels == 1].ravel(), gt[labels == 1].ravel()
    if len(s) > max_au_pixels:  # the AU-ROC sort is the slow part; a fixed random subset is enough
        idx = np.random.default_rng(seed).choice(len(s), max_au_pixels, replace=False)
        s, t = s[idx], t[idx]
    row["au_roc_005"] = float(roc_auc_score(t, s, max_fpr=0.05)) if t.any() else np.nan
    row.update({f"thr_{k}": v_ for k, v_ in thr.items()})
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["patchcore", "dinomaly_l", "efficientad", "mhpatchcore"])
    p.add_argument("--categories", nargs="+", default=CATS)
    p.add_argument("--out", type=Path, default=Path("results/seg"))
    args = p.parse_args()
    rows = []
    for model in args.models:
        for cat in args.categories:
            if not Path(f"results/maps/{model}__{cat}__val.npz").exists():
                print(f"skip {model} {cat}: no validation maps (export_maps.py --split validation)")
                continue
            rows.append(evaluate(model, cat))
            r = rows[-1]
            print(f"{model} {cat}: F1 defect-only official {r['f1_defect-only_official']:.3f} "
                  f"oracle {r['f1_defect-only_oracle']:.3f} | all official {r['f1_all_official']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "seg_eval.csv", index=False)
    cols = [c for c in df.columns if c.startswith("f1_")] + ["au_roc_005"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(df.groupby("model")[cols].mean())
    print(f"SuperADD test_public (defect-only, its own threshold): mean F1 {SUPERADD_PUBLIC_F1['mean']:.2f} %")
    print("figure:", seg_eval_figure(df, SUPERADD_PUBLIC_F1, args.out.parent / "figures" / "seg_eval.png"))


if __name__ == "__main__":
    main()
