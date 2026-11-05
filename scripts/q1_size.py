"""Q1 (small defects): is any gain from a larger input actually on the small defects?

For each model, the defective images are split into size groups by their defect area in the
full-resolution mask (terciles over the category), and each group is scored against all good
images: image AUROC with a part-level bootstrap 95 % interval, and AU-PRO@0.05 on the common
grid written by export_maps.py --eval-side (results/maps/<model>_s<side>__<category>.npz).

    uv run python scripts/q1_size.py --category can --models patchcore_256_c01 patchcore_392_c01 patchcore_512_c01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.metrics.aupro import _AUPRO
from PIL import Image
from sklearn.metrics import roc_auc_score

from industrial_inspection.plots import q1_size_figure
from industrial_inspection.q2 import bootstrap_parts, parts_and_tags

GROUPS = ("small", "medium", "large")


def defect_area(path: str) -> float:
    """Share of the image covered by the defect in the full-resolution mask file."""
    p = Path(path)
    mask = p.parent.parent / "ground_truth" / p.parent.name / (p.stem + "_mask.png")
    return float((np.array(Image.open(mask).convert("L")) > 0).mean()) if mask.exists() else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="can")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("results/q1"))
    args = p.parse_args()

    rows = []
    groups = None
    for model in args.models:
        z = np.load(f"results/maps/{model}_s{args.side}__{args.category}.npz")
        paths, y, score = z["paths"].astype(str), z["labels"], z["pred_score"]
        maps, masks = torch.from_numpy(z["maps"].astype(np.float32)), torch.from_numpy(z["masks"])
        if groups is None:  # same split for every model
            area = np.array([defect_area(q) if lab else np.nan for q, lab in zip(paths, y)])
            if np.isnan(area[y == 1]).any():
                raise SystemExit("mask files not found for some defective images")
            cuts = np.quantile(area[y == 1], [1 / 3, 2 / 3])
            groups = np.where(y == 0, "good", np.array(GROUPS)[np.digitize(np.nan_to_num(area), cuts)])
            ref_paths = paths
            parts, _ = parts_and_tags(paths)
            for g in GROUPS:
                a = area[groups == g]
                print(f"{g}: {len(a)} images, defect area {a.min():.5%} - {a.max():.5%}")
        if not np.array_equal(paths, ref_paths):
            raise SystemExit(f"{model}: images differ from the first model")
        for g in GROUPS:
            sel = (groups == "good") | (groups == g)
            boot = []
            for idx in bootstrap_parts(parts[sel], y[sel], args.draws):
                ys = y[sel][idx]
                if 0 < ys.sum() < len(ys):
                    boot.append(roc_auc_score(ys, score[sel][idx]))
            aupro = _AUPRO(fpr_limit=0.05)
            aupro.update(maps[sel], masks[sel].int())
            rows.append({"model": model, "group": g, "n_defective": int((groups == g).sum()),
                         "auroc": roc_auc_score(y[sel], score[sel]), "auroc_lo": np.quantile(boot, 0.025),
                         "auroc_hi": np.quantile(boot, 0.975), "aupro_005": float(aupro.compute())})
        aupro = _AUPRO(fpr_limit=0.05)
        aupro.update(maps, masks.int())
        rows.append({"model": model, "group": "all", "n_defective": int(y.sum()), "auroc": roc_auc_score(y, score),
                     "auroc_lo": np.nan, "auroc_hi": np.nan, "aupro_005": float(aupro.compute())})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"size__{args.category}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print("figure:", q1_size_figure(df, args.category))


if __name__ == "__main__":
    main()
