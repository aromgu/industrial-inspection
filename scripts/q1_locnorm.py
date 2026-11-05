"""Q1 step 4 (H1-d): subtract the normal variation at each location before taking the image score.

Looking at can (q1_look.py) showed why the image score fails: the defects do light up, but some
regions of every image (the holographic foil, the print) are always bright in the anomaly map, and
the image maximum comes from them. Here the anomaly maps of the good-only validation set (not used
for training, not part of test) give, for every location, its usual level (mean) and spread (std).
Test maps are then compared with that:

  max          the model's map maximum (reference)
  max(m - mu)  how far above its usual level each location is
  max z        the same in units of that location's spread, z = (m - mu) / (sigma + median sigma)

No retraining. Reported: image AUROC (part bootstrap 95 %), AU-PRO@0.05 of the normalised map, and
AUROC per defect-size group.

    uv run python scripts/export_maps.py --model patchcore_392_c01 --category can --eval-side 512 --split validation
    uv run python scripts/q1_locnorm.py --category can --models patchcore_256_c01 patchcore_392_c01 patchcore_512_c01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.metrics.aupro import _AUPRO
from sklearn.metrics import roc_auc_score

from industrial_inspection.plots import q1_locnorm_figure
from industrial_inspection.q2 import bootstrap_parts, parts_and_tags
from q1_size import GROUPS, defect_area


def variants(maps: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    floor = np.median(sigma)
    return {"max": maps, "max(m - mu)": maps - mu, "max z": (maps - mu) / (sigma + floor)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="can")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("results/q1"))
    args = p.parse_args()

    rows, groups = [], None
    for model in args.models:
        z = np.load(f"results/maps/{model}_s{args.side}__{args.category}.npz")
        v = np.load(f"results/maps/{model}_s{args.side}__{args.category}__val.npz")
        maps, masks, y, paths = z["maps"].astype(np.float32), z["masks"], z["labels"], z["paths"].astype(str)
        val = v["maps"].astype(np.float32)
        mu, sigma = val.mean(axis=0), val.std(axis=0)
        if groups is None:
            area = np.array([defect_area(q) if lab else np.nan for q, lab in zip(paths, y)])
            cuts = np.quantile(area[y == 1], [1 / 3, 2 / 3])
            groups = np.where(y == 0, "good", np.array(GROUPS)[np.digitize(np.nan_to_num(area), cuts)])
            parts, _ = parts_and_tags(paths)
        for name, m in variants(maps, mu, sigma).items():
            s = m.reshape(len(m), -1).max(axis=1)
            boot = [roc_auc_score(y[i], s[i]) for i in bootstrap_parts(parts, y, args.draws)]
            aupro = _AUPRO(fpr_limit=0.05)
            aupro.update(torch.from_numpy(m), torch.from_numpy(masks).int())
            row = {"model": model, "score": name, "n_val": len(val), "auroc": roc_auc_score(y, s),
                   "auroc_lo": np.quantile(boot, 0.025), "auroc_hi": np.quantile(boot, 0.975),
                   "aupro_005": float(aupro.compute())}
            for g in GROUPS:
                sel = (groups == "good") | (groups == g)
                row[f"auroc_{g}"] = roc_auc_score(y[sel], s[sel])
            rows.append(row)
        print(f"{model}: done ({len(val)} validation maps)", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"locnorm__{args.category}.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print("figure:", q1_locnorm_figure(df, args.category))


if __name__ == "__main__":
    main()
