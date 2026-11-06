"""Our post-processing vs SuperADD's (industrial_inspection.superadd_post) on the same saved maps.

Per category, defect-only segmentation F1 on test_public at 1/4 resolution (SuperADD's protocol), for:
  ours            the submission setting (make_submission.CONFIG): threshold rule + our post-processing
  superadd        SuperADD's threshold rule (validation 95th pct x 1.421) + SuperADD's post-processing
  ours_thr+sa     our threshold rule + SuperADD's post-processing
Thresholds come from validation maps only. Writes results/seg/compare_post.csv and
docs/figures/compare_post.png.

    .venv/Scripts/python.exe scripts/compare_post.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_submission import CONFIG, threshold_of  # noqa: E402
from seg_post import postprocess  # noqa: E402

from industrial_inspection.segeval import eval_size, ground_truth  # noqa: E402
from industrial_inspection.superadd_post import superadd_post  # noqa: E402


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum(pred & gt)
    d = 2 * tp + np.sum(pred & ~gt) + np.sum(~pred & gt)
    return float(2 * tp / d) if d else 0.0


def main() -> None:
    rows = []
    for cat, (model, rule, (close, fill, mn)) in CONFIG.items():
        z = np.load(f"results/maps/{model}__{cat}.npz")
        v = np.load(f"results/maps/{model}__{cat}__val.npz")["maps"].astype(np.float32)
        paths, y = z["paths"].astype(str), z["labels"]
        hw = eval_size(paths[0])
        maps = z["maps"].astype(np.float32)[y == 1]
        gt = np.stack([ground_truth(q, hw) for q in paths[y == 1]])
        t_ours, t_sa = threshold_of(rule, v), float(np.quantile(v, 0.95) * 1.421)
        variants = {
            "ours": np.stack([postprocess(m > t_ours, close, fill, mn) for m in maps]),
            "superadd": np.stack([superadd_post(m, t_sa) for m in maps]),
            "ours_thr+sa": np.stack([superadd_post(m, t_ours) for m in maps]),
        }
        for name, pred in variants.items():
            rows.append({"category": cat, "variant": name, "f1": f1(pred, gt) * 100})
        print(cat, {k: round(f1(p, gt) * 100, 1) for k, p in variants.items()}, flush=True)
    df = pd.DataFrame(rows)
    Path("results/seg").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/seg/compare_post.csv", index=False)
    table = df.pivot(index="category", columns="variant", values="f1")[["ours", "superadd", "ours_thr+sa"]]
    table.loc["mean"] = table.mean()
    print(table.round(1).to_string())

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(table))
    for i, (col, color) in enumerate(zip(table.columns, ["#4C78A8", "#E45756", "#54A24B"])):
        ax.bar(x + (i - 1) * 0.27, table[col], 0.27, label=col, color=color)
    ax.set_xticks(x, table.index, rotation=20)
    ax.set_ylabel("seg F1 (test_public, defect images, %)")
    ax.set_title("Post-processing: ours vs SuperADD's (same maps, thresholds from validation)")
    ax.legend()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/figures/compare_post.png", dpi=130)
    print("figure: docs/figures/compare_post.png")


if __name__ == "__main__":
    main()
