"""Results of the follow-up experiments from the 2026-11-03 review, with one figure each.

  ad1        PatchCore and MH-PatchCore on the original MVTec AD (is MH-PatchCore sound?)
  resolution PatchCore at 256 vs 392 vs Dinomaly-L at 392 (does input size explain Dinomaly's edge?)
  conformal  normal-only threshold from the good-only validation set instead of test goods
  tiles      tiled MH-PatchCore with the default vs a 9x memory bank, AUROC and AU-PRO

    uv run python scripts/fix_report.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.plots import (
    INK2, MODEL_SLOT, SERIES, SURFACE, _figure, _save, _style, model_color,
)

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
ALPHA = 0.10


def auroc(path: Path) -> float:
    d = pd.read_csv(path)
    return roc_auc_score(d.label, d.score)


def ad1() -> pd.DataFrame:
    rows = [{"model": f.stem.split("__")[0], "category": f.stem.split("__")[1], "auroc": auroc(f)}
            for f in sorted(Path("results/scores_ad1").glob("*.csv"))]
    d = pd.DataFrame(rows).pivot(index="category", columns="model", values="auroc")
    fig, (ax,) = _figure(1, 10, 4.6)
    d = d.sort_values("patchcore")
    y = np.arange(len(d))
    for off, m in ((-0.15, "patchcore"), (0.15, "mhpatchcore")):
        if m in d:
            ax.scatter(d[m], y + off, s=45, color=model_color(m), edgecolors=SURFACE, linewidths=1, label=f"{m} (mean {d[m].mean():.3f})", zorder=3)
    ax.axvline(0.99, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
    ax.set_yticks(y, d.index)
    _style(ax, "Image AUROC", "", "Original MVTec AD: PatchCore vs MH-PatchCore (dotted: published ~0.99)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    print("figure:", _save(fig, Path("results/figures/ad1_mh_check.png")))
    return d


def resolution() -> pd.DataFrame:
    m = pd.read_csv("results/map_metrics.csv")
    rows = []
    for c in CATS:
        row = {"category": c}
        for model in ("patchcore", "patchcore_392", "dinomaly_l"):
            f = Path(f"results/scores/{model}__{c}__seed0.csv")
            row[f"auroc_{model}"] = auroc(f) if f.exists() else np.nan
            hit = m[(m.model == model) & (m.category == c)]
            row[f"aupro_{model}"] = hit.aupro_005.iloc[0] if len(hit) else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    fig, axes = _figure(2, 13, 4.4)
    x = np.arange(len(CATS))
    styles = (("patchcore", "PatchCore 256", SERIES[6]), ("patchcore_392", "PatchCore 392", SERIES[0]), ("dinomaly_l", "Dinomaly-L 392 crop", SERIES[2]))
    for ax, metric, title in ((axes[0], "auroc", "Image AUROC"), (axes[1], "aupro", "AU-PRO@0.05 (model input resolution)")):
        for i, (model, label, color) in enumerate(styles):
            ax.bar(x + (i - 1) * 0.27, d[f"{metric}_{model}"], width=0.25, color=color, label=label, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, CATS, rotation=30, ha="right")
        _style(ax, "", metric.upper(), title)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.suptitle("Does input resolution explain Dinomaly-L's edge?", color="#0b0b0b", fontsize=12, x=0.01, ha="left")
    print("figure:", _save(fig, Path("results/figures/resolution_check.png")))
    return d


def conformal() -> pd.DataFrame:
    rows = []
    for model in ("patchcore", "dinomaly_l"):  # the research models; validation scores exist only for these
        for c in CATS:
            vf, tf = Path(f"results/scores_val/{model}__{c}__seed0.csv"), Path(f"results/scores/{model}__{c}__seed0.csv")
            if not (vf.exists() and tf.exists()):
                continue
            v, t = pd.read_csv(vf), pd.read_csv(tf)
            good = np.sort(v.score.to_numpy())
            thr = good[min(len(good), math.ceil((len(good) + 1) * (1 - ALPHA))) - 1]
            rows.append({"model": model, "category": c, "n_val": len(good),
                         "overkill": float((t.score[t.label == 0] >= thr).mean()),
                         "miss": float((t.score[t.label == 1] < thr).mean())})
    d = pd.DataFrame(rows)
    old = pd.read_csv("results/fewshot_report.csv")
    old = old[old.rule == "normal_only"].groupby("model")[["miss", "overkill"]].mean()
    summary = d.groupby("model")[["miss", "overkill"]].mean().join(old, rsuffix="_testgoods")
    fig, (ax,) = _figure(1, 9, 4.4)
    models = [m for m in MODEL_SLOT if m in summary.index]
    x = np.arange(len(models))
    ax.bar(x - 0.2, summary.loc[models, "overkill_testgoods"], 0.38, color=SERIES[6], label="threshold from test goods (old)", edgecolor=SURFACE)
    ax.bar(x + 0.2, summary.loc[models, "overkill"], 0.38, color=SERIES[0], label="threshold from validation goods (new)", edgecolor=SURFACE)
    ax.axhline(ALPHA, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("target 10 %", (len(models) - 0.5, ALPHA), xytext=(0, 4), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    for xi, mv in zip(x, summary.loc[models, "miss"]):
        ax.annotate(f"miss {mv:.0%}", (xi + 0.2, summary.loc[models[int(xi)], "overkill"]), xytext=(0, 4), textcoords="offset points", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, models)
    ax.set_ylim(0, max(summary.loc[models, ["overkill", "overkill_testgoods"]].max()) * 1.45)  # room for the legend
    _style(ax, "", "Overkill on test goods", "Normal-only (conformal) threshold, mean of 8 categories")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    print("figure:", _save(fig, Path("results/figures/conformal_validation.png")))
    return summary


def tiles() -> pd.DataFrame:
    tm = pd.read_csv("results/tile_study/map_metrics.csv") if Path("results/tile_study/map_metrics.csv").exists() else pd.DataFrame()
    um = pd.read_csv("results/map_metrics.csv")
    rows = []
    for c in ("can", "wallplugs"):
        row = {"category": c}
        for name, f in (("untiled", f"mhpatchcore__{c}__seed0.csv"), ("tiles_bank1000", f"mhpatchcore_tilemax__{c}__seed0.csv"),
                        ("tiles_bank9000", f"mhpatchcore_tilebigmax__{c}__seed0.csv")):
            p = Path("results/scores") / f
            row[f"auroc_{name}"] = auroc(p) if p.exists() else np.nan
        u = um[(um.model == "mhpatchcore") & (um.category == c)]
        row["aupro_untiled"] = u.aupro_005.iloc[0] if len(u) else np.nan
        for name, model in (("tiles_bank1000", "mhpatchcore_tile"), ("tiles_bank9000", "mhpatchcore_tilebig")):
            h = tm[(tm.model == model) & (tm.category == c)] if len(tm) else tm
            row[f"aupro_{name}"] = h.aupro_005.iloc[0] if len(h) else np.nan
            row[f"tile_auroc_top5_{name}"] = h.auroc_top5.iloc[0] if len(h) else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    fig, axes = _figure(2, 12, 4.2)
    names = (("untiled", "untiled", SERIES[1]), ("tiles_bank1000", "tiles, bank 1000", SERIES[3]), ("tiles_bank9000", "tiles, bank 9000", SERIES[0]))
    x = np.arange(len(d))
    for ax, metric, title in ((axes[0], "auroc", "Image AUROC (max over tiles)"), (axes[1], "aupro", "AU-PRO@0.05")):
        for i, (key, label, color) in enumerate(names):
            ax.bar(x + (i - 1) * 0.27, d[f"{metric}_{key}"], 0.25, color=color, label=label, edgecolor=SURFACE)
        ax.set_xticks(x, d.category)
        _style(ax, "", metric.upper(), title)
    axes[0].axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].set_ylim(0.3, 0.8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.suptitle("Tile experiment redone with a memory bank that grows with the number of patches", color="#0b0b0b", fontsize=12, x=0.01, ha="left")
    print("figure:", _save(fig, Path("results/figures/tile_recheck.png")))
    return d


def main() -> None:
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        # tiles() is kept for the record but not run: the tile rerun was dropped when MH-PatchCore left the research.
        for name, fn in (("ad1", ad1), ("resolution", resolution), ("conformal", conformal)):
            try:
                out = fn()
                print(f"\n== {name}\n{out.to_string()}")
                if name == "ad1":
                    print("means:", out.mean().round(4).to_dict())
            except Exception as e:  # report what is not ready yet instead of stopping
                print(f"\n== {name}: not ready ({e!r})")


if __name__ == "__main__":
    main()
