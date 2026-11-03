"""Figures for training curves and test-set performance.

Every training run draws its learning curves and every test pass draws its performance
figure; score.py and evaluate.py call these unconditionally. Colours are the dataviz skill's
reference palette (light surface); a model keeps the same colour in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from industrial_inspection.metrics import calibrate_threshold, evaluate_at_threshold  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Colour follows the entity, never its rank: a model keeps its slot whatever else is on the figure.
MODEL_SLOT = {"efficientad": 0, "mhpatchcore": 1, "dinomaly_l": 2, "patchcore": 6, "dinomaly": 4}
BUDGETS = (0.0, 0.01, 0.05)


def model_color(model: str) -> str:
    return SERIES[MODEL_SLOT.get(model, 5 + len(model) % 3)]


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def _figure(ncols: int, width: float, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def _roc_axis(ax, curves: list[tuple[str, str, np.ndarray, np.ndarray]]) -> None:
    """curves: (label, colour, scores, labels)."""
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    for label, color, s, y in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  AUROC {roc_auc_score(y, s):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _style(ax, "Overkill rate (good parts rejected)", "Detection rate (defects caught)", "ROC")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)


def _tradeoff_axis(ax, curves: list[tuple[str, str, np.ndarray, np.ndarray]]) -> None:
    """Overkill against miss rate, with the miss-budget operating points marked."""
    for label, color, s, y in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(1 - tpr, fpr, color=color, linewidth=2, label=label)
        for budget in BUDGETS:
            p = evaluate_at_threshold(s, y, calibrate_threshold(s, y, budget))
            ax.plot(p.miss_rate, p.overkill_rate, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            if len(curves) == 1:
                ax.annotate(f"{p.overkill_rate:.0%}", (p.miss_rate, p.overkill_rate), xytext=(8, -14),
                            textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlim(-0.005, 0.2)
    ax.set_ylim(0, 1.02)
    _style(ax, "Miss rate (defects accepted)", "Overkill rate", "Operating curve (dots = 0 / 1 / 5 % budgets)")
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)


def test_figure(scores_csv: Path, out: Path | None = None) -> Path:
    """ROC, operating curve and score distributions for one (model, category) score file."""
    model, category, _ = scores_csv.stem.split("__")
    df = pd.read_csv(scores_csv)
    s, y = df.score.to_numpy(), df.label.to_numpy()
    color = model_color(model)
    fig, (a1, a2, a3) = _figure(3, 15.5)
    _roc_axis(a1, [(model, color, s, y)])
    _tradeoff_axis(a2, [(model, color, s, y)])

    bins = np.histogram_bin_edges(s, bins=30)
    # Defective first, good on top: there are far fewer good parts and they would be buried.
    a3.hist(s[y == 1], bins=bins, color=SERIES[1], alpha=0.6, label="defective", edgecolor=SURFACE, linewidth=1)
    a3.hist(s[y == 0], bins=bins, color=SERIES[0], alpha=0.9, label="good", edgecolor=SURFACE, linewidth=1)
    thr = calibrate_threshold(s, y, 0.0)
    a3.axvline(thr, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    a3.annotate("threshold @ 0 % miss", (thr, a3.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                color=INK2, fontsize=9)
    _style(a3, "Anomaly score (raw)", "Images", "Score distribution")
    a3.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    fig.suptitle(f"{model} - {category} - test_public ({int((y == 0).sum())} good, {int(y.sum())} defective)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / f"{scores_csv.stem}__test.png")


def comparison_figure(scores_csvs: list[Path], category: str, out: Path | None = None) -> Path:
    """ROC and operating curves of several models on one category."""
    curves = []
    for csv in sorted(scores_csvs):
        model = csv.stem.split("__")[0]
        df = pd.read_csv(csv)
        curves.append((model, model_color(model), df.score.to_numpy(), df.label.to_numpy()))
    fig, (a1, a2) = _figure(2, 11.5)
    _roc_axis(a1, curves)
    _tradeoff_axis(a2, curves)
    fig.suptitle(f"{category} - model comparison on test_public", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / f"compare__{category}.png")


def training_figure(curve_prefix: Path, out: Path | None = None) -> Path | None:
    """Loss and periodic test AUROC against training step, from CurveLogger's CSVs."""
    loss_csv = curve_prefix.with_name(curve_prefix.name + "__loss.csv")
    auroc_csv = curve_prefix.with_name(curve_prefix.name + "__auroc.csv")
    if not loss_csv.exists():
        return None
    model, category, _ = curve_prefix.name.split("__")
    color = model_color(model)
    loss = pd.read_csv(loss_csv)
    col = next((c for c in loss.columns if c.startswith("train_loss")), None) or next(
        c for c in loss.columns if c != "step"
    )
    fig, (a1, a2) = _figure(2, 11.5)
    window = max(1, len(loss) // 25)
    a1.plot(loss.step, loss[col], color=color, linewidth=1, alpha=0.3)
    a1.plot(loss.step, loss[col].rolling(window, min_periods=1).mean(), color=color, linewidth=2)
    _style(a1, "Training step", "Training loss", "Loss  (faint: logged value, line: moving average)")

    if auroc_csv.exists() and auroc_csv.stat().st_size > 2 and not pd.read_csv(auroc_csv).dropna().empty:
        au = pd.read_csv(auroc_csv).dropna()
        a2.plot(au.step, au.test_auroc, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for _, r in au.iloc[[0, -1]].iterrows():
            a2.annotate(f"{r.test_auroc:.3f}", (r.step, r.test_auroc), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=INK2, fontsize=9)
        lo, hi = au.test_auroc.min(), au.test_auroc.max()
        pad = max(0.02, (hi - lo) * 0.3)
        a2.set_ylim(lo - pad, min(1.03, hi + pad * 1.5))  # headroom so the end labels clear the title
    _style(a2, "Training step", "Test AUROC", "Test AUROC during training (plot only)")
    fig.suptitle(f"{model} - {category} - training", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / f"{curve_prefix.name}__training.png")


def tag_figure(report: pd.DataFrame, category: str, miss_budget: float = 0.0, out: Path | None = None) -> Path:
    """Per-condition AUROC, miss/overkill under a regular-calibrated threshold, and good-score drift."""
    tags = list(dict.fromkeys(report.tag))
    x = np.arange(len(tags))
    fig, axes = _figure(3, 16.5, 4.6)
    panels = [
        ("auroc", "AUROC", "AUROC per condition", (0.0, 1.02)),
        ("overkill", "Overkill rate", f"Overkill, threshold set on regular ({miss_budget:.0%} miss)", (0.0, 1.05)),
        ("good_score_vs_regular", "Median good score / regular", "Normal parts' score vs regular", None),
    ]
    for ax, (col, ylabel, title, ylim) in zip(axes, panels):
        for model, rep in report.groupby("model", sort=False):
            rep = rep.set_index("tag").reindex(tags)
            ax.plot(x, rep[col], color=model_color(model), linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=model)
        if col == "good_score_vs_regular":
            ax.axhline(1.0, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x, tags, rotation=30, ha="right")
        _style(ax, "", ylabel, title)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    first = report.iloc[0]
    fig.suptitle(
        f"{category} - by capture condition  (per condition: {int(first.n_good)} good, {int(first.n_defective)} defective)",
        color=INK, fontsize=12, x=0.01, ha="left",
    )
    return _save(fig, out or Path("results/figures") / f"tags__{category}.png")


def data_profile_figure(table: pd.DataFrame, defects: pd.DataFrame, out: Path | None = None) -> Path:
    """Defect size per category (at model resolution) and brightness per capture condition."""
    order = table.sort_values("defect_side_px_median").category.tolist()
    fig, (a1, a2) = _figure(2, 15, 4.8)

    rng = np.random.default_rng(0)
    for i, cat in enumerate(order):
        v = defects.loc[defects.category == cat, "side_px_at_256"].clip(lower=0.1)
        a1.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=12, color=SERIES[0], alpha=0.45, linewidths=0)
        a1.plot([v.median()] * 2, [i - 0.32, i + 0.32], color=INK, linewidth=2)
    a1.axvline(8, color=SERIES[7], linewidth=1.2, linestyle=(0, (4, 3)))
    a1.annotate("8 px = one PatchCore feature cell", (8, len(order) - 0.4), xytext=(5, 0), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
    a1.set_xscale("log")
    a1.set_yticks(range(len(order)), order)
    a1.set_ylim(-0.6, len(order) - 0.2)
    _style(a1, "Defect size: side of an equal-area square on a 256 px input (log)", "",
           "Defect size per test image (bar = median)")

    cond_cols = [c for c in table.columns if c.startswith("bright_")]
    conds = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4"]
    conds = [c for c in conds if f"bright_{c}" in cond_cols]
    x = np.arange(len(conds))
    for cat in table.category:
        r = table.set_index("category").loc[cat]
        a2.plot(x, [r.get(f"bright_{c}", np.nan) for c in conds], color=SERIES[CATEGORY_SLOT[cat]], linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=cat)
    a2.axhline(1.0, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    a2.set_xticks(x, conds, rotation=30, ha="right")
    _style(a2, "", "Mean brightness / regular", "Brightness of good test images per condition")
    a2.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left", labelcolor=INK2)
    fig.suptitle("MVTec AD 2 - data profile (test_public)", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "data_profile.png")


# Fixed category -> slot, so a category keeps its colour across figures.
CATEGORY_SLOT = {c: i for i, c in enumerate(
    ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"])}


def summary_figure(report: pd.DataFrame, out: Path | None = None) -> Path:
    """AUROC and overkill at a 0 % miss budget for every (category, model), one row per category."""
    r0 = report[report.miss_budget == 0.0]
    order = r0.groupby("category").auroc.mean().sort_values().index.tolist()
    models = [m for m in MODEL_SLOT if m in set(r0.model)]
    fig, (a1, a2) = _figure(2, 14, 0.55 * len(order) + 2.2)
    offsets = np.linspace(-0.22, 0.22, len(models)) if len(models) > 1 else [0.0]
    for ax, col, xlabel, title, ref in (
        (a1, "auroc", "AUROC", "AUROC per category", 0.5),
        (a2, "heldout_overkill", "Overkill rate at 0 % miss (held-out)", "Overkill per category", None),
    ):
        for off, model in zip(offsets, models):
            rep = r0[r0.model == model].set_index("category").reindex(order)
            ax.scatter(rep[col], np.arange(len(order)) + off, s=60, color=model_color(model), edgecolors=SURFACE,
                       linewidths=1.2, label=model, zorder=3)
        if ref is not None:
            ax.axvline(ref, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate("chance", (ref, len(order) - 0.5), xytext=(4, 0), textcoords="offset points", color=MUTED,
                        fontsize=9, va="center")
        ax.set_yticks(range(len(order)), order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        _style(ax, xlabel, "", title)
    a1.set_xlim(0.35, 1.0)
    a2.set_xlim(0.0, 1.05)
    a1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    fig.suptitle("MVTec AD 2 test_public - all categories", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "summary__all_categories.png")


RULE_STYLE = {  # fixed colour per threshold rule
    "min_defect": (SERIES[0], "min defect score"),
    "margin": (SERIES[1], "min defect - 0.5 IQR(good)"),
    "normal_only": (SERIES[2], "normal only (conformal, 10%)"),
}


def fewshot_figure(overall: pd.DataFrame, out: Path | None = None) -> Path:
    """Held-out miss and overkill against the number of known defective parts, per threshold rule."""
    fig, (a1, a2) = _figure(2, 13, 4.4)
    ks = sorted(overall.k.unique())
    for ax, col, p95, ylabel, title in (
        (a1, "miss", "miss_p95", "Miss rate on unseen parts", "Miss (line = mean, band = p95)"),
        (a2, "overkill", "overkill_p95", "Overkill rate on unseen parts", "Overkill (line = mean, band = p95)"),
    ):
        for rule, (color, label) in RULE_STYLE.items():
            d = overall[overall.rule == rule].sort_values("k")
            if d.empty:
                continue
            if rule == "normal_only":  # no defects used: same at every k
                d = pd.concat([d.assign(k=k) for k in ks])
            ax.plot(d.k, d[col], color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=label)
            ax.fill_between(d.k, d[col], d[p95], color=color, alpha=0.12, linewidth=0)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.02)
        _style(ax, "Known defective parts used for calibration (k)", ylabel, title)
    a1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    fig.suptitle("Few-shot threshold calibration - mean over all models x 8 categories, 200 part-level draws each",
                 color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "fewshot_calibration.png")


def condition_threshold_figure(summary: pd.DataFrame, out: Path | None = None) -> Path:
    """Held-out miss and overkill (0 % miss budget) per threshold rule, per model."""
    rules = ["global", "regular_only", "per_condition", "normalized"]
    models = [m for m in MODEL_SLOT if m in set(summary.model)]
    fig, axes = _figure(2, 13, 4.2)
    x = np.arange(len(rules))
    width = 0.8 / len(models)
    for ax, col, ylabel in ((axes[0], "miss", "Miss rate on unseen parts"), (axes[1], "overkill", "Overkill rate on unseen parts")):
        for i, model in enumerate(models):
            v = summary[summary.model == model].groupby("rule")[col].mean().reindex(rules)
            ax.bar(x + (i - (len(models) - 1) / 2) * width, v, width=width - 0.02, color=model_color(model),
                   label=model, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, rules)
        _style(ax, "", ylabel, f"{col.capitalize()} at a 0 % miss budget (mean of 8 categories)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("Threshold rules with and without knowing the capture condition (part-level split)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "condition_thresholds.png")


def split_comparison_figure(part: pd.DataFrame, image: pd.DataFrame, out: Path | None = None) -> Path:
    """Held-out miss and overkill at a 0 % miss budget: image-level split vs part-level split."""
    k = ["model", "category"]
    m = part[part.miss_budget == 0.0].merge(image[image.miss_budget == 0.0], on=k, suffixes=("_part", "_image"))
    m = m[m.model.isin(["patchcore", "mhpatchcore", "dinomaly_l"])].sort_values(["heldout_miss_part"])
    labels = (m.model + " / " + m.category).tolist()
    y = np.arange(len(m))
    fig, axes = _figure(2, 14, 0.28 * len(m) + 2.0)
    for ax, col, title in ((axes[0], "heldout_miss", "Miss at a 0 % miss budget"),
                           (axes[1], "heldout_overkill", "Overkill at a 0 % miss budget")):
        ax.hlines(y, m[f"{col}_image"], m[f"{col}_part"], color=AXIS, linewidth=1.5)
        ax.scatter(m[f"{col}_image"], y, s=36, color=SERIES[0], label="split by image (old)", zorder=3,
                   edgecolors=SURFACE, linewidths=1)
        ax.scatter(m[f"{col}_part"], y, s=36, color=SERIES[1], label="split by part", zorder=3,
                   edgecolors=SURFACE, linewidths=1)
        ax.set_yticks(y, labels, fontsize=8)
        _style(ax, "Held-out rate (mean of 20 splits)", "", title)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    fig.suptitle("Splitting test_public by image puts the same part on both sides and hides misses",
                 color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "split_comparison.png")


def latency_figure(df: pd.DataFrame, out: Path | None = None) -> Path:
    """Median per-image latency (bar) and p95 (whisker) per model and runtime."""
    models = [m for m in MODEL_SLOT if m in set(df.model)]
    runtimes = list(dict.fromkeys(df.runtime))
    fig, (ax,) = _figure(1, 9, 4.2)
    x = np.arange(len(models))
    width = 0.8 / len(runtimes)
    for i, rt in enumerate(runtimes):
        d = df[df.runtime == rt].set_index("model").reindex(models)
        pos = x + (i - (len(runtimes) - 1) / 2) * width
        ax.bar(pos, d.median_ms, width=width - 0.03, color=SERIES[i], label=rt, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(pos, d.median_ms, yerr=[np.zeros(len(d)), d.p95_ms - d.median_ms], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=3)
        for px, v in zip(pos, d.median_ms):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (px, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            color=INK2, fontsize=8)
    ax.set_xticks(x, models)
    _style(ax, "", "Latency per image (ms, batch 1)", "Median latency (whisker = p95), pre-processing included")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    return _save(fig, out or Path("results/figures") / "latency.png")


def ad1_check_figure(per_cat: pd.DataFrame, published_mean: float, out: Path | None = None) -> Path:
    """Image AUROC of PatchCore per original MVTec AD category, against the published average."""
    d = per_cat.sort_values("auroc")
    fig, (ax,) = _figure(1, 10, 4.4)
    y = np.arange(len(d))
    ax.hlines(y, 0.85, d.auroc, color=AXIS, linewidth=1.5)
    ax.scatter(d.auroc, y, s=60, color=model_color("patchcore"), edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.axvline(d.auroc.mean(), color=model_color("patchcore"), linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axvline(published_mean, color=INK2, linewidth=1.2, linestyle=(0, (1, 2)))
    ax.annotate(f"ours mean {d.auroc.mean():.3f}", (d.auroc.mean(), len(d) - 0.6), xytext=(-4, 0),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.annotate(f"published ~{published_mean:.2f}", (published_mean, 0.4), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_yticks(y, d.category)
    ax.set_xlim(0.85, 1.005)
    _style(ax, "Image AUROC", "", "PatchCore on the original MVTec AD (pipeline sanity check)")
    return _save(fig, out or Path("results/figures") / "ad1_check.png")


def map_study_figure(agg: pd.DataFrame, aupro: pd.DataFrame, overkill: pd.DataFrame, out: Path | None = None) -> Path:
    """Image-score aggregations (mean AUROC and held-out overkill) and AU-PRO@0.05 per model."""
    models = [m for m in MODEL_SLOT if m in set(agg.model)]
    names = ["model score", "max", "top1", "top5", "mean"]
    fig, axes = _figure(3, 17, 4.6)
    x = np.arange(len(names))
    width = 0.8 / len(models)
    for ax, df, ylabel, title in ((axes[0], agg, "Mean AUROC over 8 categories", "Image score from the anomaly map"),
                                  (axes[1], overkill, "Overkill at 0 % miss (part split)", "Overkill on unseen parts")):
        for i, m in enumerate(models):
            v = df[df.model == m].set_index("aggregation").reindex(names).value
            ax.bar(x + (i - (len(models) - 1) / 2) * width, v, width=width - 0.02, color=model_color(m), label=m,
                   edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, names)
        _style(ax, "", ylabel, title)
    axes[0].set_ylim(0.4, 0.8)
    axes[1].set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2, ncol=2)
    cats = sorted(aupro.category.unique())
    yy = np.arange(len(cats))
    offs = np.linspace(-0.25, 0.25, len(models))
    for off, m in zip(offs, models):
        v = aupro[aupro.model == m].set_index("category").reindex(cats).aupro_005
        axes[2].scatter(v, yy + off, s=40, color=model_color(m), edgecolors=SURFACE, linewidths=1, zorder=3)
    axes[2].set_yticks(yy, cats)
    _style(axes[2], "AU-PRO up to 5 % FPR (model input resolution)", "", "Defect localisation")
    fig.suptitle("Anomaly-map study on MVTec AD 2 test_public", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out or Path("results/figures") / "map_study.png")
