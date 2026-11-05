"""Show test results next to the ground truth: for a few defective test_public images, the photo, the
ground-truth outline, the anomaly map with that outline, and the binarised prediction at the
official threshold compared with the ground truth (green: correctly flagged, red: false alarm,
blue: missed). Everything at seg_eval.py's evaluation size (1/4 of the original resolution).

Writes one PNG per image and a grid of all of them to results/figures/qualitative/<model>/.

    uv run python scripts/seg_examples.py --model hires_dinov2l
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from industrial_inspection.plots import INK2, SURFACE, _save, plt
from industrial_inspection.segeval import eval_size, f1_at, full_frame, ground_truth, view_of

DEFAULT_CATS = ["walnuts", "sheet_metal", "rice", "vial", "fabric"]


def pick(paths: np.ndarray, labels: np.ndarray, condition: str, index: int) -> int:
    """The index-th defective photo taken under `condition` (sorted by file name)."""
    cand = [i for i in np.argsort(paths) if labels[i] == 1 and Path(paths[i]).name.split("_", 1)[1].startswith(condition)]
    return cand[min(index, len(cand) - 1)]


def panels(axes, model: str, category: str, i: int, z, thr: float) -> float:
    path = str(z["paths"][i])
    hw = eval_size(path)
    amap = full_frame(z["maps"][i:i + 1], hw, view_of(model))[0]
    gt = ground_truth(path, hw)
    img = np.asarray(Image.open(path).convert("RGB").resize(hw[::-1], Image.BILINEAR))
    pred = amap > thr
    lo, hi = np.quantile(full_frame(z["maps"][z["labels"] == 0][:20], hw, view_of(model)), [0.5, 0.999])

    axes[0].imshow(img)
    axes[1].imshow(img)
    axes[1].contour(gt, levels=[0.5], colors="#e34948", linewidths=1.2)
    axes[2].imshow(amap, cmap="magma", vmin=lo, vmax=max(hi, amap.max()))
    axes[2].contour(gt, levels=[0.5], colors="#2a78d6", linewidths=1.0)
    overlay = (img * 0.45).astype(np.uint8)
    overlay[pred & gt] = (27, 175, 122)     # correctly flagged
    overlay[pred & ~gt] = (227, 73, 72)     # false alarm
    overlay[~pred & gt] = (42, 120, 214)    # missed
    axes[3].imshow(overlay)
    f1 = f1_at(amap.ravel(), gt.ravel(), thr)
    axes[0].set_ylabel(f"{category}\n{Path(path).name}", color=INK2, fontsize=8)
    axes[3].set_xlabel(f"F1 on this image {f1:.2f}", color=INK2, fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return f1


TITLES = ["photo", "ground truth (red outline)", "anomaly map + ground truth (blue)",
          "prediction vs truth (green hit, red false alarm, blue missed)"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hires_dinov2l")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATS)
    p.add_argument("--condition", default="regular")
    p.add_argument("--index", type=int, default=0, help="which defective photo of that condition (sorted by name)")
    p.add_argument("--out", type=Path, default=Path("results/figures/qualitative"))
    args = p.parse_args()

    ev = pd.read_csv("results/seg/seg_eval.csv").set_index(["model", "category"])
    outdir = args.out / args.model
    rows = []
    for cat in args.categories:
        z = np.load(f"results/maps/{args.model}__{cat}.npz")
        i = pick(z["paths"].astype(str), z["labels"], args.condition, args.index)
        thr = float(ev.loc[(args.model, cat), "thr_official"])
        rows.append((cat, i, z, thr))
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=SURFACE)
        f1 = panels(axes, args.model, cat, i, z, thr)
        for ax, t in zip(axes, [f"{args.model} on {cat}, threshold {thr:.3g}\n{TITLES[0]}", *TITLES[1:]]):
            ax.set_title(t, color=INK2, fontsize=9)
        print("figure:", _save(fig, outdir / f"{cat}__{Path(str(z['paths'][i])).stem}.png"), f"F1 {f1:.2f}")

    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.3 * len(rows)), facecolor=SURFACE, squeeze=False)
    for r, (cat, i, z, thr) in enumerate(rows):
        panels(axes[r], args.model, cat, i, z, thr)
    for ax, t in zip(axes[0], [f"{args.model}, {args.condition} photos,\nofficial threshold: {TITLES[0]}", *TITLES[1:]]):
        ax.set_title(t, color=INK2, fontsize=9)
    print("figure:", _save(fig, outdir / "examples_grid.png"))


if __name__ == "__main__":
    main()
