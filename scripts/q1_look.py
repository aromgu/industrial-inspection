"""Q1: look at the defects. For a few defective images per size group, a full-resolution crop around
the defect, the same crop with the mask outline, and the model's anomaly map over the whole image
(common grid from export_maps.py --eval-side) with the defect box marked. Also one good image's map,
to see what the model finds anomalous on normal parts.

    uv run python scripts/q1_look.py --category can --model patchcore_392_c01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from industrial_inspection.plots import INK, INK2, SURFACE, _save, plt


def mask_of(path: Path) -> Path:
    return path.parent.parent / "ground_truth" / path.parent.name / (path.stem + "_mask.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="can")
    p.add_argument("--model", default="patchcore_392_c01")
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--crop", type=int, default=160, help="crop side in full-resolution pixels")
    args = p.parse_args()

    z = np.load(f"results/maps/{args.model}_s{args.side}__{args.category}.npz")
    paths, y, maps = z["paths"].astype(str), z["labels"], z["maps"].astype(np.float32)
    vmin, vmax = np.quantile(maps[y == 0], [0.5, 0.999])  # same colour scale for every map
    bad = [i for i in np.flatnonzero(y == 1) if "regular" in Path(paths[i]).name]
    area = np.array([(np.array(Image.open(mask_of(Path(paths[i]))).convert("L")) > 0).mean() for i in bad])
    order = np.argsort(area)
    picks = [bad[order[j]] for j in np.linspace(0, len(order) - 1, 3 * args.per_group).round().astype(int)]
    good = [i for i in np.flatnonzero(y == 0) if "regular" in Path(paths[i]).name][:1]

    rows = len(picks) + len(good)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2.6 * rows), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 2.2]})
    for r, i in enumerate(picks + good):
        path = Path(paths[i])
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        m = np.array(Image.open(mask_of(path)).convert("L")) > 0 if y[i] else np.zeros((h, w), bool)
        cy, cx = (np.argwhere(m).mean(axis=0).astype(int) if m.any() else (h // 2, w // 2))
        c = args.crop // 2
        y0, x0 = np.clip(cy - c, 0, h - 2 * c), np.clip(cx - c, 0, w - 2 * c)
        crop, mcrop = img[y0:y0 + 2 * c, x0:x0 + 2 * c], m[y0:y0 + 2 * c, x0:x0 + 2 * c]
        axes[r, 0].imshow(crop)
        axes[r, 1].imshow(crop)
        if mcrop.any():
            axes[r, 1].contour(mcrop, levels=[0.5], colors="#e34948", linewidths=1.2)
        axes[r, 2].imshow(maps[i], cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
        sy, sx = maps[i].shape[0] / h, maps[i].shape[1] / w
        axes[r, 2].add_patch(mpatches.Rectangle((x0 * sx, y0 * sy), 2 * c * sx, 2 * c * sy, fill=False,
                                                edgecolor="#2a78d6", linewidth=1.5))
        label = f"defect {m.mean():.4%} of image" if y[i] else "good part"
        axes[r, 0].set_ylabel(f"{path.name}\n{label}", color=INK2, fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, t in zip(axes[0], (f"{args.crop}px full-res crop", "mask outline", f"{args.model} anomaly map (box = crop)")):
        ax.set_title(t, color=INK2, fontsize=9)
    fig.suptitle(f"Q1 - {args.category} defects and what the model sees (same colour scale for all maps)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    print("figure:", _save(fig, Path("results/figures") / f"q1_look__{args.category}__{args.model}.png"))


if __name__ == "__main__":
    main()
