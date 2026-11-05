"""Why cross-condition consistency works (or not): for one good part (the one the regular photo
alone flags most) and one defective part (the most anomalous in every photo) of a category, every
photo, its anomaly map, and the pixel-min map over all photos, on one colour scale.

    uv run python scripts/xcond_look.py --model dinomaly_l --category can
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_inspection.plots import INK, INK2, SURFACE, _save, plt
from industrial_inspection.q2 import parts_and_tags

# What each model looks at (resize, then centre crop), so the photo matches its map.
VIEW = {"dinomaly_l": (448, 392), "mhpatchcore": (256, 224), "patchcore": (256, 256), "efficientad": (256, 256)}


def view(path: str, model: str) -> np.ndarray:
    size, crop = VIEW[model]
    im = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
    o = (size - crop) // 2
    return im[o:o + crop, o:o + crop]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinomaly_l")
    p.add_argument("--category", default="can")
    args = p.parse_args()

    z = np.load(f"results/maps/{args.model}__{args.category}.npz")
    maps, y, paths = z["maps"].astype(np.float32), z["labels"], z["paths"].astype(str)
    parts, tags = parts_and_tags(paths)
    first = {}
    for i in np.argsort(paths):
        first.setdefault((parts[i], str(tags[i])), i)
    conds = sorted({t for _, t in first}, key=lambda c: (c != "regular", c))
    good = sorted({q for (q, _), i in first.items() if y[i] == 0})
    bad = sorted({q for (q, _), i in first.items() if y[i] == 1})
    reg_score = {q: maps[first[(q, "regular")]].max() for q in good + bad}
    pick_good = max(good, key=reg_score.get)  # the good part one photo flags most
    min_score = {q: np.min([maps[first[(q, c)]] for c in conds], axis=0).max() for q in bad}
    pick_bad = max(bad, key=min_score.get)  # the defect that stays most anomalous in every photo
    vmin, vmax = np.quantile(maps, [0.5, 0.999])

    fig, axes = plt.subplots(4, len(conds) + 1, figsize=(2.3 * (len(conds) + 1), 9.6), facecolor=SURFACE, squeeze=False)
    for r, (part, kind) in enumerate(((pick_good, "good part"), (pick_bad, "defective part"))):
        stack = []
        for c, cond in enumerate(conds):
            i = first[(part, cond)]
            stack.append(maps[i])
            axes[2 * r, c].imshow(view(paths[i], args.model))
            axes[2 * r + 1, c].imshow(maps[i], cmap="magma", vmin=vmin, vmax=vmax)
            axes[2 * r + 1, c].set_xlabel(f"max {maps[i].max():.3g}", color=INK2, fontsize=8)
            if r == 0:
                axes[0, c].set_title(cond, color=INK2, fontsize=9)
        mn = np.min(stack, axis=0)
        axes[2 * r + 1, -1].imshow(mn, cmap="magma", vmin=vmin, vmax=vmax)
        axes[2 * r + 1, -1].set_xlabel(f"max {mn.max():.3g}", color=INK2, fontsize=8)
        axes[2 * r, -1].axis("off")
        axes[2 * r, 0].set_ylabel(f"{kind}\n{part}", color=INK2, fontsize=9)
        axes[2 * r + 1, 0].set_ylabel("anomaly map", color=INK2, fontsize=9)
    axes[0, -1].set_title("pixel-min\nover all photos", color=INK2, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Cross-condition consistency, {args.model} on {args.category}: the good part the regular photo flags most, "
                 "and the defective part with the strongest consistent anomaly", color=INK, fontsize=11, x=0.01, ha="left")
    print("figure:", _save(fig, Path("results/figures") / f"xcond_look__{args.category}__{args.model}.png"))


if __name__ == "__main__":
    main()
