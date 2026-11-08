"""Figure: where seeded shape completion (industrial_inspection.complete) extends the final wallplugs prediction on the
private test photos (no ground truth -- visual sanity check). Cyan = final prediction, yellow = completion.
Writes docs/figures/wp_complete_private.png
    .venv/Scripts/python.exe scripts/wp_complete_private.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.complete import plug_cache_path, seeded_completion  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402

zv = np.load("results/maps/sub_wp__wallplugs__val.npz")
rows = []
for split in ("test_private", "test_private_mixed"):
    z = np.load(f"results/maps/sub_wp__wallplugs__{split}.npz")
    paths = z["paths"].astype(str)
    _, pred, _ = wallplugs_masks(z["maps"].astype(np.float32), paths, zv["maps"].astype(np.float32), zv["paths"].astype(str))
    for i, q in enumerate(paths):
        add = seeded_completion(q, plug_cache_path(q), pred[i])
        if add.any():
            rows.append((q, pred[i], add))
n = len(rows)
cols = min(5, n)
fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 4 * ((n + cols - 1) // cols)))
ax = np.atleast_1d(ax).ravel()
for a, (q, p, add) in zip(ax, rows):
    im = np.asarray(Image.open(q).convert("RGB").resize(p.shape[::-1], Image.BILINEAR))
    ys, xs = np.nonzero(add | p)
    ys2, xs2 = np.nonzero(add)
    cy, cx = int(ys2.mean()), int(xs2.mean())
    h = 70
    a.imshow(im)
    a.contour(p, [0.5], colors="c", linewidths=1.5)
    a.contour(add, [0.5], colors="y", linewidths=1.5)
    a.set_xlim(max(0, cx - h), min(im.shape[1], cx + h))
    a.set_ylim(min(im.shape[0], cy + h), max(0, cy - h))
    a.set_title(f"{Path(q).parent.name}/{Path(q).stem}", fontsize=9)
for a in ax:
    a.axis("off")
fig.suptitle("private test: seeded shape completion (cyan = final model, yellow = added)", fontsize=11)
fig.tight_layout()
fig.savefig("docs/figures/wp_complete_private.png", dpi=75, facecolor="white")
print("saved docs/figures/wp_complete_private.png", n)
