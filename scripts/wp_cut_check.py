"""wallplugs P5 check: does a map light up the REAL cut-off defects (parts 000, 001; 6 lighting conditions each)?
Per photo: max of the (sigma-smoothed) map inside the GT, the photo's own p99 / max outside the GT (dilated 8 px),
the GT peak's rank among the photo's pixels, and whether the GT peak beats the q90 of validation per-photo maxima
(the level a photo-independent threshold would need). Also saves a figure: photo crop, GT outline, map crop.
    .venv/Scripts/python.exe scripts/wp_cut_check.py synth_cut6 [--sigma 2]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("model")
a.add_argument("--sigma", type=float, default=2.0)
a.add_argument("--parts", nargs="+", default=["000", "001"])
args = a.parse_args()
z, zv = np.load(f"results/maps/{args.model}__wallplugs.npz"), np.load(f"results/maps/{args.model}__wallplugs__val.npz")
paths, y = z["paths"].astype(str), z["labels"]
hw = z["maps"].shape[1:]
gt = gt_of(paths, y, hw)
sm = lambda m: ndimage.gaussian_filter(m.astype(np.float32), args.sigma)
vq90 = np.quantile([sm(m).max() for m in zv["maps"]], 0.9)
print(f"== {args.model}: val per-photo max q90 = {vq90:.2f}")
print("  photo                 gt_max  out_p99  out_max  rank%   >val_q90")
rows = []
for p in args.parts:
    for i in np.flatnonzero(y == 1):
        if not Path(paths[i]).name.startswith(p + "_"):
            continue
        m, g = sm(z["maps"][i]), gt[i]
        out = ~ndimage.binary_dilation(g, iterations=8)
        gm = m[g].max()
        rank = (m[out] < gm).mean() * 100
        print(f"  {Path(paths[i]).stem:20s}  {gm:6.2f}  {np.percentile(m[out], 99):7.2f}  {m[out].max():7.2f}  {rank:5.1f}   {gm > vq90}")
        rows.append((i, gm > m[out].max()))
k = min(6, len(rows))
fig, ax = plt.subplots(2, k, figsize=(3.2 * k, 6.4))
for c, (i, top) in enumerate(rows[:k]):
    g = gt[i]
    ys, xs = np.nonzero(g)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(0, cy - 60), max(0, cx - 60)
    im = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
    m = sm(z["maps"][i])
    ax[0, c].imshow(im[y0:y0 + 120, x0:x0 + 120])
    ax[0, c].contour(g[y0:y0 + 120, x0:x0 + 120], [0.5], colors="r", linewidths=0.8)
    ax[0, c].set_title(Path(paths[i]).stem, fontsize=9)
    ax[1, c].imshow(m[y0:y0 + 120, x0:x0 + 120], cmap="jet", vmin=np.percentile(m, 50), vmax=max(m.max(), vq90))
    ax[1, c].contour(g[y0:y0 + 120, x0:x0 + 120], [0.5], colors="w", linewidths=0.8)
    ax[1, c].set_title("GT peak is photo max" if top else "not photo max", fontsize=9)
    for r in (0, 1):
        ax[r, c].axis("off")
fig.suptitle(f"{args.model}: real cut-off defects (red/white = GT)")
fig.tight_layout()
fig.savefig(f"docs/figures/{args.model}_cutcheck.png", dpi=90)
print("saved figure")
