"""wallplugs: zoom on the defects of the parts no condition detects (regular-light photos), with GT and map(s).
    .venv/Scripts/python.exe scripts/wp_missed.py --models s0_r1792_L1523 [--sigma 2] --parts 001 002 004 005 007 --out wallplugs_missed"""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.segeval import ground_truth  # noqa: E402
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf"); plt.rcParams["font.family"] = "Malgun Gothic"
a = argparse.ArgumentParser(); a.add_argument("--models", nargs="+", required=True); a.add_argument("--sigma", type=float, default=2)
a.add_argument("--parts", nargs="+", default=["001", "002", "004", "005", "007"]); a.add_argument("--out", default="wallplugs_missed")
args = a.parse_args()
zs = [np.load(f"results/maps/{m}__wallplugs.npz") for m in args.models]
paths = zs[0]["paths"].astype(str); names = [Path(p).name for p in paths]
fig, axes = plt.subplots(len(args.parts), 2 + len(args.models), figsize=(4 * (2 + len(args.models)), 4 * len(args.parts)), facecolor="white")
for r, part in enumerate(args.parts):
    i = next(k for k, p in enumerate(paths) if names[k] == f"{part}_regular.png" and zs[0]["labels"][k] == 1)
    hw = zs[0]["maps"].shape[1:]
    full = Image.open(paths[i]).convert("RGB")
    img = np.asarray(full.resize(hw[::-1], Image.BILINEAR)); gt = ground_truth(paths[i], hw)
    ys, xs = np.nonzero(gt); cy, cx = int(ys.mean()), int(xs.mean()); h = 45
    sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
    # photo crop at FULL resolution for detail
    sx, sy = full.width / hw[1], full.height / hw[0]
    crop = full.crop((sl[1].start * sx, sl[0].start * sy, sl[1].stop * sx, sl[0].stop * sy))
    axes[r, 0].imshow(crop); axes[r, 0].set_ylabel(f"{part} (결함 {gt.sum()}px)", fontsize=12)
    gv = (img * 0.5).astype(np.uint8); gv[gt] = (227, 73, 72); axes[r, 1].imshow(gv[sl])
    for c, z in enumerate(zs):
        m = z["maps"][i].astype(np.float32); m = ndimage.gaussian_filter(m, args.sigma) if args.sigma else m
        rest = m[~ndimage.binary_dilation(gt, iterations=8)]
        axes[r, 2 + c].imshow(m[sl], cmap="magma", vmin=np.quantile(m, 0.5), vmax=m.max())
        axes[r, 2 + c].contour(gt[sl], levels=[0.5], colors="#4aa3ff", linewidths=1.2)
        axes[r, 2 + c].set_xlabel(f"결함 최고 {m[gt].max():.0f} / 나머지 최고 {rest.max():.0f}", fontsize=10)
for ax, t in zip(axes[0], ["원본 (원래 해상도로 확대)", "정답"] + [f"맵: {m}" for m in args.models]):
    ax.set_title(t, fontsize=12)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"docs/figures/qualitative/{args.out}.png", dpi=70, facecolor="white"); print("saved")
