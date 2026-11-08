"""wallplugs: how small defects look under image-processing enhancements (full-resolution crops around the GT).
Columns: original | global histogram equalisation | CLAHE (clip 2, 16x16) | unsharp (1.0, sigma 2) | Sobel magnitude |
Sobel on CLAHE; green = GT outline. Writes docs/figures/wp_local_enhance.png
    .venv/Scripts/python.exe scripts/wp_local_enhance.py [--parts 002 004 007 001 009 013]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
a = argparse.ArgumentParser()
a.add_argument("--parts", nargs="+", default=["002", "004", "007", "001", "009", "013"])
a.add_argument("--cond", default="regular")
a.add_argument("--half", type=int, default=90, help="half crop size in full-resolution pixels (minimum)")
args = a.parse_args()
root = Path("datasets/MVTec_AD_2/wallplugs/test_public")


def sobel(g):
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


cols = ["원본", "히스토그램 평활화 (전역)", "대비 강화 CLAHE", "고주파 강조 (언샤프)", "소벨 크기", "CLAHE 후 소벨"]
fig, ax = plt.subplots(len(args.parts), len(cols), figsize=(3.0 * len(cols), 3.1 * len(args.parts)))
for r, p in enumerate(args.parts):
    g = cv2.imread(str(root / "bad" / f"{p}_{args.cond}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(str(root / "ground_truth" / "bad" / f"{p}_{args.cond}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    ys, xs = np.nonzero(m)
    h = max(args.half, int(max(np.ptp(ys), np.ptp(xs)) * 0.8))
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = int(np.clip(cy - h, 0, g.shape[0] - 2 * h)), int(np.clip(cx - h, 0, g.shape[1] - 2 * h))
    sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
    eq = cv2.equalizeHist(g)  # whole photo, then cropped
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(g)
    f = g.astype(np.float32)
    us = np.clip(f + 1.0 * (f - cv2.GaussianBlur(f, (0, 0), 2.0)), 0, 255)
    sb, sbc = sobel(f), sobel(cl.astype(np.float32))
    imgs = [g, eq, cl, us, sb, sbc]
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for c, im in enumerate(imgs):
        crop = im[sl].astype(np.float32)
        if c >= 4:  # Sobel: scale by the crop's own 99.5th percentile
            crop = np.clip(crop / (np.percentile(crop, 99.5) + 1e-6) * 255, 0, 255)
        rgb = np.repeat(crop[..., None], 3, 2).astype(np.uint8)
        e = edge[sl]
        rgb[e] = [0, 255, 0]
        ax[r, c].imshow(rgb)
        ax[r, c].axis("off")
        if r == 0:
            ax[r, c].set_title(cols[c], fontsize=12)
    ax[r, 0].text(-8, h, f"{p}_{args.cond}\n({m.sum()} px)", fontsize=11, ha="right", va="center")
fig.suptitle(f"wallplugs 작은 결함 주변 확대 (원본 해상도, 초록 = 정답 윤곽, {args.cond} 조명)", fontsize=14)
fig.tight_layout(rect=(0.04, 0, 1, 0.97))
out = f"docs/figures/wp_local_enhance{'' if args.cond == 'regular' else '_' + args.cond}.png"
fig.savefig(out, dpi=85, facecolor="white")
print("saved", out)
