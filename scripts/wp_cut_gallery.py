"""wallplugs cut-off plugs: how the final model (map + rule + shape completion) predicts them, every lighting condition.
Rows: bad/000_* (broken end, 6 lightings) and bad/001_regular (thin shaved edge). Columns: photo, anomaly map,
prediction (cyan = rule detection, yellow = added by shape completion) against the ground truth (red).
    .venv/Scripts/python.exe scripts/wp_cut_gallery.py  -> docs/figures/wp_cut_gallery.png
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.complete import plug_cache_path, seeded_completion  # noqa: E402
from industrial_inspection.segeval import ground_truth  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
M = "s0_r1792_L1523"
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+")
ap.add_argument("--out", default="docs/figures/wp_cut_gallery.png")
ap.add_argument("--no-title", action="store_true")
ARGS = ap.parse_args()
ROWS = ARGS.rows or ["000_regular", "000_overexposed", "000_underexposed", "000_shift_1", "000_shift_2", "000_shift_3", "001_regular"]
z, zv = np.load(f"results/maps/{M}__wallplugs.npz"), np.load(f"results/maps/{M}__wallplugs__val.npz")
paths = z["paths"].astype(str)
R, pred, (ts, te) = wallplugs_masks(z["maps"].astype(np.float32), paths, zv["maps"].astype(np.float32), zv["paths"].astype(str))
vlo, vhi = np.percentile(R, 50), np.percentile(R, 99.95)
H, W = R.shape[1:]


def contour(ax, m, color, lw=2.0):
    for q in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        q = q[:, 0]
        ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=color, lw=lw)


fig, axs = plt.subplots(len(ROWS), 3, figsize=(11.5, 3.5 * len(ROWS)), facecolor="#fcfcfb")
for r, name in enumerate(ROWS):
    i = [k for k, p in enumerate(paths) if p.replace("\\", "/").endswith(f"bad/{name}.png")][0]
    p = paths[i]
    img = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR))
    gt = ground_truth(p, (H, W)) > 0.5
    add = seeded_completion(p, plug_cache_path(p), pred[i]) & ~pred[i]
    ys, xs = np.nonzero(gt)
    pad = 70
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
    cr = lambda a: a[y0:y1, x0:x1]
    det, fin = pred[i], pred[i] | add
    cov0, cov1 = (det & gt).sum() / gt.sum(), (fin & gt).sum() / gt.sum()
    a0, a1, a2 = axs[r]
    a0.imshow(cr(img)); contour(a0, cr(gt), "#ff3b3b", 1.6)
    a1.imshow(cr(img)); a1.imshow(cr(R[i]), cmap="inferno", alpha=0.6, vmin=vlo, vmax=vhi)
    contour(a1, cr(gt), "#7dff8a", 1.4)
    a2.imshow(cr(img))
    for m, c in ((det, "#38d0ff"), (add, "#ffd23f")):
        a2.imshow(np.ma.masked_where(~cr(m), cr(m)), cmap=ListedColormap([c]), alpha=0.6)
    contour(a2, cr(gt), "#ff3b3b", 1.8)
    if add.any():
        msg = f"단면 검출 {cov0*100:.0f}% → 형태 복원 후 {cov1*100:.0f}%"
    elif det.any():
        msg = f"검출 {cov0*100:.0f}%, 형태 복원 조건 미달"
    else:
        msg = "검출 없음 → 형태 복원도 시작 못 함"
    a2.text(0.02, 0.02, msg, transform=a2.transAxes, fontsize=10.5, color="white", va="bottom",
            bbox=dict(fc="#000000aa", ec="none", pad=3))
    a0.set_ylabel({"000_regular": "보통 조명", "000_shift_1": "shift 조명"}.get(name, name.replace("_", " ", 1)), fontsize=12, fontweight="bold", color="#0b0b0b")
    for a in (a0, a1, a2):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
for a, t in zip(axs[0], ["원본 (빨강 = 정답: 없어진 부분)", "이상 점수 지도 (초록 = 정답)", "최종 예측 (하늘 = 검출, 노랑 = 복원)"]):
    a.set_title(t, fontsize=12.5, fontweight="bold", loc="left", color="#0b0b0b")
if not ARGS.no_title:
    fig.suptitle("잘려 나간 플러그를 최종 모델이 예측하는 방식 (공개 test)", x=0.01, ha="left", fontsize=16, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 1 if ARGS.no_title else 0.985))
fig.savefig(ARGS.out, dpi=100, facecolor="#fcfcfb")
print("saved", ARGS.out)
