"""wallplugs: SuperADD setting (our reproduction: DINOv3-H+ layers 7 15 23 31, 1280 px, k=1, threshold validation
p95 x 1.421 -- map hires_dv3_k1) vs our final model (map s0_r1792_L1523 + make_submission.wallplugs_masks).
Rows: defective photos (zoom around the defect) chosen automatically -- where ours gains most, where both detect,
where both miss -- plus good shift-lighting photos (whole photo) where the SuperADD setting false-alarms most.
Columns: photo | GT | SuperADD setting | ours (red = predicted, green outline = GT), per-photo F1 in the titles.
Writes docs/figures/wp_final_vs_superadd.png
    .venv/Scripts/python.exe scripts/wp_final_compare.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
from make_submission import wallplugs_masks  # noqa: E402
from wp_local import gt_of  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
z, zv = np.load("results/maps/s0_r1792_L1523__wallplugs.npz"), np.load("results/maps/s0_r1792_L1523__wallplugs__val.npz")
paths, y = z["paths"].astype(str), z["labels"]
_, ours, _ = wallplugs_masks(z["maps"], paths, zv["maps"], zv["paths"].astype(str))
zb = np.load("results/maps/hires_dv3_k1__wallplugs.npz")
assert list(zb["paths"].astype(str)) == list(paths)
mb = zb["maps"].astype(np.float32)
vb = np.load("results/maps/hires_dv3_k1__wallplugs__val.npz")["maps"].astype(np.float32)
sup = mb > np.quantile(vb, 0.95) * 1.421
if sup.shape != ours.shape:
    sup = np.stack([np.asarray(Image.fromarray(s.astype(np.uint8)).resize(ours.shape[1:][::-1], Image.NEAREST)) > 0 for s in sup])
hw = ours.shape[1:]
gt = gt_of(paths, y, hw)


def f1(p, g):
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    return 200 * tp / max(1, 2 * tp + fp + fn)


def pooled(P):
    tp, fp, fn = (P & gt).sum(), (P & ~gt).sum(), (~P & gt).sum()
    return 200 * tp / (2 * tp + fp + fn)


bad = [i for i in np.flatnonzero(y == 1) if Path(paths[i]).stem.endswith("regular")]
gain = sorted(bad, key=lambda i: f1(ours[i], gt[i]) - f1(sup[i], gt[i]), reverse=True)
both = [i for i in bad if f1(ours[i], gt[i]) > 40 and f1(sup[i], gt[i]) > 40 and i not in gain[:4]]
miss = [i for i in bad if f1(ours[i], gt[i]) < 5 and f1(sup[i], gt[i]) < 5]
good = sorted([i for i in np.flatnonzero(y == 0) if "shift" in Path(paths[i]).stem], key=lambda i: -sup[i].sum())
rows = [(i, "gain") for i in gain[:4]] + [(i, "both") for i in both[:2]] + [(i, "miss") for i in miss[:2]] + [(i, "good") for i in good[:2]]
label = {"gain": "개선", "both": "둘 다 잡음", "miss": "둘 다 놓침", "good": "정상 (shift 조명)"}

fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)))
for r, (i, kind) in enumerate(rows):
    im = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
    if kind == "good":
        sl = (slice(None), slice(None))
    else:
        ys, xs = np.nonzero(gt[i])
        h = max(48, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = int(np.clip(cy - h, 0, hw[0] - 2 * h)), int(np.clip(cx - h, 0, hw[1] - 2 * h))
        sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
    crop, g = im[sl], gt[i][sl]
    edge = g & ~ndimage.binary_erosion(g)

    def over(p):
        o = crop.astype(np.float32).copy()
        o[p] = o[p] * 0.4 + np.array([255, 0, 0]) * 0.6
        o[edge] = [0, 255, 0]
        return o.astype(np.uint8)

    ax[r, 0].imshow(crop)
    ax[r, 0].set_title(f"{Path(paths[i]).stem}  [{label[kind]}]", fontsize=10)
    ax[r, 1].imshow(g, cmap="gray")
    ax[r, 1].set_title("정답 (GT)" if kind != "good" else "정답: 결함 없음", fontsize=10)
    if kind == "good":
        ax[r, 2].imshow(over(sup[i][sl]))
        ax[r, 2].set_title(f"SuperADD 설정: 오경보 {sup[i].sum()} px", fontsize=10)
        ax[r, 3].imshow(over(ours[i][sl]))
        ax[r, 3].set_title(f"최종 모델: 오경보 {ours[i].sum()} px", fontsize=10)
    else:
        ax[r, 2].imshow(over(sup[i][sl]))
        ax[r, 2].set_title(f"SuperADD 설정  F1 {f1(sup[i], gt[i]):.0f}", fontsize=10)
        ax[r, 3].imshow(over(ours[i][sl]))
        ax[r, 3].set_title(f"최종 모델  F1 {f1(ours[i], gt[i]):.0f}", fontsize=10)
    for c in range(4):
        ax[r, c].axis("off")
fa = lambda P: P[y == 0].reshape((y == 0).sum(), -1).any(1).mean() * 100
fig.suptitle(f"wallplugs 공개 test: SuperADD 설정(재현) vs 최종 모델 (빨강 = 예측, 초록 = 정답 윤곽)\n"
             f"전체 F1  SuperADD 설정 {pooled(sup):.1f}  /  최종 {pooled(ours):.1f};   "
             f"정상 사진 오경보  {fa(sup):.0f}%  /  {fa(ours):.0f}%", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("docs/figures/wp_final_vs_superadd.png", dpi=80, facecolor="white")
print("pooled F1 superadd-setting %.1f ours %.1f; FA %.0f%% / %.0f%%" % (pooled(sup), pooled(ours), fa(sup), fa(ours)))
