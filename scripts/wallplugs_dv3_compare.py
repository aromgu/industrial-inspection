"""Same 10 wallplugs photos as wallplugs_best_worst.py (5 best, 5 worst of the submission model), now
submission model vs DINOv3-H+ k=1 (hires_dv3_k1, SuperADD's threshold rule: validation 95th pct x 1.421,
no post-processing). Columns: zoom of photo | GT | submission map | submission pred | DINOv3 map | DINOv3 pred
(maps and predictions shown on the zoom window around the defect). Writes docs/figures/qualitative/wallplugs_dv3_compare.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from final_examples import f1, load, overlay  # noqa: E402
from make_submission import CONFIG  # noqa: E402
from seg_post import postprocess  # noqa: E402
from industrial_inspection.segeval import eval_size, ground_truth  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
model, rule, post = CONFIG["wallplugs"]
zA, tA, pA = load(model, "wallplugs", rule, post)
zB = np.load("results/maps/hires_dv3_k1__wallplugs.npz")
vB = np.load("results/maps/hires_dv3_k1__wallplugs__val.npz")["maps"].astype(np.float32)
tB = float(np.quantile(vB, 0.95) * 1.421)
paths, labels = zA["paths"].astype(str), zA["labels"]
idxB = {p: i for i, p in enumerate(zB["paths"].astype(str))}
hw = eval_size(paths[0])
rows = []
for i in np.flatnonzero(labels == 1):
    gt = ground_truth(paths[i], hw)
    mA = zA["maps"][i].astype(np.float32); mB = zB["maps"][idxB[paths[i]]].astype(np.float32)
    pa = postprocess(mA > tA, *pA); pb = mB > tB
    rows.append((f1(pa, gt), f1(pb, gt), i, gt, mA, mB, pa, pb))
rows.sort(key=lambda r: -r[0])
fa, fb = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
print(f"photos with F1>0.1: submission {np.mean(fa > 0.1)*100:.0f}%, DINOv3 k1 {np.mean(fb > 0.1)*100:.0f}% | "
      f"mean per-photo F1 {fa.mean()*100:.1f} vs {fb.mean()*100:.1f}")
worst = [r for r in rows if r[0] == 0]
sel = rows[:5] + [worst[round(k * (len(worst) - 1) / 4)] for k in range(5)]
gA = zA["maps"][labels == 0][:20].astype(np.float32); gB = zB["maps"][zB["labels"] == 0][:20].astype(np.float32)
loA, hiA = np.quantile(gA, [0.5, 0.9999]); loB, hiB = np.quantile(gB, [0.5, 0.9999])
fig, axes = plt.subplots(10, 6, figsize=(19, 33), facecolor="white")
for r, (sa, sb, i, gt, mA, mB, pa, pb) in enumerate(sel):
    img = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
    ys, xs = np.nonzero(gt); cy, cx = int(ys.mean()), int(xs.mean())
    half = max(60, int(max(np.ptp(ys), np.ptp(xs)) * 1.2))
    sl = (slice(max(0, cy - half), cy + half), slice(max(0, cx - half), cx + half))
    gv = (img * 0.45).astype(np.uint8); gv[gt] = (227, 73, 72)
    panels = [img[sl], gv[sl], None, overlay(img, pa, gt)[sl], None, overlay(img, pb, gt)[sl]]
    for c, pnl in enumerate(panels):
        ax = axes[r, c]
        if c == 2:
            ax.imshow(mA[sl], cmap="magma", vmin=loA, vmax=max(hiA, float(mA.max())))
        elif c == 4:
            ax.imshow(mB[sl], cmap="magma", vmin=loB, vmax=max(hiB, float(mB.max())))
        else:
            ax.imshow(pnl)
        if c in (2, 4):
            ax.contour(gt[sl], levels=[0.5], colors="#4aa3ff", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
    axes[r, 0].set_ylabel(("잘 된 " if r < 5 else "안 된 ") + Path(paths[i]).name, fontsize=11)
    axes[r, 3].set_xlabel(f"F1 {sa:.2f}", fontsize=12); axes[r, 5].set_xlabel(f"F1 {sb:.2f}", fontsize=12)
for ax, t in zip(axes[0], ("원본 (결함 주변 확대)", "정답 GT", "이상 맵: 제출 후보\n(DINOv2-L + 플러그 비교)", "예측: 제출 후보",
                          "이상 맵: DINOv3-H+ k=1", "예측: DINOv3-H+ k=1\n(SuperADD 임계값, 후처리 없음)")):
    ax.set_title(t, fontsize=13)
fig.suptitle("wallplugs 같은 10장: 제출 후보 vs DINOv3-H+ — 초록: 맞게 잡음, 빨강: 오경보, 파랑: 놓침", fontsize=16, y=0.998)
fig.tight_layout(rect=(0, 0, 1, 0.99))
fig.savefig("docs/figures/qualitative/wallplugs_dv3_compare.png", dpi=60, facecolor="white")
print("saved")
