"""wallplugs report figures (for Notion / README):
  docs/figures/wp_progress.png       per approach: small / large region recall, good-photo false alarms, pooled F1
  docs/figures/qualitative/wp_best_vs_old.png
      10 defective regular-light photos (5 of each kind: parts the new method detects / does not), zoomed on the
      defect: photo | GT | old submission prediction | new map | new prediction
Numbers in the progress chart are copied from docs/experiment_log.md (wp_local.py / wp_hyst.py outputs).
    .venv/Scripts/python.exe scripts/wp_report_figs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from final_examples import f1, load, overlay  # noqa: E402
from make_submission import CONFIG  # noqa: E402
from seg_post import postprocess  # noqa: E402
from industrial_inspection.segeval import ground_truth  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"

# ---- 1. progress chart ----
rows = [  # name, small recall, large recall, good-photo false alarm, pooled F1 (all photos)
    ("이전 제출 후보\n(DINOv2-L + 플러그)", 16.7, 13.3, 20.0, 16.6),
    ("DINOv3\n4개 층, 1280px", 31.7, 46.7, 60.0, 49.7),
    ("+ 16·24번째 층만\n1792px, σ=2", 50.0, 66.7, 46.7, 44.0),
    ("+ 사진별 정규화\n(한 임계값)", 50.0, 40.0, 6.7, 33.9),
    ("+ 이중 임계값\n(현재 최고)", 51.7, 76.7, 6.7, 70.9),
]
fig, ax = plt.subplots(figsize=(13, 4.8), facecolor="white")
x = np.arange(len(rows))
cols = [("작은 결함 검출 %", "#4C78A8"), ("큰 결함 검출 %", "#72B7B2"), ("정상 사진 오경보 % (낮을수록 좋음)", "#E45756"), ("F1 (전체)", "#F2A541")]
for j, (lab, c) in enumerate(cols):
    v = [r[1 + j] for r in rows]
    ax.bar(x + (j - 1.5) * 0.2, v, 0.2, label=lab, color=c)
    for xi, vi in zip(x + (j - 1.5) * 0.2, v):
        ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, [r[0] for r in rows], fontsize=10)
ax.set_ylim(0, 90)
ax.set_title("wallplugs 공개 test: 단계별 변화 (임계값은 정상 검증 사진 10% 오경보 기준, 이중 임계값은 씨앗 90%·영역 99.5% 분위)", fontsize=11)
ax.legend(fontsize=9, ncol=4, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("docs/figures/wp_progress.png", dpi=110, facecolor="white")
plt.close(fig)

# ---- 2. qualitative: old vs new ----
model, rule, post = CONFIG["wallplugs"]
zA, tA, pA = load(model, "wallplugs", rule, post)
zB = np.load("results/maps/s0_r1792_L1523__wallplugs.npz")
pb = np.load("results/seg/wp_best_pred.npz")
predB = np.unpackbits(pb["pred"], axis=-1, count=int(pb["shape"][-1])).astype(bool)
paths, y = zB["paths"].astype(str), zB["labels"]
assert (pb["paths"].astype(str) == paths).all()
idxA = {p: i for i, p in enumerate(zA["paths"].astype(str))}
hw = zB["maps"].shape[1:]
gB = np.stack([ndimage.gaussian_filter(m.astype(np.float32), 2) for m in zB["maps"][y == 0][:20]])
lo, hi = np.quantile(gB, [0.5, 0.9999])
hit_parts = ["003", "008", "009", "011", "014"]
miss_parts = ["002", "004", "005", "007", "001"]
fig, axes = plt.subplots(10, 5, figsize=(17, 34), facecolor="white")
for r, part in enumerate(hit_parts + miss_parts):
    i = next(k for k, p in enumerate(paths) if y[k] == 1 and Path(p).name == f"{part}_regular.png")
    img = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
    gt = ground_truth(paths[i], hw)
    predA = postprocess(zA["maps"][idxA[paths[i]]].astype(np.float32) > tA, *pA)
    mB = ndimage.gaussian_filter(zB["maps"][i].astype(np.float32), 2)
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = max(60, int(max(np.ptp(ys), np.ptp(xs)) * 1.2))
    sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
    gv = (img * 0.45).astype(np.uint8)
    gv[gt] = (227, 73, 72)
    axes[r, 0].imshow(img[sl])
    axes[r, 1].imshow(gv[sl])
    axes[r, 2].imshow(overlay(img, predA, gt)[sl])
    axes[r, 3].imshow(mB[sl], cmap="magma", vmin=lo, vmax=max(hi, float(mB.max())))
    axes[r, 3].contour(gt[sl], levels=[0.5], colors="#4aa3ff", linewidths=1.2)
    axes[r, 4].imshow(overlay(img, predB[i], gt)[sl])
    axes[r, 0].set_ylabel(("잡는 부품 " if r < 5 else "못 잡는 부품 ") + part, fontsize=12)
    axes[r, 2].set_xlabel(f"F1 {f1(predA, gt):.2f}", fontsize=12)
    axes[r, 4].set_xlabel(f"F1 {f1(predB[i], gt):.2f}", fontsize=12)
for ax, t in zip(axes[0], ["원본 (결함 주변 확대)", "정답 (빨강)", "이전 제출 후보 예측", "새 이상 맵 (파랑 = 정답)", "새 방법 예측"]):
    ax.set_title(t, fontsize=13)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("wallplugs 보통 조명 불량 사진: 이전 제출 후보 vs 현재 최고 — 초록 맞게 잡음, 빨강 오경보, 파랑 놓침", fontsize=15, y=0.998)
fig.tight_layout(rect=(0, 0, 1, 0.99))
fig.savefig("docs/figures/qualitative/wp_best_vs_old.png", dpi=60, facecolor="white")
print("saved")
