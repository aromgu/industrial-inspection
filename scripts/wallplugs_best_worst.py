"""wallplugs: the 5 best and 5 worst defective test_public photos (per-photo F1) of the submission model.
Columns: photo (yellow box = defect) | ground truth | anomaly map (+GT outline) | binary prediction vs GT |
zoom on the defect. Prediction colours: green = hit, red = false alarm, blue = missed.
Writes docs/figures/qualitative/wallplugs_best.png and _worst.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mp
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
z, t, p = load(model, "wallplugs", rule, post)
paths, labels, maps = z["paths"].astype(str), z["labels"], z["maps"]
hw = eval_size(paths[0])
good = maps[labels == 0][:20].astype(np.float32)
lo, hi = np.quantile(good, [0.5, 0.9999])
rows = []
for i in np.flatnonzero(labels == 1):
    gt = ground_truth(paths[i], hw)
    pred = postprocess(maps[i].astype(np.float32) > t, *p)
    rows.append((f1(pred, gt), i, gt, pred))
rows.sort(key=lambda r: -r[0])
scores = np.array([r[0] for r in rows])
print(f"{len(rows)} defective photos, F1>0.1: {np.mean(scores > 0.1) * 100:.0f}%, F1=0: {np.mean(scores == 0) * 100:.0f}%")
worst = [r for r in rows if r[0] == 0]
worst = [worst[round(k * (len(worst) - 1) / 4)] for k in range(5)] if len(worst) >= 5 else rows[-5:]
for tag, sel, title in (("best", rows[:5], "잘 된 사진 5장 (사진별 F1 상위)"), ("worst", worst, "안 된 사진 5장 (F1 0, 목록에서 일정 간격)")):
    fig, axes = plt.subplots(5, 5, figsize=(22, 19), facecolor="white")
    for r, (s, i, gt, pred) in enumerate(sel):
        img = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
        m = maps[i].astype(np.float32)
        ys, xs = np.nonzero(gt)
        cy, cx = int(ys.mean()), int(xs.mean())
        half = max(40, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
        y0, y1 = max(0, cy - half), min(hw[0], cy + half)
        x0, x1 = max(0, cx - half), min(hw[1], cx + half)
        ov = overlay(img, pred, gt)
        axes[r, 0].imshow(img)
        axes[r, 0].add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=2))
        gv = (img * 0.45).astype(np.uint8); gv[gt] = (227, 73, 72)
        axes[r, 1].imshow(gv)
        axes[r, 2].imshow(m, cmap="magma", vmin=lo, vmax=max(hi, float(m.max())))
        axes[r, 2].contour(gt, levels=[0.5], colors="#4aa3ff", linewidths=1.2)
        axes[r, 3].imshow(ov)
        axes[r, 4].imshow(np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((300, 300))) if False else ov[y0:y1, x0:x1])
        axes[r, 0].set_ylabel(Path(paths[i]).name, fontsize=11)
        axes[r, 3].set_xlabel(f"이 사진 F1 {s:.2f}  (예측 {pred.sum()}px / 정답 {gt.sum()}px)", fontsize=11)
    for ax, ti in zip(axes[0], ("원본 (노란 상자 = 결함 위치)", "정답 GT (빨강)", "이상 맵 + GT(파란 윤곽)",
                                 "예측 vs 정답", "결함 부분 확대")):
        ax.set_title(ti, fontsize=14)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"wallplugs {title} — 초록: 맞게 잡음, 빨강: 오경보, 파랑: 놓침 [{model}]", fontsize=16, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = Path(f"docs/figures/qualitative/wallplugs_{tag}.png"); fig.savefig(out, dpi=70, facecolor="white"); plt.close(fig)
    print("saved", out)
