"""Item 1 figure: does the learned token projection (wp_adapt.py) lower the normal noise floor more than the defects?
Per model: seed map N = Gaussian(2) map - photo p90 (border band excluded), all values divided by the median validation
per-photo max N (so models on different scales compare). Left: median per-photo max N of good test photos per lighting
condition. Right: peak N inside the GT of the hard small defects (regular light) and of all defective photos (median).
Writes docs/figures/wp_adapt_compare.png
    .venv/Scripts/python.exe scripts/wp_adapt_compare.py s0_r1792_L1523 s0_r1792_adapt_inv [s0_r1792_adapt_cfa]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.q2 import parts_and_tags  # noqa: E402
from wp_local import gt_of  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
models = sys.argv[1:]
labels = {"s0_r1792_L1523": "기존 (얼린 DINOv3)", "s0_r1792_adapt_inv": "대조 학습 (InfoNCE)", "s0_r1792_adapt_cfa": "대조 + CFA 당기기"}
conds = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
hard = ["002", "004", "005", "007"]
res = {}
for m in models:
    z, zv = np.load(f"results/maps/{m}__wallplugs.npz"), np.load(f"results/maps/{m}__wallplugs__val.npz")
    paths, y = z["paths"].astype(str), z["labels"]
    _, tag = parts_and_tags(paths)
    g = lambda arr: np.stack([ndimage.gaussian_filter(a.astype(np.float32), 2) for a in arr])
    seed = lambda A: A - np.percentile(A.reshape(len(A), -1), 90, axis=1)[:, None, None]
    N, Nv = seed(g(z["maps"])), seed(g(zv["maps"]))
    b = np.zeros(N.shape[1:], bool)
    b[12:-12, 12:-12] = True
    ref = np.median([n[b].max() for n in Nv])
    gt = gt_of(paths, y, N.shape[1:])
    floor = [np.median([N[i][b].max() for i in np.flatnonzero((y == 0) & (tag == c))]) / ref for c in conds]
    hp = [N[next(k for k in range(len(paths)) if y[k] and Path(paths[k]).name == f"{p}_regular.png")] for p in hard]
    hv = [h[gt[next(k for k in range(len(paths)) if y[k] and Path(paths[k]).name == f"{p}_regular.png")]].max() / ref for h, p in zip(hp, hard)]
    alld = np.median([N[i][gt[i]].max() for i in np.flatnonzero(y == 1)]) / ref
    res[m] = (floor, hv + [alld])
    print(m, "floor", np.round(floor, 2), "defects", np.round(hv + [alld], 2))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
w = 0.8 / len(models)
for j, m in enumerate(models):
    ax[0].bar(np.arange(6) + j * w, res[m][0], w, label=labels.get(m, m))
    ax[1].bar(np.arange(5) + j * w, res[m][1], w, label=labels.get(m, m))
for k in (0, 1):
    ax[k].axhline(1.0, color="gray", ls="--", lw=1)
    ax[k].legend(fontsize=9)
ax[0].set_xticks(np.arange(6) + w * (len(models) - 1) / 2, conds)
ax[0].set_title("정상 사진 노이즈 바닥 (조명별, 사진당 최고점 중앙값)\n1.0 = 정상 검증 사진 최고점 중앙값", fontsize=11)
ax[1].set_xticks(np.arange(5) + w * (len(models) - 1) / 2, [f"{p} (작은 결함)" for p in hard] + ["불량 전체 중앙값"])
ax[1].set_title("결함 봉우리 높이 (같은 기준으로 나눔)\n높을수록 좋음, 노이즈 바닥보다 높아야 검출", fontsize=11)
fig.tight_layout()
fig.savefig("docs/figures/wp_adapt_compare.png", dpi=90, facecolor="white")
print("saved docs/figures/wp_adapt_compare.png")
