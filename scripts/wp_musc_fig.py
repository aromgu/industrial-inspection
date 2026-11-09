"""MuSc-style mutual scoring vs the final map (numbers from scripts/wp_eval.py, 2026-11-09; see docs/experiment_log.md).
Left: noise floor (median per-photo max of the seed map on good test photos) per lighting condition.
Right: pooled F1 on the dev / eval halves (final rule: guided extent, seeds at the 95 % quantile; no shape completion).
    .venv/Scripts/python.exe scripts/wp_musc_fig.py  -> docs/figures/wp_musc.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
C_BASE, C_MUSC, C_MIX = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COND = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
BASE = [92.5, 87.4, 87.1, 115.2, 127.4, 106.5]
MUSC = [93.3, 89.0, 92.3, 90.4, 85.4, 94.7]
F1 = {"기존 최종 지도": (70.5, 76.3), "MuSc 단독 (k=20)": (66.8, 75.5), "기존 + MuSc (반반)": (72.5, 78.0)}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]}, facecolor=SURF)
x = np.arange(len(COND))
w = 0.38
ax.bar(x - w / 2, BASE, w - 0.03, color=C_BASE, label="기존 (정상 학습 사진과 비교)", zorder=3)
ax.bar(x + w / 2, MUSC, w - 0.03, color=C_MUSC, label="MuSc (test 사진끼리 비교, k=20)", zorder=3)
for i, (b, m) in enumerate(zip(BASE, MUSC)):
    ax.text(i - w / 2, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(i + w / 2, m + 1, f"{m:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.axvspan(2.5, 5.5, color="#fde3d3", alpha=0.5, lw=0, zorder=0)
ax.set_xticks(x, COND, fontsize=10, color=INK)
ax.set_ylim(0, 145)
ax.set_title("정상 사진의 노이즈 (낮을수록 좋음): 조명 조건별", loc="left", fontsize=12.5, color=INK, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)

xs = np.arange(2)
w2 = 0.26
for j, ((name, vals), c) in enumerate(zip(F1.items(), (C_BASE, C_MUSC, C_MIX))):
    ax2.bar(xs + (j - 1) * w2, vals, w2 - 0.03, color=c, label=name, zorder=3)
    for k, v in enumerate(vals):
        ax2.text(k + (j - 1) * w2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax2.set_xticks(xs, ["개발 절반", "평가 절반"], fontsize=10.5, color=INK)
ax2.set_ylim(60, 84)
ax2.set_title("F1 (공개 test, 형태 복원 전)", loc="left", fontsize=12.5, color=INK, fontweight="bold")
ax2.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
for a in (ax, ax2):
    a.set_facecolor(SURF)
    a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    a.tick_params(axis="y", colors=INK2, labelsize=9)
    a.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(INK2)
fig.text(0.01, 0.01, "노이즈 = 정상 test 사진의 사진별 최고 점수(가장자리 제외) 중앙값. 오른쪽 그래프는 y축이 60부터 시작.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("docs/figures/wp_musc.png", dpi=130, facecolor=SURF)
print("saved docs/figures/wp_musc.png")
