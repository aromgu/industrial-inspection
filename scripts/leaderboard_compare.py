"""Our benchmark.mvtec.com submission (DINOv3-HiResKNN, 2026-11-09) vs the leaderboard's best entries (private split).
Leaderboard numbers pasted from the site: top-10 tables as of 2026-11-06, our row 2026-11-09.
  #1 entry  = the row ranked first by that metric's average (one method for all categories)
  best/cat  = best value per category among that metric's top-10 rows (different methods per category)
    .venv/Scripts/python.exe scripts/leaderboard_compare.py  -> docs/figures/leaderboard_compare.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
CATS = ["평균", "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
DATA = {  # metric: (ours, #1 entry name, #1 entry, best per category in top-10)
    "AucPro_0.05 (private)  — 리더보드 정렬 기준": (
        [57.70, 7.86, 53.85, 54.48, 68.14, 58.72, 76.06, 70.56, 71.90], "1위 YL",
        [88.73, 94.70, 98.41, 85.94, 78.98, 95.62, 84.84, 80.66, 90.73],
        [None, 94.70, 98.41, 89.60, 88.17, 95.62, 84.84, 80.66, 90.73]),
    "ClassF1 (private)  — 사진 단위 판정": (
        [69.04, 35.68, 65.54, 87.33, 52.84, 84.42, 86.78, 68.96, 70.81], "1위 CTDA+CGFA Ensemble",
        [85.89, 70.82, 87.30, 92.38, 80.85, 93.08, 92.68, 76.76, 93.23],
        [None, 70.82, 87.30, 92.38, 82.53, 93.08, 92.68, 81.15, 93.23]),
    "SegF1 (private)  — 결함 영역 분할": (
        [50.40, 1.39, 75.99, 54.31, 72.64, 42.60, 52.71, 33.63, 69.90], "1위 VAND2025 WSEG",
        [65.30, 20.45, 64.21, 78.83, 72.50, 77.82, 76.90, 51.90, 79.76],
        [None, 20.45, 91.16, 78.83, 79.29, 77.82, 76.90, 51.90, 79.76]),
}
C_OURS, C_TOP, C_BEST = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

fig, axes = plt.subplots(3, 1, figsize=(13, 12), facecolor=SURF)
x = np.arange(len(CATS))
w = 0.26
for ax, (title, (ours, top_name, top, best)) in zip(axes, DATA.items()):
    ax.set_facecolor(SURF)
    ax.axvspan(6.5, 7.5, color="#cde2fb", alpha=0.45, lw=0, zorder=0)  # wallplugs: the class we optimised
    ax.bar(x - w, ours, w - 0.02, color=C_OURS, label="우리 (DINOv3-HiResKNN)", zorder=3)
    ax.bar(x, top, w - 0.02, color=C_TOP, label=f"{top_name} (평균 1위 방법 하나)", zorder=3)
    bx = [i for i, b in enumerate(best) if b is not None]
    ax.bar(x[bx] + w, [best[i] for i in bx], w - 0.02, color=C_BEST, label="카테고리별 최고 (상위 10개 중)", zorder=3)
    for i, v in enumerate(ours):
        ax.text(i - w, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x, CATS, fontsize=10, color=INK)
    ax.set_ylim(0, 108)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", colors=INK2, labelsize=9)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":", zorder=1)
fig.suptitle("MVTec AD 2 리더보드: 우리 제출 vs 최고 성능 (private, 파란 띠 = 집중한 wallplugs)",
             x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.01, 0.005, "리더보드 상위 10개 표는 2026-11-06 기준, 우리 점수는 2026-11-09 제출. 990개 제출 중 AucPro 평균 기준 약 431위.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.015, 1, 0.97))
fig.savefig("docs/figures/leaderboard_compare.png", dpi=130, facecolor=SURF)
print("saved docs/figures/leaderboard_compare.png")
