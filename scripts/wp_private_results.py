"""wallplugs private-test results of our benchmark submission (2026-11-09) vs SuperADD and the leaderboard's best.
Sources: ours = benchmark.mvtec.com row "DINOv3-HiResKNN"; SuperADD SegF1 = paper Table 2; SuperADD AucPro / ClassF1 =
the leaderboard rows "SuperADD-DINOv3" (14.09.26) / "SuperADD" (07.11.26) (the paper gives SegF1 only); leaderboard best
= best wallplugs value among each metric's top-10 rows (2026-11-06).
    .venv/Scripts/python.exe scripts/wp_private_results.py  -> docs/figures/wp_private_results.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
METRICS = ["AucPro_0.05\n(지도 품질)", "ClassF1\n(사진 단위 판정)", "SegF1\n(결함 영역 분할)"]
OURS = [70.56, 68.96, 33.63]
SUPER = [48.28, 54.00, 26.40]
BEST = [80.66, 81.15, 51.90]
BEST_NAME = ["YL", "DINO-DDB", "VAND2025 WSEG"]
C_OURS, C_SUP, C_BEST = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [2.2, 1]}, facecolor=SURF)
x = np.arange(3)
w = 0.26
ax.set_facecolor(SURF)
for off, vals, c, lab in ((-w, OURS, C_OURS, "우리 (DINOv3-HiResKNN)"), (0, SUPER, C_SUP, "SuperADD"),
                          (w, BEST, C_BEST, "리더보드 최고 (상위 10개 중)")):
    ax.bar(x + off, vals, w - 0.03, color=c, label=lab, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i + off, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=10,
                color=INK, fontweight="bold" if c == C_OURS else "normal")
for i, n in enumerate(BEST_NAME):
    ax.text(i + w, 2, n, ha="center", va="bottom", fontsize=7.5, color="white", rotation=90, zorder=4)
ax.set_xticks(x, METRICS, fontsize=10.5, color=INK)
ax.set_ylim(0, 100)
ax.set_title("wallplugs private test (공식 서버 채점)", loc="left", fontsize=13, color=INK, fontweight="bold")
ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK)

ax2.set_facecolor(SURF)
for vals, c, lab in (((75.7, 33.63), C_OURS, "우리"), ((79.2, 26.4), C_SUP, "SuperADD")):
    ax2.plot([0, 1], vals, color=c, lw=2, marker="o", ms=8, zorder=3)
    ax2.text(1.06, vals[1], f"{lab} {vals[1]:.1f}", va="center", fontsize=10, color=INK)
    ax2.text(-0.06, vals[0], f"{vals[0]:.1f}", va="center", ha="right", fontsize=10, color=INK)
ax2.set_xticks([0, 1], ["공개 test", "private"], fontsize=10.5, color=INK)
ax2.set_xlim(-0.45, 1.6)
ax2.set_ylim(0, 100)
ax2.set_title("SegF1: 공개 → private", loc="left", fontsize=13, color=INK, fontweight="bold")
for a in (ax, ax2):
    a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    a.tick_params(axis="y", colors=INK2, labelsize=9)
    a.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(INK2)
fig.text(0.01, 0.01, "SuperADD: SegF1은 논문 Table 2, AucPro·ClassF1은 리더보드의 SuperADD 재현 제출(논문에 없음). "
         "우리 공개 test 75.7은 공개 test로 규칙을 고른 추정치.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("docs/figures/wp_private_results.png", dpi=130, facecolor=SURF)
print("saved docs/figures/wp_private_results.png")
