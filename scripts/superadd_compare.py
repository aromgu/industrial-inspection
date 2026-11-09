"""Our benchmark submission (DINOv3-HiResKNN, 2026-11-09) vs the SuperADD paper (the reference method of this project).
SegF1 (private): SuperADD paper Table 2; ours from benchmark.mvtec.com. wallplugs public SegF1: SuperADD Table 1 (79.2),
ours = the public-test estimate of the submitted rule (75.7, scripts/wp_complete.py).
    .venv/Scripts/python.exe scripts/superadd_compare.py  -> docs/figures/superadd_compare.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
CATS = ["평균", "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
OURS = [50.40, 1.39, 75.99, 54.31, 72.64, 42.60, 52.71, 33.63, 69.90]
SUPER = [57.4, 11.6, 88.5, 68.0, 73.8, 55.7, 64.4, 26.4, 70.9]
C_OURS, C_SUP = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3.2, 1]}, facecolor=SURF)
x = np.arange(len(CATS))
w = 0.38
ax.set_facecolor(SURF)
ax.axvspan(6.5, 7.5, color="#cde2fb", alpha=0.45, lw=0, zorder=0)
ax.bar(x - w / 2, OURS, w - 0.03, color=C_OURS, label="우리 (DINOv3-HiResKNN, 제출)", zorder=3)
ax.bar(x + w / 2, SUPER, w - 0.03, color=C_SUP, label="SuperADD 논문 (Table 2)", zorder=3)
for i, (o, s) in enumerate(zip(OURS, SUPER)):
    d = o - s
    ax.text(i, max(o, s) + 2.5, f"{d:+.1f}", ha="center", va="bottom", fontsize=10,
            color=INK, fontweight="bold" if d > 0 else "normal")
    ax.text(i - w / 2, o + 0.8, f"{o:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.text(i + w / 2, s + 0.8, f"{s:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, CATS, fontsize=10, color=INK)
ax.set_ylim(0, 100)
ax.axvline(0.5, color=INK2, lw=0.8, ls=":", zorder=1)
ax.set_title("SegF1 (private) — 카테고리별, 막대 위 숫자 = 우리 - SuperADD", loc="left", fontsize=12.5, color=INK, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2, labelcolor=INK)

ax2.set_facecolor(SURF)
for vals, c, lab in (((75.7, 33.63), C_OURS, "우리"), ((79.2, 26.4), C_SUP, "SuperADD")):
    ax2.plot([0, 1], vals, color=c, lw=2, marker="o", ms=8, zorder=3)
    ax2.text(1.06, vals[1], f"{lab} {vals[1]:.1f}", va="center", fontsize=10, color=INK)
    ax2.text(-0.06, vals[0], f"{vals[0]:.1f}", va="center", ha="right", fontsize=10, color=INK)
ax2.set_xticks([0, 1], ["공개 test", "private"], fontsize=10, color=INK)
ax2.set_xlim(-0.45, 1.6)
ax2.set_ylim(0, 100)
ax2.set_title("wallplugs SegF1: 공개 → private", loc="left", fontsize=12.5, color=INK, fontweight="bold")
for a in (ax, ax2):
    a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    a.tick_params(axis="y", colors=INK2, labelsize=9)
    a.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(INK2)
fig.text(0.01, 0.01, "우리 공개 test 값 75.7은 공개 test로 규칙을 고른 추정치(낙관적). SuperADD 공개 값은 논문 Table 1.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("docs/figures/superadd_compare.png", dpi=130, facecolor=SURF)
print("saved docs/figures/superadd_compare.png")
