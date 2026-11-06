"""Summary figures of the experiments for the Notion page (docs/figures/summary_*.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
R = "p95x1.2"; P = "close6+fill+small64"


def common(path, model):
    d = pd.read_csv(path)
    d = d[(d.model == model) & (d.threshold == R) & (d.post == P) & (d.protocol == "defect-only")]
    return d.set_index("category").f1 * 100


sub = pd.read_csv("results/seg/compare_post.csv").query("variant=='ours'").set_index("category").f1
v = {
    "제출 후보(16장 뱅크 k=50, wallplugs는 플러그 단위)": sub,
    "SuperADD식 뱅크 k=1": common("results/seg/sabank/seg_post__hires_sabank_k1.csv", "hires_sabank_k1"),
    "SuperADD식 뱅크 k=50": common("results/seg/sabank/seg_post__hires_sabank_k50.csv", "hires_sabank_k50"),
}
cols = ["#4C78A8", "#F2A541", "#E45756"]
fig, ax = plt.subplots(figsize=(13, 5.2))
x = np.arange(len(CATS) + 1)
for i, ((name, s), c) in enumerate(zip(v.items(), cols)):
    vals = [s.get(k, np.nan) for k in CATS]; vals.append(np.nanmean(vals))
    b = ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=name, color=c)
    for xi, y in zip(x + (i - 1) * 0.27, vals):
        ax.text(xi, y + 1, f"{y:.0f}", ha="center", fontsize=8)
ax.axhline(62.6, color="k", ls="--", lw=1); ax.text(8.45, 63.5, "SuperADD 평균 62.6(공개 test)", ha="right", fontsize=9)
ax.set_xticks(x, CATS + ["평균"]); ax.set_ylabel("결함 영역 F1 (%, 공개 test 불량 사진, 1/4 해상도)")
ax.set_title("카테고리별 결과: 뱅크·k에 따라 카테고리마다 승패가 갈린다"); ax.legend(loc="upper left", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("docs/figures/summary_by_category.png", dpi=110)

# input size probe
rows = []
for k, nm in (("k1", "k=1"), ("k50", "k=50")):
    a = common(f"results/seg/sabank/seg_post__hires_sabank_{k}.csv", f"hires_sabank_{k}")
    b = common(f"results/seg/sabank_r1280/seg_post__hires_sabank_r1280_{k}.csv", f"hires_sabank_r1280_{k}")
    for c in ["vial", "fruit_jelly", "rice"]:
        rows.append((nm, c, a[c], b[c]))
df = pd.DataFrame(rows, columns=["k", "cat", "672", "1280"])
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (k, g) in zip(axs, df.groupby("k")):
    xx = np.arange(len(g))
    ax.bar(xx - .2, g["672"], .4, label="짧은 변 672px", color="#4C78A8")
    ax.bar(xx + .2, g["1280"], .4, label="짧은 변 1280px", color="#E45756")
    for xi, (a, b) in enumerate(zip(g["672"], g["1280"])):
        ax.text(xi - .2, a + 1, f"{a:.0f}", ha="center", fontsize=9); ax.text(xi + .2, b + 1, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(xx, g["cat"]); ax.set_title(f"SuperADD식 뱅크 {k}  (평균 {g['672'].mean():.1f} → {g['1280'].mean():.1f})")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("결함 영역 F1 (%)"); axs[0].legend(fontsize=9)
fig.suptitle("입력 크기 시험: 키워도 rice만 좋아지고 vial·fruit_jelly는 나빠진다"); fig.tight_layout()
fig.savefig("docs/figures/summary_input_size.png", dpi=110)

# progress of the mean F1
steps = [("Dinomaly-L\n(공식 임계값)", 23.7), ("HiResKNN\n(공식 임계값)", 28.1), ("+공통 임계값·후처리", 40.6),
         ("+k=50", 42.2), ("+wallplugs\n플러그 단위(제출 후보)", 45.0)]
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.bar(range(len(steps)), [s[1] for s in steps], color="#4C78A8")
for i, s in enumerate(steps): ax.text(i, s[1] + 1, f"{s[1]:.1f}", ha="center")
ax.axhline(62.6, color="k", ls="--", lw=1); ax.text(0, 64, "SuperADD 62.6", fontsize=9)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=9); ax.set_ylabel("평균 결함 영역 F1 (%)")
ax.set_title("개선 단계별 평균 F1 (공개 test, 불량 사진만)"); ax.set_ylim(0, 72)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("docs/figures/summary_progress.png", dpi=110)
print(pd.DataFrame(v).round(1).to_string()); print(df.round(1).to_string())
