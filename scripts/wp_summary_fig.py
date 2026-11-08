"""wallplugs: every tried method on one chart (numbers copied from docs/experiment_log.md; all at the operating point
noted there -- hysteresis with seed quantile 0.9 unless stated). docs/figures/wp_all_experiments.png
    .venv/Scripts/python.exe scripts/wp_summary_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
# name, small-defect recall %, large-defect recall %, good-photo false alarm %, pooled F1 (all photos), verdict
rows = [
    ("이전 제출 후보 (DINOv2-L + SAM 플러그)", 16.7, 13.3, 20.0, 16.6, "old"),
    ("DINOv3, 4개 층, 1280px", 31.7, 46.7, 60.0, 49.7, "step"),
    ("+ 16·24번째 층, 1792px, σ=2", 50.0, 66.7, 46.7, 44.0, "step"),
    ("+ 사진별 정규화 (임계값 하나)", 50.0, 40.0, 6.7, 33.9, "step"),
    ("+ 이중 임계값 = 현재 최고", 51.7, 76.7, 6.7, 70.9, "best"),
    ("2048px (원본 해상도)", 53.3, 73.3, 6.7, 69.8, "no"),
    ("1792px + 좌우 뒤집기 TTA", 50.0, 76.7, 13.3, 68.7, "no"),
    ("top-hat 씨앗 추가", 50.0, 73.3, 21.7, 66.5, "no"),
    ("씨앗 게이트 (가장자리 제외)", 55.0, 80.0, 23.3, 65.0, "no"),
    ("가짜 결함 판별기 결합 (개발/평가 불안정)", 55.0, 80.0, 20.0, 72.0, "no"),
    ("소벨 재순위 (가짜 결함으로 학습)", 50.0, 76.7, 10.0, 72.1, "no"),
    ("유도 필터 스무딩 (r=4)", 48.3, 80.0, 8.3, 74.7, "partial"),
]
col = {"old": "#BBBBBB", "step": "#9DB4D3", "best": "#2E6DB4", "no": "#E8A0A0", "partial": "#F2C14E"}
names = [r[0] for r in rows][::-1]
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True, facecolor="white")
for ax, j, title, better in ((axes[0], 1, "작은 결함 검출 % (높을수록 좋음)", "high"), (axes[1], 3, "정상 사진 오경보 % (낮을수록 좋음)", "low"),
                             (axes[2], 4, "F1 전체 픽셀 (높을수록 좋음)", "high")):
    v = [r[j] for r in rows][::-1]
    c = [col[r[5]] for r in rows][::-1]
    ax.barh(range(len(v)), v, color=c)
    best = [r for r in rows if r[5] == "best"][0][j]
    ax.axvline(best, color="#2E6DB4", ls="--", lw=1)
    for k, vv in enumerate(v):
        ax.text(vv + 0.8, k, f"{vv:.0f}", va="center", fontsize=9)
    ax.set_title(title, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(names)), names, fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ("old", "step", "best", "partial", "no")]
fig.legend(handles, ["이전", "누적 개선 단계", "현재 최고", "일부 효과 (F1만)", "효과 없음"], loc="lower center", ncol=5, fontsize=10)
fig.suptitle("wallplugs: 시도한 방법 전체 (공개 test, 점선 = 현재 최고)", fontsize=14)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig("docs/figures/wp_all_experiments.png", dpi=100, facecolor="white")
print("saved")
