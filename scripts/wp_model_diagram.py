"""wallplugs final model: pipeline diagram, with the steps that differ from SuperADD highlighted.
Writes docs/figures/wp_final_model.png
    .venv/Scripts/python.exe scripts/wp_model_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
SAME, NEW, IO = "#DCE6F2", "#FBD9A6", "#EEEEEE"
fig, ax = plt.subplots(figsize=(16, 9.6), facecolor="white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.6)
ax.axis("off")


def box(x, y, w, h, title, body, color, edge="#555555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=color, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=12.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.75, body, ha="center", va="top", fontsize=10, linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.6, color="#444444"))


ax.text(0.2, 9.25, "wallplugs 최종 모델 (학습 없음: 고정 백본 + 정상 패치 메모리 뱅크 + 판정 규칙 + 형태 복원)", fontsize=16, weight="bold")
# row 1: bank
box(0.2, 6.6, 2.6, 2.2, "정상 학습 사진", "train/good\n(8장마다 1장 제외)", IO)
box(3.4, 6.6, 3.6, 2.2, "특징 추출 (같은 과정)", "DINOv3-H+ (고정)\n640px 조각, 128px 겹침", SAME)
box(7.6, 6.6, 3.6, 2.2, "메모리 뱅크", "정상 패치 10만 개\n거리 기반 선택, 밝기 0.8~1.2배", SAME)
arrow(2.8, 7.7, 3.4, 7.7)
arrow(7.0, 7.7, 7.6, 7.7)
# row 2: query
box(0.2, 3.4, 2.6, 2.6, "검사할 사진", "짧은 변 1792px\n[SuperADD: 1280px]", NEW, "#C77700")
box(3.4, 3.4, 3.6, 2.6, "① 특징 추출", "DINOv3-H+, 16·24번째 층\n[SuperADD: 8·16·24·32번째]\n16x16px 패치마다 특징", NEW, "#C77700")
box(7.6, 3.4, 3.6, 2.6, "② 이상 점수", "패치마다 뱅크에서\n가장 가까운 정상 패치까지 거리\n(k=1) → 이상 지도", SAME)
arrow(2.8, 4.7, 3.4, 4.7)
arrow(7.0, 4.7, 7.6, 4.7)
arrow(9.4, 6.6, 9.4, 6.0)
# decision
box(11.8, 5.1, 4.0, 3.7, "③-a 씨앗 (결함 위치)", "가우시안 σ=2로 부드럽게\n- 그 사진의 p90 (사진별 보정)\n> 정상 검증 사진 최고점의\n   95% 분위\n\"확실히 결함인 곳\"", NEW, "#C77700")
box(11.8, 0.9, 4.0, 3.7, "③-b 영역 (결함 범위)", "유도 필터 (원본 사진이 가이드,\n윤곽 보존 스무딩)\n> 정상 검증 픽셀 99.5% 분위\n\"결함일 수 있는 범위\"", NEW, "#C77700")
arrow(11.2, 4.9, 11.8, 6.6)
arrow(11.2, 4.5, 11.8, 2.8)
box(6.2, 0.4, 4.6, 2.2, "③ 판정: 이중 임계값", "씨앗과 연결된 영역만 결함\n[SuperADD: p95 x 1.421\n 기준 하나로 전체 판정]", NEW, "#C77700")
arrow(11.8, 2.0, 10.8, 1.6)
arrow(11.8, 6.0, 10.6, 2.6)
box(3.0, 0.2, 2.9, 2.6, "④ 형태 복원", "결함 영역이 플러그 끝을\n가로지르면(부러진 단면)\n정상 길이까지 빈 배경을 칠함\n[SuperADD: 없음]", NEW, "#C77700")
arrow(6.2, 1.5, 5.9, 1.5)
box(0.2, 0.2, 2.4, 2.6, "출력", "결함 지도 (TIFF)\n+ 결함 마스크 (PNG)\n임계값은 모두\n정상 검증 사진으로만", IO)
arrow(3.0, 1.5, 2.6, 1.5)
ax.add_patch(FancyBboxPatch((0.3, 8.95), 0.35, 0.2, boxstyle="square,pad=0", fc=SAME, ec="#555555"))
ax.text(0.75, 9.05, "SuperADD와 같음", fontsize=10, va="center")
ax.add_patch(FancyBboxPatch((2.6, 8.95), 0.35, 0.2, boxstyle="square,pad=0", fc=NEW, ec="#C77700"))
ax.text(3.05, 9.05, "바꾼 부분 ([ ] 안이 SuperADD)", fontsize=10, va="center")
fig.savefig("docs/figures/wp_final_model.png", dpi=90, facecolor="white", bbox_inches="tight")
print("saved")
