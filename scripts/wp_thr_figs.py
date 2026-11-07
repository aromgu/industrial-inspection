"""wallplugs threshold experiments, visualised (current best map: DINOv3, 1792 px, layers 16+24, sigma 2).
  docs/figures/wp_thr_conditions.png  per-photo maximum score by capture condition (good vs defective), raw map and
                                      per-photo normalised map (photo's own 90th percentile subtracted), with the
                                      validation threshold drawn
  docs/figures/wp_thr_curves.png      threshold swept over quantiles of the validation per-photo maxima: small /
                                      large region recall, good-photo false alarms, pooled F1 -- raw map, normalised
                                      map, hysteresis (seed = normalised, extent = raw map > val pixel q99.5)
  docs/figures/qualitative/wp_thr_examples.png  four photos x (raw low thr, raw high thr, normalised, hysteresis)
    .venv/Scripts/python.exe scripts/wp_thr_figs.py
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
from final_examples import overlay  # noqa: E402
from wp_local import gt_of, SMALL  # noqa: E402
from industrial_inspection.q2 import parts_and_tags  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
sm = lambda a: np.stack([ndimage.gaussian_filter(m.astype(np.float32), 2) for m in a])
z = np.load("results/maps/s0_r1792_L1523__wallplugs.npz")
R, Rv = sm(z["maps"]), sm(np.load("results/maps/s0_r1792_L1523__wallplugs__val.npz")["maps"])
N = np.load("results/maps/r1792L_s2_iq90__wallplugs.npz")["maps"].astype(np.float32)
Nv = np.load("results/maps/r1792L_s2_iq90__wallplugs__val.npz")["maps"].astype(np.float32)
paths, y = z["paths"].astype(str), z["labels"]
_, tag = parts_and_tags(paths)
gt = gt_of(paths, y, R.shape[1:])
lab = [ndimage.label(g) for g in gt]
vmaxR, vmaxN = Rv.reshape(len(Rv), -1).max(1), Nv.reshape(len(Nv), -1).max(1)
te = np.quantile(Rv, 0.995)


def hyst(i, ts):
    seed = N[i] > ts
    if not seed.any():
        return np.zeros_like(seed)
    l, _ = ndimage.label((R[i] > te) | seed)
    keep = np.unique(l[seed])
    return np.isin(l, keep[keep > 0])


def metrics(pred):
    hit, size = [], []
    for i in np.flatnonzero(y == 1):
        l, n = lab[i]
        for c in range(1, n + 1):
            r = l == c
            hit.append(pred[i][r].mean() >= 0.2)
            size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    return (hit[size < SMALL].mean() * 100, hit[size >= SMALL].mean() * 100,
            pred[y == 0].reshape((y == 0).sum(), -1).any(1).mean() * 100, 200 * tp / max(1, 2 * tp + fp + fn))


# ---- 1. per-condition maxima ----
conds = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor="white")
for ax, M, vmax, title in ((axes[0], R, vmaxR, "원래 맵: 사진별 최고점"), (axes[1], N, vmaxN, "사진별 정규화 맵 (사진 자신의 90% 분위를 뺌)")):
    mx = M.reshape(len(M), -1).max(1)
    for j, c in enumerate(conds):
        for lbl, col, off in ((0, "#4C78A8", -0.15), (1, "#E45756", 0.15)):
            s = (tag == c) & (y == lbl)
            ax.scatter(np.full(s.sum(), j + off) + np.random.default_rng(j).uniform(-0.06, 0.06, s.sum()), mx[s], s=14, color=col, alpha=0.75,
                       label=("정상" if lbl == 0 else "불량") if j == 0 else None)
    ax.scatter(np.full(len(vmax), -0.9), vmax, s=14, color="#888", alpha=0.8, label="검증(정상, 보통 조명)")
    t = np.quantile(vmax, 0.9)
    ax.axhline(t, color="k", ls="--", lw=1)
    ax.text(5.45, t, " 임계값\n (검증 최고점 90% 분위)", fontsize=8, va="center")
    ax.set_xticks([-0.9] + list(range(len(conds))), ["검증"] + conds, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("사진 한 장의 최고 점수")
fig.suptitle("왜 임계값 하나로는 안 되나: shift 조건의 정상 사진은 원래 맵에서 통째로 높다 → 정규화하면 검증과 같은 높이", fontsize=12)
fig.tight_layout()
fig.savefig("docs/figures/wp_thr_conditions.png", dpi=110, facecolor="white")
plt.close(fig)

# ---- 2. operating curves ----
qs = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
curves = {"원래 맵, 임계값 하나": [], "정규화 맵, 임계값 하나": [], "이중 임계값 (씨앗 = 정규화, 영역 = 원래 맵)": []}
for q in qs:
    curves["원래 맵, 임계값 하나"].append(metrics(R > np.quantile(vmaxR, q)))
    curves["정규화 맵, 임계값 하나"].append(metrics(N > np.quantile(vmaxN, q)))
    ts = np.quantile(vmaxN, q)
    curves["이중 임계값 (씨앗 = 정규화, 영역 = 원래 맵)"].append(metrics(np.stack([hyst(i, ts) for i in range(len(N))])))
    print("q", q, {k: np.round(v[-1], 1).tolist() for k, v in curves.items()}, flush=True)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), facecolor="white")
names = ["작은 결함 검출 %", "큰 결함 검출 %", "정상 사진 오경보 % (낮을수록 좋음)", "F1 (전체 픽셀)"]
colors = ["#888888", "#4C78A8", "#E45756"]
for j, ax in enumerate(axes):
    for (k, v), c in zip(curves.items(), colors):
        ax.plot(qs, [r[j] for r in v], "o-", color=c, label=k, lw=2 if "이중" in k else 1.3)
    ax.axvline(0.9, color="k", ls=":", lw=1)
    ax.set_title(names[j], fontsize=11)
    ax.set_xlabel("임계값 = 검증 사진 최고점의 몇 % 분위")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle("임계값을 바꿔 가며: 이중 임계값이 오경보를 낮게 유지하면서 결함 검출과 F1을 지킨다 (점선 = 채택한 90%)", fontsize=12)
fig.tight_layout()
fig.savefig("docs/figures/wp_thr_curves.png", dpi=110, facecolor="white")
plt.close(fig)

# ---- 3. examples ----
pick = [("good", "shift_2", "000"), ("good", "regular", "003"), ("bad", "shift_1", "009"), ("bad", "regular", "012")]
tq = np.quantile(vmaxR, 0.3), np.quantile(vmaxR, 0.9)
tn = np.quantile(vmaxN, 0.9)
fig, axes = plt.subplots(len(pick), 5, figsize=(21, 4.0 * len(pick)), facecolor="white")
for r, (kind, cond, part) in enumerate(pick):
    i = next((k for k, p in enumerate(paths) if f"/{kind}/" in p.replace("\\", "/") and Path(p).name.startswith(f"{part}_{cond}")), None)
    if i is None:
        i = next(k for k, p in enumerate(paths) if f"/{kind}/" in p.replace("\\", "/") and tag[k] == cond)
    img = np.asarray(Image.open(paths[i]).convert("RGB").resize(R.shape[1:][::-1], Image.BILINEAR))
    g = gt[i]
    preds = [R[i] > tq[0], R[i] > tq[1], N[i] > tn, hyst(i, tn)]
    axes[r, 0].imshow(img)
    for c, p in enumerate(preds):
        axes[r, c + 1].imshow(overlay(img, p, g))
        fa = (p & ~g).sum()
        axes[r, c + 1].set_xlabel(f"예측 {p.sum()}px, 그중 오경보 {fa}px" + (f", 정답 {g.sum()}px" if g.any() else ""), fontsize=10)
    axes[r, 0].set_ylabel(f"{'정상' if kind == 'good' else '불량'} {Path(paths[i]).name}", fontsize=12)
for ax, t in zip(axes[0], ["원본", "원래 맵, 낮은 임계값 (30%)", "원래 맵, 높은 임계값 (90%)", "정규화 맵, 임계값 하나 (90%)", "이중 임계값 (현재)"]):
    ax.set_title(t, fontsize=12)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("같은 사진, 임계값 방식만 다르게 — 초록 맞게 잡음, 빨강 오경보, 파랑 놓침", fontsize=14)
fig.tight_layout()
fig.savefig("docs/figures/qualitative/wp_thr_examples.png", dpi=60, facecolor="white")
print("saved")
