"""wallplugs shape completion, step by step on one photo (for explaining the method):
  1 photo  2 anomaly map + final-rule detection  3 SAM plug outline, axis and ends  4 break-edge check on the detection
  5 extension to the typical plug length, against the ground truth.
    .venv/Scripts/python.exe scripts/wp_complete_steps.py [--photo bad/000_regular]  -> docs/figures/wp_complete_steps.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.complete import LMED, _fg, plug_cache_path  # noqa: E402
from industrial_inspection.plugs import load_plugs  # noqa: E402
from industrial_inspection.segeval import ground_truth  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--photo", default="bad/000_regular")
a.add_argument("--model", default="s0_r1792_L1523")
a.add_argument("--pad", type=int, default=90)
args = a.parse_args()
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

z, zv = np.load(f"results/maps/{args.model}__wallplugs.npz"), np.load(f"results/maps/{args.model}__wallplugs__val.npz")
paths = z["paths"].astype(str)
i = [k for k, p in enumerate(paths) if p.replace("\\", "/").endswith(args.photo + ".png")][0]
R, pred, _ = wallplugs_masks(z["maps"].astype(np.float32), paths, zv["maps"].astype(np.float32), zv["paths"].astype(str))
photo = paths[i]
pl = load_plugs(plug_cache_path(photo))
H, W = pl[0].mask.shape
img = np.asarray(Image.open(photo).convert("RGB").resize((W, H), Image.BILINEAR))
up = lambda m, interp=cv2.INTER_NEAREST: cv2.resize(m.astype(np.float32), (W, H), interpolation=interp)
P = up(pred[i]) > 0.5
amap = up(R[i], cv2.INTER_LINEAR)
gt = up(ground_truth(photo, R.shape[1:])) > 0.5
fg = _fg(photo, (H, W))
occ = np.any([p.mask for p in pl], 0)

# the plug whose detection passes the break-edge test (same rule as industrial_inspection.complete)
chosen = None
for p in pl:
    hit = np.argwhere(P & p.mask)
    if not len(hit):
        continue
    ys, xs = np.nonzero(p.mask)
    c = np.array([xs.mean(), ys.mean()])
    X = np.stack([xs, ys], 1) - c
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    v = np.array([-u[1], u[0]])
    t, s = X @ u, X @ v
    t0, t1, wd = t.min(), t.max(), np.percentile(s, 97) - np.percentile(s, 3)
    L = t1 - t0
    Th = (hit[:, 1] - c[0]) * u[0] + (hit[:, 0] - c[1]) * u[1]
    Sh = (hit[:, 1] - c[0]) * v[0] + (hit[:, 0] - c[1]) * v[1]
    for sgn, end in ((1, t1), (-1, t0)):
        zz = sgn * (Th - end) > -0.15 * L
        if zz.any() and np.ptp(Sh[zz]) >= 0.7 * wd and np.ptp(Th) <= max(np.ptp(Sh), 1):
            chosen = dict(p=p, c=c, u=u, v=v, end=end, sgn=sgn, wd=wd, L=L, t0=t0, t1=t1,
                          cover=np.ptp(Sh[zz]) / wd, along=np.ptp(Th), across=np.ptp(Sh))
            break
    if chosen:
        break
assert chosen, "no plug passes the break-edge test in this photo"
c, u, v, wd, L, end, sgn = chosen["c"], chosen["u"], chosen["v"], chosen["wd"], chosen["L"], chosen["end"], chosen["sgn"]
ext = LMED - L
a0, a1 = c + u * end, c + u * (end + sgn * ext)
h = v * wd / 2
poly = np.round(np.array([a0 + h, a1 + h, a1 - h, a0 - h])).astype(np.int32)
rect = np.zeros((H, W), np.uint8)
cv2.fillPoly(rect, [poly], 1)
add = rect.astype(bool) & ~fg & ~occ

# crop around the plug and its extension
ys, xs = np.nonzero(chosen["p"].mask | add | gt)
y0, y1 = max(ys.min() - args.pad, 0), min(ys.max() + args.pad, H)
x0, x1 = max(xs.min() - args.pad, 0), min(xs.max() + args.pad, W)
cr = lambda m: m[y0:y1, x0:x1]
off = np.array([x0, y0])


def contour(ax, m, color, lw=2.2):
    cs, _ = cv2.findContours(cr(m).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for q in cs:
        q = q[:, 0]
        ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=color, lw=lw)


fig, axs = plt.subplots(1, 5, figsize=(22, 5.6), facecolor="#fcfcfb")
titles = ["1. 원본 사진", "2. 모델이 찾은 이상한 곳", "3. SAM으로 플러그 재기", "4. 부러진 단면인지 확인", "5. 없어진 길이만큼 칠하기"]
for ax, t in zip(axs, titles):
    ax.imshow(cr(img))
    ax.set_title(t, fontsize=15, fontweight="bold", loc="left", color="#0b0b0b")
    ax.axis("off")
# 2: anomaly map + detection
am = cr(amap)
axs[1].imshow(am, cmap="inferno", alpha=0.55, vmin=np.percentile(amap, 50), vmax=np.percentile(amap, 99.9))
contour(axs[1], P, "#38d0ff")
axs[1].text(0.02, 0.02, "하늘색 = 이상하다고 칠한 곳\n(부러진 단면만, 정답의 7%)", transform=axs[1].transAxes, fontsize=11.5,
            color="white", va="bottom", bbox=dict(fc="#000000aa", ec="none", pad=4))
# 3: SAM outline, axis, ends
contour(axs[2], chosen["p"].mask, "#ffd23f")
e0, e1 = c + u * chosen["t0"] - off, c + u * chosen["t1"] - off
axs[2].plot([e0[0], e1[0]], [e0[1], e1[1]], color="#ffd23f", lw=1.5, ls="--")
for e in (e0, e1):
    axs[2].plot(*e, "o", color="#ffd23f", ms=9, mec="black")
mid = (e0 + e1) / 2
axs[2].text(0.02, 0.02, f"노랑 = SAM 플러그 윤곽\n방향(점선), 양 끝(점), 길이 {L:.0f}px", transform=axs[2].transAxes,
            fontsize=11.5, color="white", va="bottom", bbox=dict(fc="#000000aa", ec="none", pad=4))
# 4: break-edge check
contour(axs[3], chosen["p"].mask, "#ffd23f", 1.5)
hitm = P & chosen["p"].mask
axs[3].imshow(np.ma.masked_where(~cr(hitm), cr(hitm)), cmap="cool", alpha=0.85, vmin=0, vmax=1)
axs[3].text(0.02, 0.02, f"플러그 끝에 있음 (통과)\n폭의 {chosen['cover']*100:.0f}%를 가로지름 (기준 70%) (통과)\n"
            f"얇은 띠: 길이 방향 {chosen['along']:.0f}px ≤ 폭 방향 {chosen['across']:.0f}px (통과)",
            transform=axs[3].transAxes, fontsize=11.5, color="white", va="bottom", bbox=dict(fc="#000000aa", ec="none", pad=4))
# 5: extension vs GT
from matplotlib.colors import ListedColormap
axs[4].imshow(np.ma.masked_where(~cr(add), cr(add)), cmap=ListedColormap(["#ffd23f"]), alpha=0.5)
contour(axs[4], gt, "#ff3b3b")
contour(axs[4], add | P, "#ffd23f")
cov_before = (P & gt).sum() / max(gt.sum(), 1)
cov_after = ((P | add) & gt).sum() / max(gt.sum(), 1)
axs[4].text(0.02, 0.02, f"정상 길이 {LMED:.0f} - 현재 {L:.0f} = {ext:.0f}px 연장\n빨강 = 정답, 노랑 = 최종 예측\n"
            f"정답 덮는 비율 {cov_before*100:.0f}% → {cov_after*100:.0f}%", transform=axs[4].transAxes, fontsize=11.5,
            color="white", va="bottom", bbox=dict(fc="#000000aa", ec="none", pad=4))
fig.suptitle(f"부러진 플러그 형태 복원 ({Path(photo).parent.name}/{Path(photo).stem}, 공개 test)", x=0.01, ha="left",
             fontsize=17, fontweight="bold", color="#0b0b0b")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("docs/figures/wp_complete_steps.png", dpi=110, facecolor="#fcfcfb")
print(f"saved docs/figures/wp_complete_steps.png  L={L:.0f} ext={ext:.0f} cover {cov_before:.2f}->{cov_after:.2f}")
