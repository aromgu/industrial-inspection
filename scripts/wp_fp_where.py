"""P3: where do the high peaks on NORMAL plug pixels come from? Per test photo the top local maxima of the best
map (s0_r1792_L1523, sigma 2) are labelled by context from the SAM plug masks (eval size):
  background  no plug mask
  overlap     inside >= 2 plug masks (or within 3 px of another plug)
  edge        inside one plug, within 3 px of its outline
  border      within 12 px of the photo border
  plug body   inside one plug, away from its outline
Defect peaks (inside dilated GT) are counted separately, so we compare: which contexts hold normal peaks that are
as high as defect peaks? Writes results/seg/wp_fp_where.csv and docs/figures/wp_fp_where.png"""
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of  # noqa: E402
from industrial_inspection.plugs import load_plugs  # noqa: E402
from industrial_inspection.q2 import parts_and_tags  # noqa: E402
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf"); plt.rcParams["font.family"] = "Malgun Gothic"
z = np.load("results/maps/s0_r1792_L1523__wallplugs.npz")
M = np.stack([ndimage.gaussian_filter(m.astype(np.float32), 2) for m in z["maps"]])
N = np.stack([m - np.percentile(m, 90) for m in M])  # per-photo offset, NOT clipped (the saved iq90 map is clipped at the validation max)
paths, y = z["paths"].astype(str), z["labels"]; hw = M.shape[1:]; gt = gt_of(paths, y, hw); _, tag = parts_and_tags(paths)
C = Path("results/plugs_cache/wallplugs")
rows = []
for i, p in enumerate(paths):
    f = C / Path(p).parent.parent.name / Path(p).parent.name / (Path(p).stem + ".npz")
    pls = load_plugs(f)
    masks = np.stack([cv2.resize(q.mask.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST) for q in pls]).astype(bool) if pls else np.zeros((0, *hw), bool)
    count = masks.sum(0)
    near_other = np.zeros(hw, bool); edge = np.zeros(hw, bool)
    for k in range(len(masks)):
        dil = ndimage.binary_dilation(masks[k], iterations=3); ero = ndimage.binary_erosion(masks[k], iterations=3)
        edge |= masks[k] & ~ero
        if len(masks) > 1:
            near_other |= dil & masks[np.arange(len(masks)) != k].any(0)
    border = np.zeros(hw, bool); border[:12] = border[-12:] = True; border[:, :12] = border[:, -12:] = True
    m = N[i]
    peaks = (m == ndimage.maximum_filter(m, size=15)) & (m > np.quantile(m, 0.995))
    dgt = ndimage.binary_dilation(gt[i], iterations=4)
    for yy, xx in zip(*np.nonzero(peaks)):
        if border[yy, xx]: ctx = "border"
        elif count[yy, xx] == 0: ctx = "background"
        elif count[yy, xx] >= 2 or near_other[yy, xx]: ctx = "overlap"
        elif edge[yy, xx]: ctx = "edge"
        else: ctx = "plug body"
        rows.append(dict(path=p, label=int(y[i]), tag=tag[i], ctx=ctx, defect=int(dgt[yy, xx]), norm=float(m[yy, xx]), raw=float(M[i][yy, xx])))
df = pd.DataFrame(rows); df.to_csv("results/seg/wp_fp_where.csv", index=False)
d = df[df.defect == 1]; n = df[df.defect == 0]
t_def = d.norm.quantile(0.25)  # a peak 'as high as' the lower quartile of defect peaks
print(f"defect peaks {len(d)}, normal peaks {len(n)}; defect peak norm score q25 {t_def:.1f}")
tab = pd.DataFrame({"normal peaks": n.ctx.value_counts(), f"normal peaks >= defect q25": n[n.norm >= t_def].ctx.value_counts(),
                    "defect peaks": d.ctx.value_counts()}).fillna(0).astype(int)
tab["share of high normal peaks %"] = (tab.iloc[:, 1] / tab.iloc[:, 1].sum() * 100).round(1)
print(tab.to_string())
print("\nhigh normal peaks by capture condition:\n", n[n.norm >= t_def].tag.value_counts().to_string())
fig, ax = plt.subplots(figsize=(9, 4.2))
ctxs = ["plug body", "edge", "overlap", "border", "background"]
for j, c in enumerate(ctxs):
    ax.scatter(np.full((n.ctx == c).sum(), j - 0.12) + np.random.default_rng(j).uniform(-0.07, 0.07, (n.ctx == c).sum()), n[n.ctx == c].norm, s=6, alpha=0.35, color="#4C78A8", label="정상 봉우리" if j == 0 else None)
    ax.scatter(np.full((d.ctx == c).sum(), j + 0.12), d[d.ctx == c].norm, s=18, color="#E45756", label="결함 봉우리" if j == 0 else None)
ax.axhline(t_def, color="k", ls="--", lw=1); ax.text(4.4, t_def, " 결함 봉우리\n 하위 25%", fontsize=8, va="center")
ax.set_xticks(range(len(ctxs)), ["플러그 몸통", "플러그 모서리", "플러그 겹침", "사진 가장자리", "배경"])
ax.set_ylabel("봉우리 점수 (사진별 정규화 맵)"); ax.legend(fontsize=8)
ax.set_title("P3: 정상인데 높은 봉우리는 어디서 생기나 (공개 test 150장, 사진당 상위 봉우리)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("docs/figures/wp_fp_where.png", dpi=110)
