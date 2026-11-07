"""wallplugs pipeline with the synthetic-trained Sobel re-ranker (no real defects used for training):
  1. stage-1 map: raw kNN map (--model, unsmoothed, sigma 2 here), per-photo offset N = m - photo p90 (unclipped)
  2. candidates: local maxima of N (15 px), above the photo's q99, top --top per photo
  3. candidate score c = z(N at peak) + w * z(Sobel logit), z statistics from the VALIDATION candidates
  4. seeds: candidates with c above the --qs quantile of the per-photo max c of the validation photos
  5. extent: connected regions of raw map > validation pixel q99.5 containing a seed (a 4 px disk if the raw map
     is below that at the seed)
w = 0 is the same candidate pipeline without Sobel (fair baseline). Prints small / large region recall, good-photo
false alarms, pooled F1, per weight and seed quantile; optional --half dev|eval.
    .venv/Scripts/python.exe scripts/wp_rerank.py --model s0_r1792_L1523
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of, SMALL  # noqa: E402
from wp_sobel_synth import feat, grads, synthetic_set  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--model", default="s0_r1792_L1523")
a.add_argument("--sigma", type=float, default=2.0)
a.add_argument("--top", type=int, default=8)
a.add_argument("--w", type=float, nargs="+", default=[0.0, 0.5, 1.0])
a.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.9, 0.95])
a.add_argument("--qe", type=float, default=0.995)
a.add_argument("--half", default="all", choices=["all", "dev", "eval"])
args = a.parse_args()
sm = lambda arr: np.stack([ndimage.gaussian_filter(m.astype(np.float32), args.sigma) for m in arr])
z = np.load(f"results/maps/{args.model}__wallplugs.npz")
zv = np.load(f"results/maps/{args.model}__wallplugs__val.npz")
R, Rv = sm(z["maps"]), sm(zv["maps"])
paths, vpaths, y = z["paths"].astype(str), zv["paths"].astype(str), z["labels"]
hw = R.shape[1:]
te = np.quantile(Rv, args.qe)


def cands(m):
    n = m - np.percentile(m, 90)
    pk = (n == ndimage.maximum_filter(n, size=15)) & (n > np.quantile(n, 0.99))
    ys, xs = np.nonzero(pk)
    o = np.argsort(-n[ys, xs])[: args.top]
    return ys[o], xs[o], n[ys[o], xs[o]]


def sobel_feats(path, ys, xs):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    g = grads(im)
    sy, sx = im.shape[0] / hw[0], im.shape[1] / hw[1]
    return np.array([feat(im, *g, int((yy + 0.5) * sy), int((xx + 0.5) * sx)) for yy, xx in zip(ys, xs)])


print("candidates + Sobel features ...", flush=True)
C = [cands(m) for m in R]
Cv = [cands(m) for m in Rv]
Ft = [sobel_feats(p, c[0], c[1]) for p, c in zip(paths, C)]
Fv = [sobel_feats(p, c[0], c[1]) for p, c in zip(vpaths, Cv)]
X, Y = synthetic_set()
Fv_all = np.concatenate(Fv)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(
    np.r_[X, Fv_all], np.r_[Y, np.zeros(len(Fv_all), int)])
Lt = [clf.decision_function(f) for f in Ft]
Lv = [clf.decision_function(f) for f in Fv]
nv, lv = np.concatenate([c[2] for c in Cv]), np.concatenate(Lv)

gt = gt_of(paths, y, hw)
keep = np.ones(len(paths), bool)
if args.half != "all":
    from industrial_inspection.q2 import parts_and_tags, split_parts
    dev = split_parts(parts_and_tags(paths)[0], y)
    keep = dev if args.half == "dev" else ~dev
lab = [ndimage.label(g) for g in gt]
disk = np.hypot(*np.mgrid[-4:5, -4:5]) <= 4
print(f"[{args.half}] model={args.model}\n  w     qs    rec_s  rec_l  fa_test  f1_all  f1_def")
for w in args.w:
    score = lambda c, l: (c[2] - nv.mean()) / nv.std() + w * (l - lv.mean()) / lv.std()
    vbest = np.array([score(c, l).max() if len(l) else -1e9 for c, l in zip(Cv, Lv)])
    for qs in args.qs:
        T = np.quantile(vbest, qs)
        pred = np.zeros((len(paths), *hw), bool)
        for i in np.flatnonzero(keep):
            c, l = C[i], Lt[i]
            s = score(c, l)
            seeds = np.zeros(hw, bool)
            for yy, xx, ss in zip(c[0], c[1], s):
                if ss > T:
                    y0, x0 = max(0, yy - 4), max(0, xx - 4)
                    seeds[y0:yy + 5, x0:xx + 5] |= disk[(y0 - (yy - 4)):(y0 - (yy - 4)) + min(9, hw[0] - y0), (x0 - (xx - 4)):(x0 - (xx - 4)) + min(9, hw[1] - x0)]
            if not seeds.any():
                continue
            l2, _ = ndimage.label((R[i] > te) | seeds)
            k = np.unique(l2[seeds])
            pred[i] = np.isin(l2, k[k > 0])
        hit, size = [], []
        for i in np.flatnonzero(keep & (y == 1)):
            l3, n3 = lab[i]
            for cc in range(1, n3 + 1):
                r = l3 == cc
                hit.append(pred[i][r].mean() >= 0.2)
                size.append(r.sum())
        hit, size = np.array(hit), np.array(size)
        P, G, yy_ = pred[keep], gt[keep], y[keep]
        tp, fp, fn = (P & G).sum(), (P & ~G).sum(), (~P & G).sum()
        d = yy_ == 1
        tpd, fpd, fnd = (P[d] & G[d]).sum(), (P[d] & ~G[d]).sum(), (~P[d] & G[d]).sum()
        print(f"  {w:3.1f}  {qs:4.2f}  {hit[size < SMALL].mean()*100:5.1f}  {hit[size >= SMALL].mean()*100:5.1f}  "
              f"{P[yy_ == 0].reshape((yy_ == 0).sum(), -1).any(1).mean()*100:5.1f}   {200*tp/max(1, 2*tp+fp+fn):5.1f}  {200*tpd/max(1, 2*tpd+fpd+fnd):5.1f}", flush=True)
