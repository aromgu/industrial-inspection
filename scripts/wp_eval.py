"""wallplugs one-shot evaluation of a raw (unsmoothed) anomaly map with the current best decision rule:
  smooth sigma 2; seed map N = m - photo p90 (unclipped); seeds: N > q-quantile of the validation per-photo max N;
  extent: m > validation pixel q99.5, connected to a seed (hysteresis). Prints for seed quantiles --qs:
  small / large region recall, good-photo false alarms, pooled F1 (all photos / defective only) on all, dev and eval
  halves of the physical parts; then per defective part (15) how many of its 6 photos are detected, and the noise
  floor: median per-photo max N (border band excluded) of good test photos per condition vs the defect peaks of the
  hard parts 002 004 005 007.
    .venv/Scripts/python.exe scripts/wp_eval.py --model s0_r1792_L1523 [--model2 X --mix 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of, SMALL  # noqa: E402
from industrial_inspection.q2 import parts_and_tags, split_parts  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--model", required=True)
a.add_argument("--model2", default=None, help="optional second raw map, averaged after per-map validation z-scoring")
a.add_argument("--mix", type=float, default=0.5)
a.add_argument("--sigma", type=float, default=2.0)
a.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.9, 0.95])
a.add_argument("--guided", type=float, nargs=2, default=None, metavar=("R", "EPS"),
               help="edge-aware smoothing instead of the Gaussian: guided filter (He et al. 2013) with the photo as guide")
a.add_argument("--seed-scale", type=float, default=0, help="also divide the seed map by the photo's own (p<this> - p90) spread, e.g. 99.5")
a.add_argument("--seed-gauss", action="store_true", help="with --guided: seeds from the Gaussian-smoothed map, extent from the guided one")
a.add_argument("--qe", type=float, default=0.995)
args = a.parse_args()
def guided(p: np.ndarray, I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter: output = a * I + b, locally linear in the guide I (box radius r); keeps the map's edges where
    the photo has edges, smooths elsewhere."""
    box = lambda x: ndimage.uniform_filter(x, 2 * r + 1)
    mI, mp = box(I), box(p)
    cov, var = box(I * p) - mI * mp, box(I * I) - mI * mI
    aa = cov / (var + eps)
    bb = mp - aa * mI
    return box(aa) * I + box(bb)


def smooth(maps, paths):
    if args.guided is None:
        return np.stack([ndimage.gaussian_filter(m.astype(np.float32), args.sigma) for m in maps])
    from PIL import Image
    r, eps = int(args.guided[0]), args.guided[1]
    out = []
    for m, p in zip(maps, paths):
        I = np.asarray(Image.open(p).convert("L").resize(m.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255
        m = m.astype(np.float32)
        lo, hi = np.percentile(m, [1, 99.99])
        q = (m - lo) / (hi - lo + 1e-6)  # guided filter works on a normalised map, scaled back after
        out.append(guided(q, I, r, eps) * (hi - lo) + lo)
    return np.stack(out)


def load(name):
    z, zv = np.load(f"results/maps/{name}__wallplugs.npz"), np.load(f"results/maps/{name}__wallplugs__val.npz")
    return z, smooth(z["maps"], z["paths"].astype(str)), smooth(zv["maps"], zv["paths"].astype(str))


z, R, Rv = load(args.model)
if args.model2:
    _, R2, Rv2 = load(args.model2)
    mu, sd, mu2, sd2 = Rv.mean(), Rv.std(), Rv2.mean(), Rv2.std()
    R = (1 - args.mix) * (R - mu) / sd + args.mix * (R2 - mu2) / sd2
    Rv = (1 - args.mix) * (Rv - mu) / sd + args.mix * (Rv2 - mu2) / sd2
paths, y = z["paths"].astype(str), z["labels"]
parts, tag = parts_and_tags(paths)
hw = R.shape[1:]
gt = gt_of(paths, y, hw)
S, Sv = R, Rv
if args.guided is not None and args.seed_gauss and not args.model2:
    g_ = lambda arr: np.stack([ndimage.gaussian_filter(m.astype(np.float32), args.sigma) for m in arr])
    S, Sv = g_(z["maps"]), g_(np.load(f"results/maps/{args.model}__wallplugs__val.npz")["maps"])
def seedmap(A):
    flat = A.reshape(len(A), -1)
    p90 = np.percentile(flat, 90, axis=1)[:, None, None]
    if not args.seed_scale:
        return A - p90
    if args.seed_scale < 90:  # spread from the bulk below p90 (defects cover < 10 % of a photo, so they do not move it)
        lo = np.percentile(flat, args.seed_scale, axis=1)[:, None, None]
        return (A - p90) / (p90 - lo + 1e-6)
    hi = np.percentile(flat, args.seed_scale, axis=1)[:, None, None]
    return (A - p90) / (hi - p90 + 1e-6)


N, Nv = seedmap(S), seedmap(Sv)
te = np.quantile(Rv, args.qe)
vmax = Nv.reshape(len(Nv), -1).max(1)
dev = split_parts(parts, y)
lab = [ndimage.label(g) for g in gt]


def predict(ts):
    pred = np.zeros_like(gt)
    for i in range(len(N)):
        seed = N[i] > ts
        if seed.any():
            l, _ = ndimage.label((R[i] > te) | seed)
            k = np.unique(l[seed])
            pred[i] = np.isin(l, k[k > 0])
    return pred


def metrics(pred, keep):
    hit, size = [], []
    for i in np.flatnonzero(keep & (y == 1)):
        l, n = lab[i]
        for c in range(1, n + 1):
            r = l == c
            hit.append(pred[i][r].mean() >= 0.2)
            size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    P, G, yy = pred[keep], gt[keep], y[keep]
    tp, fp, fn = (P & G).sum(), (P & ~G).sum(), (~P & G).sum()
    d = yy == 1
    tpd, fpd, fnd = (P[d] & G[d]).sum(), (P[d] & ~G[d]).sum(), (~P[d] & G[d]).sum()
    return (hit[size < SMALL].mean() * 100, hit[size >= SMALL].mean() * 100, P[yy == 0].reshape((yy == 0).sum(), -1).any(1).mean() * 100,
            200 * tp / max(1, 2 * tp + fp + fn), 200 * tpd / max(1, 2 * tpd + fpd + fnd))


name = (f"[seed scale p{args.seed_scale}] " if args.seed_scale else "") + args.model + (f" + {args.mix} x {args.model2}" if args.model2 else "") + (f" guided r{args.guided[0]:.0f} eps{args.guided[1]}" + (" (seeds Gaussian)" if args.seed_gauss else "") if args.guided else f" sigma {args.sigma}")
print(f"== {name}\n  qs    half  rec_s  rec_l  fa_test  f1_all  f1_def")
preds = {}
for qs in args.qs:
    preds[qs] = predict(np.quantile(vmax, qs))
    for h, keep in (("all", np.ones(len(y), bool)), ("dev", dev), ("eval", ~dev)):
        print(f"  {qs:4.2f}  {h:4s}  " + "  ".join(f"{v:5.1f}" for v in metrics(preds[qs], keep)))
pred = preds[0.9] if 0.9 in preds else preds[args.qs[0]]
print("  detected photos per defective part (qs 0.9, of 6):", end=" ")
for p in sorted(set(parts[y == 1])):
    idx = np.flatnonzero((parts == p) & (y == 1))
    det = sum(((pred[i] & gt[i]).sum() / max(1, gt[i].sum())) >= 0.2 for i in idx)
    print(f"{p.split('/')[-1]}:{det}", end=" ")
b = np.zeros(hw, bool)
b[12:-12, 12:-12] = True
gmax = {c: np.median([N[i][b].max() for i in np.flatnonzero((y == 0) & (tag == c))]) for c in ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]}
print("\n  noise floor (median max N of good test photos, border excluded):", {k: round(float(v), 1) for k, v in gmax.items()})
print("  val per-photo max N: median %.1f q90 %.1f" % (np.median([n[b].max() for n in Nv]), np.quantile([n[b].max() for n in Nv], 0.9)))
hard = {}
for p in ["002", "004", "005", "007"]:
    i = next(k for k in range(len(paths)) if y[k] == 1 and Path(paths[k]).name == f"{p}_regular.png")
    hard[p] = round(float(N[i][gt[i]].max()), 1)
print("  hard defects (regular light), peak N inside GT:", hard)
