"""wallplugs: one map, threshold swept over quantiles of the per-photo maxima of the VALIDATION photos.
Per threshold: small/large region recall (>= 20 % of the region flagged), share of good test photos with a
false alarm, pooled F1 over all test photos. Shows the operating curve of a map; prints a table.
    .venv/Scripts/python.exe scripts/wp_sweep.py --model r1792L_s2_iz10 [--sigma 2] [--post fill+s64]"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from seg_post import postprocess  # noqa: E402
from wp_local import gt_of, SMALL  # noqa: E402

POSTS = {"none": (0, False, 0), "fill+s64": (0, True, 64), "s16": (0, False, 16)}
a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--sigma", type=float, default=0)
a.add_argument("--post", default="none"); a.add_argument("--q", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1])
args = a.parse_args()
z = np.load(f"results/maps/{args.model}__wallplugs.npz"); zv = np.load(f"results/maps/{args.model}__wallplugs__val.npz")
maps, vm = z["maps"].astype(np.float32), zv["maps"].astype(np.float32)
if args.sigma:
    maps = np.stack([ndimage.gaussian_filter(m, args.sigma) for m in maps]); vm = np.stack([ndimage.gaussian_filter(m, args.sigma) for m in vm])
y = z["labels"]; gt = gt_of(z["paths"].astype(str), y, maps.shape[1:])
lab = [ndimage.label(g) for g in gt]
vmax = vm.reshape(len(vm), -1).max(1)
print(f"{args.model} post={args.post}\n q(valmax)  rec_s  rec_l  fa_test  f1_all  f1_def")
for q in args.q:
    t = np.quantile(vmax, min(q, 1.0)) * (q if q > 1 else 1)
    pred = np.stack([postprocess(m > t, *POSTS[args.post]) for m in maps])
    hit, size = [], []
    for i in np.flatnonzero(y == 1):
        l, n = lab[i]
        for c in range(1, n + 1):
            r = l == c; hit.append(pred[i][r].mean() >= 0.2); size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    d = y == 1; tpd = (pred[d] & gt[d]).sum(); fpd = (pred[d] & ~gt[d]).sum(); fnd = (~pred[d] & gt[d]).sum()
    print(f" {q:5.2f}     {hit[size < SMALL].mean()*100:5.1f}  {hit[size >= SMALL].mean()*100:5.1f}  "
          f"{pred[y == 0].reshape((y == 0).sum(), -1).any(1).mean()*100:5.1f}   {200*tp/max(1,2*tp+fp+fn):5.1f}  {200*tpd/max(1,2*tpd+fpd+fnd):5.1f}")
