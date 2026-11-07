"""wallplugs hysteresis decision: WHERE from a clean per-photo-normalised map (seeds: --seed model above a
threshold at quantile --qs of its validation per-photo maxima), HOW FAR from the raw map (connected regions of
--extent model above its validation-pixel quantile --qe that contain a seed). Thresholds from validation only.
    .venv/Scripts/python.exe scripts/wp_hyst.py --seed r1792L_s2_iq90 --extent s0_r1792_L1523 --sigma 2"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of, SMALL  # noqa: E402

a = argparse.ArgumentParser(); a.add_argument("--seed", required=True); a.add_argument("--extent", required=True)
a.add_argument("--sigma", type=float, default=0, help="smoothing of the extent map (saved unsmoothed)")
a.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.9, 0.95]); a.add_argument("--qe", type=float, nargs="+", default=[0.99, 0.995, 0.999])
a.add_argument("--save", default=None, help="save binary predictions for qs[0], qe[0] under this name (npz)")
args = a.parse_args()
L = lambda m: (np.load(f"results/maps/{m}__wallplugs.npz"), np.load(f"results/maps/{m}__wallplugs__val.npz"))
zs, zsv = L(args.seed); ze, zev = L(args.extent)
S, Sv = zs["maps"].astype(np.float32), zsv["maps"].astype(np.float32)
sm = (lambda a_: np.stack([ndimage.gaussian_filter(m.astype(np.float32), args.sigma) for m in a_])) if args.sigma else (lambda a_: a_.astype(np.float32))
E, Ev = sm(ze["maps"]), sm(zev["maps"])
y = zs["labels"]; gt = gt_of(zs["paths"].astype(str), y, S.shape[1:]); lab = [ndimage.label(g) for g in gt]
print(f"seed={args.seed} extent={args.extent}\n qs     qe    rec_s  rec_l  fa_test  f1_all  f1_def")
for qs in args.qs:
    ts = np.quantile(Sv.reshape(len(Sv), -1).max(1), qs)
    for qe in args.qe:
        te = np.quantile(Ev, qe)
        pred = np.zeros_like(gt)
        for i in range(len(S)):
            seed = S[i] > ts
            if not seed.any():
                continue
            l, n = ndimage.label((E[i] > te) | seed)
            keep = np.unique(l[seed]); keep = keep[keep > 0]
            pred[i] = np.isin(l, keep)
        hit, size = [], []
        for i in np.flatnonzero(y == 1):
            l, n = lab[i]
            for c in range(1, n + 1):
                r = l == c; hit.append(pred[i][r].mean() >= 0.2); size.append(r.sum())
        hit, size = np.array(hit), np.array(size)
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        d = y == 1; tpd = (pred[d] & gt[d]).sum(); fpd = (pred[d] & ~gt[d]).sum(); fnd = (~pred[d] & gt[d]).sum()
        print(f" {qs:4.2f}  {qe:5.3f}  {hit[size < SMALL].mean()*100:5.1f}  {hit[size >= SMALL].mean()*100:5.1f}  "
              f"{pred[y == 0].reshape((y == 0).sum(), -1).any(1).mean()*100:5.1f}   {200*tp/max(1,2*tp+fp+fn):5.1f}  {200*tpd/max(1,2*tpd+fpd+fnd):5.1f}", flush=True)
        if args.save and qs == args.qs[0] and qe == args.qe[0]:
            np.savez_compressed(f"results/seg/{args.save}.npz", pred=np.packbits(pred, axis=-1), shape=np.array(pred.shape), paths=zs["paths"])
