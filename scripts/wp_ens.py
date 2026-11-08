"""wallplugs map ensemble: each raw map z-scored by its own validation pixel mean/std, then averaged (weights
--w, default equal). Written as model <out> (unsmoothed, like the inputs) for wp_eval.py.
    .venv/Scripts/python.exe scripts/wp_ens.py --models s0_r1792_L1523 s0_r1792_clahe2 s0_r1792_unsharp1 --out ens3"""
import argparse
import numpy as np
a = argparse.ArgumentParser(); a.add_argument("--models", nargs="+", required=True); a.add_argument("--w", type=float, nargs="+", default=None)
a.add_argument("--out", required=True); args = a.parse_args()
w = np.array(args.w if args.w else [1.0] * len(args.models)); w = w / w.sum()
acc_t = acc_v = None
for m, wi in zip(args.models, w):
    z = dict(np.load(f"results/maps/{m}__wallplugs.npz")); zv = dict(np.load(f"results/maps/{m}__wallplugs__val.npz"))
    v = zv["maps"].astype(np.float32); mu, sd = v.mean(), v.std()
    t = (z["maps"].astype(np.float32) - mu) / sd * wi; vv = (v - mu) / sd * wi
    acc_t = t if acc_t is None else acc_t + t; acc_v = vv if acc_v is None else acc_v + vv
    base, basev = z, zv
np.savez_compressed(f"results/maps/{args.out}__wallplugs.npz", **{**base, "maps": acc_t.astype(np.float16)})
np.savez_compressed(f"results/maps/{args.out}__wallplugs__val.npz", **{**basev, "maps": acc_v.astype(np.float16)})
print("saved", args.out)
