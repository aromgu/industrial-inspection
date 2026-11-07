"""wallplugs two-stage (coarse-to-fine) scoring: stage 1 proposes candidate regions, stage 2 -- a second,
finer look (higher input resolution, or plug-level comparison) -- re-scores only inside them.
  1. both maps get the same per-photo normalisation (--norm, wp_localnorm.apply; e.g. md61+iz10)
  2. stage-2 values are quantile-matched to stage 1 on the validation pixels (same scale)
  3. candidates: stage-1 pixels above the --cand-q quantile of stage-1 validation pixels, dilated by --dilate px
  4. inside candidates the score becomes combine(s1, s2) (--mode min|mean|geo|s2); outside it stays s1
Writes results/maps/<name>__wallplugs(.npz|__val.npz) for wp_local.py / wp_tune.py.
    .venv/Scripts/python.exe scripts/wp_cascade.py --s1 hires_dv3_k1 --s2 hires_dv3r1792_k1 --mode min --name cas_r1792_min"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from wp_localnorm import apply  # noqa: E402


def load(model, norm, sigma=0.0):
    z = dict(np.load(f"results/maps/{model}__wallplugs.npz")); zv = dict(np.load(f"results/maps/{model}__wallplugs__val.npz"))
    sm = (lambda m: ndimage.gaussian_filter(m, sigma)) if sigma else (lambda m: m)
    f = (lambda m: apply(sm(m.astype(np.float32)), norm)) if norm else (lambda m: sm(m.astype(np.float32)))
    return z, zv, np.stack([f(m) for m in z["maps"]]), np.stack([f(m) for m in zv["maps"]])


def qmatch(x, src_val, dst_val):
    qs = np.linspace(0, 1, 2001)
    return np.interp(x, np.quantile(src_val, qs), np.quantile(dst_val, qs))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--s1", required=True); a.add_argument("--s2", required=True); a.add_argument("--name", required=True)
    a.add_argument("--norm", default="md61+iz10"); a.add_argument("--sigma", type=float, nargs=2, default=[0, 0]); a.add_argument("--cand-q", type=float, default=0.99)
    a.add_argument("--dilate", type=int, default=4); a.add_argument("--mode", default="min", choices=["min", "mean", "geo", "s2", "max"])
    args = a.parse_args()
    z1, zv1, t1, v1 = load(args.s1, args.norm, args.sigma[0])
    z2, zv2, t2, v2 = load(args.s2, args.norm, args.sigma[1])
    assert (z1["paths"] == z2["paths"]).all() and t1.shape == t2.shape
    t2, v2 = qmatch(t2, v2.ravel(), v1.ravel()), qmatch(v2, v2.ravel(), v1.ravel())
    thr = np.quantile(v1, args.cand_q)
    st = np.ones((2 * args.dilate + 1,) * 2, bool)

    def comb(m1, m2):
        cand = ndimage.binary_dilation(m1 > thr, st) if args.dilate else m1 > thr
        if args.mode == "min":
            c = np.minimum(m1, m2)
        elif args.mode == "mean":
            c = (m1 + m2) / 2
        elif args.mode == "max":
            c = np.maximum(m1, m2)
        elif args.mode == "geo":
            lo = min(m1.min(), m2.min())
            c = np.sqrt((m1 - lo) * (m2 - lo)) + lo
        else:
            c = m2
        return np.where(cand, c, m1).astype(np.float16)

    tm = np.stack([comb(a_, b_) for a_, b_ in zip(t1, t2)]); vm = np.stack([comb(a_, b_) for a_, b_ in zip(v1, v2)])
    np.savez_compressed(f"results/maps/{args.name}__wallplugs.npz", **{**z1, "maps": tm})
    np.savez_compressed(f"results/maps/{args.name}__wallplugs__val.npz", **{**zv1, "maps": vm})
    print("saved", args.name)


if __name__ == "__main__":
    main()
