"""wallplugs, local-defect view of saved maps (every defect REGION counts once, whatever its size).
  aupro30 / aupro05  AU-PRO up to 30 % / 5 % normal-pixel FPR, normalised to [0, 100] (MVTec's region metric)
  At a threshold set on VALIDATION photos so that a share `fa` of good validation photos has any pixel above it
  (threshold = (1 - fa) quantile of the per-photo maxima; fa = 10 % and 30 %):
    rec_s / rec_l   share of small (< 500 px at 1/4 res) / large GT regions with >= 20 % of their pixels flagged
    fa_test         share of good TEST photos with any flagged pixel
    f1_all          pooled pixel F1 over all test photos (benchmark-like)
Options: --sigma S re-smooths maps saved without smoothing (hires.py --sigma 0) before scoring.
    .venv/Scripts/python.exe scripts/wp_local.py --models hires_dv3_k1 hires_dv3_k1_th31
Writes results/seg/wp_local.csv (appends) and prints a table."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage
import sys
sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.segeval import ground_truth  # noqa: E402

SMALL = 500
_GT = {}


def gt_of(paths, y, hw):
    key = (len(paths), hw)
    if key not in _GT:
        _GT[key] = np.stack([ground_truth(p, hw) if l else np.zeros(hw, bool) for p, l in zip(paths, y)])
    return _GT[key]


def evaluate(maps, vmaps, gt, y, rng):
    normal = maps[~gt]
    sub = rng.choice(normal.size, min(5_000_000, normal.size), replace=False)
    normal = np.sort(normal.ravel()[sub])
    comps = []
    for i in np.flatnonzero(y == 1):
        lab, n = ndimage.label(gt[i])
        for c in range(1, n + 1):
            comps.append(maps[i][lab == c])
    sizes = np.array([len(c) for c in comps])
    out = {}
    for lim in (0.30, 0.05):
        f = np.linspace(0, lim, 101)
        t = normal[np.clip(((1 - f) * len(normal)).astype(int), 0, len(normal) - 1)]
        pro = np.array([np.mean([np.mean(c > tt) for c in comps]) for tt in t])
        out[f"aupro{int(lim*100):02d}"] = np.trapezoid(pro, f) / lim * 100
    vmax = vmaps.reshape(len(vmaps), -1).max(1)
    tp_all = None
    for fa in (0.1, 0.3):
        t = np.quantile(vmax, 1 - fa)
        hit = np.array([np.mean(c > t) >= 0.2 for c in comps])
        pred = maps > t
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        out[f"rec_s@{int(fa*100)}"] = hit[sizes < SMALL].mean() * 100
        out[f"rec_l@{int(fa*100)}"] = hit[sizes >= SMALL].mean() * 100
        out[f"fa_test@{int(fa*100)}"] = pred[y == 0].reshape((y == 0).sum(), -1).any(1).mean() * 100
        out[f"f1_all@{int(fa*100)}"] = 200 * tp / max(1, 2 * tp + fp + fn)
    out["n_small"], out["n_large"] = int((sizes < SMALL).sum()), int((sizes >= SMALL).sum())
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--sigma", type=float, nargs="+", default=[None]); args = a.parse_args()
    rng = np.random.default_rng(0); rows = []
    for model in args.models:
        z = np.load(f"results/maps/{model}__wallplugs.npz"); zv = np.load(f"results/maps/{model}__wallplugs__val.npz")
        paths, y = z["paths"].astype(str), z["labels"]
        for s in args.sigma:
            maps, vmaps = z["maps"].astype(np.float32), zv["maps"].astype(np.float32)
            if s:
                maps = np.stack([ndimage.gaussian_filter(m, s) for m in maps]); vmaps = np.stack([ndimage.gaussian_filter(m, s) for m in vmaps])
            gt = gt_of(paths, y, maps.shape[1:])
            r = dict(model=model if s is None else f"{model} s{s}", **evaluate(maps, vmaps, gt, y, rng)); rows.append(r)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    f = Path("results/seg/wp_local.csv"); df.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
