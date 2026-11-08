"""wallplugs item 3b: shape completion seeded ALSO by the straight-cut classifier (synth_cut6, wp_synth.py --kinds 6),
which fires on real break edges (000, even under shift light) but whose maps never covered the GT (missing region).
Cut seeds = Gaussian(2) logit map above the --qcut quantile of the validation per-photo maxima; they are not added to
the prediction themselves, only their completion (industrial_inspection.complete.seeded_completion, band/across checks),
so only an extension onto background beyond a plug end can appear. Compared with final and final + completion.
    .venv/Scripts/python.exe scripts/wp_complete_cut.py [--qcut 0.9 0.95 1.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.complete import plug_cache_path, seeded_completion  # noqa: E402
from industrial_inspection.q2 import parts_and_tags, split_parts  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402
from wp_local import SMALL, gt_of  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--cut", default="synth_cut6")
a.add_argument("--qcut", type=float, nargs="+", default=[0.9, 0.95, 1.0])
a.add_argument("--cut-across", type=float, default=0.5, help="width fraction a cut seed must span (break edges are often partly shaded)")
args = a.parse_args()


def metrics(pred, gt, y, keep):
    hit, size = [], []
    for i in np.flatnonzero(keep & (y == 1)):
        lab, n = ndimage.label(gt[i])
        for c in range(1, n + 1):
            r = lab == c
            hit.append(pred[i][r].mean() >= 0.2)
            size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    P, G, yy = pred[keep], gt[keep], y[keep]
    f1 = lambda m: 200 * (P[m] & G[m]).sum() / max(1, 2 * (P[m] & G[m]).sum() + (P[m] & ~G[m]).sum() + (~P[m] & G[m]).sum())
    return (hit[size < SMALL].mean() * 100, hit[size >= SMALL].mean() * 100,
            P[yy == 0].reshape((yy == 0).sum(), -1).any(1).mean() * 100, f1(np.ones(len(yy), bool)), f1(yy == 1))


z, zv = np.load("results/maps/s0_r1792_L1523__wallplugs.npz"), np.load("results/maps/s0_r1792_L1523__wallplugs__val.npz")
paths, y = z["paths"].astype(str), z["labels"]
_, pred, _ = wallplugs_masks(z["maps"], paths, zv["maps"], zv["paths"].astype(str))
gt = gt_of(paths, y, pred.shape[1:])
parts, _ = parts_and_tags(paths)
dev = split_parts(parts, y)
comp = np.stack([pred[i] | seeded_completion(paths[i], plug_cache_path(paths[i]), pred[i]) for i in range(len(y))])
gs = lambda arr: np.stack([ndimage.gaussian_filter(m.astype(np.float32), 2.0) for m in arr])
C, Cv = gs(np.load(f"results/maps/{args.cut}__wallplugs.npz")["maps"]), gs(np.load(f"results/maps/{args.cut}__wallplugs__val.npz")["maps"])
vmax = Cv.reshape(len(Cv), -1).max(1)
halves = (("all", np.ones(len(y), bool)), ("dev", dev), ("eval", ~dev))
print("  variant                 half  rec_s  rec_l  fa     f1_all  f1_def")
compc = np.stack([pred[i] | seeded_completion(paths[i], plug_cache_path(paths[i]), pred[i], per_component=True) for i in range(len(y))])
rows = [("final", pred), ("final + completion", comp), ("final + compl. per-comp", compc)]
for q in args.qcut:
    tc = np.quantile(vmax, q) if q < 1 else vmax.max()
    extra = np.stack([seeded_completion(paths[i], plug_cache_path(paths[i]), C[i] > tc, across=args.cut_across, per_component=True) for i in range(len(y))])
    rows.append((f"+ cut seeds q{q}", comp | extra))
    print(f"  cut q{q}: threshold {tc:.2f}; photos extended {extra.reshape(len(y), -1).any(1).sum()} "
          f"(good {extra[y == 0].reshape((y == 0).sum(), -1).any(1).sum()}); on 000/001: "
          + " ".join(f"{Path(paths[i]).stem}:{(extra[i] & gt[i]).sum() / gt[i].sum():.2f}" for i in np.flatnonzero(y == 1)
                     if Path(paths[i]).stem[:3] in ("000", "001") and extra[i].any()))
for nm, P in rows:
    for h, keep in halves:
        print(f"  {nm:22s}  {h:4s}  " + "  ".join(f"{v:5.1f}" for v in metrics(P, gt, y, keep)))
