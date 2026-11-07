"""Evaluate the silhouette residual maps (wp_silhouette.py): thresholds on residual blob area from validation photos;
region recall by size, good-photo false alarms, which defective parts are caught; residual near vs away from defect."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from wp_local import gt_of, SMALL  # noqa: E402
from industrial_inspection.q2 import parts_and_tags  # noqa: E402
name = sys.argv[1] if len(sys.argv) > 1 else "sil"
z = np.load(f"results/maps/{name}__wallplugs.npz"); v = np.load(f"results/maps/{name}__wallplugs__val.npz")["maps"].astype(np.float32)
S = z["maps"].astype(np.float32); y = z["labels"]; paths = z["paths"].astype(str); gt = gt_of(paths, y, S.shape[1:]); parts, tag = parts_and_tags(paths)
vmax = v.reshape(len(v), -1).max(1)
print(f"val per-photo max residual area: median {np.median(vmax):.0f} q90 {np.quantile(vmax, .9):.0f} max {vmax.max():.0f}")
for q in (0.5, 0.7, 0.9, 1.0):
    t = np.quantile(vmax, q); pred = S > t; hit, size, who = [], [], []
    for i in np.flatnonzero(y == 1):
        l, n = ndimage.label(gt[i])
        for c in range(1, n + 1):
            r = l == c; hit.append(pred[i][r].mean() >= 0.2); size.append(r.sum()); who.append(parts[i].split("/")[-1])
    hit, size = np.array(hit), np.array(size)
    fa = pred[y == 0].reshape((y == 0).sum(), -1).any(1).mean() * 100
    print(f"thr q{q} (area>{t:.0f}): small {hit[size < SMALL].mean()*100:.0f}% large {hit[size >= SMALL].mean()*100:.0f}% FA {fa:.0f}% | parts caught: {sorted(set(w for w, h in zip(who, hit) if h))}")
for p in ["000", "001", "005", "007", "002", "004", "012", "010"]:
    i = next(j for j, q in enumerate(paths) if q.replace("\\", "/").endswith(f"bad/{p}_regular.png"))
    d = ndimage.binary_dilation(gt[i], iterations=6)
    print(f"{p}: residual area near defect {S[i][d].max():.0f}, elsewhere {S[i][~d].max():.0f}")
