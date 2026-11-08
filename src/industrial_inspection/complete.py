"""wallplugs seeded shape completion (scripts/wp_complete.py is the experiment; this is the reusable rule).

A cut-off plug's ground truth is the MISSING part (background in the photo). Where a predicted component lies on a
SAM plug mask as a band across the plug's axis at one of its ends (a break edge), the prediction is extended beyond
that end along the axis, plug width, to the typical whole-plug length (or by default_ext x that length when the plug
is cut by the photo border and its length is unknown), on background pixels only. Never adds a detection to a photo
without one.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from industrial_inspection.plugs import load_plugs

LMED = 256.5  # median whole-plug length of the training plugs at SAM working resolution (long side 1024)


def plug_cache_path(photo: str, root: str = "results/plugs_cache/wallplugs") -> Path:
    """Where scripts/plugs_knn.py --cache-only stored the SAM masks of this photo."""
    p = Path(photo)
    return Path(root) / p.parent.parent.name / p.parent.name / (p.stem + ".npz")


def _fg(path, hw):
    im = np.asarray(Image.open(path).convert("L").resize(hw[::-1], Image.BILINEAR))
    _, b = cv2.threshold(cv2.GaussianBlur(im, (0, 0), 2), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


def _rect(c, u, v, end, sgn, width, length, H, W):
    a0, a1 = c + u * end, c + u * (end + sgn * length)
    h = v * width / 2
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.array([a0 + h, a1 + h, a1 - h, a0 - h])).astype(np.int32)], 1)
    return m.astype(bool)


def seeded_completion(photo: str, cache: str | Path, pred: np.ndarray, across: float = 0.7, band: float = 1.0,
                      end_frac: float = 0.15, default_ext: float = 0.3, per_component: bool = False) -> np.ndarray:
    """Extra predicted pixels (bool, pred's shape) for one photo. per_component: judge each connected component of
    pred on each plug separately (for seed maps that also fire weakly along whole outlines)."""
    hw = pred.shape
    if not pred.any() or not Path(cache).exists():
        return np.zeros(hw, bool)
    pl = load_plugs(cache)
    if not pl:
        return np.zeros(hw, bool)
    H, W = pl[0].mask.shape
    fg = _fg(photo, (H, W))
    occ = np.any([p.mask for p in pl], 0)
    P = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    add = np.zeros((H, W), bool)
    if per_component:
        lab, n = ndimage.label(P)
        comps = [lab == k for k in range(1, n + 1)]
    else:
        comps = [P]
    for p in pl:
        ys, xs = np.nonzero(p.mask)
        c = np.array([xs.mean(), ys.mean()])
        X = np.stack([xs, ys], 1) - c
        u = np.linalg.svd(X, full_matrices=False)[2][0]
        v = np.array([-u[1], u[0]])
        t, s = X @ u, X @ v
        t0, t1, wd = t.min(), t.max(), np.percentile(s, 97) - np.percentile(s, 3)
        L = t1 - t0
        border = xs.min() < 3 or ys.min() < 3 or xs.max() > W - 4 or ys.max() > H - 4
        ext = max(LMED - L, 0) if not border else default_ext * LMED
        if ext < 8:
            continue
        for Pc in comps:
            hit = np.argwhere(Pc & p.mask)
            if not len(hit):
                continue
            Th = (hit[:, 1] - c[0]) * u[0] + (hit[:, 0] - c[1]) * u[1]
            Sh = (hit[:, 1] - c[0]) * v[0] + (hit[:, 0] - c[1]) * v[1]
            if band and np.ptp(Th) > band * max(np.ptp(Sh), 1):
                continue
            for sgn, end in ((1, t1), (-1, t0)):
                z = sgn * (Th - end) > -end_frac * L
                if not z.any() or np.ptp(Sh[z]) < across * wd:
                    continue
                add |= _rect(c, u, v, end, sgn, wd, ext, H, W) & ~fg & ~occ
    return cv2.resize(add.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
