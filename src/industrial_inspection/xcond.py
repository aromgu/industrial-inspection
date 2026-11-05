"""Cross-condition consistency: score a part from several photos of it taken under different conditions.

Idea (2026-11-05): the false alarms found so far (walnut shadows, the can's holographic label,
sheet-metal reflections) change with the capture condition, while a physical defect stays at the
same place in every photo. With the camera and part fixed, the photos of one part are aligned, so
taking at each location a low order statistic over the photos keeps what is anomalous in all of
them and drops what is anomalous in only some.

Maps: (n_photos, H, W) for one part. Every combiner returns a part score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _tolerate(maps: np.ndarray, radius: int) -> np.ndarray:
    """Max-pool each map over a (2r+1)^2 window, so a defect that moves by up to r cells between
    photos still overlaps itself."""
    if radius <= 0:
        return maps
    t = torch.from_numpy(maps.astype(np.float32))[:, None]
    return F.max_pool2d(t, 2 * radius + 1, stride=1, padding=radius)[:, 0].numpy()


def combine(maps: np.ndarray, how: str, radius: int = 0) -> float:
    """Part score from the aligned anomaly maps of its photos.

    image-mean   mean of the per-photo image scores (map maxima): the plain multi-photo baseline
    image-max    alarm if any photo looks anomalous
    pixel-min    max over locations of the minimum over photos (anomalous in every photo)
    pixel-2nd    max over locations of the second-smallest value (anomalous in all but one photo)
    pixel-median max over locations of the median over photos
    """
    m = maps.astype(np.float32)
    if how == "image-mean":
        return float(m.reshape(len(m), -1).max(axis=1).mean())
    if how == "image-max":
        return float(m.max())
    m = _tolerate(m, radius)
    if how == "pixel-min":
        return float(m.min(axis=0).max())
    if how == "pixel-2nd":
        return float(np.sort(m, axis=0)[min(1, len(m) - 1)].max())
    if how == "pixel-median":
        return float(np.median(m, axis=0).max())
    raise ValueError(f"unknown combiner {how!r}")
