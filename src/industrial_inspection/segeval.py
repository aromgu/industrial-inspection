"""Segmentation F1 on MVTec AD 2 test_public, the metric of the official benchmark and the VAND challenges.

The official server scores pixel-level F1 of binarised anomaly maps on the private test sets; the
default threshold is the mean + 3 std of the per-pixel anomaly values of the good validation images
(MVTec AD 2 paper). Locally only test_public has ground truth. Two protocols are computed:

  defect-only  pixels of defective images only, at 1/4 of the original resolution, ground truth
               resized with bilinear interpolation and binarised at > 0. This is what SuperADD's
               released test script does (evaluate_good_images = false, evaluation_downscale = 4),
               so its numbers can be set against SuperADD's reported test_public mean of 62.61.
  all          good images included, so false alarms on good parts count, as on the server.

Anomaly maps are stored at the model's input grid (after its resize and centre crop, at most 256);
`full_frame` puts them back into the whole image before comparing with the masks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# (resize, centre crop) of each model family's pre-processor; None = no crop.
VIEW = {"dinomaly": (448, 392), "mhpatchcore": (256, 224)}
DOWNSCALE = 4


def view_of(model: str) -> tuple[int, int] | None:
    for prefix, v in VIEW.items():
        if model.startswith(prefix):
            return v
    return None


def full_frame(maps: np.ndarray, out_hw: tuple[int, int], view: tuple[int, int] | None) -> np.ndarray:
    """(N, h, w) maps on the model grid -> (N, H, W) maps over the whole image.

    With a centre crop the map covers only the central crop/resize of the (square-resized) image;
    the cut-off border is filled with each map's minimum, i.e. treated as normal.
    """
    t = torch.from_numpy(maps.astype(np.float32))[:, None]
    if view is not None:
        size, crop = view
        t = F.interpolate(t, size=(crop, crop), mode="bilinear", align_corners=False)
        pad = (size - crop) // 2
        fill = t.amin(dim=(2, 3), keepdim=True)
        canvas = fill.expand(-1, -1, size, size).clone()
        canvas[:, :, pad:pad + crop, pad:pad + crop] = t
        t = canvas
    return F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)[:, 0].numpy()


def mask_path(image_path: str) -> Path:
    p = Path(image_path)
    return p.parent.parent / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"


@lru_cache(maxsize=None)
def eval_size(image_path: str, downscale: int = DOWNSCALE) -> tuple[int, int]:
    w, h = Image.open(image_path).size
    return h // downscale, w // downscale


def ground_truth(image_path: str, out_hw: tuple[int, int]) -> np.ndarray:
    m = mask_path(image_path)
    if not m.exists():
        return np.zeros(out_hw, dtype=bool)
    small = Image.open(m).convert("L").resize(out_hw[::-1], Image.BILINEAR)
    return np.asarray(small) > 0


def f1_curve(scores: np.ndarray, truth: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """F1 at every threshold (a pixel is anomalous if score > threshold), via one histogram pass."""
    edges = np.concatenate([[-np.inf], np.sort(thresholds), [np.inf]])
    pos = np.histogram(scores[truth], bins=edges)[0]
    neg = np.histogram(scores[~truth], bins=edges)[0]
    tp = pos[::-1].cumsum()[::-1][1:]  # pixels above each threshold
    fp = neg[::-1].cumsum()[::-1][1:]
    fn = truth.sum() - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    order = np.argsort(np.argsort(thresholds))
    return f1[order]


def f1_at(scores: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = scores > threshold
    tp = np.sum(pred & truth)
    denom = 2 * tp + np.sum(pred & ~truth) + np.sum(~pred & truth)
    return float(2 * tp / denom) if denom else 0.0
