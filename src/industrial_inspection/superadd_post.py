"""SuperADD's post-processing (VAND 4.0, arXiv 2605.14808; code: github.com/LukasRoom/SuperADD,
tracks/industrial/src/industrial/post_process.py), re-implemented so it can run on our maps.

  1. threshold the map
  2. multi-oriented closing: for each of n_angles line structuring elements of the given radius,
     a closing; the union of the closings and the thresholded map is kept
  3. keep only closed pixels whose score is above lower_factor x threshold
  4. fill closed regions (external contours drawn filled)
  5. binary erosion with an ellipse of radius `erosion`

Their defaults: radius 26, 16 angles, lower_factor 0.8, erosion 1, at 1/4 of the original resolution --
the same scale our maps are stored at.
"""

from __future__ import annotations

import cv2
import numpy as np


def _line_kernel(radius: int, angle_deg: float, thickness: int = 1) -> np.ndarray:
    size = 2 * radius + 1
    k = np.zeros((size, size), np.uint8)
    c = size // 2
    dx, dy = radius * np.cos(np.deg2rad(angle_deg)), radius * np.sin(np.deg2rad(angle_deg))
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, thickness)
    return k


def superadd_post(score: np.ndarray, threshold: float, radius: int = 26, n_angles: int = 16,
                  lower_factor: float | None = 0.8, erosion: int = 1) -> np.ndarray:
    """Binary map (bool) from a continuous map, as SuperADD makes its thresholded submission images."""
    th = np.where(score > threshold, 255, 0).astype(np.uint8)
    pad = radius + 1
    th_p = np.pad(th, pad)
    out = th_p.copy()
    thick = np.ones((2, 2), np.uint8)
    for i in range(n_angles):
        k1 = _line_kernel(radius, 180.0 * i / n_angles, thickness=2)
        k2 = cv2.morphologyEx(k1, cv2.MORPH_DILATE, thick)
        closed = cv2.morphologyEx(cv2.morphologyEx(th_p, cv2.MORPH_DILATE, k2), cv2.MORPH_ERODE, k1)
        out = np.maximum(out, closed)
    out = out[pad:-pad, pad:-pad]
    if lower_factor is not None:
        out = np.where(score > threshold * lower_factor, out, 0).astype(np.uint8)
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    if erosion:
        filled = cv2.morphologyEx(filled, cv2.MORPH_ERODE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1)))
    return filled > 0
