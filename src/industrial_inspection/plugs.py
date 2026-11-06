"""Instance-level view of wallplugs: segment every plug, then look at each plug on its own.

wallplugs photos scatter 20-40 plugs at random angles, often overlapping; the defect is a broken end
of one of them. Patch matching over the whole photo (HiResKNN) compares a patch with any normal
patch, whatever part of whichever plug it came from. Here each plug is cut out, rotated so its long
axis is horizontal, and compared position by position with normal plugs cut out the same way.

This module holds the segmentation step: SAM's automatic mask generator on a downscaled photo,
filtered to masks that look like one whole plug (area and elongation close to a single plug).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class Plug:
    mask: np.ndarray          # bool, at the working resolution
    center: tuple[float, float]
    size: tuple[float, float]  # (length, width) of the min-area rectangle, length >= width
    angle: float              # degrees, of the long axis
    score: float              # SAM's predicted IoU


def load_sam(checkpoint: str, model_type: str = "vit_b", device: str = "cuda", points_per_side: int = 32):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamAutomaticMaskGenerator(sam, points_per_side=points_per_side, pred_iou_thresh=0.86,
                                     stability_score_thresh=0.9, min_mask_region_area=200)


def rect_of(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    ys, xs = np.nonzero(mask)
    (cx, cy), (w, h), a = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    if w < h:
        w, h, a = h, w, a + 90
    return (cx, cy), (w, h), a % 180


def segment_plugs(generator, image: np.ndarray, plug_area: float, area_tol: tuple[float, float] = (0.5, 1.6),
                  min_elongation: float = 2.2, max_overlap: float = 0.5) -> list[Plug]:
    """Masks that look like one plug: area within area_tol x plug_area, long/short side >= min_elongation.
    Overlapping candidates keep the higher-scoring one (masks overlapping more than max_overlap of the smaller)."""
    cands = []
    for m in generator.generate(image):
        seg = m["segmentation"]
        area = seg.sum()
        if not (area_tol[0] * plug_area <= area <= area_tol[1] * plug_area):
            continue
        c, (L, W), a = rect_of(seg)
        if W == 0 or L / W < min_elongation:
            continue
        cands.append(Plug(seg, c, (L, W), a, float(m["predicted_iou"])))
    cands.sort(key=lambda p: -p.score)
    kept: list[Plug] = []
    for p in cands:
        if all((p.mask & q.mask).sum() <= max_overlap * min(p.mask.sum(), q.mask.sum()) for q in kept):
            kept.append(p)
    return kept


def read_rgb(path: str, long_side: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    s = long_side / max(im.size)
    return np.asarray(im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR))


CROP = (448, 126)  # canonical crop (length, width) in pixels; multiples of the ViT patch (14)


def crop_affine(plug: Plug, scale_to_src: float, plug_length: float, crop: tuple[int, int] = CROP,
                margin: float = 1.15) -> np.ndarray:
    """2x3 affine from source-image coordinates to the canonical crop: the plug's long axis horizontal and
    centred. Every plug gets the same scale (plug_length, the typical whole-plug length at working
    resolution, fills the crop x margin), so a plug cut off by the border or by a break keeps its true
    size instead of being blown up. scale_to_src maps working-resolution geometry to the source image."""
    (cx, cy), a = plug.center, plug.angle
    cx, cy = cx * scale_to_src, cy * scale_to_src
    s = crop[0] / (plug_length * scale_to_src * margin)
    M = cv2.getRotationMatrix2D((cx, cy), a, s)
    M[:, 2] += np.array([crop[0] / 2 - cx, crop[1] / 2 - cy])
    return M


def canonical_crop(image: np.ndarray, plug: Plug, scale_to_src: float, plug_length: float,
                   crop: tuple[int, int] = CROP, mask_dilate: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(crop with everything outside this plug's mask blacked out, affine source->crop)."""
    M = crop_affine(plug, scale_to_src, plug_length, crop)
    out = cv2.warpAffine(image, M, crop, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    Mm = M.copy()
    Mm[:, :2] = M[:, :2] * scale_to_src  # the mask lives at working resolution
    m = cv2.warpAffine(plug.mask.astype(np.uint8), Mm, crop, flags=cv2.INTER_NEAREST)
    if mask_dilate:
        m = cv2.dilate(m, np.ones((mask_dilate, mask_dilate), np.uint8))
    return out * m[..., None].astype(out.dtype), M


def save_plugs(path, plugs: list[Plug], shape: tuple[int, int]) -> None:
    np.savez_compressed(path, shape=np.array(shape), centers=np.array([p.center for p in plugs]).reshape(-1, 2),
                        sizes=np.array([p.size for p in plugs]).reshape(-1, 2), angles=np.array([p.angle for p in plugs]),
                        scores=np.array([p.score for p in plugs]),
                        masks=np.packbits(np.stack([p.mask for p in plugs]) if plugs else np.zeros((0, *shape), bool), axis=-1))


def load_plugs(path) -> list[Plug]:
    z = np.load(path)
    h, w = z["shape"]
    masks = np.unpackbits(z["masks"], axis=-1, count=w).astype(bool)
    return [Plug(m, tuple(c), tuple(sz), float(a), float(sc))
            for m, c, sz, a, sc in zip(masks, z["centers"], z["sizes"], z["angles"], z["scores"])]
