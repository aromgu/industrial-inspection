"""Lighting-robustness tools for research question Q2 (lighting changes cause overkill).

Q2 step 1 (q2_selfnorm.py) showed that the lighting effect on the anomaly map is local (shadows,
reflections), not a global offset. These transforms act on the input image instead:

* ``LightNorm``: input normalisation (H2-b). It runs inside the model's pre-processor on every
  train and test batch, after the resize and before the ImageNet normalisation, so the model only
  ever sees normalised images.
* ``LocalLight``: lighting augmentation of the training images (H2-c), so a memory bank or a
  reconstruction model also sees lit-differently versions of the normal parts.

Both take float images in [0, 1] of shape (..., 3, H, W) and leave masks untouched.
"""

from __future__ import annotations

import math
from typing import Any

import kornia
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

NORMS = ("clahe", "histeq", "standardize", "gray", "retinex")


def _batched(fn, x: torch.Tensor) -> torch.Tensor:
    """Run fn on a (B, 3, H, W) batch whether x is batched or a single (3, H, W) image."""
    return fn(x[None])[0] if x.ndim == 3 else fn(x)


def _on_lightness(x: torch.Tensor, fn) -> torch.Tensor:
    """Apply fn to the L channel of CIELAB (scaled to [0, 1]) and keep the colour channels."""
    lab = kornia.color.rgb_to_lab(x.clamp(0, 1))
    lightness = fn((lab[:, :1] / 100).clamp(0, 1)) * 100
    return kornia.color.lab_to_rgb(torch.cat([lightness, lab[:, 1:]], dim=1)).clamp(0, 1)


def normalise(x: torch.Tensor, kind: str) -> torch.Tensor:
    """One of NORMS on a (B, 3, H, W) float batch in [0, 1]."""
    if kind == "clahe":  # local contrast equalisation, 8x8 tiles (OpenCV's default clip 2.0 is ~ kornia's 40)
        return _on_lightness(x, lambda lum: kornia.enhance.equalize_clahe(lum, clip_limit=2.0, grid_size=(8, 8)))
    if kind == "histeq":  # global histogram equalisation
        return _on_lightness(x, kornia.enhance.equalize)
    if kind == "standardize":  # per image and channel: zero mean, fixed spread (removes global brightness/contrast)
        mean, std = x.mean(dim=(-2, -1), keepdim=True), x.std(dim=(-2, -1), keepdim=True)
        return ((x - mean) / (std + 1e-6) * 0.2 + 0.45).clamp(0, 1)
    if kind == "gray":  # drop colour, which lighting changes can shift
        return kornia.color.rgb_to_grayscale(x).expand_as(x).contiguous()
    if kind == "retinex":
        # Single-scale Retinex on lightness: divide out a smooth illumination estimate (Gaussian blur with
        # sigma = 1/16 of the image side), so slow shading and broad shadows cancel and texture stays.
        def flatten(lum: torch.Tensor) -> torch.Tensor:
            sigma = max(lum.shape[-2:]) / 16
            k = int(2 * math.ceil(3 * sigma) + 1)
            illum = kornia.filters.gaussian_blur2d(lum, (k, k), (sigma, sigma), border_type="reflect")
            out = lum / (illum + 1e-3) * illum.mean(dim=(-2, -1), keepdim=True)
            return out.clamp(0, 1)

        return _on_lightness(x, flatten)
    raise ValueError(f"unknown normalisation {kind!r}; choose from {NORMS}")


class LightNorm(Transform):
    """Input normalisation as a torchvision v2 transform (images only; masks pass through)."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in NORMS:
            raise ValueError(f"unknown normalisation {kind!r}; choose from {NORMS}")
        self.kind = kind

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.Mask) or not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        out = _batched(lambda b: normalise(b, self.kind), inpt.as_subclass(torch.Tensor))
        return tv_tensors.wrap(out, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else out


def light_field(h: int, w: int, generator: torch.Generator | None = None, device=None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Random multiplicative illumination field, additive highlight field (both (1, H, W)) and a gamma.

    Ranges follow what the test conditions do to good parts (data_profile.py: mean brightness moves
    -39 % to +18 % against "regular"), plus local effects the global jitter of patchcore_aug lacked:
      * directional ramp: 1 + a * (direction . position), a in [0.1, 0.4]
      * 0-2 soft shadows: multiply by 1 - d * gaussian, d in [0.2, 0.5], sigma 10-30 % of the side
      * 0-2 highlights: add s * gaussian, s in [0.1, 0.4], sigma 3-10 % of the side
      * gamma in [0.7, 1.4]
    The fields are drawn on a 64x64 grid and upsampled, so the cost does not grow with a 5 MP image.
    """
    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=generator))

    g = 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g), indexing="ij")
    theta, strength = u(0, 2 * math.pi), u(0.1, 0.4)
    gain = 1 + strength * (math.cos(theta) * xx + math.sin(theta) * yy)
    highlight = torch.zeros(g, g)

    def blob(sigma: float) -> torch.Tensor:
        cy, cx = u(-1, 1), u(-1, 1)
        return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (2 * sigma) ** 2))  # grid spans 2 units

    for _ in range(int(torch.randint(0, 3, (), generator=generator))):
        gain = gain * (1 - u(0.2, 0.5) * blob(u(0.10, 0.30)))
    for _ in range(int(torch.randint(0, 3, (), generator=generator))):
        highlight = highlight + u(0.1, 0.4) * blob(u(0.03, 0.10))
    fields = torch.stack([gain, highlight])[None]
    fields = F.interpolate(fields, size=(h, w), mode="bilinear", align_corners=True)[0].to(device)
    return fields[:1], fields[1:], u(0.7, 1.4)


def relight(x: torch.Tensor, gain: torch.Tensor, highlight: torch.Tensor, gamma: float) -> torch.Tensor:
    return ((x * gain).clamp(0, 1) ** gamma + highlight).clamp(0, 1)


class LocalLight(Transform):
    """Training augmentation: with probability p, relight the image with a random light_field()."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"apply": float(torch.rand(())) < self.p}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if (not params["apply"] or isinstance(inpt, tv_tensors.Mask) or not isinstance(inpt, torch.Tensor)
                or not inpt.is_floating_point()):
            return inpt
        gain, highlight, gamma = light_field(*inpt.shape[-2:], device=inpt.device)
        out = relight(inpt.as_subclass(torch.Tensor), gain, highlight, gamma)
        return tv_tensors.wrap(out, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else out
