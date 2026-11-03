import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from industrial_inspection.lighting import NORMS, LightNorm, LocalLight, normalise
from industrial_inspection.q2 import split_parts


@pytest.mark.parametrize("kind", NORMS)
def test_normalise_keeps_shape_and_range(kind):
    x = torch.rand(2, 3, 64, 48)
    y = normalise(x, kind)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_standardize_removes_global_brightness():
    x = torch.rand(1, 3, 32, 32) * 0.5
    assert torch.allclose(normalise(x, "standardize"), normalise(x * 0.6 + 0.1, "standardize"), atol=1e-4)


def test_transforms_leave_masks_alone():
    img = tv_tensors.Image(torch.rand(3, 32, 32))
    mask = tv_tensors.Mask(torch.zeros(1, 32, 32, dtype=torch.bool))
    for t in (LightNorm("clahe"), LocalLight(p=1.0)):
        out_img, out_mask = t(img, mask)
        assert isinstance(out_img, tv_tensors.Image) and out_img.shape == img.shape
        assert torch.equal(out_mask, mask)


def test_dev_and_eval_halves_share_no_part():
    parts = np.array([f"good/{i // 6}" for i in range(36)] + [f"bad/{i // 6}" for i in range(60)])
    labels = np.array([0] * 36 + [1] * 60)
    dev = split_parts(parts, labels)
    assert not set(parts[dev]) & set(parts[~dev])
    assert dev[labels == 0].sum() == 18 and dev[labels == 1].sum() == 30
