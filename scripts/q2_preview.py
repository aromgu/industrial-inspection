"""Look before training: what each Q2 input normalisation and the lighting augmentation do to the
same good part under every capture condition. A normalisation that works should make the columns of
its row look alike.

    uv run python scripts/q2_preview.py --category walnuts
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from industrial_inspection.lighting import NORMS, light_field, normalise, relight
from industrial_inspection.plots import q2_preview_figure
from industrial_inspection.q2 import TAG_RE


def load(path: Path, size: int) -> torch.Tensor:
    x = TF.to_dtype(TF.to_image(Image.open(path).convert("RGB")), torch.float32, scale=True)
    return TF.resize(x, [size, size], antialias=True)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.permute(1, 2, 0).clamp(0, 1).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--part", default=None, help="leading number of the good part to show (default: the first)")
    args = p.parse_args()

    good = sorted((args.root / args.category / "test_public" / "good").glob("*.png"))
    part = args.part or good[0].name.split("_")[0]
    by_tag = {}
    for f in good:
        m = re.match(TAG_RE, f.name)
        if f.name.split("_")[0] == part and m:
            by_tag.setdefault(m.group(1), f)
    tags = sorted(by_tag, key=lambda t: (t != "regular", t))
    batch = torch.stack([load(by_tag[t], args.size) for t in tags])

    rows = [("original", [to_np(x) for x in batch])]
    for kind in NORMS:
        rows.append((kind, [to_np(x) for x in normalise(batch, kind)]))
    reg = batch[0]
    g = torch.Generator().manual_seed(0)
    aug = [relight(reg, *light_field(*reg.shape[-2:], generator=g)) for _ in tags]
    rows.append(("LocalLight\n(regular, random)", [to_np(x) for x in aug]))
    out = Path("results/figures") / f"q2_preview__{args.category}.png"
    print("figure:", q2_preview_figure(rows, tags, f"Q2 input normalisations and lighting augmentation - {args.category}, good part {part}", out))


if __name__ == "__main__":
    main()
