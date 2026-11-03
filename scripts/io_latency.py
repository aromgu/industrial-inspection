"""Time the part of a line inspection that latency.py leaves out: reading the camera-resolution
image from disk and shrinking it to the model's input size (CPU, one image at a time).

    uv run python scripts/io_latency.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from industrial_inspection.plots import io_latency_figure

CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("results/io_latency.csv"))
    args = p.parse_args()
    torch.set_num_threads(4)
    rows = []
    for c in CATEGORIES:
        paths = sorted((args.root / c / "test_public" / "good").glob("*.png"))[: args.images]
        for path in paths:
            t0 = time.perf_counter()
            im = Image.open(path).convert("RGB")
            t1 = time.perf_counter()
            x = TF.to_image(im)
            x = TF.to_dtype(x, torch.float32, scale=True)
            for size in (256, 392):
                t2 = time.perf_counter()
                TF.resize(x, [size, size], antialias=True)
                rows.append({"category": c, "size": size, "decode_ms": (t1 - t0) * 1000,
                             "resize_ms": (time.perf_counter() - t2) * 1000, "megapixels": im.width * im.height / 1e6})
    df = pd.DataFrame(rows)
    summary = df.groupby(["category", "size"]).agg(megapixels=("megapixels", "first"), decode_ms=("decode_ms", "median"),
                                                  resize_ms=("resize_ms", "median")).reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    with pd.option_context("display.float_format", "{:.1f}".format):
        print(summary.to_string(index=False))
    print("figure:", io_latency_figure(summary))


if __name__ == "__main__":
    main()
