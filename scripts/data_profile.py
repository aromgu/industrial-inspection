"""Profile MVTec AD 2: split sizes, capture conditions, defect geometry and brightness per category.

CPU only and low priority, so it can run beside a training job.

    uv run python scripts/data_profile.py --root datasets/MVTec_AD_2
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from industrial_inspection.plots import data_profile_figure

CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
TAG_RE = r"^\d+_(.+?)(?:-\d+)?\.png$"
KEEP = 0.875  # fraction of each side kept by the models' centre crop (224/256, 392/448)


def tags(paths: list[Path]) -> Counter:
    return Counter(pd.Series([p.name for p in paths]).str.extract(TAG_RE)[0].fillna("?"))


def profile(root: Path, category: str, brightness_samples: int) -> tuple[dict, pd.DataFrame]:
    d = root / category
    train = sorted((d / "train" / "good").glob("*.png"))
    val = sorted((d / "validation" / "good").glob("*.png"))
    good = sorted((d / "test_public" / "good").glob("*.png"))
    bad = sorted((d / "test_public" / "bad").glob("*.png"))
    first = Image.open(train[0])
    w, h = first.size

    defects = []
    for m in sorted((d / "test_public" / "ground_truth" / "bad").glob("*_mask.png")):
        a = np.asarray(Image.open(m).convert("L")) > 0
        H, W = a.shape
        y0, y1, x0, x1 = int(H * (1 - KEEP) / 2), int(H * (1 + KEEP) / 2), int(W * (1 - KEEP) / 2), int(W * (1 + KEEP) / 2)
        _, n_regions = ndimage.label(a)
        defects.append({
            "category": category,
            "area_frac": a.mean(),
            "regions": n_regions,
            "kept_in_crop": a[y0:y1, x0:x1].sum() / max(a.sum(), 1),
        })
    dd = pd.DataFrame(defects)
    # A square of the same area, measured on a 256 x 256 model input.
    dd["side_px_at_256"] = np.sqrt(dd.area_frac) * 256

    # Brightness of good test images per capture condition, relative to "regular".
    by_tag: dict[str, list[float]] = {}
    for p in good:
        tag = pd.Series([p.name]).str.extract(TAG_RE)[0].iloc[0]
        if len(by_tag.setdefault(tag, [])) < brightness_samples:
            im = Image.open(p).convert("L")
            im.thumbnail((512, 512))
            by_tag[tag].append(float(np.asarray(im).mean()))
    ref = np.mean(by_tag.get("regular", [np.nan]))
    brightness = {f"bright_{t}": np.mean(v) / ref for t, v in sorted(by_tag.items())}

    train_tags = tags(train)
    row = {
        "category": category,
        "mode": first.mode,
        "size": f"{w}x{h}",
        "aspect": w / h,
        "train": len(train),
        "val": len(val),
        "test_good": len(good),
        "test_bad": len(bad),
        "train_conditions": ", ".join(f"{k} {v}" for k, v in sorted(train_tags.items())),
        "test_conditions": len(tags(good + bad)),
        "defect_area_median_pct": dd.area_frac.median() * 100,
        "defect_side_px_median": dd.side_px_at_256.median(),
        "defect_side_px_p10": dd.side_px_at_256.quantile(0.1),
        "defect_side_px_p90": dd.side_px_at_256.quantile(0.9),
        "multi_region_defects": int((dd.regions > 1).sum()),
        "defects_lost_to_crop": int((dd.kept_in_crop == 0).sum()),
        "defects_half_lost_to_crop": int((dd.kept_in_crop < 0.5).sum()),
        **brightness,
    }
    return row, dd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--brightness-samples", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/data_profile.csv"))
    args = p.parse_args()
    if sys.platform == "win32":  # BELOW_NORMAL: stay out of the way of training and games
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)

    rows, defects = [], []
    for c in CATEGORIES:
        row, dd = profile(args.root, c, args.brightness_samples)
        rows.append(row)
        defects.append(dd)
        print(f"{c}: done", flush=True)
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(table.drop(columns=["train_conditions"]).to_string(index=False))
        print(table[["category", "train_conditions"]].to_string(index=False))
    print("figure:", data_profile_figure(table, pd.concat(defects)))


if __name__ == "__main__":
    main()
