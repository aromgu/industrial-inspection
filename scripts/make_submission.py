"""Build the benchmark.mvtec.com submission from the private-test maps already exported by
hires_private.py (7 categories) and plugs_knn.py/plugs_fuse.py --split test_private[_mixed] (wallplugs).

Per category/split, picks the threshold + post-processing settled by seg_post.py / the wallplugs
ablations (docs: README.md "wallplugs: 플러그를 하나씩 잘라서 비교하기"), both computed only from that
category's *validation* maps (never from a test split, private or public):

  7 categories  hires_k50, threshold = validation 95th percentile x 1.2 (SuperADD-style rule, the
                setting chosen across all 8 categories in seg_post.py), post = close(r=6) + fill holes
                + drop components < 64px
  wallplugs     plugs_rect_max, threshold = mean over validation images of that image's map maximum
                (the fixed rule needed because the optimal threshold sits at the extreme tail of the
                map -- percentile rules miss it by a wide margin, see README), post = fill holes +
                drop components < 256px

Writes submission_dir/anomaly_images/<cat>/<split>/<stem>.tiff (float16) and
submission_dir/anomaly_images_thresholded/<cat>/<split>/<stem>.png (0/255), validates file counts and
formats against the official per-category counts, and packages submission_dir into a .tar.gz.

    .venv/Scripts/python.exe scripts/make_submission.py
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from seg_post import POSTS, postprocess

SPLITS = ["test_private", "test_private_mixed"]
EXPECTED_COUNTS = {"can": 321, "fabric": 314, "fruit_jelly": 255, "rice": 277, "sheet_metal": 142,
                   "vial": 276, "wallplugs": 232, "walnuts": 228}

# category -> (model name, threshold rule, post-processing)
P95X12 = POSTS["close6+fill+small64"]
WALLPLUGS_POST = POSTS["fill+small256"]
CONFIG = {cat: ("hires_k50", "p95x1.2", P95X12) for cat in EXPECTED_COUNTS if cat != "wallplugs"}
CONFIG["wallplugs"] = ("plugs_rect_max", "valimgmax", WALLPLUGS_POST)


def threshold_of(rule: str, val_maps: np.ndarray) -> float:
    if rule == "p95x1.2":
        return float(np.quantile(val_maps, 0.95) * 1.2)
    if rule == "valimgmax":
        return float(val_maps.reshape(len(val_maps), -1).max(1).mean())
    raise ValueError(rule)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/submission"))
    p.add_argument("--archive", type=Path, default=Path("results/submission.tar.gz"))
    args = p.parse_args()

    img_root = args.out / "anomaly_images"
    thr_root = args.out / "anomaly_images_thresholded"
    for cat, (model, rule, (close, fill, min_area)) in CONFIG.items():
        val = np.load(f"results/maps/{model}__{cat}__val.npz")["maps"].astype(np.float32)
        thr = threshold_of(rule, val)
        for split in SPLITS:
            z = np.load(f"results/maps/{model}__{cat}__{split}.npz")
            maps, paths = z["maps"].astype(np.float32), z["paths"].astype(str)
            (img_root / cat / split).mkdir(parents=True, exist_ok=True)
            (thr_root / cat / split).mkdir(parents=True, exist_ok=True)
            for m, path in zip(maps, paths):
                stem = Path(path).stem
                tifffile.imwrite(img_root / cat / split / f"{stem}.tiff", m.astype(np.float16))
                binary = postprocess(m > thr, close, fill, min_area)
                Image.fromarray((binary * 255).astype(np.uint8)).save(thr_root / cat / split / f"{stem}.png")
        print(f"{cat}: model {model}, threshold {rule}={thr:.4g}, post {CONFIG[cat][2]}", flush=True)

    n_bad = validate(args.out)
    if n_bad:
        print(f"{n_bad} problem(s) found, see above -- not archiving", flush=True)
        return
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(args.out, arcname=args.out.name)
    print(f"wrote {args.archive} ({args.archive.stat().st_size / 1e6:.0f} MB)", flush=True)


def validate(root: Path) -> int:
    """Re-checks what benchmark.mvtec.com's own checker checks: per-category/-split file counts, name
    pattern, TIFF single-channel float16, PNG single-channel {0, 255}. Returns the number of problems."""
    import re

    problems = []
    pattern = re.compile(r"^(\d{3})_(regular|mixed)$")
    for cat, n in EXPECTED_COUNTS.items():
        for split, suffix in zip(SPLITS, ("regular", "mixed")):
            tiffs = sorted((root / "anomaly_images" / cat / split).glob("*.tiff"))
            pngs = sorted((root / "anomaly_images_thresholded" / cat / split).glob("*.png"))
            if len(tiffs) != n:
                problems.append(f"{cat}/{split}: expected {n} tiffs, found {len(tiffs)}")
            if len(pngs) != n:
                problems.append(f"{cat}/{split}: expected {n} pngs, found {len(pngs)}")
            seen = set()
            for f in tiffs:
                m = pattern.match(f.stem)
                if not m or m.group(2) != suffix:
                    problems.append(f"{f}: bad name")
                    continue
                seen.add(int(m.group(1)))
                arr = tifffile.imread(f)
                if arr.ndim != 2 or arr.dtype != np.float16:
                    problems.append(f"{f}: expected 2D float16, got {arr.ndim}D {arr.dtype}")
            if seen != set(range(n)):
                problems.append(f"{cat}/{split}: index set mismatch, missing {sorted(set(range(n)) - seen)[:5]}")
            for f in pngs:
                arr = np.asarray(Image.open(f))
                if arr.ndim != 2 or not set(np.unique(arr).tolist()).issubset({0, 255}):
                    problems.append(f"{f}: expected 2D {{0,255}}, got {arr.ndim}D values {np.unique(arr)[:5]}")
    for pr in problems:
        print("PROBLEM:", pr, flush=True)
    if not problems:
        print("validation OK: all counts, names and formats match.", flush=True)
    return len(problems)


if __name__ == "__main__":
    main()
