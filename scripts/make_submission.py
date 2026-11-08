"""Build the benchmark.mvtec.com submission from private-test maps written by
scripts/hires.py --private (DINOv3-H+, 2026-11-09):

  7 categories  sub_dv3 (layers 7 15 23 31, 640 px tiles, k=1), threshold = validation p95 x 1.2
                (x 1.421 for rice, walnuts), post = close(r=6) + fill holes + drop components < 64 px
                (choice on test_public: scripts/sub_compare.py)
  wallplugs     sub_wp (layers 15 23, short side 1792, k=1), final rule of docs/experiment_log.md 2026-11-08:
                Gaussian-seed hysteresis with photo-p90 offset, guided-filter extent (wallplugs_masks)

All thresholds from each category's validation maps only. Writes submission_dir/anomaly_images/<cat>/<split>/<stem>.tiff
(float16) and anomaly_images_thresholded/<cat>/<split>/<stem>.png (0/255), validates counts and formats against the
official checker's rules, and packages a .tar.gz. Upload only after the user confirms (2 uploads per 7 days).

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

# category -> (model name, threshold rule, post-processing). 2026-11-09: DINOv3-H+ for every category (better than
# DINOv2-L on test_public in all 7 others, scripts/sub_compare.py); threshold factor per category from validation p95
# (x1.421 = SuperADD's factor where it scored higher on test_public, else x1.2). wallplugs: final config of
# docs/experiment_log.md (2026-11-08), see wallplugs_masks().
P95X12 = POSTS["close6+fill+small64"]
CONFIG = {cat: ("sub_dv3", "p95x1.2", P95X12) for cat in EXPECTED_COUNTS if cat != "wallplugs"}
for _c in ("rice", "walnuts"):
    CONFIG[_c] = ("sub_dv3", "p95x1.421", P95X12)
CONFIG["wallplugs"] = ("sub_wp", "hysteresis", None)


def _guided(p, I, r, eps):
    from scipy import ndimage
    box = lambda x: ndimage.uniform_filter(x, 2 * r + 1)
    mI, mp = box(I), box(p)
    aa = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    return box(aa) * I + box(mp - aa * mI)


def wallplugs_smooth(maps, paths, r=4, eps=0.01):
    """Extent map: guided filter (photo as guide) on the raw map, as wp_eval.py --guided 4 0.01."""
    out = []
    for m, p in zip(maps, paths):
        I = np.asarray(Image.open(p).convert("L").resize(m.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255
        m = m.astype(np.float32)
        lo, hi = np.percentile(m, [1, 99.99])
        out.append(_guided((m - lo) / (hi - lo + 1e-6), I, r, eps) * (hi - lo) + lo)
    return np.stack(out)


def wallplugs_masks(maps, paths, val_maps, val_paths, qs=0.95, qe=0.995, sigma=2.0):
    """Final wallplugs rule: seeds = Gaussian(sigma) map minus the photo's own p90 above the qs quantile of the
    validation per-photo maxima; extent = guided map above the validation pixel qe quantile, connected to a seed."""
    from scipy import ndimage
    g = lambda arr: np.stack([ndimage.gaussian_filter(m.astype(np.float32), sigma) for m in arr])
    seed = lambda A: A - np.percentile(A.reshape(len(A), -1), 90, axis=1)[:, None, None]
    R, Rv = wallplugs_smooth(maps, paths), wallplugs_smooth(val_maps, val_paths)
    N, Nv = seed(g(maps)), seed(g(val_maps))
    te, ts = np.quantile(Rv, qe), np.quantile(Nv.reshape(len(Nv), -1).max(1), qs)
    pred = np.zeros(maps.shape, bool)
    for i in range(len(maps)):
        s = N[i] > ts
        if s.any():
            lab, _ = ndimage.label((R[i] > te) | s)
            k = np.unique(lab[s])
            pred[i] = np.isin(lab, k[k > 0])
    return R, pred, (ts, te)


def threshold_of(rule: str, val_maps: np.ndarray) -> float:
    if rule == "p95x1.2":
        return float(np.quantile(val_maps, 0.95) * 1.2)
    if rule == "p95x1.421":
        return float(np.quantile(val_maps, 0.95) * 1.421)
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
    for cat, (model, rule, post) in CONFIG.items():
        zv = np.load(f"results/maps/{model}__{cat}__val.npz")
        val = zv["maps"].astype(np.float32)
        thr = threshold_of(rule, val) if rule != "hysteresis" else None
        for split in SPLITS:
            z = np.load(f"results/maps/{model}__{cat}__{split}.npz")
            maps, paths = z["maps"].astype(np.float32), z["paths"].astype(str)
            (img_root / cat / split).mkdir(parents=True, exist_ok=True)
            (thr_root / cat / split).mkdir(parents=True, exist_ok=True)
            if rule == "hysteresis":
                maps, binaries, thr = wallplugs_masks(maps, paths, val, zv["paths"].astype(str))
            else:
                binaries = [postprocess(m > thr, *post) for m in maps]
            for m, b, path in zip(maps, binaries, paths):
                stem = Path(path).stem
                tifffile.imwrite(img_root / cat / split / f"{stem}.tiff", m.astype(np.float16))
                Image.fromarray((np.asarray(b) * 255).astype(np.uint8)).save(thr_root / cat / split / f"{stem}.png")
            print(f"{cat}/{split}: model {model}, rule {rule}, threshold {thr}, flagged photos "
                  f"{np.mean([np.asarray(b).any() for b in binaries]) * 100:.0f}%", flush=True)

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
