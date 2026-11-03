"""Shared protocol for research question Q2 (lighting changes cause overkill).

* physical parts are split once per category into a DEV half (compare and choose methods) and an
  EVAL half (touched only for the final report);
* main metric: threshold set on "regular" images at a 0 % miss budget, applied unchanged to the
  shift_* conditions -> overkill and miss there; plus AUROC and how far the median good score moves
  under shift (drift, 1.0 = no drift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from industrial_inspection.metrics import calibrate_threshold, evaluate_at_threshold

TAG_RE = r"^\d+_(.+?)(?:-\d+)?\.png$"
SPLIT_SEED = 2026


def parts_and_tags(image_paths) -> tuple[np.ndarray, np.ndarray]:
    """Physical-part id (class folder + leading number) and capture condition of each image."""
    paths = pd.Series(image_paths).astype(str).str.replace("\\", "/", regex=False)
    name = paths.str.rsplit("/", n=1).str[-1]
    parts = (paths.str.rsplit("/", n=2).str[-2] + "/" + name.str.split("_").str[0]).to_numpy()
    return parts, name.str.extract(TAG_RE)[0].to_numpy()


def split_parts(parts: np.ndarray, labels: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """Boolean mask of the DEV half, drawn per class over physical parts."""
    rng = np.random.default_rng(seed)
    dev = np.zeros(len(parts), dtype=bool)
    for c in (0, 1):
        u = rng.permutation(np.unique(parts[labels == c]))
        dev |= np.isin(parts, u[: len(u) // 2])
    return dev


def half_mask(parts: np.ndarray, labels: np.ndarray, half: str) -> np.ndarray:
    dev = split_parts(parts, labels)
    return dev if half == "dev" else ~dev


def evaluate(s: np.ndarray, y: np.ndarray, tag: np.ndarray) -> dict[str, float]:
    reg, sh = tag == "regular", np.char.startswith(tag.astype(str), "shift")
    thr = calibrate_threshold(s[reg], y[reg], 0.0)
    op = evaluate_at_threshold(s[sh], y[sh], thr)
    good_reg, good_sh = np.median(s[reg & (y == 0)]), np.median(s[sh & (y == 0)])
    return {"auroc_all": roc_auc_score(y, s), "auroc_shift": roc_auc_score(y[sh], s[sh]),
            "overkill_shift": op.overkill_rate, "miss_shift": op.miss_rate,
            "drift": good_sh / good_reg if good_reg != 0 else np.nan}


def bootstrap_parts(parts: np.ndarray, labels: np.ndarray, draws: int, seed: int = 0):
    """Yield index arrays of part-level bootstrap resamples (per class, a part brings all its images)."""
    rng = np.random.default_rng(seed)
    pools = {c: np.unique(parts[labels == c]) for c in (0, 1)}
    index = {p: np.flatnonzero(parts == p) for p in np.unique(parts)}
    for _ in range(draws):
        yield np.concatenate([index[p] for c in (0, 1) for p in rng.choice(pools[c], size=pools[c].size)])
