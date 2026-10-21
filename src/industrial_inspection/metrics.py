"""Operating-point metrics for line inspection.

AUROC says how well scores rank; a production line needs a single threshold.
The question asked on the line is: "if we may miss at most X% of defects,
what fraction of good parts do we wrongly reject (overkill)?"

Convention: label 1 = defective, 0 = good. A part is rejected when
score >= threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    miss_rate: float  # defective parts accepted / defective parts
    overkill_rate: float  # good parts rejected / good parts


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (good) or 1 (defective)")
    return s, y


def calibrate_threshold(scores, labels, target_miss_rate: float) -> float:
    """Highest threshold whose miss rate on (scores, labels) is <= target.

    Highest, because among thresholds meeting the miss budget it rejects the
    fewest good parts.
    """
    s, y = _as_arrays(scores, labels)
    if not 0.0 <= target_miss_rate < 1.0:
        raise ValueError("target_miss_rate must be in [0, 1)")
    defect = np.sort(s[y == 1])
    if defect.size == 0:
        raise ValueError("need at least one defective sample to calibrate")
    allowed_misses = int(np.floor(target_miss_rate * defect.size))
    # Rejecting score >= defect[k] misses exactly the k lowest-scored defects
    # (fewer if there are ties at defect[k]).
    return float(defect[allowed_misses])


def evaluate_at_threshold(scores, labels, threshold: float) -> OperatingPoint:
    s, y = _as_arrays(scores, labels)
    rejected = s >= threshold
    n_defect, n_good = int((y == 1).sum()), int((y == 0).sum())
    miss = float((~rejected & (y == 1)).sum() / n_defect) if n_defect else float("nan")
    overkill = float((rejected & (y == 0)).sum() / n_good) if n_good else float("nan")
    return OperatingPoint(threshold=float(threshold), miss_rate=miss, overkill_rate=overkill)


def stratified_split(labels, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (calib, eval) with the defect ratio preserved in both."""
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    calib, evals = [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_calib = round(calib_fraction * idx.size)
        calib.append(idx[:n_calib])
        evals.append(idx[n_calib:])
    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(evals))


def held_out_operating_point(
    scores, labels, target_miss_rate: float, calib_fraction: float = 0.5, seed: int = 0
) -> tuple[OperatingPoint, OperatingPoint]:
    """(calibration-set point, held-out point) for one stratified split.

    The gap between them is the optimism you get from tuning the threshold
    on the same data you report on.
    """
    s, y = _as_arrays(scores, labels)
    calib, evals = stratified_split(y, calib_fraction, seed)
    thr = calibrate_threshold(s[calib], y[calib], target_miss_rate)
    return evaluate_at_threshold(s[calib], y[calib], thr), evaluate_at_threshold(s[evals], y[evals], thr)


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    scores,
    labels,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) via stratified percentile bootstrap.

    Resampling good and defective parts separately keeps every resample's
    defect count equal to the original, so rate statistics stay defined.
    """
    s, y = _as_arrays(scores, labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == cls) for cls in (0, 1)]
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups if g.size])
        stats[i] = statistic(s[idx], y[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(statistic(s, y)), float(lo), float(hi)
