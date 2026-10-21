import numpy as np
import pytest

from industrial_inspection.metrics import (
    bootstrap_ci,
    calibrate_threshold,
    evaluate_at_threshold,
    held_out_operating_point,
    stratified_split,
)


def test_threshold_respects_miss_budget_and_minimises_overkill():
    # 10 defects scored 1..10, goods scored 0..9.5.
    defect = np.arange(1, 11, dtype=float)
    good = np.arange(0, 10, 0.5)
    s = np.concatenate([good, defect])
    y = np.concatenate([np.zeros_like(good), np.ones_like(defect)]).astype(int)

    thr = calibrate_threshold(s, y, target_miss_rate=0.2)
    assert thr == 3.0  # may miss 2 defects (scores 1, 2)
    op = evaluate_at_threshold(s, y, thr)
    assert op.miss_rate == pytest.approx(0.2)
    assert op.overkill_rate == pytest.approx(14 / 20)  # goods 3.0..9.5


def test_zero_miss_budget_catches_every_defect():
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
    y = np.array([0, 0, 1, 1, 1])
    op = evaluate_at_threshold(s, y, calibrate_threshold(s, y, 0.0))
    assert op.miss_rate == 0.0
    assert op.overkill_rate == 0.0


def test_ties_never_exceed_budget():
    s = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([1, 1, 1, 0])
    op = evaluate_at_threshold(s, y, calibrate_threshold(s, y, 0.4))
    assert op.miss_rate <= 0.4


def test_stratified_split_preserves_ratio_and_is_disjoint():
    y = np.array([0] * 80 + [1] * 20)
    calib, evals = stratified_split(y, 0.5, seed=1)
    assert set(calib).isdisjoint(evals)
    assert len(calib) + len(evals) == 100
    assert y[calib].sum() == 10 and y[evals].sum() == 10


def test_held_out_is_not_more_optimistic_on_average():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 1, 400), rng.normal(2, 1, 100)])
    y = np.array([0] * 400 + [1] * 100)
    gaps = []
    for seed in range(50):
        calib_op, eval_op = held_out_operating_point(s, y, 0.05, seed=seed)
        assert calib_op.miss_rate <= 0.05
        gaps.append(eval_op.miss_rate - calib_op.miss_rate)
    assert np.mean(gaps) > 0  # tuning on the reported set understates misses


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 1, 300), rng.normal(2, 1, 60)])
    y = np.array([0] * 300 + [1] * 60)
    thr = 1.0
    def stat(s_, y_):
        return evaluate_at_threshold(s_, y_, thr).overkill_rate

    point, lo, hi = bootstrap_ci(stat, s, y, n_resamples=500)
    assert lo <= point <= hi
    assert hi - lo < 0.15


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, 0.2], [0, 0], 0.1)
    with pytest.raises(ValueError):
        evaluate_at_threshold([0.1], [2], 0.5)
