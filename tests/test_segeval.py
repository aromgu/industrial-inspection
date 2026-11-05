import numpy as np

from industrial_inspection.segeval import f1_at, f1_curve, full_frame


def test_f1_curve_matches_direct_f1():
    rng = np.random.default_rng(0)
    truth = rng.random(5000) < 0.1
    scores = rng.random(5000) + truth * 0.5
    thresholds = np.array([0.9, 0.3, 0.6, 1.2])
    curve = f1_curve(scores, truth, thresholds)
    for th, f in zip(thresholds, curve):
        assert abs(f - f1_at(scores, truth, th)) < 1e-3


def test_full_frame_puts_a_cropped_map_back_in_the_centre():
    m = np.zeros((1, 8, 8), dtype=np.float32)
    m[0, 4, 4] = 1.0
    out = full_frame(m, (32, 32), view=(16, 8))  # crop is the central half of the resized image
    peak = np.unravel_index(out[0].argmax(), out[0].shape)
    assert 14 <= peak[0] <= 20 and 14 <= peak[1] <= 20
    assert out[0, 0, 0] == m.min()
