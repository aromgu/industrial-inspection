import numpy as np

from industrial_inspection.xcond import combine


def test_pixel_min_keeps_a_consistent_defect_and_drops_a_moving_highlight():
    maps = np.zeros((3, 16, 16), dtype=np.float32)
    maps[:, 4, 4] = 5.0  # defect: same place in every photo
    maps[0, 10, 10] = maps[1, 12, 2] = maps[2, 1, 14] = 9.0  # reflection: moves with the light
    assert combine(maps, "pixel-min") == 5.0
    assert combine(maps, "image-mean") == 9.0


def test_tolerance_lets_a_slightly_shifted_defect_survive():
    maps = np.zeros((2, 16, 16), dtype=np.float32)
    maps[0, 5, 5] = maps[1, 6, 6] = 3.0
    assert combine(maps, "pixel-min") == 0.0
    assert combine(maps, "pixel-min", radius=1) == 3.0
