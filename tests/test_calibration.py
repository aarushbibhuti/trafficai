from dataclasses import asdict

import numpy as np
import pytest

from trafficai.calibration import Calibration


def test_known_plane_coordinates_and_save_load(calibration, tmp_path):
    assert calibration.project((50, 100)) == pytest.approx((5, 10), abs=1e-6)
    assert calibration.project((200, 200)) == pytest.approx((20, 20), abs=1e-6)
    assert calibration.project((201, 100)) is None
    assert calibration.project((float("nan"), 0)) is None
    path = tmp_path / "cal.json"
    calibration.save(path)
    assert Calibration.load(path).project((50, 100)) == pytest.approx((5, 10))
    with pytest.raises(FileExistsError):
        calibration.save(path)


def test_perspective_transform():
    calibration = Calibration(
        [401, 401],
        [[100, 100], [300, 100], [400, 400], [0, 400]],
        10,
        30,
        "a" * 64,
        "Synthetic trapezoid",
    )
    # On the centerline, x_road=5 and y_road=60*(v-100)/(v+200).
    # Thus image v=200 is road y=15, not one third of the road length.
    assert calibration.project((200, 100)) == pytest.approx((5, 0), abs=1e-5)
    assert calibration.project((200, 400)) == pytest.approx((5, 30), abs=1e-5)
    assert calibration.project((200, 200)) == pytest.approx((5, 15), abs=1e-5)


@pytest.mark.parametrize(
    "change",
    [
        {"width_m": 0},
        {"length_m": -5},
        {"length_m": np.nan},
        {"source_points": [[0, 0], [0, 0], [100, 100], [0, 100]]},
        {"source_points": [[0, 0], [100, 100], [100, 0], [0, 100]]},
        {"source_points": [[0, 0], [50, 50], [100, 100], [150, 150]]},
        {"source_points": [[0, 0], [250, 0], [200, 200], [0, 200]]},
        {"source_points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        {"dimensions_source": ""},
        {"schema_version": 2},
        {"image_size": [201.5, 201]},
    ],
)
def test_invalid_calibration(calibration, change):
    with pytest.raises(ValueError):
        Calibration(**(asdict(calibration) | change))


def test_compatibility(calibration):
    calibration.check_compatibility((201, 201), "a" * 64)
    with pytest.raises(ValueError, match="resolution"):
        calibration.check_compatibility((400, 200), "a" * 64, True)
    with pytest.raises(ValueError, match="different footage"):
        calibration.check_compatibility((201, 201), "b" * 64)
    calibration.check_compatibility((201, 201), "b" * 64, True)
