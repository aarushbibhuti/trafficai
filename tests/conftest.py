import numpy as np
import pytest

from trafficai.calibration import Calibration, fingerprint
from trafficai.video import VideoWriter


@pytest.fixture
def calibration():
    return Calibration(
        [201, 201],
        [[0, 0], [200, 0], [200, 200], [0, 200]],
        20,
        20,
        "a" * 64,
        "Synthetic known dimensions",
    )


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "source.mp4"
    with VideoWriter(path, 320, 240, 20) as writer:
        for index in range(70):
            writer.write(np.full((240, 320, 3), 70, np.uint8), index / 20)
    calibration = Calibration(
        [320, 240],
        [[0, 0], [319, 0], [319, 239], [0, 239]],
        63.8,
        47.8,
        fingerprint(path),
        "Synthetic 0.2 m/pixel",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration.save(calibration_path)
    return path, calibration_path
