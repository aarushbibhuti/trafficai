import numpy as np
import pytest

from trafficai.config import Config
from trafficai.motion import MotionEstimator


@pytest.mark.parametrize("velocity", [(10, 0), (0, 10), (6, 8), (0, 0)])
def test_physical_speed_all_directions(calibration, velocity):
    estimator = MotionEstimator(Config())
    for frame in range(21):
        t = frame / 20
        # 10 pixels per meter; timestamps span exactly 1 s.
        pixel = (10 + velocity[0] * t * 10, 10 + velocity[1] * t * 10)
        measured = estimator.update(1, t, frame, calibration.project(pixel))
    assert measured.speed_mps == pytest.approx(np.linalg.norm(velocity), abs=1e-6)
    assert measured.acceleration_mps2 == pytest.approx(0, abs=1e-6)


def test_gaps_use_real_time_not_observation_count():
    estimator = MotionEstimator(Config())
    for frame in (0, 1, 3, 4, 7, 9, 10):
        t = frame / 10
        measured = estimator.update(1, t, frame, (12 * t, 0))
    assert measured.speed_mps == pytest.approx(12, abs=1e-9)
    assert measured.acceleration_mps2 is None
    assert "gap_in_motion_window" in measured.flags


def test_irregular_timestamps():
    estimator = MotionEstimator(Config())
    for frame, t in enumerate([0, 0.07, 0.2, 0.31, 0.49, 0.62, 0.81]):
        measured = estimator.update(1, t, frame, (3 * t, 4 * t))
    assert measured.speed_mps == pytest.approx(5, abs=1e-9)


@pytest.mark.parametrize("acceleration,v0", [(4, 2), (-4, 15)])
def test_acceleration_from_known_quadratic_motion(acceleration, v0):
    estimator = MotionEstimator(Config())
    for frame in range(61):
        t = frame / 20
        measured = estimator.update(1, t, frame, (v0 * t + 0.5 * acceleration * t * t, 0))
    assert measured.acceleration_mps2 == pytest.approx(acceleration, abs=1e-8)
    assert measured.speed_mps == pytest.approx(
        v0 + acceleration * measured.effective_time, abs=1e-8
    )


def test_noise_is_smoothed():
    rng = np.random.default_rng(7)
    estimator = MotionEstimator(Config())
    speeds = []
    for frame in range(80):
        t = frame / 20
        m = estimator.update(1, t, frame, (8 * t + rng.normal(0, 0.025), 0))
        if m.speed_mps is not None:
            speeds.append(m.speed_mps)
    assert np.max(np.abs(np.array(speeds) - 8)) < 0.2


def test_jump_reacquisition_invalid_position_and_cleanup():
    estimator = MotionEstimator(Config())
    for frame in range(30):
        estimator.update(1, frame / 20, frame, (frame / 2, 0))
    m = estimator.update(1, 1.5, 30, (100, 0))
    assert m.speed_mps is None and "track_jump" in m.flags
    segment = m.segment
    m = estimator.update(1, 3.0, 60, (101, 0))
    assert m.speed_mps is None and "reacquired" in m.flags and m.segment > segment
    m = estimator.update(1, 3.05, 61, None, "outside_road_region")
    assert m.speed_mps is None and m.flags == ["outside_road_region"]
    assert estimator.expire(10) == [1]
    assert not estimator.states


def test_short_history_and_invalid_time():
    estimator = MotionEstimator(Config())
    assert estimator.update(1, 0, 0, (0, 0)).speed_mps is None
    with pytest.raises(ValueError, match="increase"):
        estimator.update(1, 0, 1, (1, 0))


def test_bounded_histories():
    estimator = MotionEstimator(Config())
    for frame in range(5000):
        estimator.update(1, frame / 1000, frame, (frame / 1000, 0))
    state = estimator.states[1]
    assert len(state.positions) <= 512 and len(state.speeds) <= 512
