import pytest

from trafficai.config import Config
from trafficai.events import EventDetector
from trafficai.motion import Measurement, MotionEstimator


def measured(t, acceleration, segment=0, flags=None):
    return Measurement(
        speed_mps=10,
        acceleration_mps2=acceleration,
        effective_time=t,
        flags=flags or [],
        segment=segment,
        position=(t, 0),
        support_start=max(0, t - 1),
        acceleration_time=t,
    )


@pytest.mark.parametrize("acceleration,kind", [(4, "acceleration"), (-4, "braking")])
def test_duration_hysteresis_deduplication(acceleration, kind):
    detector = EventDetector(Config())
    result = []
    for index in range(21):
        t = index / 10
        result += detector.update(1, t, measured(t, acceleration), (10, 20))
    assert not result
    result += detector.update(1, 2.1, measured(2.1, acceleration / 2), (10, 20))
    assert not result  # below entry but still above release
    result += detector.update(1, 2.2, measured(2.2, 0), (10, 20))
    assert len(result) == 1 and result[0]["kind"] == kind
    assert result[0]["confirmed_seconds"] == pytest.approx(0.5)
    assert detector.flush() == []


def test_short_impulse_gap_and_reset_do_not_create_events():
    detector = EventDetector(Config())
    for index in range(3):
        detector.update(1, index / 10, measured(index / 10, -9), (0, 0))
    assert detector.interrupt_missing(set()) == []
    assert detector.update(1, 1, measured(1, None), (0, 0)) == []
    detector.update(1, 1.1, measured(1.1, -9), (0, 0))
    assert detector.update(1, 1.2, measured(1.2, -9, segment=1), (0, 0)) == []
    assert not detector.active


def test_track_jump_cannot_trigger_motion_event():
    motion, events = MotionEstimator(Config()), EventDetector(Config())
    emitted = []
    for frame in range(100):
        t = frame / 20
        position = (8 * t + (100 if frame >= 40 else 0), 0)
        m = motion.update(1, t, frame, position)
        emitted += events.update(1, t, m, (0, 0))
    emitted += events.flush()
    assert emitted == []
