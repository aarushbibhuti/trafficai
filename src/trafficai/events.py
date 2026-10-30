"""Candidate events: sustained threshold entry, hysteresis release, gap interruption."""

from dataclasses import asdict, dataclass

from .config import Config
from .motion import Measurement


@dataclass
class Candidate:
    track_id: int
    segment: int
    kind: str
    start_seconds: float
    end_seconds: float
    confirmed_seconds: float | None
    peak_acceleration_mps2: float
    start_speed_mps: float
    end_speed_mps: float
    pixel_position: tuple[float, float]
    road_position_m: tuple[float, float] | None
    support_start_seconds: float
    last_observed_seconds: float
    quality_flags: list[str]
    peak_observed_seconds: float


class EventDetector:
    def __init__(self, config: Config, calibration_flags=None):
        self.config = config
        self.calibration_flags = calibration_flags or ["manual_calibration"]
        self.active: dict[int, Candidate] = {}

    def finish(self, track_id, reason):
        candidate = self.active.pop(track_id, None)
        if candidate is None or candidate.confirmed_seconds is None:
            return []
        candidate.quality_flags.append(reason)
        event = asdict(candidate)
        event["classification"] = "candidate_safety_event"
        event["units"] = {"time": "s", "speed": "m/s", "acceleration": "m/s^2", "road": "m"}
        return [event]

    def update(self, track_id, timestamp, measurement: Measurement, pixel):
        emitted = []
        current = self.active.get(track_id)
        acceleration = measurement.acceleration_mps2
        event_time = measurement.acceleration_time
        if (
            acceleration is None
            or event_time is None
            or measurement.flags
            or (current and current.segment != measurement.segment)
        ):
            return self.finish(track_id, "interrupted_measurement")
        if current:
            signed = acceleration if current.kind == "acceleration" else -acceleration
            threshold = (
                self.config.acceleration_threshold
                if current.kind == "acceleration"
                else self.config.braking_threshold
            )
            if signed < threshold * self.config.release_ratio:
                emitted += self.finish(track_id, "hysteresis_release")
                current = None
            else:
                current.end_seconds = event_time
                current.last_observed_seconds = timestamp
                current.end_speed_mps = measurement.speed_mps
                if abs(acceleration) > abs(current.peak_acceleration_mps2):
                    current.peak_acceleration_mps2 = acceleration
                    current.pixel_position = pixel
                    current.road_position_m = measurement.position
                    current.peak_observed_seconds = timestamp
                if (
                    current.confirmed_seconds is None
                    and current.end_seconds - current.start_seconds
                    >= self.config.event_duration - 1e-9
                ):
                    current.confirmed_seconds = timestamp
        if current is None:
            kind = (
                "acceleration"
                if acceleration >= self.config.acceleration_threshold
                else "braking"
                if acceleration <= -self.config.braking_threshold
                else None
            )
            if kind:
                self.active[track_id] = Candidate(
                    track_id,
                    measurement.segment,
                    kind,
                    event_time,
                    event_time,
                    None,
                    acceleration,
                    measurement.speed_mps,
                    measurement.speed_mps,
                    pixel,
                    measurement.position,
                    measurement.support_start,
                    timestamp,
                    ["smoothed_estimate", *self.calibration_flags],
                    timestamp,
                )
        return emitted

    def interrupt_missing(self, visible):
        emitted = []
        for track_id in list(self.active):
            if track_id not in visible:
                emitted += self.finish(track_id, "observation_gap")
        return emitted

    def flush(self, reason="end_of_video"):
        emitted = []
        for track_id in list(self.active):
            emitted += self.finish(track_id, reason)
        return emitted
