"""Timed planar velocity and acceleration; no frame-count-derived elapsed times."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config


@dataclass
class Measurement:
    speed_mps: float | None = None
    acceleration_mps2: float | None = None
    effective_time: float | None = None
    flags: list[str] = field(default_factory=list)
    segment: int = 0
    position: tuple[float, float] | None = None
    support_start: float | None = None
    acceleration_time: float | None = None
    provenance_flags: list[str] = field(default_factory=list)


@dataclass
class TrackState:
    positions: deque = field(default_factory=lambda: deque(maxlen=512))
    speeds: deque = field(default_factory=lambda: deque(maxlen=512))
    last_seen: float = 0
    last_frame: int = -1
    segment: int = 0
    velocity: np.ndarray | None = None
    gap_until: float = -1


def fit_slope(times, values):
    """Least-squares slope in real seconds; works for scalar or vector values."""
    times = np.asarray(times, dtype=float)
    centered = times - times.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        raise ValueError("Insufficient distinct timestamps")
    return centered @ np.asarray(values, dtype=float) / denominator


class MotionEstimator:
    def __init__(self, config: Config):
        self.config = config
        self.states: dict[int, TrackState] = {}
        self.next_segment = 0

    def new_segment(self):
        self.next_segment += 1
        return self.next_segment

    def expire(self, timestamp):
        expired = [
            key
            for key, state in self.states.items()
            if timestamp - state.last_seen > self.config.track_ttl
        ]
        for key in expired:
            del self.states[key]
        return expired

    def reset(self, state):
        state.positions.clear()
        state.speeds.clear()
        state.velocity = None
        state.segment = self.new_segment()
        state.gap_until = -1

    def update(self, track_id, timestamp, frame_index, position, unavailable=None):
        if track_id not in self.states:
            self.states[track_id] = TrackState(segment=self.new_segment())
        state = self.states[track_id]
        flags = []
        cfg = self.config
        if not np.isfinite(timestamp):
            raise ValueError("Motion timestamp must be finite")
        if state.positions and timestamp <= state.last_seen:
            raise ValueError("Motion timestamps must increase per track")
        if position is None or not np.isfinite(position).all():
            self.reset(state)
            state.last_seen, state.last_frame = timestamp, frame_index
            return Measurement(flags=[unavailable or "invalid_projection"], segment=state.segment)
        if state.positions:
            dt = timestamp - state.last_seen
            delta = np.asarray(position) - state.positions[-1][1]
            if dt > cfg.max_gap:
                self.reset(state)
                flags.append("reacquired")
            elif np.linalg.norm(delta) > cfg.max_speed * dt + cfg.jump_tolerance:
                self.reset(state)
                flags.append("track_jump")
            elif (
                state.velocity is not None
                and np.linalg.norm(delta - state.velocity * dt) > cfg.jump_tolerance + 10 * dt**2
            ):
                self.reset(state)
                flags.append("track_jump")
            elif frame_index != state.last_frame + 1:
                # Keep timed position history for speed but restart derivative support.
                state.speeds.clear()
                state.gap_until = timestamp + cfg.motion_window
                flags.append("observation_gap")
        state.last_seen, state.last_frame = timestamp, frame_index
        state.positions.append((timestamp, np.asarray(position, dtype=float)))
        while state.positions and timestamp - state.positions[0][0] > cfg.motion_window + 1e-9:
            state.positions.popleft()
        measurement = Measurement(flags=flags, segment=state.segment, position=position)
        if len(state.positions) < 3 or timestamp - state.positions[0][0] < cfg.min_span - 1e-9:
            measurement.flags.append("short_history")
            return measurement
        times, points = zip(*state.positions, strict=True)
        times, points = np.array(times), np.array(points)
        velocity = fit_slope(times, points)
        residual = np.sqrt(
            np.mean(
                np.sum(
                    (points - points.mean(axis=0) - (times - times.mean())[:, None] * velocity)
                    ** 2,
                    axis=1,
                )
            )
        )
        speed = float(np.linalg.norm(velocity))
        if residual > cfg.max_residual or speed > cfg.max_speed:
            self.reset(state)
            measurement.segment = state.segment
            measurement.flags.append(
                "noisy_motion" if residual > cfg.max_residual else "implausible_speed"
            )
            return measurement
        state.velocity = velocity
        effective = float(times.mean())
        measurement.speed_mps, measurement.effective_time = speed, effective
        measurement.support_start = float(times[0])
        if timestamp <= state.gap_until:
            measurement.flags.append("gap_in_motion_window")
            state.speeds.clear()
            return measurement
        state.speeds.append((effective, speed, float(times[0])))
        while state.speeds and effective - state.speeds[0][0] > cfg.acceleration_window + 1e-9:
            state.speeds.popleft()
        if len(state.speeds) >= 3 and effective - state.speeds[0][0] >= cfg.min_span - 1e-9:
            measurement.acceleration_mps2 = float(
                fit_slope([x[0] for x in state.speeds], [x[1] for x in state.speeds])
            )
            measurement.support_start = state.speeds[0][2]
            measurement.acceleration_time = float(np.mean([x[0] for x in state.speeds]))
        else:
            measurement.flags.append("acceleration_warmup")
        return measurement
