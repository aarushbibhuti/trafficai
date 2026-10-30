"""Validated, serializable run settings (all internal measurements are SI)."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass
class Config:
    model: str = "yolo26n.pt"
    device: str = "cpu"
    confidence: float = 0.1
    iou: float = 0.7
    tracker: str = "bytetrack.yaml"
    classes: tuple[int, ...] = (2, 3, 5, 7)
    units: str = "kmh"
    preview: bool = False
    max_frames: int | None = None
    motion_window: float = 0.7
    min_span: float = 0.4
    acceleration_window: float = 0.8
    max_gap: float = 0.5
    track_ttl: float = 3.0
    max_speed: float = 60.0
    max_residual: float = 0.75
    jump_tolerance: float = 2.0
    braking_threshold: float = 3.0
    acceleration_threshold: float = 3.0
    event_duration: float = 0.5
    release_ratio: float = 0.5
    clip_before: float = 3.0
    clip_after: float = 3.0
    clips: bool = True
    time_bin: float = 60.0
    grid_size: int = 100

    def validate(self):
        positive = (
            "motion_window",
            "min_span",
            "acceleration_window",
            "max_gap",
            "track_ttl",
            "max_speed",
            "max_residual",
            "jump_tolerance",
            "braking_threshold",
            "acceleration_threshold",
            "event_duration",
            "time_bin",
            "grid_size",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("confidence", "iou", "release_ratio"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("clip_before", "clip_after"):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_span > self.motion_window or self.min_span > self.acceleration_window:
            raise ValueError("min_span cannot exceed motion/acceleration windows")
        if self.track_ttl < self.max_gap:
            raise ValueError("track_ttl must be at least max_gap")
        if self.units not in ("kmh", "mph", "mps"):
            raise ValueError("units must be kmh, mph or mps")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if not self.classes or any(c < 0 for c in self.classes):
            raise ValueError("classes must contain nonnegative class IDs")

    def as_dict(self):
        return asdict(self)


def display_speed(speed: float | None, units: str) -> str:
    if speed is None:
        return "speed unavailable"
    factor, label = {"kmh": (3.6, "km/h"), "mph": (2.2369362921, "mph"), "mps": (1.0, "m/s")}[units]
    return f"{speed * factor:.1f} {label}"
