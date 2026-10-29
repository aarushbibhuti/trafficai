"""Vehicle tracking adapter; model-free callers can inject the same small interface."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str

    @property
    def contact(self):
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class Detector(Protocol):
    info: dict

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloTracker:
    def __init__(self, config):
        # Lazy import: calibration and numerical tests never initialize/download a model.
        # Isolate upstream settings from other YOLO applications/versions on this machine.
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "trafficai" / "ultralytics"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.config = config
        device = config.device
        if device == "auto":
            device = (
                "cuda:0"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; use --device cpu")
        if (device.startswith("cuda") or device.isdigit()) and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; install a matching PyTorch build or use cpu")
        self.device = device
        try:
            self.model = YOLO(config.model)
            if self.model.task != "detect":
                raise ValueError("Model must be an object detection model")
            self.model.to(device)
        except Exception as exc:
            raise ValueError(
                f"Cannot load model {config.model!r} on {device}: {exc}. "
                "Supply a local compatible .pt model or check download access"
            ) from exc
        tracker = Path(config.tracker)
        if not tracker.is_file():
            tracker = Path(ultralytics.__file__).parent / "cfg" / "trackers" / config.tracker
        if not tracker.is_file():
            raise ValueError(f"Tracker YAML not found: {config.tracker}")
        # This baseline intentionally limits association algorithms to the reviewed choices.
        import yaml

        tracker_config = yaml.safe_load(tracker.read_text())
        if tracker_config.get("tracker_type") not in ("bytetrack", "botsort"):
            raise ValueError("Supported tracker_type values: bytetrack, botsort")
        # Do not enable appearance/identity matching, even through a supplied YAML.
        if tracker_config.get("with_reid", False):
            raise ValueError("Appearance re-identification is not supported; set with_reid: false")
        weights = Path(getattr(self.model, "ckpt_path", config.model))
        from .calibration import fingerprint

        self.info = {
            "backend": "ultralytics",
            "version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "model": config.model,
            "device": device,
            "weights_sha256": fingerprint(weights) if weights.is_file() else None,
            "tracker_config": tracker_config,
            "tracker_path": str(tracker),
        }
        self.tracker_path = str(tracker)

    def detect(self, frame):
        try:
            result = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_path,
                device=self.device,
                conf=self.config.confidence,
                iou=self.config.iou,
                classes=list(self.config.classes),
                verbose=False,
                save=False,
                show=False,
            )[0]
        except Exception as exc:
            raise ValueError(f"Detection/tracking failed on {self.device}: {exc}") from exc
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        output = []
        for box, track_id, confidence, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.int().cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            strict=True,
        ):
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            output.append(
                Detection(
                    track_id,
                    tuple(map(float, box)),
                    confidence,
                    class_id,
                    result.names.get(class_id, str(class_id)),
                )
            )
        return output
