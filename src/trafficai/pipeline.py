"""One-pass analysis followed by bounded-memory evidence exports."""

import platform
import signal
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from . import __version__
from .annotation import annotate
from .calibration import Calibration, fingerprint
from .clips import export_clips
from .config import Config
from .detection import YoloTracker
from .events import EventDetector
from .exports import Exports, write_json
from .features import Features
from .motion import MotionEstimator
from .report import render_report
from .video import VideoReader, VideoWriter


class RunFailed(ValueError):
    pass


def analyze(
    video,
    output,
    config: Config,
    calibration_path=None,
    features_path=None,
    allow_calibration_reuse=False,
    detector=None,
):
    config.validate()
    if config.preview:
        from .preview import require_display

        require_display()
    calibration = Calibration.load(calibration_path) if calibration_path else None
    input_hash = fingerprint(video)
    with VideoReader(video) as reader:
        if calibration:
            calibration.check_compatibility(
                (reader.width, reader.height), input_hash, allow_calibration_reuse
            )
        features = (
            Features.load(features_path, (reader.width, reader.height))
            if features_path
            else Features()
        )
        backend = detector if detector is not None else YoloTracker(config)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
        directory = Path(output) / f"run_{run_id}"
        directory.mkdir(parents=True, exist_ok=False)
        warnings = [
            "No field measurement accuracy validation has been performed for this run.",
            "Bottom-center boxes approximate tire contact; occlusion, shadows, camera motion, "
            "curves, grade and lens distortion can bias measurements.",
            "Track fragmentation inflates volume; ID switches can create false events.",
            "Audio is omitted from annotated video and clips.",
        ]
        if calibration is None:
            warnings.append("No calibration: metric speed, acceleration and events unavailable.")
        elif calibration.source_sha256 != input_hash:
            warnings.append(
                "Calibration reused by explicit operator assertion of unchanged camera view."
            )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "trafficai_version": __version__,
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("av", "numpy", "opencv-python")},
            "source_sha256": input_hash,
            "config": config.as_dict(),
            "model": backend.info,
            "video": reader.info,
            "calibration": asdict(calibration) if calibration else None,
            "features": features.data,
            "status": "running",
            "frames": 0,
            "duration_seconds": 0.0,
            "warnings": warnings,
            "unavailable_categories": {
                "red_light": "No reliable signal-state input or validated signal association",
                "prohibited_turn": (
                    "Configured movements are descriptive only; no validated trajectory rule engine"
                ),
                "near_miss": "No validated conflict or time-to-collision model",
            },
        }
        if calibration is None:
            metadata["unavailable_categories"]["motion_events"] = "Metric calibration missing"
        write_json(directory / "run.json", metadata)
        exports = Exports(directory, config, features)
        motion, events = MotionEstimator(config), EventDetector(config)
        error = None
        try:
            with VideoWriter(
                directory / "annotated.mp4", reader.width, reader.height, reader.fps
            ) as writer:
                for index, timestamp, frame in reader.frames():
                    if index == 0:
                        import cv2

                        success, scene = cv2.imencode(".jpg", frame)
                        if not success:
                            raise ValueError("Cannot encode scene reference image")
                        (directory / "scene.jpg").write_bytes(scene.tobytes())
                    for expired in motion.expire(timestamp):
                        for event in events.finish(expired, "track_expired"):
                            exports.event(event)
                    detections = backend.detect(frame)
                    visible = {d.track_id for d in detections}
                    for event in events.interrupt_missing(visible):
                        exports.event(event)
                    observations = []
                    for detection in detections:
                        point = calibration.project(detection.contact) if calibration else None
                        measurement = motion.update(
                            detection.track_id,
                            timestamp,
                            index,
                            point,
                            "outside_road_region" if calibration else "uncalibrated",
                        )
                        for event in events.update(
                            detection.track_id, timestamp, measurement, detection.contact
                        ):
                            exports.event(event)
                        exports.measurement(index, timestamp, detection, measurement)
                        observations.append((detection, measurement))
                    output_frame = annotate(
                        frame,
                        observations,
                        timestamp,
                        calibration,
                        features,
                        config.units,
                        events.active,
                    )
                    writer.write(output_frame, timestamp)
                    metadata["frames"] += 1
                    metadata["duration_seconds"] = timestamp
                    if config.preview:
                        from .preview import show

                        if not show(output_frame):
                            metadata["status"] = "cancelled"
                            break
                    if config.max_frames and metadata["frames"] >= config.max_frames:
                        metadata["status"] = "limited"
                        break
            if not metadata["frames"]:
                raise ValueError("Input has no decodable video frames")
            if metadata["status"] == "running":
                metadata["status"] = "completed"
                declared = reader.info["declared_frames"]
                if declared and declared != metadata["frames"]:
                    metadata["warnings"].append(
                        f"Decoded {metadata['frames']} frames but container declared {declared}; "
                        "source coverage may be incomplete or metadata inaccurate."
                    )
        except KeyboardInterrupt:
            metadata["status"] = "cancelled"
        except Exception as exc:
            metadata["status"], metadata["error"] = "failed", str(exc)
            error = exc
        finally:
            if config.preview:
                from .preview import close

                close()
        try:
            for event in events.flush(metadata["status"]):
                exports.event(event)
            exports.finish()
            # Interrupt a second time to skip remaining clips; already-written evidence survives.
            if config.clips and metadata["frames"] and metadata["status"] != "failed":
                try:
                    export_clips(exports, reader.info, metadata["duration_seconds"], config)
                except KeyboardInterrupt:
                    metadata["status"] = "cancelled"
                    metadata["warnings"].append("Clip export interrupted; some clips unavailable")
            exports.finish()
            metadata["summary"] = exports.summary()
            clip_failures = sum(1 for e in exports.events() if "clip_failed" in e["quality_flags"])
            if clip_failures:
                metadata["warnings"].append(f"{clip_failures} event clips failed; see events.json")
            render_report(exports, metadata)
            write_json(directory / "run.json", metadata)
        except Exception as exc:
            metadata["status"], metadata["error"] = "failed", f"Finalizing outputs: {exc}"
            write_json(directory / "run.json", metadata)
            error = error or exc
        finally:
            exports.close()
        if error:
            raise RunFailed(f"Analysis failed: {error}. Partial outputs: {directory}") from error
        return directory


def install_signal_handlers():
    """CLI only: SIGTERM follows the same cleanup path as Ctrl-C."""

    def cancel(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
