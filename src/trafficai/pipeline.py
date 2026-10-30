"""One-pass analysis followed by bounded-memory evidence exports."""

import platform
import signal
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
    auto_options=None,
    ai_predictor=None,
):
    config.validate()
    if calibration_path and auto_options is not None:
        raise ValueError("Choose saved calibration or automatic calibration, not both")
    if auto_options is not None:
        auto_options.validate()
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
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
        directory = Path(output) / f"run_{run_id}"
        directory.mkdir(parents=True, exist_ok=False)
        print(f"Run output: {directory}", flush=True)
        auto_diagnostics = None
        if auto_options is not None:
            from .auto_calibration import build_auto_calibration

            write_json(
                directory / "run.json",
                {"status": "running", "phase": "ai_calibration", "frames": 0, "run_id": run_id},
            )
            try:
                result = build_auto_calibration(
                    video,
                    auto_options,
                    predictor=ai_predictor,
                    progress=lambda text: print(text, flush=True),
                )
            except KeyboardInterrupt:
                write_json(
                    directory / "run.json",
                    {"status": "cancelled", "phase": "ai_calibration", "frames": 0},
                )
                return directory
            calibration, auto_diagnostics = result.calibration, result.diagnostics
            write_json(directory / "auto_calibration.json", auto_diagnostics)
            if result.preview is not None:
                import cv2

                success, encoded = cv2.imencode(".png", result.preview)
                if not success:
                    raise ValueError("Cannot encode AI calibration preview")
                (directory / "calibration.preview.png").write_bytes(encoded.tobytes())
        # Release depth-model resources before loading the vehicle detector.
        try:
            backend = detector if detector is not None else YoloTracker(config)
        except (Exception, KeyboardInterrupt) as exc:
            write_json(
                directory / "run.json",
                {
                    "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "phase": "detector_initialization",
                    "frames": 0,
                    "error": str(exc),
                    "auto_calibration": auto_diagnostics,
                },
            )
            raise
        calibration_flags = getattr(calibration, "measurement_flags", [])
        if calibration:
            calibration.save(directory / "calibration.json")
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
        if calibration_flags:
            warnings.append(
                "EXPERIMENTAL: AI-estimated scale and geometry are unvalidated. "
                "Numerical consistency is not a physical accuracy guarantee."
            )
        if auto_diagnostics and calibration is None:
            warnings.append("AI calibration rejected: " + auto_diagnostics["reason"])
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
            "calibration": calibration.metadata() if calibration else None,
            "auto_calibration": auto_diagnostics,
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
        motion, events = MotionEstimator(config), EventDetector(config, calibration_flags)
        view_rejected, last_view_check = False, -1.0
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
                    if calibration_flags and not view_rejected and timestamp - last_view_check >= 1:
                        from .plane_calibration import GeometryRejected

                        last_view_check = timestamp
                        try:
                            calibration.check_view(frame)
                        except GeometryRejected as exc:
                            view_rejected = True  # Latch: never silently resume a moved camera.
                            metadata["calibration_runtime_rejection"] = {
                                "timestamp_seconds": timestamp,
                                "reason": str(exc),
                            }
                            metadata["warnings"].append(
                                f"AI calibration invalidated at {timestamp:.2f}s: {exc}"
                            )
                    for expired in motion.expire(timestamp):
                        for event in events.finish(expired, "track_expired"):
                            exports.event(event)
                    detections = backend.detect(frame)
                    visible = {d.track_id for d in detections}
                    for event in events.interrupt_missing(visible):
                        exports.event(event)
                    observations = []
                    for detection in detections:
                        point = (
                            calibration.project(detection.contact)
                            if calibration and not view_rejected
                            else None
                        )
                        measurement = motion.update(
                            detection.track_id,
                            timestamp,
                            index,
                            point,
                            "ai_camera_view_unreliable"
                            if view_rejected
                            else "outside_road_region"
                            if calibration
                            else "ai_calibration_rejected"
                            if auto_diagnostics
                            else "uncalibrated",
                        )
                        measurement.provenance_flags = calibration_flags
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
