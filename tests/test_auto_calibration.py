"""Known camera geometry and fake models: no weights or optional dependencies needed."""

from dataclasses import asdict

import cv2
import numpy as np
import pytest

from trafficai.auto_calibration import (
    AutoOptions,
    build_auto_calibration,
    combine_predictions,
    fit_plane,
    prediction_plane,
)
from trafficai.calibration import Calibration, fingerprint
from trafficai.config import Config
from trafficai.motion import MotionEstimator
from trafficai.plane_calibration import GeometryRejected, check_fixed_view
from trafficai.video import VideoWriter


def known_prediction(width=320, height=240, camera_height=4.0):
    # Independent pinhole camera: ground y=4 m; fx=fy=200 px, principal point centered.
    focal = width * 200 / 320
    yy, _ = np.indices((height, width))
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = camera_height * focal / (yy - height / 2)
    probability = (yy > height / 2 + 4).astype(float)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], float)
    return depth, intrinsics, probability


def predictions():
    depth, intrinsics, probability = known_prediction()
    normal, offset, mask, info = prediction_plane(depth, intrinsics, probability, AutoOptions())
    return [(normal, offset, mask, intrinsics, info) for _ in range(3)]


def texture():
    return np.random.default_rng(3).integers(0, 256, (240, 320), dtype=np.uint8)


@pytest.fixture
def plane():
    return combine_predictions(
        predictions(),
        (320, 240),
        "a" * 64,
        texture(),
        AutoOptions(),
        {"backend": "known_geometry_test"},
    )


def test_known_plane_distances_motion_and_roundtrip(plane, tmp_path):
    # At z=40 m, y_pixel=140, and one image pixel = 40/200 = 0.2 m.
    assert plane.project((160, 140)) == pytest.approx((0, 40), abs=1e-6)
    assert plane.project((180, 140)) == pytest.approx((4, 40), abs=1e-6)
    assert plane.project((160, 0)) is None  # sky cannot become a road measurement
    estimator = MotionEstimator(Config())
    for i in range(21):
        t = i / 20
        m = estimator.update(1, t, i, plane.project((160 + 20 * t, 140)))
    assert m.speed_mps == pytest.approx(4, abs=1e-6)
    assert m.acceleration_mps2 == pytest.approx(0, abs=1e-6)
    path = tmp_path / "ai.json"
    plane.save(path)
    loaded = Calibration.load(path)
    assert loaded.project((180, 140)) == pytest.approx((4, 40), abs=1e-6)
    assert loaded.measurement_flags == ["ai_scale_unvalidated", "ai_geometry_unvalidated"]
    assert "mask_png" not in loaded.metadata()
    with pytest.raises(FileExistsError):
        plane.save(path)
    with pytest.raises(ValueError, match="different footage"):
        loaded.check_compatibility((320, 240), "b" * 64)
    with pytest.raises(ValueError, match="resolution"):
        loaded.check_compatibility((640, 480), "a" * 64)


def test_mask_preserves_unsupported_holes():
    samples = predictions()
    for sample in samples:
        sample[2][150:190, 140:180] = 0
    plane = combine_predictions(samples, (320, 240), "a" * 64, texture(), AutoOptions(), {})
    assert plane.project((160, 170)) is None
    assert plane.project((200, 170)) is not None


@pytest.mark.parametrize("condition", ["no_road", "invalid_depth", "bad_focal", "wrong_shape"])
def test_invalid_predictions_rejected(condition):
    depth, k, road = known_prediction()
    if condition == "no_road":
        road[:] = 0
    elif condition == "invalid_depth":
        depth[:] = np.nan
    elif condition == "bad_focal":
        k[0, 0] = np.nan
    else:
        depth = depth[:10]
    with pytest.raises(GeometryRejected):
        prediction_plane(depth, k, road, AutoOptions())


def test_nonplanar_and_vertical_surface_rejected():
    rng = np.random.default_rng(13)
    with pytest.raises(GeometryRejected, match="planar"):
        fit_plane(rng.uniform(-10, 10, (1000, 3)), 0.35, 0.65)
    points = rng.uniform(0, 20, (1000, 3))
    points[:, 0] = 4
    with pytest.raises(GeometryRejected, match="orientation"):
        fit_plane(points, 0.35, 0.65)


@pytest.mark.parametrize("change", ["scale", "normal", "focal"])
def test_temporal_inconsistency_rejected(change):
    samples = predictions()
    n, d, mask, k, info = samples[-1]
    if change == "scale":
        d *= 2
    elif change == "normal":
        n = np.array([0.5, -np.sqrt(0.75), 0])
    else:
        k = k.copy()
        k[0, 0] *= 1.5
    samples[-1] = n, d, mask, k, info
    with pytest.raises(GeometryRejected, match="varies|inconsistent"):
        combine_predictions(samples, (320, 240), "a" * 64, texture(), AutoOptions(), {})


def test_camera_view_check():
    reference = texture()
    assert check_fixed_view(reference, reference) < 1e-6
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 12], [0, 1, 0]]), (320, 240))
    with pytest.raises(GeometryRejected, match="changed"):
        check_fixed_view(reference, shifted)
    with pytest.raises(GeometryRejected, match="features"):
        check_fixed_view(np.zeros_like(reference), np.zeros_like(reference))


class FakePredictor:
    info = {"backend": "known_camera_fixture", "scale_status": "ai_estimated_unvalidated"}

    def predict(self, frame):
        return known_prediction(frame.shape[1], frame.shape[0])

    def close(self):
        pass


@pytest.fixture
def textured_video(tmp_path):
    path = tmp_path / "scene.mp4"
    frame = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
    with VideoWriter(path, 320, 240, 20) as writer:
        for i in range(70):
            writer.write(frame, i / 20)
    return path


def test_model_free_builder(textured_video):
    result = build_auto_calibration(
        textured_video, predictor=FakePredictor(), progress=lambda _: None
    )
    assert result.calibration is not None
    assert result.diagnostics["status"] == "accepted"
    assert result.calibration.source_sha256 == fingerprint(textured_video)
    assert len(result.diagnostics["samples"]) == 3


def test_short_video_rejected(textured_video):
    result = build_auto_calibration(
        textured_video, AutoOptions(interval=3), predictor=FakePredictor(), progress=lambda _: None
    )
    assert result.calibration is None
    assert "too short" in result.diagnostics["reason"]


def test_loaded_bad_calibration_rejected(plane, tmp_path):
    import json

    data = asdict(plane)
    data["diagnostics"]["status"] = "rejected"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="did not pass"):
        Calibration.load(path)


def test_cli_ai_save_load_and_protect(textured_video, tmp_path, monkeypatch):
    from trafficai import ai_models
    from trafficai.cli import main

    monkeypatch.setattr(ai_models, "DepthRoadModels", lambda *a, **kw: FakePredictor())
    path = tmp_path / "camera.json"
    args = ["auto-calibrate", str(textured_video), "--output", str(path)]
    assert main(args) == 0
    assert path.with_suffix(".diagnostics.json").is_file()
    assert path.with_suffix(".preview.png").is_file()
    assert main(args) == 1
    assert (
        main(
            [
                "calibrate",
                str(textured_video),
                "--load",
                str(path),
                "--output",
                str(tmp_path / "manual.json"),
            ]
        )
        == 1
    )


def test_auto_pipeline_marks_measurements_events_and_reuses_without_models(
    textured_video,
    tmp_path,
):
    import csv
    import json

    from trafficai.detection import Detection
    from trafficai.pipeline import analyze

    class Detector:
        info = {"backend": "synthetic"}
        frame = 0

        def detect(self, frame):
            t = self.frame / 20
            self.frame += 1
            x = 80 + 5 * (2 * t + 2 * t * t)  # 0.2 m/pixel => 4 m/s²
            return [Detection(1, (x - 5, 130, x + 5, 140), 0.99, 2, "car")]

    directory = analyze(
        textured_video,
        tmp_path,
        Config(clips=False),
        detector=Detector(),
        auto_options=AutoOptions(),
        ai_predictor=FakePredictor(),
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["auto_calibration"]["status"] == "accepted"
    assert metadata["summary"]["rows_with_speed"] > 20
    assert not metadata.get("calibration_runtime_rejection")
    with (directory / "measurements.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all("ai_scale_unvalidated" in row["quality_flags"] for row in rows)
    events = json.loads((directory / "events.json").read_text())
    assert len(events) == 1
    assert events[0]["peak_acceleration_mps2"] == pytest.approx(4, abs=1e-6)
    assert "manual_calibration" not in events[0]["quality_flags"]
    assert "ai_scale_unvalidated" in events[0]["quality_flags"]
    assert "EXPERIMENTAL" in (directory / "report.html").read_text()
    # Saved geometry works with no model initialization or ai extra.
    repeated = analyze(
        textured_video,
        tmp_path,
        Config(max_frames=3),
        directory / "calibration.json",
        detector=Detector(),
    )
    assert json.loads((repeated / "run.json").read_text())["status"] == "limited"


def test_auto_failure_retains_tracking(textured_video, tmp_path):
    import json

    from trafficai.detection import Detection
    from trafficai.pipeline import analyze

    class Detector:
        info = {"backend": "fake"}

        def detect(self, frame):
            return [Detection(1, (10, 10, 30, 30), 0.9, 2, "car")]

    class BadPredictor(FakePredictor):
        def predict(self, frame):
            raise RuntimeError("model unavailable")

    directory = analyze(
        textured_video,
        tmp_path,
        Config(max_frames=5),
        detector=Detector(),
        auto_options=AutoOptions(),
        ai_predictor=BadPredictor(),
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["calibration"] is None
    assert "model unavailable" in metadata["auto_calibration"]["reason"]
    assert metadata["summary"]["tracked_vehicles"] == 1
    assert metadata["summary"]["rows_with_speed"] == 0


def test_runtime_camera_change_latches_unavailable(textured_video, tmp_path, monkeypatch):
    import csv
    import json

    from trafficai.detection import Detection
    from trafficai.pipeline import analyze
    from trafficai.plane_calibration import PlaneCalibration

    result = build_auto_calibration(
        textured_video, predictor=FakePredictor(), progress=lambda _: None
    )
    path = tmp_path / "saved.json"
    result.calibration.save(path)
    checks = 0

    def view_check(self, frame):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise GeometryRejected("Camera moved")
        return 0

    monkeypatch.setattr(PlaneCalibration, "check_view", view_check)

    class Detector:
        info = {"backend": "fake"}

        def detect(self, frame):
            return [Detection(1, (150, 130, 170, 140), 0.9, 2, "car")]

    directory = analyze(textured_video, tmp_path, Config(), path, detector=Detector())
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["calibration_runtime_rejection"]["timestamp_seconds"] == 1
    with (directory / "measurements.csv").open() as f:
        later = [row for row in csv.DictReader(f) if float(row["timestamp_seconds"]) >= 1]
    assert all(row["speed_mps"] == "" for row in later)
    assert all("ai_camera_view_unreliable" in row["quality_flags"] for row in later)
    assert checks == 2


def test_rejected_replacement_preserves_saved_calibration(textured_video, tmp_path, monkeypatch):
    from trafficai import ai_models
    from trafficai.cli import main

    path = tmp_path / "camera.json"
    monkeypatch.setattr(ai_models, "DepthRoadModels", lambda *a, **kw: FakePredictor())
    args = ["auto-calibrate", str(textured_video), "--output", str(path)]
    assert main(args) == 0
    original = path.read_bytes()

    class BadPredictor(FakePredictor):
        def predict(self, frame):
            raise RuntimeError("geometry unavailable")

    monkeypatch.setattr(ai_models, "DepthRoadModels", lambda *a, **kw: BadPredictor())
    assert main([*args, "--overwrite"]) == 1
    assert path.read_bytes() == original


def test_png_size_checked_before_decoding():
    import base64

    from trafficai.plane_calibration import decode_png

    malicious = b"\x89PNG\r\n\x1a\n" + bytes(8) + (50000).to_bytes(4, "big") * 2
    with pytest.raises(ValueError, match="dimensions"):
        decode_png(base64.b64encode(malicious).decode())
