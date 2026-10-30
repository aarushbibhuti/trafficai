import csv
import json

import av
import cv2
import numpy as np
import pytest

from trafficai.cli import main
from trafficai.config import Config
from trafficai.detection import Detection
from trafficai.pipeline import RunFailed, analyze
from trafficai.video import VideoReader, VideoWriter


class FakeDetector:
    info = {"backend": "synthetic_test", "model": "none"}

    def __init__(self, error=None):
        self.index = 0
        self.error = error

    def detect(self, frame):
        if self.error and self.index == 3:
            raise self.error
        t = self.index / 20
        self.index += 1
        x = 20 + 5 * (2 * t + 2 * t * t)  # 0.2 m/px => acceleration 4 m/s²
        return [Detection(1, (x - 8, 100, x + 8, 120), 0.99, 2, "car")]


@pytest.fixture
def no_gui(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Headless code made a GUI call")

    for name in (
        "imshow",
        "waitKey",
        "waitKeyEx",
        "namedWindow",
        "destroyAllWindows",
        "setMouseCallback",
        "destroyWindow",
    ):
        monkeypatch.setattr(cv2, name, forbidden)


def test_calibrated_end_to_end(synthetic_video, tmp_path, no_gui):
    video, calibration = synthetic_video
    directory = analyze(
        video, tmp_path / "runs", Config(time_bin=1), calibration, detector=FakeDetector()
    )
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["status"] == "completed" and metadata["frames"] == 70
    assert metadata["summary"]["tracked_vehicles"] == 1
    assert metadata["summary"]["candidate_events"] == 1
    assert metadata["summary"]["events_per_tracked_vehicle"] == 1
    events = json.loads((directory / "events.json").read_text())
    event = events[0]
    assert event["peak_acceleration_mps2"] == pytest.approx(4, abs=1e-6)
    assert event["clip"] is not None and event["clip"]["frames"] > 20
    with av.open(str(directory / event["clip"]["path"])) as clip:
        assert sum(1 for _ in clip.decode(video=0)) == event["clip"]["frames"]
    with VideoReader(directory / "annotated.mp4") as reader:
        frames = list(reader.frames())
        assert len(frames) == 70 and frames[-1][1] == pytest.approx(69 / 20, abs=1e-5)
    with (directory / "measurements.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 70 and rows[0]["speed_mps"] == ""
    assert float(rows[-1]["acceleration_mps2"]) == pytest.approx(4, abs=1e-6)
    assert "candidate" in (directory / "report.html").read_text().lower()
    assert (directory / "aggregates.csv").is_file()


def test_uncalibrated_and_collision_free(synthetic_video, tmp_path, no_gui):
    video, _ = synthetic_video
    outputs = [
        analyze(video, tmp_path / "runs", Config(max_frames=5), detector=FakeDetector())
        for _ in range(2)
    ]
    assert outputs[0] != outputs[1]
    for directory in outputs:
        rows = list(csv.DictReader((directory / "measurements.csv").open()))
        assert all(
            row["speed_mps"] == "" and row["quality_flags"] == "uncalibrated" for row in rows
        )
        assert json.loads((directory / "events.json").read_text()) == []


@pytest.mark.parametrize(
    "error,status", [(KeyboardInterrupt(), "cancelled"), (ValueError("fake failure"), "failed")]
)
def test_partial_outputs_and_cleanup(synthetic_video, tmp_path, error, status):
    video, _ = synthetic_video
    if status == "failed":
        with pytest.raises(RunFailed, match="fake failure"):
            analyze(video, tmp_path / "runs", Config(), detector=FakeDetector(error))
    else:
        analyze(video, tmp_path / "runs", Config(), detector=FakeDetector(error))
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "run.json").read_text())["status"] == status
    with VideoReader(directory / "annotated.mp4") as reader:
        assert len(list(reader.frames())) == 3


def test_writer_failure_is_reported(synthetic_video, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(VideoWriter, "write", fail)
    with pytest.raises(RunFailed, match="disk full"):
        analyze(synthetic_video[0], tmp_path / "runs", Config(), detector=FakeDetector())
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "run.json").read_text())["status"] == "failed"


def test_variable_frame_timestamps_preserved(tmp_path):
    path = tmp_path / "vfr.mp4"
    times = [0, 0.1, 0.31, 0.47, 0.8]
    with VideoWriter(path, 64, 48, 10) as writer:
        for t in times:
            writer.write(np.zeros((48, 64, 3), np.uint8), t)
    with VideoReader(path) as reader:
        assert [t for _, t, _ in reader.frames()] == pytest.approx(times, abs=1e-6)


def test_cli_calibration_and_frame(synthetic_video, tmp_path, no_gui):
    video, _ = synthetic_video
    image = tmp_path / "frame.png"
    assert main(["frame", str(video), "--at", "1", "--output", str(image)]) == 0
    assert image.is_file()
    output = tmp_path / "manual.json"
    args = [
        "calibrate",
        str(video),
        "--output",
        str(output),
        "--points",
        "0,0 319,0 319,239 0,239",
        "--width-m",
        "63.8",
        "--length-m",
        "47.8",
        "--dimensions-source",
        "Synthetic reference",
    ]
    assert main(args) == 0
    assert main(args) == 1  # protect existing calibration
    assert (
        main(
            [
                "calibrate",
                str(video),
                "--load",
                str(output),
                "--width-m",
                "60",
                "--output",
                str(tmp_path / "corrected.json"),
            ]
        )
        == 0
    )


def test_bad_video_and_bad_config(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        VideoReader(tmp_path / "missing.mp4")
    with pytest.raises(ValueError, match="confidence"):
        Config(confidence=float("nan")).validate()


@pytest.mark.parametrize(
    "timestamps,message", [([None], "no presentation"), ([0, 0], "Nonmonotonic")]
)
def test_missing_or_nonmonotonic_pts_rejected(timestamps, message):
    from fractions import Fraction
    from types import SimpleNamespace

    frames = []
    for timestamp in timestamps:
        frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
        frame.pts = timestamp
        frame.time_base = Fraction(1, 10)
        frames.append(frame)
    reader = VideoReader.__new__(VideoReader)
    reader.container = SimpleNamespace(decode=lambda stream: iter(frames))
    reader.stream = None
    reader.width, reader.height, reader.info = 64, 48, {}
    with pytest.raises(ValueError, match=message):
        list(reader.frames())
