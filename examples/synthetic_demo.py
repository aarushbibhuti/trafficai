"""Generate reviewable, model-free evidence with known acceleration and braking.

Run: uv run python examples/synthetic_demo.py --output runs/synthetic-demo
This is a numerical/integration demonstration, not detector or field validation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from trafficai.calibration import Calibration, fingerprint
from trafficai.config import Config
from trafficai.detection import Detection
from trafficai.pipeline import analyze
from trafficai.video import VideoWriter


def position(t):
    if t <= 2.5:
        return 2 * t + 2 * t**2  # acceleration +4 m/s²
    if t <= 5:
        elapsed = t - 2.5
        return 17.5 + 12 * elapsed - 2 * elapsed**2  # braking -4 m/s²
    return 35 + 2 * (t - 5)


class SyntheticDetector:
    info = {"backend": "synthetic_demo", "model": "known_trajectory_no_inference"}

    def __init__(self):
        self.frame = 0

    def detect(self, frame):
        x = 20 + 5 * position(self.frame / 20)
        self.frame += 1
        return [Detection(1, (x - 8, 100, x + 8, 120), 1.0, 2, "synthetic vehicle")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    video = args.output / "synthetic.mp4"
    with VideoWriter(video, 320, 240, 20) as writer:
        for index in range(140):
            frame = np.full((240, 320, 3), 50, np.uint8)
            x = round(20 + 5 * position(index / 20))
            cv2.rectangle(frame, (x - 8, 100), (x + 8, 120), (200, 200, 200), -1)
            writer.write(frame, index / 20)
    calibration = Calibration(
        [320, 240],
        [[0, 0], [319, 0], [319, 239], [0, 239]],
        63.8,
        47.8,
        fingerprint(video),
        "Synthetic known 0.2 m/pixel",
    )
    calibration_path = args.output / "calibration.json"
    calibration.save(calibration_path)
    run = analyze(
        video, args.output, Config(time_bin=2), calibration_path, detector=SyntheticDetector()
    )
    print(f"Synthetic evidence report: {run / 'report.html'}")


if __name__ == "__main__":
    main()
