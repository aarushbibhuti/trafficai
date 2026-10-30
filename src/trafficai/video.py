"""Timestamp-preserving file video I/O. No GUI calls live in this module."""

from fractions import Fraction
from math import isfinite
from pathlib import Path

import av


class VideoReader:
    def __init__(self, path):
        self.path = Path(path)
        self.container = None
        if not self.path.is_file():
            raise ValueError(f"Input video does not exist: {path}")
        try:
            self.container = av.open(str(path))
            if not self.container.streams.video:
                raise ValueError("Input has no video stream")
            self.stream = self.container.streams.video[0]
            self.width, self.height = self.stream.width, self.stream.height
            self.fps = float(self.stream.average_rate or self.stream.guessed_rate or 0)
            if self.width < 2 or self.height < 2 or not isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Invalid video dimensions or frame-rate metadata")
            self.info = {
                "width": self.width,
                "height": self.height,
                "nominal_fps": self.fps,
                "declared_frames": self.stream.frames,
                "timestamp_source": "decoded_pts",
                "path": str(self.path.resolve()),
            }
        except Exception as exc:
            self.close()
            raise ValueError(f"Cannot open video {path}: {exc}") from exc

    def frames(self):
        origin = previous = None
        for index, frame in enumerate(self.container.decode(self.stream)):
            if frame.pts is None or not frame.time_base:
                raise ValueError(
                    f"Frame {index} has no presentation timestamp; cannot measure time"
                )
            timestamp = float(frame.pts * frame.time_base)
            if not isfinite(timestamp) or (previous is not None and timestamp <= previous):
                raise ValueError(f"Nonmonotonic/invalid timestamp at frame {index}")
            if frame.width != self.width or frame.height != self.height:
                raise ValueError("Video resolution changed during decoding")
            if origin is None:
                origin = timestamp
                self.info["first_pts_seconds"] = origin
            previous = timestamp
            yield index, timestamp - origin, frame.to_ndarray(format="bgr24")

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class VideoWriter:
    """H.264 MP4, no audio; preserves relative video PTS, including variable cadence."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.container = None
        self.count = 0
        self.width, self.height = width, height
        try:
            self.container = av.open(str(path), mode="w")
            self.stream = self.container.add_stream(
                "libx264", rate=Fraction(fps).limit_denominator()
            )
            # yuv444p accommodates odd-sized videos without resizing calibration coordinates.
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"
            self.stream.time_base = Fraction(1, 1_000_000)
            self.stream.codec_context.time_base = self.stream.time_base
            self.stream.options = {"crf": "20", "preset": "fast"}
        except Exception as exc:
            if self.container:
                self.container.close()
            raise ValueError(f"Cannot initialize video writer {path}: {exc}") from exc

    def write(self, frame, timestamp):
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Output frame dimensions changed")
        output = av.VideoFrame.from_ndarray(frame, format="bgr24")
        output.pts = round(timestamp * 1_000_000)
        output.time_base = Fraction(1, 1_000_000)
        for packet in self.stream.encode(output):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        if self.container is None:
            return
        container, self.container = self.container, None
        try:
            for packet in self.stream.encode():
                container.mux(packet)
        finally:
            container.close()
        if self.count:
            # Encoding/muxing errors raise; additionally ensure the resulting file decodes.
            with av.open(str(self.path)) as check:
                if next(check.decode(video=0), None) is None:
                    raise ValueError(f"Output video is unreadable: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
