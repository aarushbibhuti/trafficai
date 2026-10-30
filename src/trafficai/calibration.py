"""Explicit rectangular road-plane calibration, restricted to its measured polygon."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Calibration:
    image_size: list[int]
    source_points: list[list[float]]
    width_m: float
    length_m: float
    source_sha256: str
    dimensions_source: str
    assumptions: str = "Fixed camera; planar road; negligible lens distortion"
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported calibration schema_version")
        if len(self.image_size) != 2 or any(
            not isinstance(x, int) or x < 2 for x in self.image_size
        ):
            raise ValueError("image_size must contain integer width and height")
        dims = np.asarray([self.width_m, self.length_m], dtype=float)
        if not np.isfinite(dims).all() or (dims <= 0).any():
            raise ValueError("Real-world width and length must be finite and positive meters")
        if not self.dimensions_source.strip():
            raise ValueError("Record how dimensions were measured or assumed")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            raise ValueError("Calibration requires a source video SHA256")
        points = np.asarray(self.source_points, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Exactly four finite pixel points are required")
        if (points < 0).any() or (points >= self.image_size).any():
            raise ValueError("Calibration points must lie inside the image")
        contour = points.astype(np.float32)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 100:
            raise ValueError("Points must form a convex ordered rectangle footprint of >=100 px²")
        if min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) < 5:
            raise ValueError("Calibration corners are too close together")
        target = np.float32(
            [[0, 0], [self.width_m, 0], [self.width_m, self.length_m], [0, self.length_m]]
        )
        self.matrix = cv2.getPerspectiveTransform(contour, target)
        denominators = np.c_[points, np.ones(4)] @ self.matrix[2]
        if (
            not np.isfinite(self.matrix).all()
            or np.linalg.cond(self.matrix) > 1e10
            or np.any(abs(denominators) < 1e-8)
            or np.any(np.sign(denominators) != np.sign(denominators[0]))
        ):
            raise ValueError("Ill-conditioned homography or projection horizon within road region")

    def project(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        if cv2.pointPolygonTest(np.float32(self.source_points), tuple(point), False) < 0:
            return None
        projected = self.matrix @ np.r_[point, 1.0]
        if not np.isfinite(projected).all() or abs(projected[2]) < 1e-8:
            return None
        result = projected[:2] / projected[2]
        if (result < -1e-5).any() or (result > [self.width_m + 1e-5, self.length_m + 1e-5]).any():
            return None
        return tuple(float(x) for x in result)

    def check_compatibility(self, size, source_sha256, allow_other_video=False):
        if list(size) != self.image_size:
            raise ValueError("Calibration resolution differs; recalibrate for this crop/resolution")
        if source_sha256 != self.source_sha256 and not allow_other_video:
            raise ValueError(
                "Calibration belongs to different footage. Inspect the overlay, then use "
                "--allow-calibration-reuse only for the same unchanged camera view"
            )

    def save(self, path, overwrite=False):
        with open(path, "w" if overwrite else "x") as handle:
            json.dump(asdict(self), handle, indent=2, allow_nan=False)
            handle.write("\n")

    def metadata(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        try:
            data = json.loads(Path(path).read_text())
            if data.get("method") == "ai_depth_plane":
                from .plane_calibration import PlaneCalibration

                return PlaneCalibration(**data)
            return cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid calibration schema: {exc}") from exc


def draw_calibration(frame, calibration):
    if hasattr(calibration, "draw"):
        return calibration.draw(frame)
    preview = frame.copy()
    points = np.int32(calibration.source_points)
    cv2.polylines(preview, [points], True, (0, 255, 255), 2)
    for index, point in enumerate(points):
        cv2.circle(preview, tuple(point), 5, (0, 255, 255), -1)
        cv2.putText(
            preview, f"P{index + 1}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
    return preview


def select_points(frame, initial=None):
    """Four ordered corners, U undo, R reset, Enter accept, Escape cancel."""
    from .preview import require_display

    require_display()
    points = list(initial or [])
    window = "Calibration: click corners | U undo | R reset | Enter accept | Esc cancel"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, click)
        while True:
            view = frame.copy()
            for index, point in enumerate(points):
                cv2.circle(view, tuple(map(int, point)), 5, (0, 255, 255), -1)
                cv2.putText(
                    view,
                    str(index + 1),
                    tuple(map(int, point)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow(window, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13) and len(points) == 4:
                return points
            if key == ord("u") and points:
                points.pop()
            if key == ord("r"):
                points.clear()
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Calibration cancelled")
    finally:
        cv2.destroyAllWindows()
