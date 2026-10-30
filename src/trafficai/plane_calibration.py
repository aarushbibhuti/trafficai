"""Saved experimental plane geometry with a restricted pixel mask and view checks."""

import base64
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


class GeometryRejected(ValueError):
    """An explicit failure to establish usable geometry; never substitute a scale."""


def encode_png(array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise ValueError("Could not encode calibration reference")
    return base64.b64encode(encoded).decode("ascii")


def decode_png(encoded):
    if not isinstance(encoded, str) or len(encoded) > 4_000_000:
        raise ValueError("Invalid/oversized calibration image")
    payload = base64.b64decode(encoded, validate=True)
    if (
        payload[:8] != b"\x89PNG\r\n\x1a\n"
        or len(payload) < 24
        or not 1 <= int.from_bytes(payload[16:20], "big") <= 1024
        or not 1 <= int.from_bytes(payload[20:24], "big") <= 1024
    ):
        raise ValueError("Invalid/oversized calibration PNG dimensions")
    array = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    if array is None or array.size > 1_000_000:
        raise ValueError("Invalid calibration reference image")
    return array


def check_fixed_view(reference, current, tolerance_px=4.0):
    """Robust background registration; uncertainty rejects rather than assuming stability."""
    if reference.shape != current.shape:
        raise GeometryRejected("Camera reference resolution differs")
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, descriptors1 = orb.detectAndCompute(reference, None)
    kp2, descriptors2 = orb.detectAndCompute(current, None)
    if descriptors1 is None or descriptors2 is None:
        raise GeometryRejected("Too few scene features to verify a fixed camera")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors1, descriptors2)
    matches = sorted((m for m in matches if m.distance < 55), key=lambda m: m.distance)[:400]
    if len(matches) < 30:
        raise GeometryRejected("Too few matching scene features to verify a fixed camera")
    source = np.float32([kp1[m.queryIdx].pt for m in matches])
    target = np.float32([kp2[m.trainIdx].pt for m in matches])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if homography is None or inliers is None or inliers.mean() < 0.6:
        raise GeometryRejected("Scene registration is unreliable; camera stability unavailable")
    height, width = reference.shape
    samples = np.float32(
        [[[width * x, height * y] for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)]]
    )
    warped = cv2.perspectiveTransform(samples, homography)
    shift = float(np.linalg.norm(warped - samples, axis=2).max())
    if not np.isfinite(shift) or shift > tolerance_px:
        raise GeometryRejected(f"Camera view changed ({shift:.1f} px > {tolerance_px:.1f} px)")
    return shift


@dataclass
class PlaneCalibration:
    image_size: list[int]
    source_sha256: str
    homography: list[list[float]]
    mask_png: str
    reference_png: str
    diagnostics: dict
    models: dict
    assumptions: str = (
        "AI-estimated metric scale and camera intrinsics; unvalidated. Fixed camera, "
        "one planar road, square pixels, centered principal point, negligible lens distortion."
    )
    method: str = "ai_depth_plane"
    schema_version: int = 2

    def __post_init__(self):
        if self.schema_version != 2 or self.method != "ai_depth_plane":
            raise ValueError("Unsupported AI calibration schema")
        if len(self.image_size) != 2 or any(
            not isinstance(x, int) or x < 2 for x in self.image_size
        ):
            raise ValueError("Invalid AI calibration image_size")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(x not in "0123456789abcdef" for x in self.source_sha256)
        ):
            raise ValueError("Invalid source fingerprint")
        self.matrix = np.asarray(self.homography, dtype=float)
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or np.linalg.cond(self.matrix) > 1e10
        ):
            raise ValueError("Invalid AI road homography")
        self.mask, self.reference = decode_png(self.mask_png), decode_png(self.reference_png)
        if self.mask.shape != self.reference.shape or min(self.mask.shape) < 16:
            raise ValueError("Invalid road-mask/reference dimensions")
        if np.count_nonzero(self.mask) < 100:
            raise ValueError("AI road region is too small")
        if self.diagnostics.get("status") != "accepted":
            raise ValueError("AI calibration did not pass geometry checks")
        ys, xs = np.where(self.mask > 0)
        pixels = np.c_[
            xs * self.image_size[0] / self.mask.shape[1],
            ys * self.image_size[1] / self.mask.shape[0],
            np.ones(len(xs)),
        ]
        denominator = pixels @ self.matrix[2]
        if np.any(abs(denominator) < 1e-8) or np.any(
            np.sign(denominator) != np.sign(denominator[0])
        ):
            raise ValueError("Projection horizon intersects the valid road mask")

    @property
    def measurement_flags(self):
        return ["ai_scale_unvalidated", "ai_geometry_unvalidated"]

    def metadata(self):
        data = asdict(self)
        data.pop("mask_png")
        data.pop("reference_png")
        data["valid_mask_pixels"] = int(np.count_nonzero(self.mask))
        data["mask_size"] = list(reversed(self.mask.shape))
        return data

    def save(self, path, overwrite=False):
        with open(path, "w" if overwrite else "x") as handle:
            json.dump(asdict(self), handle, indent=2, allow_nan=False)
            handle.write("\n")

    def check_compatibility(self, size, source_sha256, allow_other_video=False):
        if list(size) != self.image_size:
            raise ValueError("Calibration resolution differs; regenerate for this camera view")
        if source_sha256 != self.source_sha256 and not allow_other_video:
            raise ValueError(
                "AI calibration belongs to different footage; inspect the camera view "
                "before using --allow-calibration-reuse"
            )

    def check_view(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.reference.shape[1], self.reference.shape[0]))
        return check_fixed_view(self.reference, gray)

    def project(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        if (point < 0).any() or (point >= self.image_size).any():
            return None
        x, y = (point * [self.mask.shape[1], self.mask.shape[0]] / self.image_size).astype(int)
        if not self.mask[y, x]:
            return None
        result = self.matrix @ np.r_[point, 1.0]
        if not np.isfinite(result).all() or abs(result[2]) < 1e-8:
            return None
        return tuple(float(x) for x in result[:2] / result[2])

    def draw(self, frame):
        output = frame.copy()
        mask = cv2.resize(self.mask, tuple(self.image_size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 180, 255), 2)
        return output
