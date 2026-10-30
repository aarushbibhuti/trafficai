"""Experimental semantic-road + metric-depth calibration of one fixed camera plane."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .calibration import fingerprint
from .plane_calibration import (
    GeometryRejected,
    PlaneCalibration,
    check_fixed_view,
    encode_png,
)
from .video import VideoReader


@dataclass
class AutoOptions:
    device: str = "auto"
    samples: int = 3
    interval: float = 1.0
    at: float = 0.0
    local_files_only: bool = False
    road_probability: float = 0.7
    max_depth_m: float = 100.0
    plane_tolerance_m: float = 0.35
    min_inlier_ratio: float = 0.65
    max_scale_spread: float = 0.2
    max_normal_degrees: float = 8.0

    def validate(self):
        if not isinstance(self.samples, int) or not 3 <= self.samples <= 10:
            raise ValueError("AI calibration needs 3 to 10 sampled frames")
        for name in (
            "interval",
            "max_depth_m",
            "plane_tolerance_m",
            "max_scale_spread",
            "max_normal_degrees",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"AI {name} must be finite and positive")
        if not np.isfinite(self.at) or self.at < 0:
            raise ValueError("AI start time must be finite and nonnegative")
        if not 0 < self.road_probability < 1 or not 0 < self.min_inlier_ratio < 1:
            raise ValueError("AI road probability and inlier ratio must be between 0 and 1")


@dataclass
class AutoResult:
    calibration: PlaneCalibration | None
    diagnostics: dict
    preview: np.ndarray | None = None


def fit_plane(points, tolerance, min_inlier_ratio, seed=0):
    """Deterministic bounded RANSAC, followed by orthogonal least-squares refinement."""
    if len(points) < 100 or not np.isfinite(points).all():
        raise GeometryRejected("Insufficient finite road points for a plane")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(250):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ a)
        inliers = abs(points @ normal + offset) <= tolerance
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < min_inlier_ratio:
        raise GeometryRejected(f"Road is not sufficiently planar ({best.mean():.0%} inliers)")
    for _ in range(2):
        selected = points[best]
        center = selected.mean(axis=0)
        _, singular, vectors = np.linalg.svd(selected - center, full_matrices=False)
        if singular[1] < 0.01:
            raise GeometryRejected("Road support is collinear")
        normal = vectors[-1]
        offset = -float(normal @ center)
        if offset < 0:
            normal, offset = -normal, -offset
        best = abs(points @ normal + offset) <= tolerance
        if best.mean() < min_inlier_ratio:
            raise GeometryRejected("Refined road plane has insufficient support")
    # Normal should face toward an above-road camera; vertical facades must not become roads.
    if normal[1] >= -0.25 or not 0.5 <= offset <= 40:
        raise GeometryRejected("Implausible road orientation or estimated camera height")
    return normal, offset, best


def prediction_plane(depth, intrinsics, road_probability, options, seed=0):
    depth, intrinsics, probability = map(np.asarray, (depth, intrinsics, road_probability))
    if depth.ndim != 2 or probability.shape != depth.shape or intrinsics.shape != (3, 3):
        raise GeometryRejected("Invalid model output shapes")
    height, width = depth.shape
    if (
        not np.isfinite(intrinsics).all()
        or not 0.25 * width < intrinsics[0, 0] < 5 * width
        or not np.isclose(intrinsics[0, 0], intrinsics[1, 1])
    ):
        raise GeometryRejected("Estimated focal length is invalid or implausible")
    valid = (
        np.isfinite(depth)
        & (depth >= 2)
        & (depth <= options.max_depth_m)
        & np.isfinite(probability)
        & (probability >= options.road_probability)
    )
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.where(valid)
    if len(xs) < max(150, depth.size * 0.02):
        raise GeometryRejected("Too little confidently segmented, finite-depth road")
    indices = np.random.default_rng(seed).choice(len(xs), min(len(xs), 6000), replace=False)
    pixels = np.c_[xs[indices], ys[indices], np.ones(len(indices))]
    rays = pixels @ np.linalg.inv(intrinsics).T
    points = rays * depth[ys[indices], xs[indices], None]
    normal, offset, inliers = fit_plane(
        points, options.plane_tolerance_m, options.min_inlier_ratio, seed
    )
    # Keep only directly supported pixels; no convex hull across islands, vehicles or medians.
    all_pixels = np.c_[xs, ys, np.ones(len(xs))]
    all_points = (all_pixels @ np.linalg.inv(intrinsics).T) * depth[ys, xs, None]
    accepted = abs(all_points @ normal + offset) <= options.plane_tolerance_m
    mask = np.zeros(depth.shape, np.uint8)
    mask[ys[accepted], xs[accepted]] = 255
    return (
        normal,
        offset,
        mask,
        {
            "inlier_ratio": float(inliers.mean()),
            "median_plane_residual_m": float(np.median(abs(points[inliers] @ normal + offset))),
            "estimated_camera_height_m": offset,
            "focal_length_px": float(intrinsics[0, 0]),
            "normal": normal.tolist(),
            "supported_pixels": int(np.count_nonzero(mask)),
        },
    )


def combine_predictions(predictions, image_size, source_hash, reference, options, models):
    """Check across-frame consistency and freeze a single ray/plane mapping."""
    normals = np.array([p[0] for p in predictions])
    offsets = np.array([p[1] for p in predictions])
    intrinsics = np.array([p[3] for p in predictions])
    focal = intrinsics[:, 0, 0]
    height_spread = float(np.ptp(offsets) / np.median(offsets))
    focal_spread = float(np.ptp(focal) / np.median(focal))
    angles = np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1)))
    if height_spread > options.max_scale_spread or focal_spread > 0.15:
        raise GeometryRejected(
            f"AI scale varies across frames (height {height_spread:.0%}, "
            f"focal length {focal_spread:.0%})"
        )
    if angles.max() > options.max_normal_degrees:
        raise GeometryRejected("Road-plane orientation is inconsistent across sampled frames")
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    offset = float(np.median(offsets))
    intrinsic = np.median(intrinsics, axis=0)
    # Require repeated road support, never fill holes or extrapolate over unseen regions.
    votes = sum((p[2] > 0).astype(int) for p in predictions)
    mask = (votes >= max(2, int(np.ceil(len(predictions) * 2 / 3)))).astype(np.uint8) * 255
    working_height, working_width = mask.shape
    yy, xx = np.indices(mask.shape)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixels @ np.linalg.inv(intrinsic).T
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        distance_z = -offset / denominator
        sample_depths = np.array(
            [-p[1] / ((pixels @ np.linalg.inv(p[3]).T) @ p[0]) for p in predictions]
        )
        ray_spread = np.ptp(sample_depths, axis=0) / np.median(sample_depths, axis=0)
    mask[(denominator >= -0.03) | (distance_z < 2) | (distance_z > options.max_depth_m)] = 0
    mask[
        (~np.isfinite(ray_spread))
        | (ray_spread > options.max_scale_spread)
        | (sample_depths.min(axis=0) <= 0)
    ] = 0
    # Remove poorly supported disconnected specks, preserving separate road components.
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for label in range(1, components):
        if stats[label, cv2.CC_STAT_AREA] < 100:
            mask[labels == label] = 0
    ys, xs = np.where(mask)
    if (
        len(xs) < mask.size * 0.02
        or np.ptp(xs) < working_width * 0.15
        or np.ptp(ys) < working_height * 0.1
    ):
        raise GeometryRejected("Too little spatially distributed road support after consensus")
    axis_x = np.array([1.0, 0, 0]) - normal[0] * normal
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    homography = np.vstack([-offset * axis_x, -offset * axis_y, normal]) @ np.linalg.inv(intrinsic)
    homography = homography @ np.diag(
        [working_width / image_size[0], working_height / image_size[1], 1]
    )
    diagnostics = {
        "status": "accepted",
        "scale_status": "ai_estimated_unvalidated",
        "height_spread_fraction": height_spread,
        "focal_spread_fraction": focal_spread,
        "maximum_normal_difference_degrees": float(angles.max()),
        "estimated_camera_height_m": offset,
        "focal_length_working_px": float(intrinsic[0, 0]),
        "working_size": [working_width, working_height],
        "normal": normal.tolist(),
        "road_coverage_fraction": float(np.count_nonzero(mask) / mask.size),
        "samples": [p[4] for p in predictions],
        "options": asdict(options),
        "warning": "Internal consistency checks do not establish physical scale accuracy.",
    }
    return PlaneCalibration(
        list(image_size),
        source_hash,
        homography.tolist(),
        encode_png(mask),
        encode_png(reference),
        diagnostics,
        models,
    )


def build_auto_calibration(video, options=None, predictor=None, progress=print):
    options = options or AutoOptions()
    options.validate()
    model = predictor
    owned = predictor is None
    diagnostics = {"status": "rejected", "options": asdict(options), "samples": []}
    preview = reference = None
    predictions = []
    try:
        with VideoReader(video) as reader:
            source_hash = fingerprint(video)
            targets = [options.at + i * options.interval for i in range(options.samples)]
            for _, timestamp, frame in reader.frames():
                if timestamp + 1e-9 < targets[len(predictions)]:
                    continue
                scale = min(1.0, 640 / max(reader.width, reader.height))
                working = cv2.resize(
                    frame, (round(reader.width * scale), round(reader.height * scale))
                )
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                if reference is None:
                    reference = gray
                    preview = frame.copy()
                else:
                    check_fixed_view(reference, gray)
                if model is None:
                    from .ai_models import DepthRoadModels

                    model = DepthRoadModels(options.device, options.local_files_only, progress)
                progress(
                    f"AI calibration: estimating road geometry for sample "
                    f"{len(predictions) + 1}/{options.samples} at {timestamp:.2f}s..."
                )
                depth, intrinsics, probability = model.predict(working)
                normal, offset, mask, metrics = prediction_plane(
                    depth, intrinsics, probability, options, seed=len(predictions)
                )
                metrics["timestamp_seconds"] = timestamp
                diagnostics["samples"].append(metrics)
                predictions.append((normal, offset, mask, intrinsics, metrics))
                if len(predictions) == options.samples:
                    break
            if len(predictions) != options.samples:
                raise GeometryRejected("Video is too short for the requested calibration samples")
            calibration = combine_predictions(
                predictions,
                (reader.width, reader.height),
                source_hash,
                reference,
                options,
                model.info,
            )
        progress("AI calibration accepted for experimental use. Metric scale remains unvalidated.")
        return AutoResult(calibration, calibration.diagnostics, calibration.draw(preview))
    except Exception as exc:
        diagnostics["reason"] = str(exc)
        if model is not None:
            diagnostics["models"] = model.info
        progress(f"AI calibration unavailable: {exc}")
        return AutoResult(None, diagnostics, preview)
    finally:
        if owned and model is not None:
            model.close()
