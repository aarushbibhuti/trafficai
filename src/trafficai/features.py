"""Manually configured image-space context; these features do not infer violations."""

import json
from pathlib import Path

import cv2
import numpy as np


class Features:
    def __init__(self, data=None):
        self.data = data or {"schema_version": 1, "features": [], "permitted_movements": []}
        self.items = self.data.get("features", [])

    @classmethod
    def load(cls, path, size):
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1 or data.get("image_size") != list(size):
            raise ValueError("Feature schema/resolution mismatch; expected schema 1 and image_size")
        names = set()
        for feature in data.get("features", []):
            name, kind = feature.get("name"), feature.get("kind")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Feature names must be unique nonempty strings")
            names.add(name)
            if kind not in ("lane", "crosswalk", "zone", "stop_line"):
                raise ValueError(f"Unsupported feature kind: {kind}")
            points = np.asarray(feature.get("points"), dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not np.isfinite(points).all()
                or (points < 0).any()
                or (points >= size).any()
            ):
                raise ValueError(f"Invalid feature coordinates: {name}")
            if kind == "stop_line":
                if len(points) != 2 or np.linalg.norm(points[1] - points[0]) < 1:
                    raise ValueError("stop_line requires two distinct points")
            elif len(points) < 3 or not cv2.isContourConvex(points.astype(np.float32)):
                raise ValueError("Area features require convex ordered polygons")
        lanes = {f["name"] for f in data.get("features", []) if f["kind"] == "lane"}
        for movement in data.get("permitted_movements", []):
            if movement.get("from") not in lanes or movement.get("to") not in lanes:
                raise ValueError("Permitted movements must reference configured lane names")
        return cls(data)

    def zones(self, point):
        return [
            item["name"]
            for item in self.items
            if item["kind"] != "stop_line"
            and cv2.pointPolygonTest(np.float32(item["points"]), tuple(point), False) >= 0
        ]

    def draw(self, frame):
        for item in self.items:
            points = np.int32(item["points"])
            cv2.polylines(frame, [points], item["kind"] != "stop_line", (200, 150, 50), 2)
            cv2.putText(
                frame,
                item["name"],
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 150, 50),
                1,
            )
