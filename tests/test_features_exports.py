import json

import pytest

from trafficai.config import Config
from trafficai.detection import Detection
from trafficai.exports import Exports
from trafficai.features import Features
from trafficai.motion import Measurement


def test_feature_validation_and_grouping(tmp_path):
    data = {
        "schema_version": 1,
        "image_size": [100, 100],
        "features": [
            {"name": "lane", "kind": "lane", "points": [[0, 0], [90, 0], [90, 90], [0, 90]]},
            {"name": "stop", "kind": "stop_line", "points": [[0, 50], [90, 50]]},
        ],
        "permitted_movements": [{"from": "lane", "to": "lane"}],
    }
    path = tmp_path / "features.json"
    path.write_text(json.dumps(data))
    features = Features.load(path, (100, 100))
    assert features.zones((50.0, 50.0)) == ["lane"]
    assert features.zones((95.0, 95.0)) == []
    with pytest.raises(ValueError, match="resolution"):
        Features.load(path, (200, 200))
    data["permitted_movements"][0]["from"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lane names"):
        Features.load(path, (100, 100))


def test_exact_volume_denominators_and_time_bins(tmp_path):
    exports = Exports(tmp_path, Config(time_bin=1), Features())
    try:
        for index, t in enumerate([0, 0.5, 1.0, 1.5]):
            for track_id in [1, 2]:
                exports.measurement(
                    index,
                    t,
                    Detection(track_id, (0, 0, 20, 20), 0.9, 2, "car"),
                    Measurement(speed_mps=10 if track_id == 1 else None),
                )
        # Two distinct candidates for one vehicle is a count, not a probability.
        for _ in range(2):
            exports.event({"start_seconds": 0.5, "pixel_position": (10, 20), "kind": "braking"})
        summary = exports.summary()
        assert summary["tracked_vehicles"] == 2
        assert summary["vehicles_with_speed"] == 1
        assert summary["events_per_tracked_vehicle"] == 1
        assert summary["events_per_vehicle_with_speed"] == 2
        rows = {r["bucket"]: r for r in exports.aggregates() if r["category"] == "time"}
        assert rows["0"]["tracked_vehicles"] == rows["1"]["tracked_vehicles"] == 2
        assert rows["0"]["events"] == rows["0"]["braking_events"] == 2
        assert rows["1"]["events"] == 0
    finally:
        exports.close()
