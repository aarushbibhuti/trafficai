"""Streaming measurement export and disk-backed exact run-level denominators."""

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

FIELDS = [
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "segment",
    "class_id",
    "label",
    "confidence",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "contact_x_px",
    "contact_y_px",
    "road_x_m",
    "road_y_m",
    "speed_mps",
    "acceleration_mps2",
    "effective_time_seconds",
    "acceleration_time_seconds",
    "support_start_seconds",
    "quality_flags",
    "zones",
]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Exports:
    def __init__(self, directory, config, features):
        self.directory, self.config, self.features = Path(directory), config, features
        self.db = sqlite3.connect(self.directory / "analysis.sqlite")
        self.db.executescript("""
            PRAGMA cache_size = -2048;
            PRAGMA temp_store = FILE;
            CREATE TABLE tracks(id INTEGER PRIMARY KEY, measured INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE exposure(category TEXT, bucket TEXT, track_id INTEGER, measured INTEGER,
                                  PRIMARY KEY(category,bucket,track_id));
            CREATE TABLE events(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE counts(category TEXT, bucket TEXT, kind TEXT, count INTEGER,
                                PRIMARY KEY(category,bucket,kind));
        """)
        self.file = (self.directory / "measurements.csv").open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(FIELDS)
        self.rows = self.measured_rows = 0
        self.acceleration_rows = 0
        self.flags = Counter()

    def buckets(self, timestamp, point):
        x, y = point
        return [
            ("time", str(int(timestamp // self.config.time_bin))),
            ("grid", f"{int(x // self.config.grid_size)},{int(y // self.config.grid_size)}"),
            *[("feature", name) for name in self.features.zones(point)],
        ]

    def measurement(self, frame_index, timestamp, detection, measurement):
        road = measurement.position or (None, None)
        self.writer.writerow(
            [
                frame_index,
                timestamp,
                detection.track_id,
                measurement.segment,
                detection.class_id,
                detection.label,
                detection.confidence,
                *detection.bbox,
                *detection.contact,
                *road,
                measurement.speed_mps,
                measurement.acceleration_mps2,
                measurement.effective_time,
                measurement.acceleration_time,
                measurement.support_start,
                "|".join(measurement.flags + measurement.provenance_flags),
                json.dumps(self.features.zones(detection.contact)),
            ]
        )
        measured = int(measurement.speed_mps is not None)
        self.db.execute(
            "INSERT INTO tracks VALUES (?,?) ON CONFLICT(id) DO UPDATE SET "
            "measured=max(measured,excluded.measured)",
            (detection.track_id, measured),
        )
        for category, bucket in self.buckets(timestamp, detection.contact):
            self.db.execute(
                "INSERT INTO exposure VALUES (?,?,?,?) ON CONFLICT DO UPDATE SET "
                "measured=max(measured,excluded.measured)",
                (category, bucket, detection.track_id, measured),
            )
        self.rows += 1
        self.measured_rows += measured
        self.acceleration_rows += int(measurement.acceleration_mps2 is not None)
        self.flags.update(measurement.flags + measurement.provenance_flags)
        if self.rows % 500 == 0:
            self.db.commit()
            self.file.flush()

    def event(self, event):
        cursor = self.db.execute("INSERT INTO events(payload) VALUES ('{}')")
        event["event_id"] = cursor.lastrowid
        event["zones"] = self.features.zones(event["pixel_position"])
        event["clip"] = None
        self.update_event(event)
        for category, bucket in self.buckets(event["start_seconds"], event["pixel_position"]):
            self.db.execute(
                "INSERT INTO counts VALUES (?,?,?,1) ON CONFLICT DO UPDATE SET count=count+1",
                (category, bucket, event["kind"]),
            )

    def update_event(self, event):
        self.db.execute(
            "UPDATE events SET payload=? WHERE id=?",
            (json.dumps(event, allow_nan=False), event["event_id"]),
        )

    def events(self):
        for (payload,) in self.db.execute("SELECT payload FROM events ORDER BY id"):
            yield json.loads(payload)

    def export_events(self):
        with (self.directory / "events.json").open("w") as handle:
            handle.write("[\n")
            for index, event in enumerate(self.events()):
                if index:
                    handle.write(",\n")
                json.dump(event, handle, allow_nan=False)
            handle.write("\n]\n")

    def aggregates(self):
        query = """SELECT e.category,e.bucket,count(*),sum(e.measured),
                   coalesce((SELECT sum(c.count) FROM counts c
                     WHERE c.category=e.category AND c.bucket=e.bucket),0),
                   coalesce((SELECT c.count FROM counts c WHERE c.category=e.category
                     AND c.bucket=e.bucket AND c.kind='braking'),0),
                   coalesce((SELECT c.count FROM counts c WHERE c.category=e.category
                     AND c.bucket=e.bucket AND c.kind='acceleration'),0)
                   FROM exposure e GROUP BY e.category,e.bucket ORDER BY e.category,e.bucket"""
        for category, bucket, vehicles, measured, events, braking, acceleration in self.db.execute(
            query
        ):
            yield {
                "category": category,
                "bucket": bucket,
                "tracked_vehicles": vehicles,
                "vehicles_with_speed": measured,
                "events": events,
                "braking_events": braking,
                "acceleration_events": acceleration,
                "events_per_tracked_vehicle": events / vehicles,
                "events_per_vehicle_with_speed": events / measured if measured else None,
            }

    def summary(self):
        count, measured = self.db.execute(
            "SELECT count(*),coalesce(sum(measured),0) FROM tracks"
        ).fetchone()
        events = self.db.execute("SELECT count(*) FROM events").fetchone()[0]
        return {
            "tracked_vehicles": count,
            "vehicles_with_speed": measured,
            "candidate_events": events,
            "events_per_tracked_vehicle": events / count if count else None,
            "events_per_vehicle_with_speed": events / measured if measured else None,
            "measurement_rows": self.rows,
            "rows_with_speed": self.measured_rows,
            "rows_with_acceleration": self.acceleration_rows,
            "acceleration_coverage_fraction": self.acceleration_rows / self.rows
            if self.rows
            else None,
            "speed_coverage_fraction": self.measured_rows / self.rows if self.rows else None,
            "quality_flag_counts": dict(self.flags),
        }

    def finish(self):
        self.db.commit()
        self.file.flush()
        self.export_events()
        fields = [
            "category",
            "bucket",
            "tracked_vehicles",
            "vehicles_with_speed",
            "events",
            "braking_events",
            "acceleration_events",
            "events_per_tracked_vehicle",
            "events_per_vehicle_with_speed",
        ]
        with (self.directory / "aggregates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.aggregates())

    def close(self):
        self.file.close()
        self.db.commit()
        self.db.close()
