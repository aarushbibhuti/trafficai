"""Self-contained, escaped, static HTML evidence report; no third-party services."""

import html
import json


def render_report(exports, metadata):
    esc = lambda value: html.escape(str(value))  # noqa: E731
    summary = metadata["summary"]
    with (exports.directory / "report.html").open("w") as out:
        out.write("""<!doctype html><html lang="en"><meta charset="utf-8">
<title>TrafficAI evidence review</title><style>
body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#182b3a}
table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #ccd}
th{text-align:left;background:#eef3f7}video{max-width:100%}pre{white-space:pre-wrap}
.notice{padding:16px;background:#fff1d0}svg{max-width:100%;background:#edf1f5}
</style><h1>TrafficAI evidence review</h1>
<p class="notice">Candidate braking and acceleration events are observations for human review,
not violations, proof of causation, or proof that infrastructure is unsafe. Track IDs apply only
within this run. Detection errors and ID switches can inflate counts or create false events.</p>""")
        out.write(
            f"<p>Run status: <strong>{esc(metadata['status'])}</strong>. "
            f"Coverage: {metadata['frames']} decoded frames, "
            f"{metadata['duration_seconds']:.2f} seconds from first video frame.</p>"
        )
        out.write(
            f"<p>{summary['tracked_vehicles']} tracked vehicle IDs; "
            f"{summary['vehicles_with_speed']} with at least one usable speed; "
            f"{summary['candidate_events']} candidate events. Events per tracked vehicle: "
            f"{esc(summary['events_per_tracked_vehicle'])}.</p>"
        )
        out.write(
            "<p>Counts are tracker IDs, not verified traffic volume. Overlapping zones and time "
            "bins can count the same ID repeatedly; denominators are unique within each bucket. "
            "Zero events in areas without valid speed coverage does not establish safety. "
            "Elapsed video time is not wall-clock time; recording date/time is unknown.</p>"
        )
        if (metadata.get("calibration") or {}).get("method") == "ai_depth_plane":
            out.write(
                '<p class="notice"><strong>EXPERIMENTAL: AI-estimated scale — unvalidated.'
                "</strong> Road scale and camera geometry were predicted, not surveyed. "
                "Consistency checks do not establish physical accuracy. Speeds, acceleration "
                "and candidate events must not be treated as validated measurements.</p>"
            )
        out.write('<video controls preload="metadata" src="annotated.mp4"></video>')
        if metadata.get("auto_calibration"):
            out.write(
                "<h2>AI calibration diagnostics</h2><pre>"
                + esc(json.dumps(metadata["auto_calibration"], indent=2))
                + "</pre>"
            )
        out.write(
            "<h2>Coverage and measurement assumptions</h2><pre>"
            + esc(
                json.dumps(
                    {
                        "calibration": metadata["calibration"],
                        "summary": summary,
                        "warnings": metadata["warnings"],
                        "unavailable_categories": metadata["unavailable_categories"],
                    },
                    indent=2,
                )
            )
            + "</pre>"
        )
        width, height = metadata["video"]["width"], metadata["video"]["height"]
        out.write(
            "<h2>Candidate event locations</h2><p>Image-space grid; cells show raw event "
            "counts at peak acceleration observation positions. Perspective distorts area.</p>"
        )
        out.write(
            f'<svg role="img" aria-label="Event locations in camera coordinates" '
            f'viewBox="0 0 {width} {height}">'
        )
        out.write(f'<image href="scene.jpg" width="{width}" height="{height}"/>')
        grid = metadata["config"]["grid_size"]
        for row in exports.aggregates():
            if row["category"] == "grid" and row["events"]:
                x, y = (int(x) * grid for x in row["bucket"].split(","))
                out.write(
                    f'<rect x="{x}" y="{y}" width="{grid}" height="{grid}" '
                    'fill="#e47c58" fill-opacity="0.65" stroke="white"/>'
                )
                out.write(f'<text x="{x + 5}" y="{y + 20}" font-size="18">{row["events"]}</text>')
        out.write(
            "</svg><h2>Patterns and denominators</h2><p>Time buckets are elapsed intervals of "
            f"{metadata['config']['time_bin']} seconds; grid buckets are column,row at "
            f"{grid} pixels per cell. Event time bucket uses event start.</p>"
        )
        out.write(
            "<table><tr><th>Group</th><th>Bucket</th><th>Vehicle IDs</th>"
            "<th>IDs with speed</th><th>Events</th><th>Braking</th><th>Acceleration</th>"
            "<th>Events/ID</th></tr>"
        )
        for row in exports.aggregates():
            out.write(
                "<tr>"
                + "".join(
                    f"<td>{esc(row[key])}</td>"
                    for key in (
                        "category",
                        "bucket",
                        "tracked_vehicles",
                        "vehicles_with_speed",
                        "events",
                        "braking_events",
                        "acceleration_events",
                        "events_per_tracked_vehicle",
                    )
                )
                + "</tr>"
            )
        out.write(
            "</table><h2>Review individual events</h2><table><tr><th>ID / track</th>"
            "<th>Type / time (s)</th><th>Peak m/s²</th><th>Evidence / quality</th></tr>"
        )
        for event in exports.events():
            clip = event.get("clip")
            link = f'<a href="{esc(clip["path"])}">Review clip</a>' if clip else "Clip unavailable"
            if clip and clip["context_truncated"]:
                link += " (context truncated at video/run boundary)"
            out.write(
                f"<tr><td>{event['event_id']} / #{event['track_id']}</td>"
                f"<td>{esc(event['kind'])}: {event['start_seconds']:.2f}–"
                f"{event['end_seconds']:.2f}</td><td>{event['peak_acceleration_mps2']:.2f}</td>"
                f"<td>{link}<br>{esc(', '.join(event['quality_flags']))}</td></tr>"
            )
        out.write(
            '</table><p>Machine-readable files: <a href="measurements.csv">measurements.csv</a>, '
            '<a href="events.json">events.json</a>, <a href="aggregates.csv">aggregates.csv</a>, '
            '<a href="run.json">run.json</a>. Preserve the source footage for review.</p></html>'
        )
