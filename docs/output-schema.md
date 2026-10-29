# Output schema (version 1)

Every analysis writes a fresh `run_<UTC timestamp>_<random suffix>/` directory. Video
and CSV are incremental; a caught failure/cancellation finalizes available outputs.
`run.json.status` is `completed`, `limited` (`--max-frames`), `cancelled` or `failed`.
An abrupt OS kill or power loss may leave `running`, partial CSV or an unfinalized MP4.
Do not treat incomplete runs as complete surveys. Existing runs are never reused.

## measurements.csv

One row per tracked detection; detections without an assigned ID are not exported.
Missing numeric values are empty CSV cells, never zero or fabricated values.

| Fields | Meaning |
|---|---|
| frame_index, timestamp_seconds | Zero-based decoded frame index and real presentation time relative to the first frame. No wall-clock time inferred. |
| track_id, segment | Run-scoped tracker ID; globally allocated motion segment ID changes on reset. No personal or cross-camera identity. |
| class_id, label, confidence | Detector class/name/confidence; default COCO IDs 2,3,5,7. Confidence is not speed accuracy. |
| x1_px, y1_px, x2_px, y2_px | Image-space box corners. |
| contact_x_px, contact_y_px | Bottom-center proxy for road contact. |
| road_x_m, road_y_m | Position inside calibrated road polygon; blank otherwise. |
| speed_mps | Magnitude of fitted planar velocity, meters/second. |
| acceleration_mps2 | Slope of smoothed speeds over actual time, meters/second squared. Positive is speeding up, negative slowing. Not lateral/centripetal acceleration. |
| effective_time_seconds | Mean timestamp of positions used for velocity fit. |
| acceleration_time_seconds | Mean timestamp of speeds used for acceleration fit; events use this estimate time. |
| support_start_seconds | Earliest observation supporting the current derivative (or velocity). Support ends at timestamp_seconds. |
| quality_flags | Pipe-separated flags; empty means checks passed, not verified physical accuracy. |
| zones | JSON array of containing named polygon features. Stop lines are overlays, not area groups. |

Flags: `uncalibrated`, `outside_road_region`, `invalid_projection`, `short_history`,
`reacquired`, `track_jump`, `observation_gap`, `gap_in_motion_window`,
`noisy_motion`, `implausible_speed`, `acceleration_warmup`.
Gaps can leave usable average speed while acceleration/events are unavailable. Invalid
projection, excessive gap, jump or excessive residual starts a fresh motion segment.
Histories are limited to configured time windows and at most 512 samples each;
inactive state is removed after `track_ttl` seconds. Very high frame rates may not fit
minimum support into that bound and therefore correctly yield no speed.

## events.json

JSON array (empty for no candidates). Each event contains:

- `event_id`, `track_id`, `segment`, `kind` (`braking` or `acceleration`),
  `classification: "candidate_safety_event"`.
- `start_seconds`, `end_seconds`: smoothed acceleration estimate times above the
  hysteresis release threshold. Smoothing limits onset precision; these are not exact
  physical braking onset/offset times.
- `confirmed_seconds`: observation time at which minimum duration was satisfied;
  `last_observed_seconds`: last supporting detection time.
- `support_start_seconds`: earliest supporting observation for the onset estimate.
- `peak_acceleration_mps2`, `start_speed_mps`, `end_speed_mps`, explicit `units` object.
  Speeds refer to contemporaneous speed fits, whose centers differ from acceleration centers.
- `pixel_position`, `road_position_m`, `peak_observed_seconds`: contact location and
  observation time associated with the peak derivative estimate. This is an approximate
  event location with smoothing lag, not a reconstructed exact onset position.
- `zones`: containing configured polygons at that peak observation position.
- `quality_flags`: `smoothed_estimate`, `manual_calibration` plus termination reason
  (`hysteresis_release`, `observation_gap`, `interrupted_measurement`, `track_expired`,
  `completed`, `limited`, `cancelled`, `failed`). A terminal boundary may truncate an event.
- `clip`: null when disabled/unavailable; otherwise relative `path`, actual source
  `start_seconds`, `end_seconds`, frame count, `audio: false`, `context_truncated`.
  Clip begins at supporting-history start minus `clip_before` and ends at last
  observation plus `clip_after`, clamped to the processed video. Clip time starts at zero.
  Seek/encode failures add `clip_failed` and `clip_error`; see run warnings.

Entry defaults: acceleration >=3 m/s² or braking <=-3 m/s², lasting >=0.5 s in the
active band. An active event releases below half its entry magnitude (hysteresis).
No duplicate is emitted each frame. Interrupted history closes confirmed candidates;
unconfirmed impulses are discarded. Reacquisition requires derivative warmup again.
These defaults are screening choices, not legal or engineering standards.

## aggregates.csv and analysis.sqlite

`category` is `time`, `grid` or `feature`. `bucket` is a zero-based elapsed time bin
(`--time-bin`, default 60 s), image grid `column,row` (`--grid-size`, default 100 px),
or configured feature name. Event time bins use smoothed event start; spatial bins
use peak observation position. Buckets include exposure without events.

`tracked_vehicles` counts distinct IDs observed within that bucket;
`vehicles_with_speed` counts IDs with any valid speed there. `events`, `braking_events`,
`acceleration_events` are raw candidate counts. `events_per_tracked_vehicle` and
`events_per_vehicle_with_speed` divide by those denominators (blank for zero).
The latter denominator does not guarantee enough continuous acceleration coverage;
inspect row quality and clips. These are screening rates, not validated risk rates.
IDs can span multiple bins/features, so bucket denominators must not be summed to
obtain run totals. Multiple events per ID are possible. ID fragmentation inflates
counts; missed detections and ID switches bias both counts and rates.

SQLite backs exact unique-ID aggregation without retaining all vehicles/events in RAM.
Tables: `tracks`, `exposure`, `counts`, `events` (JSON payloads). The stable exchange
formats are CSV/JSON; SQLite tables are an implementation detail.

## run.json and report.html

Metadata includes schema/app/Python/dependency versions, source SHA256/path, video
metadata and timestamp source, exact configuration, model/version/device/weights
SHA256 and resolved tracker settings, full calibration assumptions, feature settings,
status, processed frames, duration (last-minus-first timestamp), summary and warnings.
The summary reports unique track IDs, IDs/rows with speed, speed and acceleration coverage fractions,
flag counts, candidate counts and normalized counts. It does not estimate undetected
vehicles or invent missing observations. Signal/turn/conflict categories explicitly
remain unavailable, even when descriptive movement rules are present.

The static HTML report links annotated H.264 MP4, individual clips, source reference
`scene.jpg`, machine-readable files and an image-space event grid. Image-space cell
areas are not equal road areas. Audio is omitted. Full-resolution video is retained;
odd-sized frames use H.264 yuv444p, which some browsers cannot play (use a video player).
