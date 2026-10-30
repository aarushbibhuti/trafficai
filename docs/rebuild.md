# Repository audit and rebuild design

Inspected 2026-10-29 after fetching origin. Clean starting working tree; ignored local
videos and weights preserved. Rebuild branch: `codex/trafficai-rebuild`, based on
`origin/main` e06bba4. No merge into main.

## Existing behavior and confirmed defects

- Initial main 9ee88f5: hardcoded YOLO26/ByteTrack vehicle runner; greeting-only main.py.
- speed-tracker 3fcc56f: useful frame selection, four-corner homography, bottom-center
  anchors, unit conversion and annotation prototype. Hardcoded file/model/settings.
- Speed uses only absolute transformed Y displacement, so transverse motion disappears.
  Elapsed time is observation count / FPS (also off by one); missed detections distort it.
- No road-region gate, calibration compatibility validation, continuity checks, inactive
  history cleanup, event detector, structured exports, tests or CI.
- supervision is imported but undeclared. Headless OpenCV conflicts with mandatory GUI;
  Ultralytics also depends on the regular OpenCV wheel, risking two cv2 installations.
- torch is imported only in the script guard; importing and calling main fails.
- Calibration accepts unchecked point geometry and unmeasured US/EU lane assumptions.
- Latest main e06bba4 only adds README claims: nonexistent scripts, automatic depth
  calibration, event export and performance numbers. Neither branch implements them.

## Proposed architecture (implemented in this rebuild)

Small installable Python 3.12 package, argparse CLI; no service or web framework.
`video`: PyAV decode/encode with presentation timestamps and bounded frame memory.
`detection`: lazy Ultralytics persistent ByteTrack adapter; injectable detector for tests.
`calibration`: validated four-corner measured rectangle, footage fingerprint, saved
JSON, still-image/interactive correction workflow and explicit assumptions.
`motion`: bounded per-track timed road coordinates, least-squares velocity, smoothed
speed derivative, discontinuity/gap gates; no invented metric values.
`events`: duration/hysteresis state machine scoped to continuous track segments.
`features`: image-space lanes/crosswalks/stop lines and descriptive movement rules.
`annotation`: boxes, quality-aware labels, calibration and configured features.
`exports`/`report`: streaming CSV, disk-backed aggregation, events JSON, metadata,
review HTML, spatial/time counts and exposure denominators; second-pass event clips.
`pipeline`: resource ownership, validation, cancellation and partial-run status.

Retain COCO vehicle classes 2/3/5/7 and default YOLO26 nano/ByteTrack. Regular
opencv-python is the sole cv2 distribution; GUI is opt-in and never called headlessly.
PyAV adds a dependency to obtain actual media timestamps, including variable frame
rate; absent/nonmonotonic timestamps fail explicitly instead of guessing elapsed time.
SQLite bounds aggregation memory as video length and track count grow. Only active
motion/event histories remain in RAM. Clips are generated one at a time after analysis.

Manual features are overlays and spatial grouping, not inferred rules. Red-light,
prohibited-turn and conflict/near-miss judgments remain unavailable. No identity tools.


## Optional AI extension (2026-10-30)

`ai_models` lazily loads pinned local depth/road models through the optional `ai`
extra. `auto_calibration` samples frames, fits/checks plane geometry, and returns an
explicit accepted/rejected result. `plane_calibration` provides schema-2 persistence,
masked projection and camera-view checks alongside the unchanged manual schema.
The existing pipeline accepts saved geometry or performs calibration before loading
vehicle tracking. AI provenance is separate from temporal validity, so valid but
unvalidated estimates can produce explicitly labeled candidate events. Rejected
geometry retains tracking without metric outputs. See `ai-calibration.md` for details.
