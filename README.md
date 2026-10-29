# TrafficAI

TrafficAI tracks vehicles in intersection footage, estimates road-plane speed when
valid manual calibration is supplied, and exports **candidate** sudden braking and
acceleration events for human review. It is an infrastructure analysis tool for urban
planners and community advocates, not a policing or driver identification system.

Implemented: YOLO vehicle detection, persistent ByteTrack IDs, calibrated planar
motion, timestamp-aware event screening, annotated video, contextual event clips,
CSV/JSON exports and an HTML report with location/time summaries and events per
tracked vehicle. Without calibration, tracking works and metric speed is explicitly
unavailable. Red-light, prohibited-turn, conflict and automatic depth analysis are
not implemented; reports mark those categories unavailable.

## Install

Supported and tested Python: **3.12**. Use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/aarushbibhuti/trafficai.git
cd trafficai
# Use the rebuild branch until these changes are integrated:
git switch codex/trafficai-rebuild
uv sync --locked --python 3.12
uv run trafficai --help
```

The rebuild branch is local until the repository owner pushes it. In an existing
checkout containing these changes, start with `uv sync`. Standard installation is
also supported with `python3.12 -m pip install .`; uv's lock gives reproducible versions.
On minimal Debian/Ubuntu, install `libgl1 libglib2.0-0` for the OpenCV wheel. CPU works
without a GPU; use a compatible PyTorch installation for CUDA. Model inference adds
substantial dependencies, but calibration and tests never download model weights.

Only **opencv-python** is declared; do not also install opencv-python-headless. If
upgrading the old environment breaks `import cv2`, repair overlapping old wheel files:

```bash
uv sync --locked --reinstall-package opencv-python
```

Headless means no GUI calls and is the default. The regular wheel permits optional
preview and matches Ultralytics' dependency. On macOS, the OpenCV/PyAV wheels may log
duplicate FFmpeg AVFoundation class warnings; file decoding/encoding was tested on
Apple Silicon, but camera capture is not supported or tested.

## Analyze a video

```bash
uv run trafficai analyze /path/to/footage.mp4 --output runs
# Use existing weights and explicit hardware; no model download is needed:
uv run trafficai analyze data/videos/trafficshort.mp4 \
  --model ./yolo26n.pt --device cpu --output runs
# Calibrated analysis with optional local GUI preview:
uv run trafficai analyze /path/to/footage.mp4 --calibration camera.json \
  --device auto --units mph --preview --output runs
```

Default model `yolo26n.pt` downloads through Ultralytics if not present (network
required). Use trusted compatible detection weights. `--device` accepts `cpu` (default),
`auto`, `mps`, or a supported CUDA device such as `cuda:0`. Failures identify model/device
problems. Default vehicle classes are COCO car, motorcycle, bus and truck (2,3,5,7).
For custom-trained models configure `--classes` for that model's class mapping.

Useful controls:

```bash
uv run trafficai analyze footage.mp4 --calibration camera.json \
  --confidence 0.1 --iou 0.7 --tracker examples/bytetrack.yaml \
  --braking-threshold 3 --acceleration-threshold 3 --event-duration 0.5 \
  --clip-before 3 --clip-after 3 --time-bin 60 --grid-size 100
uv run trafficai analyze footage.mp4 --max-frames 300 --no-clips
uv run trafficai analyze --help
```

Thresholds use m/s², motion distances meters, and durations seconds regardless of
`--units kmh|mph|mps` (display only). Tracker YAML exposes association thresholds and
lost-track buffer. `botsort.yaml` is supported with ReID disabled; appearance/identity
matching is rejected. Higher detector confidence can remove low-score detections
that ByteTrack would otherwise use. Ctrl-C/SIGTERM or Q in preview finalizes partial
outputs when possible; the metadata identifies cancellation. No GUI calls occur
without `--preview` or calibration's `--interactive` option.

## Calibrate and inspect

Use a **stationary camera and a genuinely planar road rectangle with known dimensions**.
Pick four ground points around that rectangle in order: P1, P2, P3, P4 around its
boundary. P1→P2 is the measured width; P2→P3 is the measured length. They map to
(0,0), (width,0), (width,length), (0,length) meters. The polygon is the valid measurement
region; tracks outside it remain visible but have no metric speed.

1. Extract any frame to inspect in an image editor (pixel origin is top-left):

   ```bash
   uv run trafficai frame footage.mp4 --at 5 --output frame.png
   ```

2. Select points and enter dimensions, either headlessly or interactively:

   ```bash
   # EXAMPLE numbers only: replace every point and dimension with your own measurements.
   uv run trafficai calibrate footage.mp4 --at 5 --output camera.json \
     --points "400,250 600,250 850,650 150,650" \
     --width-m 7.2 --length-m 20 \
     --dimensions-source "Surveyed road rectangle; describe measurement method/date"

   uv run trafficai calibrate footage.mp4 --at 5 --output camera.json \
     --interactive --width-m 7.2 --length-m 20 \
     --dimensions-source "Surveyed control points"
   ```

   Interactive controls: click four corners, U undo, R reset, Enter accept, Escape
   cancel. Choose `--at` to change the reference frame. Points must form an ordered
   convex footprint, sufficiently large and inside the image. Positive physical
   dimensions and a numerically valid projection are required.

3. Inspect `camera.preview.png`. Correct points/dimensions, saving a new version:

   ```bash
   uv run trafficai calibrate footage.mp4 --load camera.json --interactive \
     --width-m 7.1 --output camera-v2.json
   uv run trafficai analyze footage.mp4 --calibration camera-v2.json --output runs
   ```

   JSON is editable and validated again on load. `--overwrite` is required to replace
   a calibration/image. Matching video SHA256 and image dimensions are checked. For
   another recording from the same unchanged camera, inspect the view and explicitly
   use `--allow-calibration-reuse`. Resolution mismatch is always rejected; recalibrate
   after cropping, camera movement, zoom or lens changes. Matching dimensions alone
   cannot prove identical camera geometry.

Lane widths/dash cycles are **not measurements**. No road-standard preset is silently
applied. If using an assumed dimension, enter it explicitly, identify it as an
assumption in `--dimensions-source`, and replace it with a measurement when available.
Four matching corners cannot validate scale accuracy. Validate with held-out measured
road distances and independently timed passages before interpreting quantitative results.

## Intersection context

Copy and edit [examples/features.json](examples/features.json), including `image_size`,
then pass `--features my-intersection.json`. It supports convex lane, crosswalk and
zone polygons, stop-line segments, and permitted movements between named lanes.
Coordinates are pixels in the source frame. The example is illustrative, not a map
of the supplied footage. Polygons annotate the video and group events/exposure.
Stop lines and permitted movements are descriptive context only: there is no
red-light, stop-compliance or prohibited-turn classifier. Missing signal/rule evidence
is never guessed. There is no face, plate, driver or cross-camera identity analysis.

## Results and interpretation

Each run creates a fresh `runs/run_<UTC>_<suffix>/` directory:

- `annotated.mp4`: original cadence, track IDs, confidence, usable speed or “unavailable”,
  and candidate event labels after confirmation; no audio.
- `measurements.csv`: per-detection positions, timestamp, SI speed/acceleration,
  supporting times and measurement-quality flags. Missing numbers are empty cells.
- `events.json` and `clips/event_000001.mp4`: candidate event measurements, location,
  units and quality; clips include supporting history plus configured before/after
  context, truncated at run boundaries when necessary.
- `aggregates.csv`: candidate counts by elapsed time, image grid and configured zones,
  unique tracked IDs, IDs with usable speed, and events per tracked vehicle.
- `run.json`: configuration, model/weights, calibration assumptions, video coverage,
  processing status, missing-data flags and summary.
- `report.html`, `scene.jpg`: readable evidence review and event-location grid.
- `analysis.sqlite`: disk-backed aggregation and event storage.

Open `report.html` in a browser or open the MP4s in a video player. Full schemas and
flag meanings: [docs/output-schema.md](docs/output-schema.md).

Speed is the magnitude of a least-squares fit of **both** road-coordinate components
against decoded video timestamps. Bottom-center boxes approximate tire contact.
Default speed support is 0.7 s with >=3 observations spanning >=0.4 s; acceleration
fits these smoothed speeds over 0.8 s with >=0.4 s of support. Actual timestamps
include detection gaps and variable-rate footage. No FPS/observation-count clock is
used. Missing or nonmonotonic timestamps stop processing rather than invent time.

Short histories, bad projections, implausible motion and long gaps yield unavailable
speed. Brief detection gaps can retain average speed but suppress acceleration until
continuous support is rebuilt. Large jumps reset history. Events require sustained
threshold entry and hysteresis release, with no per-frame duplicate exports.

Smoothing delays and blurs onset; turns bias the magnitude of fitted vector velocity,
contact-box jitter biases derivatives, and identity switches can evade jump checks.
Occlusion, shadows, rolling shutter, slope, lens distortion and camera movement are
not solved by homography. Thresholds are configurable screening choices, not laws.
Tracker IDs approximate observed traffic volume; fragmentation and missed detections
bias rates. Time bins are video-relative, not clock time. A zero count with missing
measurement coverage does not establish safety. Reports support further investigation;
the software alone establishes neither causation nor a funding case.

## Verification and development

```bash
uv sync --locked --python 3.12
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

To generate a small model-free demonstration with known +4 and -4 m/s² motion,
two candidate events and review clips:

```bash
uv run python examples/synthetic_demo.py --output runs/synthetic-demo
```

Choose a new output directory on subsequent demo runs; existing data is protected.

Tests generate tiny videos and inject known trajectories; **no model download** is
needed. They cover homography validation, known distances/times, transverse motion,
missed observations, noisy motion, acceleration, track resets, deduplication, headless
processing, variable timestamps, writer failure, cancellation, and readable exports/clips.
Exact synthetic coordinates use 1e-6 m/s tolerance; ideal quadratic motion uses 1e-8
m/s² tolerance. Seeded 2.5 cm noise is checked within 0.2 m/s on its fixture. These
numerical tolerances are not field accuracy claims. CI runs checks on Python 3.12.

[Repository audit and architecture](docs/rebuild.md),
[research and alternatives](docs/research.md),
[validation results and remaining checks](docs/validation.md).
