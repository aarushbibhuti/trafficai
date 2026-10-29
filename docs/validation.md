# Validation record — 2026-10-29

Environment: macOS Apple Silicon, CPython 3.12.13. Locked stack includes Ultralytics
8.4.67, PyTorch 2.12.0, OpenCV 4.13.0.92, PyAV 19.0.2, NumPy 2.4.6 and lap 0.5.13.

## Automated checks

- `uv run pytest -q`: **42 passed** (latest full run 0.80 s).
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv lock --check`: passed. Lockfile was regenerated with `uv lock`.
- `uv pip check`: passed.
- `uv build`: wheel and source distribution built successfully.
- `git diff --check`: passed.
- Separate empty environment: `UV_PROJECT_ENVIRONMENT=/tmp/trafficai-locked-rebuild-validation
  uv sync --locked --python 3.12`; all 42 tests passed in that environment (1.78 s).
- Wheel installed into another empty Python 3.12 virtualenv using `uv pip install`.
  Dependency check, CLI help and a 5-frame CPU inference run passed. This unpinned
  install selected newer compatible versions (including Ultralytics 8.4.192,
  PyTorch 2.15.0 and OpenCV 4.14.0.94). Use the lockfile for the main tested stack.

Tests include independently specified distances and timestamps, axis-aligned and
diagonal motion, trapezoidal perspective, zero speed, irregular time intervals,
missed observations, positive/negative quadratic motion, seeded measurement noise,
short histories, track jumps, reacquisition, inactive cleanup, bounded histories,
threshold duration/hysteresis/deduplication, feature validation and exact aggregation
with repeated IDs. Tiny encoded video tests cover calibration correction/save/load,
headless operation with all GUI calls patched to fail, missing/nonmonotonic timestamps,
variable-rate timestamp preservation, cancellation, simulated writer failure, complete
CSV/JSON/video exports, event clips and collision-free run directories.

## Real model execution on available footage

No video or weights were added to Git. Existing ignored local assets were preserved.
Weights: local `yolo26n.pt`, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

```bash
uv run trafficai analyze data/videos/trafficshort.mp4 \
  --model yolo26n.pt --device cpu --output data/rebuild-validation/final
uv run trafficai analyze data/videos/newtraffic.mkv \
  --model yolo26n.pt --device mps --max-frames 50 --no-clips \
  --output data/rebuild-validation
```

- MP4 / CPU: **376 frames**, 1280×720, nominal 25 FPS, 15.0 s first-to-last
  presentation-time span; **3,282 tracked observation rows**, **77 distinct tracker IDs**.
  Annotated output was decoded end-to-end: 376 frames and final timestamp 15.0 s.
  These are tracker outputs, not manually verified vehicle counts.
- MKV / MPS: **50 frames**, 1.96 s time span, **607 observation rows**, **21 IDs**.
  Metadata correctly labels this deliberately limited run as `limited`.
- No measured real calibration was available. Both real runs correctly leave every
  metric speed/acceleration unavailable, emit zero candidate events, and identify
  motion-event analysis as unavailable. This does not demonstrate safe traffic.
- A representative annotated MP4 frame was visually inspected. Labels were subsequently
  given dark backgrounds for readability against bright sky and pavement.

## Synthetic physical/integration demonstration

```bash
uv run python examples/synthetic_demo.py \
  --output data/rebuild-validation/final-synthetic
```

Known 0.2 m/pixel plane; one run-scoped track; 140 frames at 20 FPS. Motion accelerates
at +4 m/s², brakes at -4 m/s², then continues at constant speed. Output contains
**two candidates**, one acceleration and one braking. Peak estimates were
approximately +3.99999995 and -3.99999995 m/s². Both contextual H.264 clips decoded;
context truncation at the recording boundary is explicitly marked. The report links
clips and displays per-location/time exposure counts. Exact ideal motion tests and
this integration fixture validate arithmetic and plumbing, not real detector accuracy.

## Limits and remaining verification

- No surveyed scene dimensions, independently timed passages or speed reference was
  available. Real-world speed/acceleration error, event precision/recall, and traffic
  volume accuracy remain **unvalidated**. Do not use synthetic tolerance as a field claim.
- CPU and MPS execution were tested; CUDA hardware was unavailable.
- Headless calibration was tested. Interactive click/undo/preview GUI workflows were
  not exercised manually; they require an available graphical session.
- Linux CI was added but has not run remotely; Windows installation was not tested.
- Model auto-download was not exercised because compatible local weights existed.
- Memory is structurally bounded for frames and histories; SQLite stores cumulative
  counts/events on disk. The locked ByteTrack implementation bounds removed-track
  history to 1,000 entries. A multi-hour process-memory soak test was not run.
- macOS wheels log duplicate FFmpeg AVFoundation class warnings when OpenCV and PyAV
  coexist. File processing passed, but native camera capture is unsupported. The old
  overlapping cv2 install required `uv sync --reinstall-package opencv-python`; a fresh
  environment installed without that repair.
- Advanced depth, conflict, red-light and prohibited-turn analysis are intentionally
  unavailable. Follow-up validation work is scoped in `research.md`.
