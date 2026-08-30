# trafficAI

Processes intersection/traffic camera footage to track vehicles, estimate speed, and flag sudden braking or acceleration. Built for urban planners and traffic analysts. Outputs an annotated video plus a JSON log of events.

---

## Install

```bash
pip install ultralytics opencv-python transformers torch Pillow
```

On Windows with a GPU, swap the torch line for the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/). On Mac (Apple Silicon), the standard install already uses MPS. On Azure ML, torch is preinstalled, just add `transformers` and `Pillow`.

---

## Usage

**Just point it at a video:**
```bash
python trafficAI.py --video footage.mp4
```

This runs in depth mode by default, Depth Anything V2 fits a ground plane to the road surface and uses lane markings as a scale reference. No camera calibration needed.

**If you know the road type, say so (improves accuracy):**
```bash
python trafficAI.py --video footage.mp4 --road-standard eu_motorway
```

| `--road-standard` | Where |
|---|---|
| `eu_motorway` | EU/UK motorways — default |
| `eu_urban` | EU urban roads and A-roads |
| `us_highway` | US freeways |
| `us_urban` | US city intersections |

**Headless / server:**
```bash
python trafficAI.py --video footage.mp4 --no-preview
```
Automatically set when running on Azure ML or any Linux box without a display.

**If you have a calibrated camera (more accurate):**
```bash
python trafficAI.py --video footage.mp4 --mode homography --calibration cam1.json
```
Run `calibrate_camera.py` first to generate the JSON — it walks you through clicking four corners of a known ground rectangle on a frame.

**Debug depth output:**
```bash
python trafficAI.py --video footage.mp4 --show-depth
```
Overlays the depth heatmap on the output video so you can see what the model is reading.

---

## Output

Saved to `data/processed/track_TIMESTAMP/`:

- `annotated.mp4` — original footage with bounding boxes, IDs, confidence, and speed
- `events.json` — every rapid acceleration or braking event with timestamp, tracker ID, and delta km/h

Bounding box labels show: `#ID  confidence  speed km/h`. Boxes turn red on a flagged event.

---

## How the speed works

Depth Anything V2 produces a disparity map (closer = brighter). The script fits a plane to the road surface using RANSAC — this handles camera tilt, perspective, and mounting height without any manual setup. Vehicle positions are projected onto that plane frame-to-frame, and lane dash lengths (which have known physical sizes by road standard) convert the relative displacement into km/h.

The only thing you need to tell it is the road standard, because that's a physical property of the road, not the camera.

---

## Notes

- First run downloads the depth model (~200 MB) from HuggingFace
- Depth mode runs ~2–4 fps on CPU, real-time on MPS/CUDA
- Speed readings stabilize after ~5 frames of tracking per vehicle
- If dashes aren't visible (night, heavy rain, no markings), scale anchoring won't update — existing estimate carries forward
