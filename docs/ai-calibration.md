# Experimental AI calibration

Implemented 2026-10-30. This is an optional scale estimate, not independently measured
calibration. It does not change the manual rectangle workflow or its schema.

## Models and dependencies

The `ai` extra installs Transformers 4.57.6 (locked) and Hugging Face Hub 0.36.2.
No extra dependencies or model downloads are required to use a previously saved
AI calibration. Models execute locally; only model downloads use the network.

| Role | Pinned model and revision | Weight size / runtime | License information |
|---|---|---|---|
| Metric camera-Z depth and focal length | `apple/DepthPro-hf` at `de816c8ce7168afcb231f96d501d72b869d0beda` | 1.904 GB; half precision on MPS/CUDA, float32 on CPU | Model card identifies `apple-amlr`; follow the model's linked terms. |
| Semantic road probability | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` at `21b3847fae21ddee674abd31129307b6a1235bd9` | 14.96 MB; float32 | Linked NVIDIA SegFormer license limits use to noncommercial research/evaluation. |

Models are loaded through Transformers implementations without `trust_remote_code`.
The depth model's FOV head is enabled. Its processed metric depth and focal length
are used directly, without display normalization. The segmentation model's pinned
label 0 is road. Weights stay in the user's Hugging Face cache, not Git.

Primary sources checked 2026-10-30:

- [Depth Pro paper/code](https://github.com/apple-aiml-research/ml-depth-pro),
  [official HF model](https://huggingface.co/apple/DepthPro-hf), and
  [Transformers Depth Pro documentation](https://huggingface.co/docs/transformers/model_doc/depth_pro).
  Metric depth and focal-length estimation are supported, but their benchmark results
  do not validate speeds from this camera. The original package declares NumPy <2;
  the Transformers implementation avoids that conflict with TrafficAI's baseline.
- [NVIDIA road-segmentation checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
  and [linked license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).
  This is an optional research/evaluation path. It is not a blanket license for
  production/commercial deployment; select appropriately licensed models before that use.
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) also estimates metric depth and
  camera properties, but its documented setup centers on Linux/CUDA. Depth Pro through
  Transformers was selected and exercised on this Apple Silicon machine instead.
- Relative-depth-only models cannot supply meters by themselves. Depth Anything V2
  was not added as a substitute with a fabricated conversion factor.

## Geometry and rejection rules

1. Decode at least three distinct frames using actual timestamps (default 0, 1, 2 s).
   Registration compares later samples against the first with ORB matches and a robust
   image homography. Insufficient matches, unreliable inliers or more than 4 pixels
   of estimated view displacement at the working resolution rejects the attempt.
2. Resize reference frames to at most 640 pixels on their longest side. Depth Pro retains its
   pretrained internal preprocessing resolution. Use square pixels, centered principal
   point and predicted focal length to back-project each valid depth pixel:
   `P = Z * inverse(K) * [u, v, 1]`. These intrinsics are assumptions/estimates.
3. Select finite depths from 2–100 estimated meters with road probability >=0.7.
   Erode boundaries and require sufficient image coverage. No bounding rectangle or
   convex hull fills vehicle holes, medians, sidewalk gaps or unseen regions.
4. Fit a deterministic RANSAC plane using at most 6,000 points and 250 proposals,
   then refine by orthogonal least squares. Require >=65% support within 0.35 estimated
   meters, non-collinear support, an oblique road-facing orientation, and estimated
   camera-plane distance of 0.5–40 m. These are rejection heuristics, not confidence bounds.
5. Across samples, require camera-height spread <=20%, focal spread <=15%, and normal
   difference <=8 degrees. Keep pixels supported in at least two thirds of samples
   (minimum two), with positive ray intersections and <=20% variation in ray depth.
   Remove horizon-adjacent pixels, tiny components and insufficiently distributed regions.
6. Use median plane normal/offset and focal length to form one fixed mapping. For
   unit plane normal n and offset d, intersect each camera ray r at `-d*r/(n·r)`.
   Project the resulting point onto two orthonormal road-plane axes. The origin is
   the perpendicular projection of the camera onto the estimated plane; it is not
   georeferenced. This yields metric coordinates only to the extent AI scale is right.
7. Release depth models before loading vehicle tracking. Existing motion/history/event
   logic then runs unchanged, with AI provenance flags added separately from temporal
   validity flags. Every second of processed video, compare the camera view against
   the saved reference. Once a view check fails, further metric output stays unavailable.

The saved valid mask is authoritative; orange outlines are a visual guide. Interior
mask holes may be less visible than external contours. Field geometry must still be
reviewed. A stationary camera may be rejected in fog, darkness, low texture or heavy
occlusion. The orientation check can reject near-vertical/top-down views. Conversely,
a consistent wrong plane or wrong physical scale can pass every check. Slow camera
drift below registration thresholds and incorrect road segmentation can also escape.

All frames use one fixed mapping, so per-frame depth changes cannot directly create
acceleration. This does not eliminate detector jitter, tracking errors, curved-road
bias, grade changes, distortion, or scale bias. No calibration uncertainty interval is
claimed. Road probability and RANSAC inlier fraction are not speed confidence scores.

## Files and status

Manual calibration stays schema 1. AI calibration uses `schema_version: 2` and
`method: ai_depth_plane`, a homography, input dimensions/hash, compressed PNG road
mask/reference image, model revisions, assumptions, options and diagnostic statistics.
The compact `run.json.calibration` omits encoded images; the complete artifact is
`calibration.json`. JSON schema 2 is loaded by the same `--calibration` option.

`auto_calibration.json` records `accepted` or `rejected`, per-sample diagnostics and
rejection reason. Accepted means the heuristic consistency checks passed, not that
physical accuracy has been established. On rejection, combined analysis retains
tracking and marks all rows `ai_calibration_rejected`; standalone calibration exits 1.
If interrupted during calibration, `run.json` records a cancelled calibration phase;
no video has been analyzed yet. Do not interpret that directory as completed coverage.

AI rows and events carry `ai_scale_unvalidated` and `ai_geometry_unvalidated`.
They must not be represented as manual calibration. A later view failure adds
`ai_camera_view_unreliable`; run metadata records its time and reason and closes
current events through the existing interruption logic. The pipeline never substitutes
an assumed road width, a pixel velocity labeled mph, or a failed depth output.

Model-free tests construct a known camera above a plane and verify distance/speed,
serialization, mask holes, temporal inconsistency, missing road, bad depth, vertical
surfaces, view motion, saved-calibration reuse and failure retaining tracking.
