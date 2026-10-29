# Approach selection and evidence

Primary documentation checked 2026-10-29. These comparisons are architectural judgments,
not TrafficAI accuracy or speed benchmarks. Published detection/tracking benchmarks do
not validate this camera, its calibration, speed estimates, or candidate event accuracy.

| Choice | Accuracy considerations | Runtime / hardware | Dependency and license considerations | Decision |
|---|---|---|---|---|
| Ultralytics YOLO26 nano + ByteTrack | Preserve current vehicle detector; low-score association can recover tracks. ID switches and occlusion remain. COCO classes may miss unusual vehicles. | Small model, CPU supported; optional CUDA/MPS. Association adds no appearance network. No promised FPS. | Existing PyTorch + Ultralytics stack; lap solver. Ultralytics distribution uses AGPL-3.0 or commercial licensing. Original ByteTrack repo is MIT; that does not change the integrated distribution's terms. | Default baseline. Explicitly select ByteTrack rather than inheriting upstream tracker defaults. |
| YOLO11 or larger YOLO26 weights | Useful controlled comparison; larger detector may improve recall, but camera-specific validation is required. | Same adapter/dependencies; larger weights generally require more compute and memory. | Same Ultralytics licensing considerations. | User-selectable compatible detection model; no accuracy superiority claim. |
| BoT-SORT, appearance ReID disabled | Camera motion compensation and association can improve some sequences. It cannot make a fixed homography valid after camera movement. | Additional camera motion computation; no added embedding network in supported mode. | Available within existing Ultralytics dependency. Original BoT-SORT repo is MIT; integrated code terms still apply. | Optional `--tracker botsort.yaml`; recalibrate if camera geometry changes. ReID is rejected. |
| Four-point measured homography | Metric scale comes from physical dimensions; valid only on the calibrated plane and inside the polygon. Sensitive to corner placement, camera distortion, and contact-point errors. | Negligible relative to detection; CPU only is sufficient. | Existing OpenCV dependency (Apache-2.0 for current OpenCV); no learned geometry model. | Implemented baseline. |
| Depth Anything V2 | Relative depth alone cannot establish physical scale. Metric models have learned/domain-specific priors; traffic ground-plane scale needs independent validation. Temporal consistency and occlusion add problems. | Additional transformer inference, weights and memory; hardware-specific throughput must be measured. | Extra model/dependencies. Official repository distinguishes Apache-2.0 Small weights from CC-BY-NC-4.0 Base/Large/Giant; check each chosen checkpoint. | Experimental follow-up only. No depth runtime or automatic calibration claims. |

Sources:

- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/):
  persistent frame-sequence tracking, custom YAML settings, thresholds, ByteTrack and
  BoT-SORT behavior. TrafficAI passes `persist=True`, `show=False`, `save=False` and
  the explicitly selected tracker. Default detection confidence is 0.1 so ByteTrack's
  low-confidence association is not preemptively removed by a 0.3 detector cutoff.
- [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/):
  model family and evaluation context. No advertised benchmark is reused as an
  end-to-end TrafficAI result.
- [Ultralytics licensing](https://www.ultralytics.com/license),
  [original ByteTrack repository](https://github.com/FoundationVision/ByteTrack),
  [original BoT-SORT repository](https://github.com/NirAharon/BoT-SORT).
  Deployment must account for the licenses of the actual code and weights selected;
  this rebuild does not assign a new license to the owner's repository.
- [OpenCV homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html):
  planar point correspondence and perspective correction. Four correspondences fit
  exactly; that is not independent accuracy validation.
- [OpenCV wheel guidance](https://github.com/opencv/opencv-python): install one cv2
  distribution. We use regular opencv-python, already required by Ultralytics, with
  GUI calls opt-in. Do not install headless and regular wheels together.
- [OpenCV VideoWriter reference](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
  and [PyAV time documentation](https://pyav.basswood.io/docs/stable/api/time.html):
  ordinary OpenCV writer configuration takes a fixed FPS; PyAV exposes frame PTS and
  time bases. PyAV was selected to retain variable-rate video time, detect invalid
  timestamps, and report encoding/muxing failures. Output H.264 is decoded once to
  verify readability when the writer closes.
- [Depth Anything V2 official repository](https://github.com/DepthAnything/Depth-Anything-V2)
  and [project page](https://depth-anything-v2.github.io/): relative and metric variants,
  weight-specific licenses, image-based depth. Neither establishes calibrated vehicle
  speed accuracy for this footage.

## Follow-up work requiring evidence

1. Field validation: surveyed control points, held-out road distances, independently
   timed vehicle passages, speed reference instrument where appropriate; quantify bias,
   error by image location, speed band, lighting and occlusion. Tune event thresholds
   against blinded human review, reporting false positives and missed events.
2. Camera intrinsics/lens correction, multi-point robust homography and explicit
   uncertainty propagation. Separate planes for sloped approaches; detect camera drift.
3. Better tire/contact anchors or segmentation; compare detector sizes and trackers on
   labeled intersection trajectories without driver identity or cross-camera matching.
4. Lane-based trajectories, configured movement validation, and signal-state input with
   synchronized timing before any turn/red-light classification. Do not infer these
   from speed changes. Near-miss/conflict analysis needs separate validated geometry
   and interaction models.
5. Experimental metric depth only after scale anchoring, temporal validation and
   comparison with the measured homography baseline; relative depth is not meters.
6. Wall-clock alignment and repeat surveys across recordings with documented coverage,
   reviewed events and uncertainty. These would support investigation, not establish
   infrastructure causation or a funding case by themselves.
