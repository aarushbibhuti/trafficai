"""Optional local model inference. No model code executes outside Transformers."""

import gc
from importlib.metadata import version

import cv2
import numpy as np

DEPTH_MODEL = "apple/DepthPro-hf"
DEPTH_REVISION = "de816c8ce7168afcb231f96d501d72b869d0beda"
ROAD_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
ROAD_REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"


class DepthRoadModels:
    """Infer camera-Z depth in meters and semantic road probability on a small frame."""

    def __init__(self, device="auto", local_files_only=False, progress=print):
        try:
            import torch
            from transformers import (
                DepthProForDepthEstimation,
                DepthProImageProcessor,
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as exc:
            raise ValueError("AI dependencies missing. Run: uv sync --locked --extra ai") from exc
        if device == "auto":
            device = (
                "cuda:0"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable for AI calibration; use --ai-device cpu")
        self.torch, self.device = torch, device
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        self.depth = self.road = None
        self.info = {
            "depth_model": DEPTH_MODEL,
            "depth_revision": DEPTH_REVISION,
            "road_model": ROAD_MODEL,
            "road_revision": ROAD_REVISION,
            "transformers_version": version("transformers"),
            "device": device,
            "precision": str(self.dtype),
            "scale_status": "ai_estimated_unvalidated",
            "licenses": {"depth": "apple-amlr", "road": "NVIDIA research/evaluation-only"},
        }
        progress(
            f"AI calibration: loading {DEPTH_MODEL} and road segmentation on {device}. "
            "First use downloads approximately 1.9 GB of weights; inference stays local."
        )
        try:
            self.depth_processor = DepthProImageProcessor.from_pretrained(
                DEPTH_MODEL, revision=DEPTH_REVISION, local_files_only=local_files_only
            )
            self.depth = (
                DepthProForDepthEstimation.from_pretrained(
                    DEPTH_MODEL,
                    revision=DEPTH_REVISION,
                    local_files_only=local_files_only,
                    dtype=self.dtype,
                    use_fov_model=True,
                    attn_implementation="sdpa",
                )
                .to(device)
                .eval()
            )
            processor_config, _ = SegformerImageProcessor.get_image_processor_dict(
                ROAD_MODEL, revision=ROAD_REVISION, local_files_only=local_files_only
            )
            # Translate this pinned checkpoint's legacy processor keys explicitly.
            processor_config.pop("feature_extractor_type", None)
            processor_config["do_reduce_labels"] = processor_config.pop("reduce_labels", False)
            self.road_processor = SegformerImageProcessor(**processor_config)
            self.road = (
                SegformerForSemanticSegmentation.from_pretrained(
                    ROAD_MODEL,
                    revision=ROAD_REVISION,
                    local_files_only=local_files_only,
                )
                .to(device)
                .eval()
            )
        except Exception as exc:
            self.close()
            raise ValueError(
                f"AI model loading failed: {exc}. Check network/cache and free memory; "
                "--ai-device cpu is available if GPU inference fails"
            ) from exc

    def predict(self, frame):
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        with torch.inference_mode():
            road_input = self.road_processor(images=rgb, return_tensors="pt").to(self.device)
            logits = self.road(**road_input).logits.float()
            probability = (
                torch.nn.functional.interpolate(
                    logits, size=(height, width), mode="bilinear", align_corners=False
                )
                .softmax(dim=1)[0, 0]
                .cpu()
                .numpy()
            )  # pinned Cityscapes label 0 = road
            del logits, road_input
            inputs = self.depth_processor(images=rgb, return_tensors="pt").to(
                self.device, dtype=self.dtype
            )
            outputs = self.depth(**inputs)
            result = self.depth_processor.post_process_depth_estimation(
                outputs, target_sizes=[(height, width)]
            )[0]
            depth = result["predicted_depth"].float().cpu().numpy()
            focal = float(result["focal_length"].item())
        intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        return depth, intrinsics, probability

    def close(self):
        self.depth = self.road = None
        gc.collect()
        if getattr(self, "device", None) == "mps":
            self.torch.mps.empty_cache()
