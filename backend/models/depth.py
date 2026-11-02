import os
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from backend.config import config

logger = logging.getLogger(__name__)

def resolve_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if req == "auto":
        return "cuda:0" if has_cuda else "cpu"
    elif req.startswith("cuda"):
        if has_cuda:
            return req
        logger.warning("CUDA device '%s' requested for depth estimation, but torch.cuda.is_available() is False. Falling back to CPU.", req)
        return "cpu"
    return "cpu"

class DepthEstimator(ABC):
    """
    Abstract interface for relative depth estimation.
    Convention: Relative inverse depth where larger value = closer object.
    Min-max normalized to [0.0, 1.0] per frame.
    Depth categories: NEAR (>=0.66), MID-RANGE (0.33 to 0.66), FAR (<0.33), UNKNOWN.
    Never outputs metric measurements (metres/centimetres).
    """
    @abstractmethod
    def estimate_frame(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes a normalized relative depth map in [0, 1] resized to the original image dimensions.
        Returns None if estimation is unavailable or depth map is degenerate/flat.
        """
        pass

    @abstractmethod
    def evaluate_object_depth(
        self,
        depth_map: Optional[np.ndarray],
        bbox: Any,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Calculates categorical and scalar relative depth for a detected object bounding box.
        """
        pass

class DepthAnythingV2Estimator(DepthEstimator):
    """
    Real Depth Anything V2 Small model integration.
    Supports Hugging Face transformers pipeline, official PyTorch DepthAnythingV2 architecture,
    and ONNX Runtime. Thread-safe, CPU/CUDA compatible, loaded once at startup.
    """
    def __init__(self, model_name: Optional[str] = None, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.DEPTH_MODEL_NAME
        self.weights_path = weights_path or config.DEPTH_WEIGHTS_PATH
        self.requested_device = device or config.DEPTH_DEVICE
        self.device = resolve_device(self.requested_device)
        self.backend_type: Optional[str] = None
        self.model: Any = None
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.last_inference_error: Optional[str] = None
        self.last_inference_ms: Optional[float] = None
        self.input_size = config.DEPTH_INPUT_SIZE
        self._inference_lock = threading.Lock()
        self._load()

    def _load(self):
        # 1. Try ONNX runtime model if an ONNX file exists
        onnx_candidates = [
            Path(self.weights_path).with_suffix(".onnx"),
            config.WEIGHTS_DIR / "depth_anything_v2_small.onnx",
            config.WEIGHTS_DIR / "depth_anything_v2_vits.onnx",
            Path("depth_anything_v2_small.onnx")
        ]
        for onnx_file in onnx_candidates:
            if onnx_file.exists():
                try:
                    import onnxruntime as ort
                    providers = ["CPUExecutionProvider"]
                    if self.device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                        providers.insert(0, "CUDAExecutionProvider")
                    self.model = ort.InferenceSession(str(onnx_file), providers=providers)
                    self.backend_type = "onnx"
                    self.loaded = True
                    self.weights_path = str(onnx_file)
                    logger.info("Loaded Depth Anything V2 ONNX model from %s on %s.", onnx_file, self.device)
                    self._warmup()
                    return
                except Exception as e:
                    logger.warning("Failed loading ONNX depth model %s: %s", onnx_file, e)

        # 2. Try Hugging Face transformers pipeline
        try:
            from transformers import pipeline
            logger.info("Attempting to load Depth Anything V2 from Hugging Face: %s on %s...", self.model_name, self.device)
            dev_id = 0 if self.device.startswith("cuda") else -1
            self.model = pipeline("depth-estimation", model=self.model_name, device=dev_id)
            self.backend_type = "transformers"
            self.loaded = True
            logger.info("Successfully loaded Depth Anything V2 via Hugging Face pipeline.")
            self._warmup()
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Could not load Depth Anything V2 via transformers (%s: %s).", type(e).__name__, e)

        # 3. Try official PyTorch model weights (.pth) if available
        pth_path = Path(self.weights_path)
        if not pth_path.exists():
            for alt in [config.WEIGHTS_DIR / "depth_anything_v2_vits.pth", Path("depth_anything_v2_vits.pth")]:
                if alt.exists():
                    pth_path = alt
                    break

        checkpoint_note = ""
        if pth_path.exists():
            checkpoint_note = (
                f" Found checkpoint '{pth_path}', but this runtime has no compatible "
                "Depth Anything V2 architecture loader for standalone .pth state dictionaries."
            )
            logger.warning("%s", checkpoint_note.strip())

        # Model weights / packages are missing in this environment
        self.loaded = False
        self.load_error = (
            f"Depth Anything V2 Small model is unavailable. "
            f"To enable real depth estimation: "
            f"1) Install dependencies: pip install torch torchvision transformers\n"
            f"2) Pretrained model will load automatically from '{self.model_name}' "
            f"or place a compatible ONNX model in weights/.{checkpoint_note}"
        )

    def _warmup(self):
        """Warm-up pass on a small blank frame at startup."""
        if not self.loaded or self.model is None:
            return
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.estimate_frame(dummy)
            logger.info("Depth Anything V2 warm-up completed on %s.", self.device)
        except Exception as e:
            logger.warning("Depth Anything warm-up error: %s", e)

    def estimate_frame(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs Depth Anything V2 inference.
        Returns a normalized relative depth map in [0.0, 1.0] with dimensions matching image_bgr (H, W).
        Returns None if estimation is unavailable or depth map is flat/invalid.
        """
        if not self.loaded or self.model is None:
            return None

        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return None

        orig_h, orig_w = image_bgr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return None

        with self._inference_lock:
            started_at = time.perf_counter()
            self.last_inference_error = None
            self.last_inference_ms = None
            try:
                # 1. Convert BGR to RGB
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                raw_depth: Optional[np.ndarray] = None

                # Branch A: Hugging Face transformers pipeline
                if self.backend_type == "transformers":
                    from PIL import Image
                    pil_img = Image.fromarray(image_rgb)
                    output = self.model(pil_img)
                    # output is dict with "depth" PIL Image
                    raw_depth = np.array(output["depth"], dtype=np.float32)

                # Branch B: ONNX Runtime
                elif self.backend_type == "onnx":
                    # Resize to input_size (e.g. 518x518)
                    resized = cv2.resize(image_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
                    # Normalize ImageNet mean/std
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    norm_img = ((resized / 255.0) - mean) / std
                    # NCHW layout
                    blob = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
                    input_name = self.model.get_inputs()[0].name
                    output = self.model.run(None, {input_name: blob})[0]
                    raw_depth = np.squeeze(output).astype(np.float32)

                if raw_depth is None or raw_depth.size == 0:
                    self.last_inference_error = "Depth model returned an empty map."
                    self.last_inference_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    return None

                # 2. Resize depth map back to original image dimensions (W, H)
                if raw_depth.shape[:2] != (orig_h, orig_w):
                    raw_depth = cv2.resize(raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                # 3. Check for flat / degenerate depth map
                d_min = float(np.min(raw_depth))
                d_max = float(np.max(raw_depth))
                if (d_max - d_min) < config.FLAT_DEPTH_EPSILON:
                    self.last_inference_error = "Depth model returned a flat map; relative depth is unknown."
                    self.last_inference_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    return None

                # 4. Min-max normalize per frame to [0.0, 1.0] where 1.0 = closest
                norm_depth = (raw_depth - d_min) / (d_max - d_min + 1e-8)
                norm_depth = np.clip(norm_depth, 0.0, 1.0).astype(np.float32)
                self.last_inference_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                return norm_depth

            except Exception as e:
                self.last_inference_error = f"{type(e).__name__}: {e}"
                self.last_inference_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                logger.exception("Depth estimation failed.")
                return None

    def evaluate_object_depth(
        self,
        depth_map: Optional[np.ndarray],
        bbox: Any,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Calculates relative depth category and median value for a detected object.
        Uses the central 50% region of the bounding box to prevent background bleed.
        Categories: NEAR (>=0.66), MID-RANGE (0.33 to 0.66), FAR (<0.33), UNKNOWN.
        Never outputs metric distance in metres.
        """
        unknown_result = {
            "depth_value": None,
            "relative_depth_value": None,
            "depth_category": "UNKNOWN",
            "depth_class": "UNKNOWN",
            "depth_confidence": "low"
        }

        if depth_map is None:
            return unknown_result

        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            return unknown_result

        # Parse bbox coordinates
        if isinstance(bbox, dict):
            bx1, by1 = bbox.get("x1", 0.0), bbox.get("y1", 0.0)
            bx2, by2 = bbox.get("x2", 0.0), bbox.get("y2", 0.0)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
        elif hasattr(bbox, "bbox"):
            return self.evaluate_object_depth(depth_map, bbox.bbox, frame_shape)
        else:
            return unknown_result

        x1 = max(0, int(round(float(bx1))))
        y1 = max(0, int(round(float(by1))))
        x2 = min(w, int(round(float(bx2))))
        y2 = min(h, int(round(float(by2))))

        box_w = x2 - x1
        box_h = y2 - y1

        # Reject tiny or degenerate bounding boxes (< 10 px or area < 100)
        if box_w < 10 or box_h < 10 or (box_w * box_h) < 100:
            return unknown_result

        # Verify depth map is not flat
        d_min = float(np.min(depth_map))
        d_max = float(np.max(depth_map))
        if (d_max - d_min) < config.FLAT_DEPTH_EPSILON:
            return unknown_result

        # Extract central 50% region (avoid edge pixels prone to background bleed)
        crop_x1 = int(round(x1 + 0.25 * box_w))
        crop_x2 = int(round(x1 + 0.75 * box_w))
        crop_y1 = int(round(y1 + 0.25 * box_h))
        crop_y2 = int(round(y1 + 0.75 * box_h))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return unknown_result

        region = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        if region.size == 0:
            return unknown_result

        # Robust central median depth statistic
        median_val = float(np.median(region))

        # Categorize
        if median_val >= config.NEAR_THRESHOLD:
            category = "NEAR"
        elif median_val >= config.MID_THRESHOLD:
            category = "MID-RANGE"
        else:
            category = "FAR"

        return {
            "depth_value": round(median_val, 3),
            "relative_depth_value": round(median_val, 3),
            "depth_category": category,
            "depth_class": category,
            "depth_confidence": "high"
        }

    def cleanup(self):
        """Releases depth model resources on shutdown."""
        self.model = None
        self.loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Depth estimator cleaned up.")
