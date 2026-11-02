import os
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from backend.config import config
from backend.exceptions import ModelUnavailableException

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def resolve_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if req == "auto":
        dev = "cuda:0" if has_cuda else "cpu"
    elif req.startswith("cuda"):
        if has_cuda:
            dev = req
        else:
            logger.warning("CUDA device '%s' requested but torch.cuda.is_available() is False. Falling back to CPU.", req)
            dev = "cpu"
    else:
        dev = "cpu"
    return dev

class ObjectDetection:
    def __init__(
        self,
        label: str,
        confidence: float,
        bbox: Dict[str, float], # pixels: x1, y1, x2, y2
        bbox_norm: Dict[str, float], # normalized [0, 1]: x1, y1, x2, y2
        class_id: int,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None
    ):
        self.label = label
        self.confidence = float(confidence)
        self.bbox = bbox
        self.bbox_norm = bbox_norm
        self.class_id = int(class_id)
        self.center_x = float(center_x if center_x is not None else (bbox["x1"] + bbox["x2"]) / 2.0)
        self.center_y = float(center_y if center_y is not None else (bbox["y1"] + bbox["y2"]) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        x1 = round(self.bbox["x1"], 1)
        y1 = round(self.bbox["y1"], 1)
        x2 = round(self.bbox["x2"], 1)
        y2 = round(self.bbox["y2"], 1)
        return {
            "label": self.label,
            "class_name": self.label,
            "model_class_name": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "bounding_box": [x1, y1, x2, y2],
            "box": [x1, y1, x2, y2],
            "bbox_norm": {
                "x1": round(self.bbox_norm["x1"], 4),
                "y1": round(self.bbox_norm["y1"], 4),
                "x2": round(self.bbox_norm["x2"], 4),
                "y2": round(self.bbox_norm["y2"], 4)
            },
            "class_id": self.class_id,
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1)
        }

class ObjectDetector:
    """
    Production YOLO11n Object Detector.
    Supports Ultralytics PyTorch YOLO and ONNX Runtime.
    Loaded once at startup, reused across all requests, thread-safe bounded inference.
    """
    _offline_checked: bool = False

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.requested_device = device or config.YOLO_DEVICE
        self.device = resolve_device(self.requested_device)
        self.backend_type: Optional[str] = None
        self.model: Any = None
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.file_size_bytes: int = 0
        self.file_sha256: str = ""
        self.last_metadata: Dict[str, Any] = {}
        self._inference_lock = threading.Lock()
        self._load()

    def _attempt_download_weights(self, target_path: Path) -> bool:
        """
        Attempts download once with timeout if network is reachable.
        Never blocks startup indefinitely.
        """
        if ObjectDetector._offline_checked:
            return False

        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        timeout_sec = config.YOLO_DOWNLOAD_TIMEOUT_SECONDS
        try:
            import urllib.request
            target_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Checking network for YOLO11n weights download from %s...", url)
            req = urllib.request.Request(url, headers={"User-Agent": "BYAKU/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                content = response.read()
                target_path.write_bytes(content)
                logger.info("Successfully downloaded YOLO11n weights to %s (%d bytes).", target_path, len(content))
                return True
        except Exception as e:
            ObjectDetector._offline_checked = True
            logger.warning("Could not download weights automatically (%s: %s).", type(e).__name__, e)
            return False

    def _load(self):
        path_obj = Path(self.model_path)
        is_configured_model = path_obj == Path(config.YOLO_MODEL_PATH)

        # Check alternative default locations if primary not found
        if not path_obj.exists() and is_configured_model:
            for alt in [config.WEIGHTS_DIR / "yolo11n.pt", Path("yolo11n.pt"), Path("weights/yolo11n.onnx"), Path("yolo11n.onnx")]:
                if alt.exists():
                    self.model_path = str(alt)
                    path_obj = alt
                    break

        if not path_obj.exists():
            downloaded = self._attempt_download_weights(path_obj) if is_configured_model else False
            if not downloaded or not path_obj.exists():
                self.loaded = False
                self.load_error = (
                    f"Weight file '{self.model_path}' not found. "
                    f"To enable real object detection, place yolo11n.pt in weights/ "
                    f"(curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) "
                    f"or set YOLO_MODEL_PATH."
                )
                return

        # Record file metadata
        try:
            data = path_obj.read_bytes()
            self.file_size_bytes = len(data)
            self.file_sha256 = hashlib.sha256(data).hexdigest()
        except Exception as e:
            logger.warning("Could not compute file hash: %s", e)

        # ONNX Runtime branch
        if str(self.model_path).endswith(".onnx"):
            try:
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"]
                if self.device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                self.model = ort.InferenceSession(str(self.model_path), providers=providers)
                self.backend_type = "onnx"
                self.loaded = True
                self._warmup()
                return
            except Exception as e:
                self.loaded = False
                self.load_error = f"Failed to load ONNX model '{self.model_path}': {e}"
                return

        # Ultralytics PyTorch branch
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.backend_type = "ultralytics"
            self.loaded = True
            self._warmup()
        except ImportError:
            self.loaded = False
            self.load_error = (
                "Neither ultralytics nor torch could be imported. "
                "Install ultralytics with: pip install torch torchvision ultralytics"
            )
        except Exception as e:
            self.loaded = False
            self.load_error = f"Error loading YOLO weights from '{self.model_path}': {e}"

    def _warmup(self):
        """Run one warm-up inference on a blank frame at startup and exclude from timings."""
        if not self.loaded or self.model is None:
            return
        try:
            blank = np.zeros((config.YOLO_IMGSZ, config.YOLO_IMGSZ, 3), dtype=np.uint8)
            if self.backend_type == "ultralytics":
                try:
                    import torch
                    with torch.inference_mode():
                        self.model.predict(
                            blank,
                            conf=config.YOLO_CONF_THRESHOLD,
                            iou=config.YOLO_IOU_THRESHOLD,
                            imgsz=config.YOLO_IMGSZ,
                            device=self.device,
                            max_det=1,
                            verbose=False
                        )
                except ImportError:
                    self.model.predict(
                        blank,
                        conf=config.YOLO_CONF_THRESHOLD,
                        iou=config.YOLO_IOU_THRESHOLD,
                        imgsz=config.YOLO_IMGSZ,
                        device=self.device,
                        max_det=1,
                        verbose=False
                    )
            elif self.backend_type == "onnx":
                blob = cv2.dnn.blobFromImage(blank, 1/255.0, (config.YOLO_IMGSZ, config.YOLO_IMGSZ), swapRB=True, crop=False)
                input_name = self.model.get_inputs()[0].name
                self.model.run(None, {input_name: blob})
            logger.info("YOLO detector warmup completed on %s.", self.device)
        except Exception as e:
            logger.warning("YOLO detector warmup encountered an issue: %s", e)

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: Optional[int] = None
    ) -> List[ObjectDetection]:
        dets, _ = self.predict_with_meta(
            image_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det
        )
        return dets

    def predict_with_meta(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: Optional[int] = None
    ) -> Tuple[List[ObjectDetection], Dict[str, Any]]:
        """
        Executes bounded, thread-safe inference under torch.inference_mode.
        Returns: (List[ObjectDetection], metadata_dict)
        """
        if not self.loaded or self.model is None:
            raise ModelUnavailableException(
                model_name=str(self.model_path),
                hint=self.load_error or "Model weights missing. Place yolo11n.pt in weights/ or set YOLO_MODEL_PATH."
            )

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty or None.")

        orig_h, orig_w = image_bgr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

        conf = float(conf_threshold if conf_threshold is not None else config.YOLO_CONF_THRESHOLD)
        iou = float(iou_threshold if iou_threshold is not None else config.YOLO_IOU_THRESHOLD)
        m_det = int(max_det if max_det is not None else config.YOLO_MAX_DET)
        imgsz = int(config.YOLO_IMGSZ)

        detections: List[ObjectDetection] = []
        inference_ms: float = 0.0

        with self._inference_lock:
            if self.backend_type == "ultralytics":
                has_torch = False
                try:
                    import torch
                    has_torch = True
                except ImportError:
                    has_torch = False

                t0 = time.perf_counter()
                if has_torch:
                    with torch.inference_mode():
                        results = self.model.predict(
                            image_bgr,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=self.device,
                            max_det=m_det,
                            verbose=False
                        )
                else:
                    results = self.model.predict(
                        image_bgr,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=self.device,
                        max_det=m_det,
                        verbose=False
                    )
                inference_ms = (time.perf_counter() - t0) * 1000.0

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        raw_xyxy = box.xyxy[0]
                        xyxy = raw_xyxy.cpu().numpy() if hasattr(raw_xyxy, "cpu") else raw_xyxy
                        raw_conf = box.conf[0]
                        confidence_value = raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else raw_conf
                        confidence = float(np.asarray(confidence_value).item())
                        raw_cls = box.cls[0]
                        class_value = raw_cls.cpu().numpy() if hasattr(raw_cls, "cpu") else raw_cls
                        cls_id = int(np.asarray(class_value).item())

                        # Flatten if array
                        if hasattr(xyxy, "flatten"):
                            xyxy = xyxy.flatten()

                        # Ultralytics letterboxes internally and returns coords in original image space
                        # Clamp to [0, W-1] x [0, H-1]
                        x1 = max(0.0, min(float(orig_w - 1), float(xyxy[0])))
                        y1 = max(0.0, min(float(orig_h - 1), float(xyxy[1])))
                        x2 = max(0.0, min(float(orig_w - 1), float(xyxy[2])))
                        y2 = max(0.0, min(float(orig_h - 1), float(xyxy[3])))

                        # Drop degenerate boxes (zero width or zero height)
                        if (x2 <= x1) or (y2 <= y1):
                            continue

                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        norm_bbox = {
                            "x1": x1 / max(1.0, float(orig_w)),
                            "y1": y1 / max(1.0, float(orig_h)),
                            "x2": x2 / max(1.0, float(orig_w)),
                            "y2": y2 / max(1.0, float(orig_h))
                        }
                        label = self.model.names.get(cls_id, COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}")

                        detections.append(ObjectDetection(
                            label=label,
                            confidence=confidence,
                            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            bbox_norm=norm_bbox,
                            class_id=cls_id,
                            center_x=cx,
                            center_y=cy
                        ))

            elif self.backend_type == "onnx":
                # ONNX pipeline
                blob = cv2.dnn.blobFromImage(image_bgr, 1/255.0, (imgsz, imgsz), swapRB=True, crop=False)
                input_name = self.model.get_inputs()[0].name
                t0 = time.perf_counter()
                outputs = self.model.run(None, {input_name: blob})[0]
                inference_ms = (time.perf_counter() - t0) * 1000.0

                predictions = np.squeeze(outputs).T # shape [8400, 84]
                boxes_list = []
                confidences_list = []
                class_ids_list = []

                x_scale = float(orig_w) / float(imgsz)
                y_scale = float(orig_h) / float(imgsz)

                for row in predictions:
                    scores = row[4:]
                    cls_id = int(np.argmax(scores))
                    score = float(scores[cls_id])
                    if score >= conf:
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        x1 = (cx - 0.5 * w) * x_scale
                        y1 = (cy - 0.5 * h) * y_scale
                        bw = w * x_scale
                        bh = h * y_scale
                        boxes_list.append([int(x1), int(y1), int(bw), int(bh)])
                        confidences_list.append(score)
                        class_ids_list.append(cls_id)

                indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, conf, iou)
                if len(indices) > 0:
                    for idx in indices.flatten()[:m_det]:
                        bx, by, bw, bh = boxes_list[idx]
                        x1 = max(0.0, min(float(orig_w - 1), float(bx)))
                        y1 = max(0.0, min(float(orig_h - 1), float(by)))
                        x2 = max(0.0, min(float(orig_w - 1), float(bx + bw)))
                        y2 = max(0.0, min(float(orig_h - 1), float(by + bh)))

                        if (x2 <= x1) or (y2 <= y1):
                            continue

                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        cls_id = class_ids_list[idx]
                        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

                        detections.append(ObjectDetection(
                            label=label,
                            confidence=confidences_list[idx],
                            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            bbox_norm={"x1": x1/orig_w, "y1": y1/orig_h, "x2": x2/orig_w, "y2": y2/orig_h},
                            class_id=cls_id,
                            center_x=cx,
                            center_y=cy
                        ))

        metadata = {
            "image_width": orig_w,
            "image_height": orig_h,
            "inference_ms": round(inference_ms, 2),
            "device": self.device,
            "model": Path(self.model_path).name,
            "num_detections": len(detections)
        }
        self.last_metadata = metadata
        return detections, metadata

    def cleanup(self):
        """Release model and free CUDA cache on lifespan shutdown."""
        self.model = None
        self.loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Detector resources successfully cleaned up.")
