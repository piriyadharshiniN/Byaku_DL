import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import cv2

import pytest
from backend.models.detector import ObjectDetector, ObjectDetection
from backend.config import config
from backend.exceptions import ModelUnavailableException
from backend.validation import validate_image_bytes

# Helper class to mock ultralytics Result and Box objects
class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self._xyxy = np.array([xyxy], dtype=np.float32)
        self._conf = np.array([conf], dtype=np.float32)
        self._cls = np.array([cls], dtype=np.float32)

    @property
    def xyxy(self):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = self._xyxy
        return [m]

    @property
    def conf(self):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = self._conf
        return [m]

    @property
    def cls(self):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = self._cls
        return [m]

class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

# ==============================================================================
# GROUP 1: @pytest.mark.mocked
# ==============================================================================
class TestMockedYOLO(unittest.TestCase):
    pytest_marks = {"mocked"}

    def setUp(self):
        self.detector = ObjectDetector(model_path="weights/yolo11n.pt")
        self.detector.loaded = True
        self.detector.backend_type = "ultralytics"
        self.detector.model = MagicMock()
        self.detector.model.names = {0: "person", 39: "bottle", 56: "chair"}

    @pytest.mark.mocked
    def test_detection_structure_and_coordinates(self):
        # Image 800x600 (W=800, H=600)
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        fake_box = FakeBox([100.0, 150.0, 300.0, 450.0], 0.85432, 0)
        self.detector.model.predict.return_value = [FakeResult([fake_box])]

        dets, meta = self.detector.predict_with_meta(img)
        self.assertEqual(len(dets), 1)
        det = dets[0]

        # Verify exact fields
        self.assertEqual(det.label, "person")
        self.assertEqual(det.class_id, 0)
        self.assertEqual(round(det.confidence, 4), 0.8543)
        self.assertEqual(det.bbox["x1"], 100.0)
        self.assertEqual(det.bbox["y1"], 150.0)
        self.assertEqual(det.bbox["x2"], 300.0)
        self.assertEqual(det.bbox["y2"], 450.0)
        self.assertEqual(det.center_x, 200.0)
        self.assertEqual(det.center_y, 300.0)

        # Dictionary representation for frontend
        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_name"], "person")
        self.assertEqual(d_dict["model_class_name"], "person")
        self.assertEqual(d_dict["bounding_box"], [100.0, 150.0, 300.0, 450.0])
        self.assertEqual(d_dict["box"], [100.0, 150.0, 300.0, 450.0])
        self.assertEqual(d_dict["center_x"], 200.0)
        self.assertEqual(d_dict["center_y"], 300.0)

        # Metadata
        self.assertEqual(meta["image_width"], 800)
        self.assertEqual(meta["image_height"], 600)
        self.assertEqual(meta["num_detections"], 1)
        self.assertIn("inference_ms", meta)
        self.assertIn("device", meta)
        self.assertIn("model", meta)

    @pytest.mark.mocked
    def test_coordinate_clamping_and_degenerate_dropping(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Box 1: extends out of bounds -> clamped to [0, W-1] x [0, H-1]
        out_of_bounds = FakeBox([-20.0, -10.0, 700.0, 500.0], 0.9, 39)
        # Box 2: degenerate (zero width) -> dropped
        zero_w = FakeBox([50.0, 50.0, 50.0, 100.0], 0.8, 0)
        # Box 3: degenerate (negative height) -> dropped
        neg_h = FakeBox([50.0, 100.0, 100.0, 50.0], 0.8, 56)

        self.detector.model.predict.return_value = [FakeResult([out_of_bounds, zero_w, neg_h])]

        dets = self.detector.predict(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].bbox["x1"], 0.0)
        self.assertEqual(dets[0].bbox["y1"], 0.0)
        self.assertEqual(dets[0].bbox["x2"], 639.0)
        self.assertEqual(dets[0].bbox["y2"], 479.0)

    @pytest.mark.mocked
    def test_empty_results(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.model.predict.return_value = [FakeResult([])]

        dets, meta = self.detector.predict_with_meta(img)
        self.assertEqual(dets, [])
        self.assertEqual(meta["num_detections"], 0)
        self.assertEqual(meta["image_width"], 640)
        self.assertEqual(meta["image_height"], 480)

    @pytest.mark.mocked
    def test_multiple_detections(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        b1 = FakeBox([10.0, 10.0, 100.0, 100.0], 0.95, 0) # person
        b2 = FakeBox([200.0, 200.0, 300.0, 350.0], 0.80, 39) # bottle
        b3 = FakeBox([400.0, 100.0, 550.0, 400.0], 0.75, 56) # chair
        self.detector.model.predict.return_value = [FakeResult([b1, b2, b3])]

        dets = self.detector.predict(img)
        self.assertEqual(len(dets), 3)
        labels = [d.label for d in dets]
        self.assertEqual(labels, ["person", "bottle", "chair"])

    @pytest.mark.mocked
    def test_missing_weights_handling_and_503(self):
        unloaded_detector = ObjectDetector(model_path="weights/non_existent_file.pt")
        self.assertFalse(unloaded_detector.loaded)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ModelUnavailableException) as ctx:
            unloaded_detector.predict(img)
        self.assertEqual(ctx.exception.code, "MODEL_UNAVAILABLE")
        self.assertEqual(ctx.exception.status_code, 503)

# ==============================================================================
# GROUP 2: @pytest.mark.synthetic
# ==============================================================================
class TestSyntheticYOLO(unittest.TestCase):
    pytest_marks = {"synthetic"}

    def setUp(self):
        self.detector = ObjectDetector(model_path="weights/yolo11n.pt")
        self.detector.loaded = True
        self.detector.backend_type = "ultralytics"
        self.detector.model = MagicMock()
        self.detector.model.names = {0: "person", 39: "bottle"}

    @pytest.mark.synthetic
    def test_blank_array(self):
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.model.predict.return_value = [FakeResult([])]
        dets, meta = self.detector.predict_with_meta(blank)
        self.assertEqual(dets, [])
        self.assertEqual(meta["num_detections"], 0)

    @pytest.mark.synthetic
    def test_noise_array(self):
        noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.detector.model.predict.return_value = [FakeResult([])]
        dets = self.detector.predict(noise)
        self.assertEqual(dets, [])

    @pytest.mark.synthetic
    def test_tiny_and_huge_arrays(self):
        # Tiny
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        self.detector.model.predict.return_value = [FakeResult([])]
        dets_t, meta_t = self.detector.predict_with_meta(tiny)
        self.assertEqual(meta_t["image_width"], 10)
        self.assertEqual(meta_t["image_height"], 10)

        # Huge non-square
        huge = np.zeros((1080, 1920, 3), dtype=np.uint8)
        box = FakeBox([100.0, 100.0, 500.0, 900.0], 0.9, 0)
        self.detector.model.predict.return_value = [FakeResult([box])]
        dets_h, meta_h = self.detector.predict_with_meta(huge)
        self.assertEqual(meta_h["image_width"], 1920)
        self.assertEqual(meta_h["image_height"], 1080)
        self.assertEqual(len(dets_h), 1)

    @pytest.mark.synthetic
    def test_grayscale_converted_to_bgr(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.detector.model.predict.return_value = [FakeResult([])]
        dets = self.detector.predict(bgr)
        self.assertEqual(dets, [])

    @pytest.mark.synthetic
    def test_validation_corrupted_and_wrong_extension(self):
        from backend.exceptions import ValidationException
        # Corrupted bytes
        with self.assertRaises(ValidationException):
            validate_image_bytes(b"NOT_A_VALID_JPEG_HEADER", filename="test.jpg")
        # Wrong extension
        valid_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        with self.assertRaises(ValidationException):
            validate_image_bytes(valid_jpeg, filename="test.pdf")

# ==============================================================================
# GROUP 3: @pytest.mark.realmodel
# ==============================================================================
class TestRealModelYOLO(unittest.TestCase):
    pytest_marks = {"realmodel"}

    def setUp(self):
        # Check if real ultralytics and weights exist
        self.weights_path = Path("weights/yolo11n.pt")
        try:
            import torch
            from ultralytics import YOLO
            has_ultralytics = True
        except ImportError:
            has_ultralytics = False

        if not self.weights_path.exists() or not has_ultralytics:
            pytest.skip("weights/yolo11n.pt or ultralytics package not available in offline environment")

        self.detector = ObjectDetector(model_path=str(self.weights_path))
        if not self.detector.loaded:
            pytest.skip(f"Detector failed to load: {self.detector.load_error}")

    @pytest.mark.realmodel
    def test_real_model_loading_and_structure(self):
        self.assertTrue(self.detector.loaded)
        self.assertIsNotNone(self.detector.model)

    @pytest.mark.realmodel
    def test_real_inference_on_known_image(self):
        # Look for real image asset in repo
        sample_img_path = Path("public/assets/byakugan-logo.jpg")
        if not sample_img_path.exists():
            pytest.skip("No real test photo found on disk")

        img = cv2.imread(str(sample_img_path))
        dets, meta = self.detector.predict_with_meta(img)
        self.assertIsInstance(dets, list)
        self.assertGreater(meta["inference_ms"], 0.0)

    @pytest.mark.realmodel
    def test_confidence_threshold_filtering(self):
        sample_img_path = Path("public/assets/byakugan-logo.jpg")
        if not sample_img_path.exists():
            pytest.skip("No real test photo found on disk")
        img = cv2.imread(str(sample_img_path))
        dets_low = self.detector.predict(img, conf_threshold=0.1)
        dets_high = self.detector.predict(img, conf_threshold=0.8)
        self.assertGreaterEqual(len(dets_low), len(dets_high))

    @pytest.mark.realmodel
    def test_boxes_inside_bounds(self):
        sample_img_path = Path("public/assets/byakugan-logo.jpg")
        if not sample_img_path.exists():
            pytest.skip("No real test photo found on disk")
        img = cv2.imread(str(sample_img_path))
        h, w = img.shape[:2]
        dets = self.detector.predict(img)
        for d in dets:
            self.assertGreaterEqual(d.bbox["x1"], 0.0)
            self.assertGreaterEqual(d.bbox["y1"], 0.0)
            self.assertLessEqual(d.bbox["x2"], float(w))
            self.assertLessEqual(d.bbox["y2"], float(h))

    @pytest.mark.realmodel
    def test_live_frame_and_video_endpoints(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        sample_img_path = Path("public/assets/byakugan-logo.jpg")
        if not sample_img_path.exists():
            pytest.skip("No real test photo found on disk")
        img_bytes = sample_img_path.read_bytes()
        with TestClient(app) as client:
            res_live = client.post("/api/sessions/live/start")
            sid = res_live.json()["data"]["session_id"]
            files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
            res_frame = client.post(f"/api/live/{sid}/frame", files=files)
            self.assertEqual(res_frame.status_code, 200)
            self.assertIn("objects", res_frame.json()["data"])
            client.post(f"/api/sessions/live/{sid}/stop")
