import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
import cv2

import pytest
from backend.models.depth import DepthAnythingV2Estimator, DepthEstimator
from backend.config import config

# ==============================================================================
# GROUP 1: @pytest.mark.mocked
# ==============================================================================
class TestMockedDepthEstimator(unittest.TestCase):
    pytest_marks = {"mocked"}

    def setUp(self):
        self.estimator = DepthAnythingV2Estimator()
        self.estimator.loaded = True
        self.estimator.backend_type = "transformers"
        self.estimator.model = MagicMock()

    @pytest.mark.mocked
    def test_model_initialization(self):
        self.assertTrue(self.estimator.loaded)
        self.assertIsNotNone(self.estimator.model)
        self.assertEqual(self.estimator.device, "cpu")

    @pytest.mark.mocked
    def test_missing_weights_fallback(self):
        unloaded = DepthAnythingV2Estimator(model_name="nonexistent_model", weights_path="nonexistent.pth")
        # In offline environment with no weights, loaded should be False
        self.assertFalse(unloaded.loaded)
        self.assertIsNotNone(unloaded.load_error)
        # Should return None for frame and UNKNOWN for object depth
        res_frame = unloaded.estimate_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertIsNone(res_frame)
        res_obj = unloaded.evaluate_object_depth(None, {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, (100, 100))
        self.assertEqual(res_obj["depth_category"], "UNKNOWN")
        self.assertEqual(res_obj["depth_confidence"], "low")

    @pytest.mark.mocked
    def test_cpu_inference_with_mocked_pipeline(self):
        # Mock transformers pipeline output with a 518x518 depth gradient
        dummy_depth = np.linspace(10, 100, 518 * 518, dtype=np.float32).reshape(518, 518)
        self.estimator.model.return_value = {"depth": dummy_depth}

        input_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
        depth_map = self.estimator.estimate_frame(input_bgr)

        self.assertIsNotNone(depth_map)
        # Verify output matches original input dimensions (H=300, W=400)
        self.assertEqual(depth_map.shape, (300, 400))
        # Verify min-max normalization in [0.0, 1.0]
        self.assertAlmostEqual(float(np.min(depth_map)), 0.0, places=3)
        self.assertAlmostEqual(float(np.max(depth_map)), 1.0, places=3)

# ==============================================================================
# GROUP 2: @pytest.mark.synthetic
# ==============================================================================
class TestSyntheticDepthEstimator(unittest.TestCase):
    pytest_marks = {"synthetic"}

    def setUp(self):
        self.estimator = DepthAnythingV2Estimator()

    @pytest.mark.synthetic
    def test_invalid_and_empty_images(self):
        self.estimator.loaded = True
        self.estimator.model = MagicMock()

        # None input
        self.assertIsNone(self.estimator.estimate_frame(None))
        # Empty array
        self.assertIsNone(self.estimator.estimate_frame(np.array([], dtype=np.uint8)))
        # Zero dimension image
        self.assertIsNone(self.estimator.estimate_frame(np.zeros((0, 100, 3), dtype=np.uint8)))

    @pytest.mark.synthetic
    def test_output_dimensions_match_input(self):
        self.estimator.loaded = True
        self.estimator.backend_type = "transformers"
        dummy_pred = np.ones((518, 518), dtype=np.float32)
        dummy_pred[0, 0] = 0.0 # prevent flat map
        self.estimator.model = MagicMock(return_value={"depth": dummy_pred})

        # Test with multiple non-square aspect ratios
        for h, w in [(480, 640), (1080, 1920), (250, 750)]:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            dmap = self.estimator.estimate_frame(img)
            self.assertIsNotNone(dmap)
            self.assertEqual(dmap.shape, (h, w))

    @pytest.mark.synthetic
    def test_relative_depth_categories(self):
        # Create a synthetic 100x100 depth map with known regions
        depth_map = np.full((100, 100), 0.5, dtype=np.float32)
        depth_map[0, 0] = 0.0
        depth_map[99, 99] = 1.0

        # Region with value 0.85 -> NEAR (>= 0.66)
        depth_map[10:50, 10:50] = 0.85
        res_near = self.estimator.evaluate_object_depth(depth_map, {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, (100, 100))
        self.assertEqual(res_near["depth_category"], "NEAR")
        self.assertEqual(res_near["depth_class"], "NEAR")
        self.assertAlmostEqual(res_near["depth_value"], 0.85, places=2)
        self.assertEqual(res_near["depth_confidence"], "high")

        # Region with value 0.50 -> MID-RANGE (0.33 to 0.66)
        depth_map[55:95, 10:50] = 0.50
        res_mid = self.estimator.evaluate_object_depth(depth_map, {"x1": 10, "y1": 55, "x2": 50, "y2": 95}, (100, 100))
        self.assertEqual(res_mid["depth_category"], "MID-RANGE")
        self.assertAlmostEqual(res_mid["depth_value"], 0.50, places=2)

        # Region with value 0.15 -> FAR (< 0.33)
        depth_map[10:50, 55:95] = 0.15
        res_far = self.estimator.evaluate_object_depth(depth_map, {"x1": 55, "y1": 10, "x2": 95, "y2": 50}, (100, 100))
        self.assertEqual(res_far["depth_category"], "FAR")
        self.assertAlmostEqual(res_far["depth_value"], 0.15, places=2)

    @pytest.mark.synthetic
    def test_unknown_fallbacks(self):
        depth_map = np.full((100, 100), 0.5, dtype=np.float32)
        depth_map[0, 0] = 0.0
        depth_map[99, 99] = 1.0

        # Tiny box (width < 10)
        res_tiny = self.estimator.evaluate_object_depth(depth_map, {"x1": 20, "y1": 20, "x2": 25, "y2": 80}, (100, 100))
        self.assertEqual(res_tiny["depth_category"], "UNKNOWN")
        self.assertIsNone(res_tiny["depth_value"])

        # Degenerate box (x2 <= x1)
        res_degen = self.estimator.evaluate_object_depth(depth_map, {"x1": 50, "y1": 20, "x2": 40, "y2": 80}, (100, 100))
        self.assertEqual(res_degen["depth_category"], "UNKNOWN")

        # Flat depth map (max - min < epsilon)
        flat_map = np.full((100, 100), 0.4, dtype=np.float32)
        res_flat = self.estimator.evaluate_object_depth(flat_map, {"x1": 20, "y1": 20, "x2": 80, "y2": 80}, (100, 100))
        self.assertEqual(res_flat["depth_category"], "UNKNOWN")

# ==============================================================================
# GROUP 3: @pytest.mark.realmodel
# ==============================================================================
class TestRealDepthAnythingV2(unittest.TestCase):
    pytest_marks = {"realmodel"}

    def setUp(self):
        self.estimator = DepthAnythingV2Estimator()
        if not self.estimator.loaded:
            pytest.skip("Depth Anything V2 weights or transformers package not available in offline environment")

    @pytest.mark.realmodel
    def test_real_model_inference_on_disk_image(self):
        sample_path = Path("public/assets/byakugan-logo.jpg")
        if not sample_path.exists():
            pytest.skip("Test image public/assets/byakugan-logo.jpg not found on disk")
        img = cv2.imread(str(sample_path))
        dmap = self.estimator.estimate_frame(img)
        self.assertIsNotNone(dmap)
        self.assertEqual(dmap.shape[:2], img.shape[:2])
        self.assertTrue(0.0 <= np.min(dmap) <= 0.1)
        self.assertTrue(0.9 <= np.max(dmap) <= 1.0)
