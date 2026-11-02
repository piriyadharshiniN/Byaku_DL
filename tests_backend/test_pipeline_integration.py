import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

import pytest
from fastapi.testclient import TestClient

from backend.main import app, init_models, models_state
from backend.models.detector import ObjectDetector, ObjectDetection
from backend.models.depth import DepthAnythingV2Estimator

# ==============================================================================
# GROUP: @pytest.mark.mocked
# ==============================================================================
class TestPipelineIntegration(unittest.TestCase):
    pytest_marks = {"mocked"}

    @classmethod
    def setUpClass(cls):
        init_models()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    @pytest.mark.mocked
    def test_health_endpoint_model_status(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        data = body["data"]
        self.assertIn("models_loaded", data)
        self.assertIn("yolo_detector", data["models_loaded"])
        self.assertIn("depth_anything_v2", data["models_loaded"])
        # In this offline environment without weights on disk, both report false truthfully
        self.assertIsInstance(data["models_loaded"]["yolo_detector"], bool)
        self.assertIsInstance(data["models_loaded"]["depth_anything_v2"], bool)
        self.assertTrue(data["database_ok"])

    @pytest.mark.mocked
    def test_api_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("backend", body)
        self.assertEqual(body["backend"], "connected")
        self.assertIn("stage_status", body)
        self.assertIn("detection", body["stage_status"])
        self.assertIn("tracking", body["stage_status"])
        self.assertIn("depth", body["stage_status"])
        self.assertIn("hitl", body)

    @pytest.mark.mocked
    def test_live_frame_reports_unavailable_detector_as_503(self):
        session_id = self.client.post("/api/sessions/live/start").json()["data"]["session_id"]
        original_detector = app.state.detector
        app.state.detector = ObjectDetector(model_path="weights/missing-test-model.pt")
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", image)

        try:
            response = self.client.post(
                f"/api/live/{session_id}/frame",
                files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            )
        finally:
            app.state.detector = original_detector
            self.client.post(f"/api/sessions/live/{session_id}/stop")

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "MODEL_UNAVAILABLE")
        self.assertEqual(body["error"]["details"]["stage"], "detection")

    @pytest.mark.mocked
    def test_pipeline_flow_detections_to_depth_and_spatial(self):
        # Mock detector and depth estimator on app.state
        mock_detector = ObjectDetector(model_path="weights/yolo11n.pt")
        mock_detector.loaded = True
        mock_detector.predict_with_meta = MagicMock(return_value=([
            ObjectDetection("person", 0.95, {"x1": 50, "y1": 50, "x2": 150, "y2": 350}, {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.7}, 0),
            ObjectDetection("chair", 0.88, {"x1": 200, "y1": 150, "x2": 380, "y2": 420}, {"x1": 0.4, "y1": 0.3, "x2": 0.75, "y2": 0.85}, 56)
        ], {"inference_ms": 12.5, "device": "cpu", "model": "yolo11n.pt", "num_detections": 2}))

        mock_depth = DepthAnythingV2Estimator()
        mock_depth.loaded = True
        synthetic_dmap = np.full((480, 640), 0.5, dtype=np.float32)
        synthetic_dmap[50:350, 50:150] = 0.85 # NEAR for person
        synthetic_dmap[150:420, 200:380] = 0.45 # MID-RANGE for chair
        mock_depth.estimate_frame = MagicMock(return_value=synthetic_dmap)

        app.state.detector = mock_detector
        app.state.depth_estimator = mock_depth
        models_state["detector"] = mock_detector
        models_state["depth_estimator"] = mock_depth

        # Send test frame to /api/image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)

        res = self.client.post("/api/image", files={"file": ("frame.jpg", enc.tobytes(), "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        data = body["data"]

        # 1. Detection verified
        self.assertEqual(len(data["objects"]), 2)
        person_obj = next(o for o in data["objects"] if o["label"] == "person")
        chair_obj = next(o for o in data["objects"] if o["label"] == "chair")

        # 2. Relative depth connected
        self.assertEqual(person_obj["depth_category"], "NEAR")
        self.assertAlmostEqual(person_obj["depth_value"], 0.85, places=1)
        self.assertEqual(chair_obj["depth_category"], "MID-RANGE")
        self.assertAlmostEqual(chair_obj["depth_value"], 0.45, places=1)

        # 3. Spatial reasoning connected
        self.assertEqual(person_obj["position"]["horizontal"], "LEFT")
        self.assertEqual(person_obj["position"]["combined"], "MIDDLE-LEFT")
        self.assertEqual(chair_obj["position"]["horizontal"], "CENTER")

        # 4. Structured Scene graph present and verified
        self.assertIn("scene_graph", data)
        self.assertIn("graph", data) # frontend alias
        sg = data["scene_graph"]
        self.assertEqual(len(sg["objects"]), 2)
        self.assertTrue(len(sg["relationships"]) >= 1)

        # 5. Frontend contract preserved
        self.assertIn("bounding_box", person_obj)
        self.assertIsInstance(person_obj["bounding_box"], list)
        self.assertEqual(len(person_obj["bounding_box"]), 4)
        self.assertIn("class_name", person_obj)
        self.assertIn("horizontal_position", person_obj)

    @pytest.mark.mocked
    def test_live_camera_session_scene_memory_and_qa_integration(self):
        # 1. Start live camera session
        res_start = self.client.post("/api/sessions/live/start")
        sid = res_start.json()["data"]["session_id"]

        # Mock detector with bottle on left
        mock_detector = ObjectDetector()
        mock_detector.loaded = True
        mock_detector.predict_with_meta = MagicMock(return_value=([
            ObjectDetection("bottle", 0.96, {"x1": 50, "y1": 100, "x2": 120, "y2": 300}, {"x1": 0.1, "y1": 0.2, "x2": 0.25, "y2": 0.6}, 39)
        ], {"inference_ms": 10.0, "device": "cpu", "model": "yolo11n.pt", "num_detections": 1}))

        mock_depth = DepthAnythingV2Estimator()
        mock_depth.loaded = True
        dmap = np.full((480, 640), 0.2, dtype=np.float32)
        dmap[100:300, 50:120] = 0.85 # NEAR for bottle
        mock_depth.estimate_frame = MagicMock(return_value=dmap)

        app.state.detector = mock_detector
        app.state.depth_estimator = mock_depth
        models_state["detector"] = mock_detector
        models_state["depth_estimator"] = mock_depth

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)

        # 2. Process frame
        res_frame = self.client.post(f"/api/live/{sid}/frame", files={"file": ("cam.jpg", enc.tobytes(), "image/jpeg")})
        self.assertEqual(res_frame.status_code, 200)
        frame_data = res_frame.json()["data"]
        self.assertIn("scene_graph", frame_data)
        self.assertIn("bottle", frame_data["narration"].lower())

        # 3. Voice QA strictly from scene memory
        res_qa1 = self.client.post(f"/api/sessions/{sid}/ask", json={"question": "Where is my bottle?"})
        self.assertEqual(res_qa1.status_code, 200)
        ans1 = res_qa1.json()["data"]["answer"]
        self.assertIn("bottle", ans1.lower())
        self.assertIn("left", ans1.lower())

        res_qa2 = self.client.post(f"/api/sessions/{sid}/ask", json={"question": "What objects are nearby?"})
        self.assertEqual(res_qa2.status_code, 200)
        ans2 = res_qa2.json()["data"]["answer"]
        self.assertIn("bottle", ans2.lower())

        res_qa3 = self.client.post(f"/api/sessions/{sid}/ask", json={"question": "Which object is closest?"})
        self.assertEqual(res_qa3.status_code, 200)
        ans3 = res_qa3.json()["data"]["answer"]
        self.assertIn("closest", ans3.lower())

        res_qa4 = self.client.post(f"/api/sessions/{sid}/ask", json={"question": "What is on my left?"})
        self.assertEqual(res_qa4.status_code, 200)
        ans4 = res_qa4.json()["data"]["answer"]
        self.assertIn("bottle", ans4.lower())

        res_qa5 = self.client.post(f"/api/sessions/{sid}/ask", json={"question": "Has anything changed?"})
        self.assertEqual(res_qa5.status_code, 200)
        ans5 = res_qa5.json()["data"]["answer"]
        self.assertTrue("appeared" in ans5 or "unchanged" in ans5)

        # 4. Stop session
        self.client.post(f"/api/sessions/live/{sid}/stop")
