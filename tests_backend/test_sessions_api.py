import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

class TestSessionsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

        # Generate a test JPEG
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[50:150, 50:150] = [120, 200, 255]
        _, enc = cv2.imencode(".jpg", img)
        cls.test_jpeg = enc.tobytes()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["data"]["database_ok"])

    def test_live_session_lifecycle(self):
        # 1. Start live session
        res_start = self.client.post("/api/sessions/live/start")
        self.assertEqual(res_start.status_code, 200)
        body_start = res_start.json()
        self.assertTrue(body_start["ok"])
        session_id = body_start["data"]["session_id"]
        self.assertTrue(session_id.startswith("live_"))

        # 2. Send live frame
        files = {"file": ("frame.jpg", self.test_jpeg, "image/jpeg")}
        res_frame = self.client.post(f"/api/live/{session_id}/frame", files=files, data={"frame_id": 1})
        self.assertEqual(res_frame.status_code, 200)
        body_frame = res_frame.json()
        self.assertTrue(body_frame["ok"])
        self.assertEqual(body_frame["data"]["frame_id"], 1)

        # 3. Query scene
        res_scene = self.client.get(f"/api/sessions/{session_id}/scene")
        self.assertEqual(res_scene.status_code, 200)
        body_scene = res_scene.json()
        self.assertTrue(body_scene["ok"])

        # 4. Ask question
        res_ask = self.client.post(f"/api/sessions/{session_id}/ask", json={"question": "What is in front?"})
        self.assertEqual(res_ask.status_code, 200)
        body_ask = res_ask.json()
        self.assertTrue(body_ask["ok"])
        self.assertIn("answer", body_ask["data"])

        # 5. Stop live session
        res_stop = self.client.post(f"/api/sessions/live/{session_id}/stop")
        self.assertEqual(res_stop.status_code, 200)
        body_stop = res_stop.json()
        self.assertTrue(body_stop["ok"])
        self.assertEqual(body_stop["data"]["status"], "terminated")

        # 6. Subsequent requests to stopped session return 404
        res_after = self.client.get(f"/api/sessions/{session_id}/scene")
        self.assertEqual(res_after.status_code, 404)

    def test_image_upload_endpoint(self):
        files = {"file": ("upload.jpg", self.test_jpeg, "image/jpeg")}
        res = self.client.post("/api/image", files=files)
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["media_type"], "image")
        self.assertIn("scene_description", body["data"])
        self.assertIn("annotated_image", body["data"])

    def test_invalid_file_handling(self):
        # Empty file
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        res = self.client.post("/api/image", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.json()["ok"])
        self.assertEqual(res.json()["error"]["code"], "EMPTY_FILE")

        # Corrupt file
        files = {"file": ("corrupt.jpg", b"corrupt_non_image_bytes", "image/jpeg")}
        res = self.client.post("/api/image", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "INVALID_MAGIC_BYTES")

if __name__ == "__main__":
    unittest.main()
