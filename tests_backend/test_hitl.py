import unittest
import time
import uuid
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.hitl.database import init_database, get_db_connection
from backend.hitl.review import submit_review, undo_review
from backend.hitl.memory import query_reviews, query_audit_events, export_reviews_coco
from backend.hitl.cache import save_cache_frame, get_cache_frame, sweep_review_cache
from backend.exceptions import ValidationException, GoneException

class TestHITLReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_database()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_1_confirmed_review_stored(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        res = submit_review(
            session_id=sid,
            submission_id=f"sub_1",
            original_label="person",
            original_confidence=0.92,
            bbox={"x1": 50, "y1": 50, "x2": 150, "y2": 250},
            action="confirmed",
            input_source="live_camera"
        )
        self.assertEqual(res["action"], "confirmed")
        self.assertEqual(res["original_label"], "person")
        self.assertIsNone(res["corrected_label"])

        # Verify in DB
        rows = query_reviews(session_id=sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "confirmed")
        self.assertEqual(rows[0]["original_label"], "person")
        self.assertIsNone(rows[0]["corrected_label"])

    def test_2_corrected_review_keeps_both_labels(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        corrected_bbox = {"x1": 70, "y1": 90, "x2": 170, "y2": 210}
        res = submit_review(
            session_id=sid,
            submission_id="sub_2",
            original_label="dog",
            original_confidence=0.74,
            bbox={"x1": 60, "y1": 80, "x2": 180, "y2": 220},
            action="corrected",
            corrected_label="cat",
            corrected_bbox=corrected_bbox,
            input_source="live_camera"
        )
        self.assertEqual(res["action"], "corrected")
        self.assertEqual(res["original_label"], "dog")
        self.assertEqual(res["corrected_label"], "cat")

        # Verify DB preserves both immutable original and corrected label
        rows = query_reviews(session_id=sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["original_label"], "dog")
        self.assertEqual(rows[0]["corrected_label"], "cat")
        self.assertEqual(rows[0]["corrected_bbox"], corrected_bbox)
        self.assertEqual(rows[0]["action"], "corrected")

        dataset = export_reviews_coco()
        image = next(
            item for item in dataset["images"]
            if item["file_name"] == f"{sid}.jpg"
        )
        annotation = next(
            item for item in dataset["annotations"]
            if item["image_id"] == image["id"]
            and item["original_label"] == "dog"
            and item["corrected_label"] == "cat"
        )
        category = next(
            item for item in dataset["categories"]
            if item["id"] == annotation["category_id"]
        )
        self.assertEqual(category["name"], "cat")
        self.assertEqual(annotation["bbox"], [70.0, 90.0, 100.0, 120.0])

    def test_3_rejected_review_recorded(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        res = submit_review(
            session_id=sid,
            submission_id="sub_3",
            original_label="frisbee",
            original_confidence=0.45,
            bbox={"x1": 10, "y1": 20, "x2": 50, "y2": 60},
            action="rejected",
            input_source="live_camera"
        )
        self.assertEqual(res["action"], "rejected")
        rows = query_reviews(session_id=sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "rejected")

    def test_4_restart_persistence(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        res = submit_review(
            session_id=sid,
            submission_id="sub_persist",
            original_label="chair",
            original_confidence=0.89,
            bbox={"x1": 100, "y1": 100, "x2": 200, "y2": 300},
            action="confirmed",
            input_source="live_camera"
        )
        review_id = res["review_id"]

        # Simulate backend restart by re-initializing DB and querying via a fresh connection
        init_database()
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["review_id"], review_id)
        self.assertEqual(row["original_label"], "chair")

    def test_5_live_and_upload_sessions_separated(self):
        sid_live = f"live_{uuid.uuid4().hex[:8]}"
        sid_upload = f"img_{uuid.uuid4().hex[:8]}"

        submit_review(
            session_id=sid_live,
            submission_id="sub_live",
            original_label="person",
            original_confidence=0.9,
            bbox={"x1": 10, "y1": 10, "x2": 100, "y2": 100},
            action="confirmed",
            input_source="live_camera"
        )
        submit_review(
            session_id=sid_upload,
            submission_id="sub_upload",
            original_label="car",
            original_confidence=0.85,
            bbox={"x1": 20, "y1": 20, "x2": 200, "y2": 200},
            action="confirmed",
            input_source="uploaded_image"
        )

        live_reviews = query_reviews(source="live_camera", session_id=sid_live)
        upload_reviews = query_reviews(source="uploaded_image", session_id=sid_upload)

        self.assertEqual(len(live_reviews), 1)
        self.assertEqual(live_reviews[0]["input_source"], "live_camera")
        self.assertEqual(len(upload_reviews), 1)
        self.assertEqual(upload_reviews[0]["input_source"], "uploaded_image")

    def test_6_cache_expiry_and_410_gone(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        sub = submit_review(
            session_id=sid,
            submission_id="sub_cache",
            original_label="bottle",
            original_confidence=0.9,
            bbox={"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            action="confirmed",
            input_source="live_camera"
        )
        frame_token = f"test_frame_{uuid.uuid4().hex[:8]}"
        frame_path = save_cache_frame(frame_token, b"dummy_jpeg_data")

        # Before expiry: retrievable
        data, mime = get_cache_frame(frame_token)
        self.assertEqual(data, b"dummy_jpeg_data")

        # Fast forward time on file (make mtime 200 seconds in past, TTL is 180s)
        old_time = time.time() - 250
        import os
        os.utime(str(frame_path), (old_time, old_time))

        # Sweep cache
        deleted = sweep_review_cache()
        self.assertTrue(deleted >= 1)

        # Reviews and audit events are preserved
        reviews_after = query_reviews(session_id=sid)
        self.assertEqual(len(reviews_after), 1)

        # Accessing expired frame returns 410 Gone
        with self.assertRaises(GoneException):
            get_cache_frame(frame_token)

        # Via API client
        res_http = self.client.get(f"/api/review-frames/{frame_token}")
        self.assertEqual(res_http.status_code, 410)

    def test_7_undo_appends_audit_and_deletes_nothing(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        res = submit_review(
            session_id=sid,
            submission_id="sub_undo",
            original_label="cup",
            original_confidence=0.8,
            bbox={"x1": 10, "y1": 10, "x2": 50, "y2": 50},
            action="confirmed",
            input_source="live_camera"
        )
        rev_id = res["review_id"]

        audit_before = query_audit_events(review_id=rev_id)
        self.assertEqual(len(audit_before), 1)

        # Perform undo
        undo_res = undo_review(rev_id)
        self.assertEqual(undo_res["status"], "undone")

        # Review record still exists! Status changed to 'undone', original_label intact
        reviews = query_reviews(session_id=sid)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["status"], "undone")
        self.assertEqual(reviews[0]["original_label"], "cup")

        # New 'undone' audit event appended (total audit count is now 2)
        audit_after = query_audit_events(review_id=rev_id)
        self.assertEqual(len(audit_after), 2)
        self.assertEqual(audit_after[0]["event_type"], "undone")

    def test_8_dashboard_api_returns_actual_sqlite_rows(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"
        # Insert via API
        payload = {
            "session_id": sid,
            "submission_id": "sub_api_dash",
            "original_label": "laptop",
            "original_confidence": 0.94,
            "bbox": {"x1": 40, "y1": 40, "x2": 240, "y2": 180},
            "action": "confirmed",
            "input_source": "live_camera"
        }
        res_post = self.client.post("/api/reviews", json=payload)
        self.assertEqual(res_post.status_code, 201)
        rev_id = res_post.json()["data"]["review_id"]

        # Fetch via API
        res_get = self.client.get(f"/api/reviews?session_id={sid}")
        self.assertEqual(res_get.status_code, 200)
        items = res_get.json()["data"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["review_id"], rev_id)
        self.assertEqual(items[0]["original_label"], "laptop")
        self.assertEqual(items[0]["action"], "confirmed")

        # Direct DB query to compare
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (rev_id,)).fetchone()
        conn.close()
        self.assertEqual(row["original_label"], items[0]["original_label"])
        self.assertEqual(row["action"], items[0]["action"])

    def test_constraint_violations(self):
        sid = f"live_{uuid.uuid4().hex[:8]}"

        # Corrected without label must fail
        with self.assertRaises(ValidationException) as ctx1:
            submit_review(
                session_id=sid,
                submission_id="bad_1",
                original_label="person",
                original_confidence=0.8,
                bbox={"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                action="corrected",
                corrected_label=None,
                input_source="live_camera"
            )
        self.assertEqual(ctx1.exception.code, "CORRECTED_LABEL_REQUIRED")

        # Confirmed with corrected_label must fail
        with self.assertRaises(ValidationException) as ctx2:
            submit_review(
                session_id=sid,
                submission_id="bad_2",
                original_label="person",
                original_confidence=0.8,
                bbox={"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                action="confirmed",
                corrected_label="something",
                input_source="live_camera"
            )
        self.assertEqual(ctx2.exception.code, "UNEXPECTED_CORRECTED_LABEL")

        # Confidence out of range [0, 1] must fail
        with self.assertRaises(ValidationException) as ctx3:
            submit_review(
                session_id=sid,
                submission_id="bad_3",
                original_label="person",
                original_confidence=1.5,
                bbox={"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                action="confirmed",
                input_source="live_camera"
            )
        self.assertEqual(ctx3.exception.code, "INVALID_CONFIDENCE")

if __name__ == "__main__":
    unittest.main()
