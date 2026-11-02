import unittest
import numpy as np
from backend.models.detector import ObjectDetection
from backend.models.depth import DepthAnythingV2Estimator
from backend.models.tracker import ByteTrackSessionTracker
from backend.voice.narration import generate_scene_narration

class TestVision(unittest.TestCase):
    def test_detection_structure(self):
        det = ObjectDetection(
            label="bottle",
            confidence=0.885,
            bbox={"x1": 50.0, "y1": 100.0, "x2": 150.0, "y2": 300.0},
            bbox_norm={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6},
            class_id=39
        )
        d_dict = det.to_dict()
        self.assertEqual(d_dict["label"], "bottle")
        self.assertEqual(d_dict["confidence"], 0.885)
        self.assertEqual(d_dict["class_id"], 39)
        self.assertIn("bbox", d_dict)
        self.assertIn("bounding_box", d_dict)
        self.assertIn("box", d_dict)
        self.assertEqual(d_dict["bounding_box"], [50.0, 100.0, 150.0, 300.0])

    def test_empty_detections(self):
        narration, sig = generate_scene_narration([])
        self.assertEqual(narration, "I don't see any objects I can identify.")
        self.assertEqual(sig, "")

    def test_depth_estimation_range_and_unknown(self):
        estimator = DepthAnythingV2Estimator() # Unloaded in offline environment

        # 1. Model unavailable -> UNKNOWN
        dummy_bbox = {"x1": 50, "y1": 50, "x2": 200, "y2": 200}
        res = estimator.evaluate_object_depth(None, dummy_bbox, (480, 640))
        self.assertEqual(res["depth_category"], "UNKNOWN")
        self.assertEqual(res["depth_confidence"], "low")
        self.assertIsNone(res["depth_value"])

        # 2. Synthetic depth map: values in [0, 1]
        depth_map = np.full((480, 640), 0.8, dtype=np.float32)
        depth_map[0, 0] = 0.0 # Make sure map is not flat

        # Object in high depth region (>=0.66) -> NEAR
        res_near = estimator.evaluate_object_depth(depth_map, {"x1": 100, "y1": 100, "x2": 300, "y2": 300}, (480, 640))
        self.assertEqual(res_near["depth_category"], "NEAR")
        self.assertEqual(res_near["depth_confidence"], "high")
        self.assertAlmostEqual(res_near["depth_value"], 0.8, places=2)

        # Object in mid depth region (0.33 to 0.66) -> MID-RANGE
        depth_map[100:300, 100:300] = 0.5
        res_mid = estimator.evaluate_object_depth(depth_map, {"x1": 100, "y1": 100, "x2": 300, "y2": 300}, (480, 640))
        self.assertEqual(res_mid["depth_category"], "MID-RANGE")

        # Object in far depth region (<0.33) -> FAR
        depth_map[100:300, 100:300] = 0.2
        res_far = estimator.evaluate_object_depth(depth_map, {"x1": 100, "y1": 100, "x2": 300, "y2": 300}, (480, 640))
        self.assertEqual(res_far["depth_category"], "FAR")

        # 3. Tiny box check (< 10 px) -> UNKNOWN
        res_tiny = estimator.evaluate_object_depth(depth_map, {"x1": 100, "y1": 100, "x2": 105, "y2": 105}, (480, 640))
        self.assertEqual(res_tiny["depth_category"], "UNKNOWN")

        # 4. Flat depth map check (difference < epsilon) -> UNKNOWN
        flat_map = np.full((480, 640), 0.5, dtype=np.float32)
        res_flat = estimator.evaluate_object_depth(flat_map, {"x1": 100, "y1": 100, "x2": 300, "y2": 300}, (480, 640))
        self.assertEqual(res_flat["depth_category"], "UNKNOWN")

    def test_tracker_moving_object_and_stability(self):
        tracker = ByteTrackSessionTracker()
        # Frame 1: Person at (100, 100)
        det_f1 = [{"bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 250}, "confidence": 0.9, "class_id": 0, "label": "person"}]
        t1 = tracker.update(det_f1)
        self.assertEqual(t1[0]["track_id"], 1)

        # Frame 2: Person moves to (115, 105)
        det_f2 = [{"bbox": {"x1": 115, "y1": 105, "x2": 165, "y2": 255}, "confidence": 0.92, "class_id": 0, "label": "person"}]
        t2 = tracker.update(det_f2)
        self.assertEqual(t2[0]["track_id"], 1)

        # Frame 3: Person moves further to (130, 110)
        det_f3 = [{"bbox": {"x1": 130, "y1": 110, "x2": 180, "y2": 260}, "confidence": 0.91, "class_id": 0, "label": "person"}]
        t3 = tracker.update(det_f3)
        self.assertEqual(t3[0]["track_id"], 1)

    def test_tracker_temporarily_missing_and_reappearance(self):
        tracker = ByteTrackSessionTracker(track_buffer=10)
        # Frame 1: Bottle at (200, 200)
        det_f1 = [{"bbox": {"x1": 200, "y1": 200, "x2": 250, "y2": 350}, "confidence": 0.95, "class_id": 39, "label": "bottle"}]
        t1 = tracker.update(det_f1)
        self.assertEqual(t1[0]["track_id"], 1)

        # Frame 2 & 3: Bottle missing (occluded)
        tracker.update([])
        tracker.update([])

        # Frame 4: Bottle reappears near previous location
        det_f4 = [{"bbox": {"x1": 205, "y1": 202, "x2": 255, "y2": 352}, "confidence": 0.93, "class_id": 39, "label": "bottle"}]
        t4 = tracker.update(det_f4)
        # Reappeared bottle recovers persistent ID 1
        self.assertEqual(t4[0]["track_id"], 1)

    def test_tracker_multiple_objects_same_class(self):
        tracker = ByteTrackSessionTracker()
        # Two distinct persons side by side
        det_f1 = [
            {"bbox": {"x1": 50, "y1": 100, "x2": 150, "y2": 400}, "confidence": 0.94, "class_id": 0, "label": "person"},
            {"bbox": {"x1": 350, "y1": 100, "x2": 450, "y2": 400}, "confidence": 0.91, "class_id": 0, "label": "person"}
        ]
        t1 = tracker.update(det_f1)
        self.assertEqual(len(t1), 2)
        id_1 = t1[0]["track_id"]
        id_2 = t1[1]["track_id"]
        self.assertNotEqual(id_1, id_2)

        # Frame 2: both persons move slightly
        det_f2 = [
            {"bbox": {"x1": 55, "y1": 102, "x2": 155, "y2": 402}, "confidence": 0.93, "class_id": 0, "label": "person"},
            {"bbox": {"x1": 355, "y1": 102, "x2": 455, "y2": 402}, "confidence": 0.90, "class_id": 0, "label": "person"}
        ]
        t2 = tracker.update(det_f2)
        self.assertEqual(len(t2), 2)
        self.assertEqual(t2[0]["track_id"], id_1)
        self.assertEqual(t2[1]["track_id"], id_2)

    def test_tracker_session_isolation_and_reset(self):
        # Two independent tracker sessions must not interfere or share track IDs
        tracker_a = ByteTrackSessionTracker()
        tracker_b = ByteTrackSessionTracker()

        det_a1 = [{"bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "confidence": 0.9, "class_id": 0, "label": "person"}]
        tracked_a1 = tracker_a.update(det_a1)
        self.assertEqual(tracked_a1[0]["track_id"], 1)

        det_b1 = [{"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "confidence": 0.9, "class_id": 39, "label": "bottle"}]
        tracked_b1 = tracker_b.update(det_b1)
        self.assertEqual(tracked_b1[0]["track_id"], 1)

        # Reset tracker A
        tracker_a.reset()
        self.assertEqual(len(tracker_a.tracked_stracks), 0)
        self.assertEqual(tracker_a.next_id, 1)

        # Tracker B remains untouched
        self.assertEqual(len(tracker_b.tracked_stracks), 1)
        self.assertEqual(tracker_b.tracked_stracks[0].track_id, 1)
        self.assertEqual(tracker_b.tracked_stracks[0].label, "bottle")

if __name__ == "__main__":
    unittest.main()
