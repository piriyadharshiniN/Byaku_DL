import unittest
from backend.spatial.reasoning import compute_direction, format_direction_phrase, build_scene_graph, extract_relationships
from backend.spatial.scene_memory import SceneMemory
from backend.voice.narration import generate_scene_narration, compute_scene_signature
from backend.voice.qa import SceneQAEngine

class TestSpatialAndVoice(unittest.TestCase):
    def test_direction_boundaries(self):
        w, h = 600, 600
        # Left boundary: center_x / 600 < 0.33 -> center_x < 198
        h_left, v_top = compute_direction({"x1": 50, "y1": 50, "x2": 150, "y2": 150}, w, h)
        self.assertEqual(h_left, "LEFT")
        self.assertEqual(v_top, "TOP")

        # Center boundary: 0.33 <= center_x / 600 <= 0.67 -> center_x in [198, 402]
        h_center, v_mid = compute_direction({"x1": 250, "y1": 250, "x2": 350, "y2": 350}, w, h)
        self.assertEqual(h_center, "CENTER")
        self.assertEqual(v_mid, "MIDDLE")

        # Right boundary: center_x / 600 > 0.67 -> center_x > 402
        h_right, v_bot = compute_direction({"x1": 450, "y1": 450, "x2": 550, "y2": 550}, w, h)
        self.assertEqual(h_right, "RIGHT")
        self.assertEqual(v_bot, "BOTTOM")

    def test_scene_relationships(self):
        # Two objects: cup on table
        table = {
            "track_id": 1,
            "label": "dining table",
            "bbox": {"x1": 100, "y1": 300, "x2": 500, "y2": 500}
        }
        cup = {
            "track_id": 2,
            "label": "cup",
            "bbox": {"x1": 200, "y1": 260, "x2": 260, "y2": 320} # Resting on top edge of table
        }
        graph = build_scene_graph([table, cup], 640, 480)
        relations = extract_relationships(graph)
        on_relations = [r for r in relations if r["relation"] == "on"]
        self.assertTrue(len(on_relations) >= 1)
        self.assertEqual(on_relations[0]["source"], 2) # cup on table
        self.assertEqual(on_relations[0]["target"], 1)

    def test_scene_memory_retention_and_eviction(self):
        mem = SceneMemory(session_id="test_sess", retention_frames=5)

        # Frame 1: bottle detected
        det_f1 = [{
            "track_id": 1,
            "label": "bottle",
            "bbox": {"x1": 50, "y1": 100, "x2": 100, "y2": 200},
            "direction_h": "LEFT",
            "direction_v": "MIDDLE",
            "depth_class": "NEAR",
            "depth_value": 0.8,
            "confidence": 0.95
        }]
        mem.update(frame_number=1, detected_objects=det_f1)
        objs_f1 = mem.get_all()
        self.assertEqual(len(objs_f1), 1)
        self.assertEqual(objs_f1[0].status, "current")
        self.assertEqual(objs_f1[0].missed_frames, 0)

        # Frames 2 to 6 (5 missed frames): bottle not detected
        for f in range(2, 7):
            mem.update(frame_number=f, detected_objects=[])
            objs = mem.get_all()
            self.assertEqual(len(objs), 1, f"Failed at frame {f}")
            self.assertEqual(objs[0].status, "recent")
            self.assertEqual(objs[0].missed_frames, f - 1)

        # Frame 7 (6th missed frame > retention_frames=5): bottle evicted!
        mem.update(frame_number=7, detected_objects=[])
        objs_f7 = mem.get_all()
        self.assertEqual(len(objs_f7), 0, "Object should have been evicted after 5 missed frames.")

    def test_narration_deduplication(self):
        obj1 = {
            "label": "bottle",
            "direction_h": "LEFT",
            "direction_v": "MIDDLE",
            "depth_class": "NEAR"
        }
        text1, sig1 = generate_scene_narration([obj1])
        self.assertIn("bottle", text1)
        self.assertIn("left", text1)

        # Same scene, same signature
        text2, sig2 = generate_scene_narration([obj1])
        self.assertEqual(sig1, sig2)

        # Changed scene
        obj2 = {
            "label": "chair",
            "direction_h": "CENTER",
            "direction_v": "MIDDLE",
            "depth_class": "FAR"
        }
        text3, sig3 = generate_scene_narration([obj1, obj2])
        self.assertNotEqual(sig1, sig3)

    def test_voice_qa_intents(self):
        qa = SceneQAEngine()
        mem = SceneMemory(session_id="test_qa", retention_frames=5)

        # Empty scene
        ans_empty = qa.answer_question("What is around me?", mem)
        self.assertEqual(ans_empty, "I do not see any objects in the scene right now.")

        # Populate memory with bottle and sofa
        mem.update(frame_number=1, detected_objects=[
            {
                "track_id": 1,
                "label": "bottle",
                "bbox": {"x1": 50, "y1": 100, "x2": 100, "y2": 200},
                "direction_h": "LEFT",
                "direction_v": "MIDDLE",
                "depth_class": "NEAR",
                "depth_value": 0.85,
                "confidence": 0.95
            },
            {
                "track_id": 2,
                "label": "couch",
                "bbox": {"x1": 250, "y1": 200, "x2": 450, "y2": 400},
                "direction_h": "CENTER",
                "direction_v": "MIDDLE",
                "depth_class": "MID-RANGE",
                "depth_value": 0.5,
                "confidence": 0.90
            }
        ])

        # Intent 1: Locate object
        ans_locate = qa.answer_question("Where is my bottle?", mem)
        self.assertIn("bottle", ans_locate.lower())
        self.assertIn("left", ans_locate.lower())

        # Intent 2: Synonym mapping ("couch" -> "sofa")
        ans_sofa = qa.answer_question("Where is the sofa?", mem)
        self.assertIn("couch", ans_sofa.lower())

        # Intent 3: Nearest object
        ans_nearest = qa.answer_question("Where is the closest bottle?", mem)
        self.assertIn("closest", ans_nearest.lower())

        # Intent 4: What is in front
        ans_front = qa.answer_question("What is in front of me?", mem)
        self.assertIn("couch", ans_front.lower())

        # Intent 5: List everything
        ans_all = qa.answer_question("What do you see?", mem)
        self.assertIn("bottle", ans_all.lower())
        self.assertIn("couch", ans_all.lower())

        # Intent 6: Object not found
        ans_missing = qa.answer_question("Where are my keys?", mem)
        self.assertEqual(ans_missing, "I cannot currently identify a keys in the scene.")

        # Intent 7: Stale / recent object
        # Advance 2 frames with no objects
        mem.update(frame_number=2, detected_objects=[])
        ans_stale = qa.answer_question("Where is my bottle?", mem)
        self.assertIn("a moment ago", ans_stale)
        self.assertIn("can't confirm it now", ans_stale)

    def test_voice_qa_extended_intents(self):
        qa = SceneQAEngine()
        mem = SceneMemory(session_id="test_qa_extended", retention_frames=5)

        mem.update(frame_number=1, detected_objects=[
            {
                "track_id": 1,
                "label": "bottle",
                "bbox": {"x1": 50, "y1": 100, "x2": 100, "y2": 200},
                "direction_h": "LEFT",
                "direction_v": "MIDDLE",
                "depth_class": "NEAR",
                "depth_value": 0.85,
                "confidence": 0.95
            },
            {
                "track_id": 2,
                "label": "chair",
                "bbox": {"x1": 250, "y1": 200, "x2": 350, "y2": 400},
                "direction_h": "CENTER",
                "direction_v": "MIDDLE",
                "depth_class": "MID-RANGE",
                "depth_value": 0.5,
                "confidence": 0.88
            }
        ])

        # 1. Existence queries: positive and negative
        ans_exist_pos = qa.answer_question("Is there a bottle?", mem)
        self.assertIn("yes", ans_exist_pos.lower())
        self.assertIn("bottle", ans_exist_pos.lower())

        ans_exist_neg = qa.answer_question("Is there a laptop?", mem)
        self.assertIn("no", ans_exist_neg.lower())
        self.assertIn("laptop", ans_exist_neg.lower())

        # 2. Nearby query
        ans_nearby = qa.answer_question("What objects are nearby?", mem)
        self.assertIn("bottle", ans_nearby.lower())
        self.assertIn("left", ans_nearby.lower())

        # 3. Closest query
        ans_closest = qa.answer_question("Which object is closest?", mem)
        self.assertIn("closest", ans_closest.lower())
        self.assertIn("bottle", ans_closest.lower())

        # 4. Directional queries: Left vs Right
        ans_left = qa.answer_question("What is on my left?", mem)
        self.assertIn("bottle", ans_left.lower())

        ans_right = qa.answer_question("What is on my right?", mem)
        self.assertIn("no objects on your right", ans_right.lower())

        # 5. Scene unchanged check
        ans_changed = qa.answer_question("Has anything changed?", mem)
        # In frame 1, new objects were added so recent_changes has events
        self.assertTrue(len(mem.recent_changes) > 0)
        ans_change_event = qa.answer_question("What changed?", mem)
        self.assertTrue("appeared" in ans_change_event or "unchanged" in ans_change_event)

    def test_scene_memory_change_events_and_qa(self):
        mem = SceneMemory(session_id="test_changes", retention_frames=5)
        qa = SceneQAEngine()

        # Frame 1: Bottle appears on the left
        mem.update(frame_number=1, detected_objects=[{
            "track_id": 10,
            "label": "bottle",
            "bbox": {"x1": 50, "y1": 100, "x2": 100, "y2": 200},
            "direction_h": "LEFT",
            "direction_v": "MIDDLE",
            "depth_class": "NEAR",
            "depth_value": 0.85
        }])
        self.assertEqual(len(mem.recent_changes), 1)
        self.assertEqual(mem.recent_changes[0]["type"], "new_object")
        ans_f1 = qa.answer_question("Did anything move?", mem)
        self.assertIn("appeared", ans_f1)
        self.assertIn("bottle", ans_f1)

        # Frame 2: Bottle moves to the center
        mem.update(frame_number=2, detected_objects=[{
            "track_id": 10,
            "label": "bottle",
            "bbox": {"x1": 250, "y1": 100, "x2": 300, "y2": 200},
            "direction_h": "CENTER",
            "direction_v": "MIDDLE",
            "depth_class": "NEAR",
            "depth_value": 0.85
        }])
        self.assertEqual(mem.recent_changes[-1]["type"], "object_moved")
        ans_f2 = qa.answer_question("Has anything changed?", mem)
        self.assertIn("moved", ans_f2)

        # Frames 3 to 8 (6 frames with no objects): object disappears and gets evicted
        for f in range(3, 9):
            mem.update(frame_number=f, detected_objects=[])
        self.assertEqual(mem.recent_changes[-1]["type"], "object_disappeared")
        ans_f8 = qa.answer_question("What changed?", mem)
        self.assertIn("no longer visible", ans_f8)

if __name__ == "__main__":
    unittest.main()
