import unittest
import pytest

from backend.spatial.reasoning import (
    compute_horizontal_position, compute_vertical_position,
    compute_combined_position, compute_direction,
    compute_object_relationships, build_scene_graph
)

# ==============================================================================
# GROUP: @pytest.mark.synthetic
# ==============================================================================
class TestSyntheticSpatialReasoning(unittest.TestCase):
    pytest_marks = {"synthetic"}

    @pytest.mark.synthetic
    def test_horizontal_position_boundaries(self):
        w = 900
        # Left: cx < 0.33 * 900 = 297
        self.assertEqual(compute_horizontal_position(100.0, w), "LEFT")
        self.assertEqual(compute_horizontal_position(296.0, w), "LEFT")

        # Center: 297 <= cx <= 0.67 * 900 = 603
        self.assertEqual(compute_horizontal_position(300.0, w), "CENTER")
        self.assertEqual(compute_horizontal_position(450.0, w), "CENTER")
        self.assertEqual(compute_horizontal_position(600.0, w), "CENTER")

        # Right: cx > 603
        self.assertEqual(compute_horizontal_position(604.0, w), "RIGHT")
        self.assertEqual(compute_horizontal_position(800.0, w), "RIGHT")

    @pytest.mark.synthetic
    def test_vertical_position_boundaries(self):
        h = 600
        # Top: cy < 0.33 * 600 = 198
        self.assertEqual(compute_vertical_position(50.0, h), "TOP")
        self.assertEqual(compute_vertical_position(195.0, h), "TOP")

        # Middle: 198 <= cy <= 0.67 * 600 = 402
        self.assertEqual(compute_vertical_position(200.0, h), "MIDDLE")
        self.assertEqual(compute_vertical_position(300.0, h), "MIDDLE")
        self.assertEqual(compute_vertical_position(400.0, h), "MIDDLE")

        # Bottom: cy > 402
        self.assertEqual(compute_vertical_position(405.0, h), "BOTTOM")
        self.assertEqual(compute_vertical_position(550.0, h), "BOTTOM")

    @pytest.mark.synthetic
    def test_combined_position_generation(self):
        self.assertEqual(compute_combined_position("LEFT", "TOP"), "UPPER-LEFT")
        self.assertEqual(compute_combined_position("CENTER", "TOP"), "UPPER-CENTER")
        self.assertEqual(compute_combined_position("RIGHT", "TOP"), "UPPER-RIGHT")
        self.assertEqual(compute_combined_position("LEFT", "MIDDLE"), "MIDDLE-LEFT")
        self.assertEqual(compute_combined_position("CENTER", "MIDDLE"), "CENTER")
        self.assertEqual(compute_combined_position("RIGHT", "MIDDLE"), "MIDDLE-RIGHT")
        self.assertEqual(compute_combined_position("LEFT", "BOTTOM"), "LOWER-LEFT")
        self.assertEqual(compute_combined_position("CENTER", "BOTTOM"), "LOWER-CENTER")
        self.assertEqual(compute_combined_position("RIGHT", "BOTTOM"), "LOWER-RIGHT")

    @pytest.mark.synthetic
    def test_above_and_below_relationships(self):
        w, h = 640, 480
        # Object A is directly above Object B with horizontal overlap
        obj_a = {"id": "obj_a", "label": "cup", "bounding_box": [200, 50, 300, 150]}
        obj_b = {"id": "obj_b", "label": "table", "bounding_box": [150, 140, 400, 400]}

        rels = compute_object_relationships([obj_a, obj_b], w, h)
        a_to_b = [r["relation"] for r in rels if r["subject"] == "obj_a" and r["object"] == "obj_b"]
        b_to_a = [r["relation"] for r in rels if r["subject"] == "obj_b" and r["object"] == "obj_a"]

        self.assertIn("ABOVE", a_to_b)
        self.assertIn("BELOW", b_to_a)

    @pytest.mark.synthetic
    def test_left_of_and_right_of_relationships(self):
        w, h = 800, 600
        # Object A is to the left of Object B within the same vertical band
        obj_left = {"id": "obj_1", "label": "chair", "bounding_box": [50, 200, 200, 450]}
        obj_right = {"id": "obj_2", "label": "desk", "bounding_box": [210, 180, 500, 480]}

        rels = compute_object_relationships([obj_left, obj_right], w, h)
        l_to_r = [r["relation"] for r in rels if r["subject"] == "obj_1" and r["object"] == "obj_2"]
        r_to_l = [r["relation"] for r in rels if r["subject"] == "obj_2" and r["object"] == "obj_1"]

        self.assertIn("LEFT_OF", l_to_r)
        self.assertIn("RIGHT_OF", r_to_l)

    @pytest.mark.synthetic
    def test_overlapping_relationship(self):
        w, h = 640, 480
        # Heavy overlap between person and chair
        person = {"id": "person_1", "label": "person", "bounding_box": [100, 100, 300, 400]}
        chair = {"id": "chair_1", "label": "chair", "bounding_box": [150, 200, 320, 420]}

        rels = compute_object_relationships([person, chair], w, h)
        p_to_c = [r["relation"] for r in rels if r["subject"] == "person_1" and r["object"] == "chair_1"]
        self.assertIn("OVERLAPPING", p_to_c)

    @pytest.mark.synthetic
    def test_near_and_far_in_image(self):
        w, h = 1000, 1000
        # Close objects (center distance ~ 100 px / diagonal ~ 1414 px = ~0.07 <= 0.25)
        obj_1 = {"id": "1", "label": "bottle", "bounding_box": [100, 100, 150, 250]}
        obj_2 = {"id": "2", "label": "cup", "bounding_box": [160, 110, 220, 230]}
        rels_near = compute_object_relationships([obj_1, obj_2], w, h)
        self.assertTrue(any(r["relation"] == "NEAR_IN_IMAGE" for r in rels_near))

        # Far objects (top-left vs bottom-right, center distance > 800 px / 1414 px = ~0.60 >= 0.55)
        obj_far = {"id": "3", "label": "clock", "bounding_box": [850, 850, 950, 950]}
        rels_far = compute_object_relationships([obj_1, obj_far], w, h)
        self.assertTrue(any(r["relation"] == "FAR_IN_IMAGE" for r in rels_far))

    @pytest.mark.synthetic
    def test_scene_graph_structure(self):
        w, h = 640, 480
        objs = [
            {
                "id": "object_1",
                "label": "person",
                "confidence": 0.92,
                "bounding_box": [50, 100, 180, 400],
                "depth_category": "NEAR",
                "depth_value": 0.85
            },
            {
                "id": "object_2",
                "label": "chair",
                "confidence": 0.88,
                "bounding_box": [220, 200, 400, 450],
                "depth_category": "MID-RANGE",
                "depth_value": 0.45
            }
        ]
        graph = build_scene_graph(objs, w, h)

        self.assertIn("objects", graph)
        self.assertIn("relationships", graph)
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)

        # Check object 1 structure matches prompt requirement exactly
        o1 = graph["objects"][0]
        self.assertEqual(o1["id"], "object_1")
        self.assertEqual(o1["label"], "person")
        self.assertEqual(o1["confidence"], 0.92)
        self.assertEqual(o1["position"]["horizontal"], "LEFT")
        self.assertEqual(o1["position"]["vertical"], "MIDDLE")
        self.assertEqual(o1["position"]["combined"], "MIDDLE-LEFT")
        self.assertEqual(o1["depth"]["category"], "NEAR")
        self.assertEqual(o1["depth"]["value"], 0.85)

        # Check relationships structure
        self.assertTrue(len(graph["relationships"]) >= 1)
        r0 = graph["relationships"][0]
        self.assertIn("subject", r0)
        self.assertIn("relation", r0)
        self.assertIn("object", r0)

    @pytest.mark.synthetic
    def test_empty_and_invalid_detections(self):
        # Empty detections
        empty_graph = build_scene_graph([], 640, 480)
        self.assertEqual(empty_graph["objects"], [])
        self.assertEqual(empty_graph["relationships"], [])

        # Invalid bounding box
        invalid_obj = [{"id": "bad", "label": "test", "bounding_box": None}]
        graph = build_scene_graph(invalid_obj, 640, 480)
        self.assertEqual(len(graph["objects"]), 1)
        self.assertEqual(graph["relationships"], [])
