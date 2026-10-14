"""
Unit tests for M2 LayoutPlanner
Verifies spatial coordinate planning, minimum face sizes, and OpenPose canvas synthesis.
"""
import unittest
import numpy as np
from consistency_v2.layout_planner import LayoutPlanner, CharacterLayoutBox


class TestLayoutPlanner(unittest.TestCase):

    def test_single_character_medium_shot(self):
        positions = {"young_ebright": "center"}
        boxes = LayoutPlanner.plan_scene_layout(positions, shot_type="medium shot")
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertEqual(b.character_id, "young_ebright")
        self.assertEqual(b.target_zone, "center")
        # Strict requirement: face width >= 96px for medium shots
        self.assertGreaterEqual(b.face_bbox["w"], 96)
        # Centered horizontally
        cx = b.face_bbox["x"] + b.face_bbox["w"] / 2
        self.assertAlmostEqual(cx / 1024.0, 0.50, delta=0.08)

    def test_two_character_spatial_binding(self):
        positions = {
            "young_ebright": "sitting on the left",
            "mother": "sitting on the right"
        }
        boxes = LayoutPlanner.plan_scene_layout(positions, shot_type="two-shot")
        self.assertEqual(len(boxes), 2)

        left_box = [b for b in boxes if b.character_id == "young_ebright"][0]
        right_box = [b for b in boxes if b.character_id == "mother"][0]

        # Verify left character is on left
        self.assertEqual(left_box.target_zone, "left")
        left_cx = (left_box.face_bbox["x"] + left_box.face_bbox["w"] / 2) / 1024.0
        self.assertLessEqual(left_cx, 0.40)

        # Verify right character is on right
        self.assertEqual(right_box.target_zone, "right")
        right_cx = (right_box.face_bbox["x"] + right_box.face_bbox["w"] / 2) / 1024.0
        self.assertGreaterEqual(right_cx, 0.60)

        # Non-overlapping horizontal regions
        self.assertLess(left_box.normalized_region[3], right_box.normalized_region[2] + 0.05)

    def test_wide_shot_minimum_face_size(self):
        positions = {"college_ebright": "center"}
        boxes = LayoutPlanner.plan_scene_layout(positions, shot_type="wide shot")
        self.assertEqual(len(boxes), 1)
        # Strict requirement: face width >= 64px for wide shots
        self.assertGreaterEqual(boxes[0].face_bbox["w"], 64)

    def test_openpose_canvas_synthesis(self):
        positions = {"young_ebright": "center"}
        boxes = LayoutPlanner.plan_scene_layout(positions, shot_type="medium shot")
        canvas = LayoutPlanner.render_openpose_canvas(boxes)

        self.assertEqual(canvas.shape, (1024, 1024, 3))
        # Non-empty canvas with lines and joints
        colored_pixels = np.count_nonzero(canvas)
        self.assertGreater(colored_pixels, 1000)

    def test_depth_hint_synthesis(self):
        positions = {"young_ebright": "center"}
        boxes = LayoutPlanner.plan_scene_layout(positions, shot_type="medium shot")
        depth = LayoutPlanner.render_depth_layout_hint(boxes)

        self.assertEqual(depth.shape, (1024, 1024, 3))
        # Foreground figure is brighter than dark background
        fb = boxes[0].face_bbox
        face_depth_val = depth[fb["y"] + fb["h"] // 2, fb["x"] + fb["w"] // 2, 0]
        bg_depth_val = depth[100, 100, 0]
        self.assertGreater(face_depth_val, bg_depth_val + 50)


if __name__ == "__main__":
    unittest.main()
