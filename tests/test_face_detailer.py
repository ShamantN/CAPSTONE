"""
Unit tests for Method M6: Identity-Conditioned Face Detailer (face_detailer.py).
Uses Python standard library unittest.
"""
import unittest
import numpy as np
from consistency_v2.face_detailer import FaceDetailer, FaceDetailerConfig
from consistency_v2.layout_planner import CharacterLayoutBox


class TestFaceDetailer(unittest.TestCase):

    def setUp(self):
        self.detailer = FaceDetailer()

    def test_build_face_mask_geometry(self):
        shape = (1024, 1024, 3)
        bbox = {"x": 200, "y": 150, "w": 100, "h": 120}
        mask = self.detailer.build_face_mask(shape, bbox, blur_radius=21)

        self.assertEqual(mask.shape, (1024, 1024))
        self.assertEqual(mask.dtype, np.uint8)

        # Center should be high intensity
        cx = bbox["x"] + bbox["w"] // 2
        cy = bbox["y"] + bbox["h"] // 2
        self.assertGreater(mask[cy, cx], 200)

        # Far corners should be 0
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[1023, 1023], 0)

    def test_match_face_to_layout_box(self):
        boxes = [
            CharacterLayoutBox("young_ebright", "left", {}, {}, [0.0, 1.0, 0.0, 0.52], "medium"),
            CharacterLayoutBox("mother", "right", {}, {}, [0.0, 1.0, 0.48, 1.0], "medium")
        ]

        left_face = {"x": 220, "y": 150, "w": 100, "h": 120}
        right_face = {"x": 700, "y": 150, "w": 100, "h": 120}

        match_l = self.detailer.match_face_to_layout_box(left_face, boxes, 1024)
        match_r = self.detailer.match_face_to_layout_box(right_face, boxes, 1024)

        self.assertIsNotNone(match_l)
        self.assertEqual(match_l.character_id, "young_ebright")

        self.assertIsNotNone(match_r)
        self.assertEqual(match_r.character_id, "mother")

    def test_disabled_config_bypasses(self):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cfg = FaceDetailerConfig(enabled=False)
        out = self.detailer.detail_image(dummy_img, {}, [], None, config=cfg)
        self.assertTrue(np.array_equal(dummy_img, out))


if __name__ == "__main__":
    unittest.main()
