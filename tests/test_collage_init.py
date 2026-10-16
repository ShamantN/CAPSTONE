"""
Unit tests for Method M12: Collage-Init Composition
Verifies mask feathering, flag toggles, pixel modification at layout coordinates,
and minimum face scale enforcement.
"""
import unittest
from pathlib import Path
import numpy as np
import cv2

from consistency_v2.collage_init import CollageInitConfig, CollageInitEngine
from consistency_v2.layout_planner import CharacterLayoutBox


class MockBankManager:
    """Mock bank manager providing synthetic reference images."""
    def __init__(self):
        # Create 256x256 reference image with synthetic face circle
        self.ref_img = np.full((256, 256, 3), 180, dtype=np.uint8)
        cv2.circle(self.ref_img, (128, 128), 60, (120, 140, 200), -1)

    def get_primary_reference_image(self, character_id: str):
        return self.ref_img.copy()


class TestCollageInit(unittest.TestCase):

    def test_feathered_mask_properties(self):
        w, h = 100, 120
        mask = CollageInitEngine.create_feathered_ellipse_mask(w, h, feather_radius=10)
        self.assertEqual(mask.shape, (h, w))
        self.assertGreaterEqual(float(np.min(mask)), 0.0)
        self.assertLessEqual(float(np.max(mask)), 1.0)
        # Center should be high confidence (1.0)
        self.assertAlmostEqual(float(mask[h // 2, w // 2]), 1.0, places=1)
        # Corners should be 0.0
        self.assertEqual(float(mask[0, 0]), 0.0)
        self.assertEqual(float(mask[h - 1, w - 1]), 0.0)

    def test_disabled_by_default(self):
        config = CollageInitConfig(enabled=False)
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        box = CharacterLayoutBox(
            character_id="char_a",
            target_zone="left",
            face_bbox={"x": 50, "y": 40, "w": 80, "h": 90},
            body_bbox={"x": 20, "y": 40, "w": 180, "h": 400},
            normalized_region=[0.0, 1.0, 0.0, 0.5],
            scale_category="medium"
        )
        res, rep = CollageInitEngine.compose_collage(canvas, [box], MockBankManager(), config)
        self.assertFalse(rep["collage_applied"])
        self.assertEqual(rep["num_characters_collaged"], 0)
        self.assertTrue(np.array_equal(res, canvas))

    def test_enabled_modifies_designated_region(self):
        config = CollageInitConfig(enabled=True, min_face_scale_px=96, alpha_blend=0.90)
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        box1 = CharacterLayoutBox(
            character_id="char_left",
            target_zone="left",
            face_bbox={"x": 40, "y": 30, "w": 80, "h": 80},
            body_bbox={"x": 10, "y": 30, "w": 180, "h": 400},
            normalized_region=[0.0, 1.0, 0.0, 0.5],
            scale_category="medium"
        )
        box2 = CharacterLayoutBox(
            character_id="char_right",
            target_zone="right",
            face_bbox={"x": 300, "y": 30, "w": 80, "h": 80},
            body_bbox={"x": 270, "y": 30, "w": 180, "h": 400},
            normalized_region=[0.0, 1.0, 0.5, 1.0],
            scale_category="medium"
        )
        res, rep = CollageInitEngine.compose_collage(canvas, [box1, box2], MockBankManager(), config)
        self.assertTrue(rep["collage_applied"])
        self.assertEqual(rep["num_characters_collaged"], 2)
        # Canvas was zero; collaged canvas must now have non-zero pixels
        diff = float(np.mean(np.abs(res.astype(np.float32) - canvas.astype(np.float32))))
        self.assertGreater(diff, 1.0)

        # Scale enforcement check: face box was 80px, should be clamped to >= min_face_scale_px (96px)
        for char_info in rep["characters"]:
            tb = char_info["target_bbox"]
            self.assertGreaterEqual(tb["w"], 96)
            self.assertGreaterEqual(tb["h"], 96)


if __name__ == "__main__":
    unittest.main()
