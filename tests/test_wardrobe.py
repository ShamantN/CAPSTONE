"""
Unit tests for Wardrobe and Palette Lock (M7)
"""
import unittest
import numpy as np
import cv2
from consistency_v2.wardrobe import (
    WardrobeLock,
    delta_e_cielab,
    bgr_to_standard_cielab,
    standard_cielab_to_bgr,
    REFERENCE_LAB_COLORS
)


class TestWardrobeLock(unittest.TestCase):
    def setUp(self):
        self.wardrobe_lock = WardrobeLock(delta_e_threshold=35.0)

    def test_delta_e_cielab(self):
        lab1 = np.array([50.0, 10.0, 20.0], dtype=np.float32)
        lab2 = np.array([50.0, 10.0, 20.0], dtype=np.float32)
        self.assertAlmostEqual(delta_e_cielab(lab1, lab2), 0.0, places=4)

        lab3 = np.array([53.0, 14.0, 20.0], dtype=np.float32)
        # sqrt(3^2 + 4^2) = 5.0
        self.assertAlmostEqual(delta_e_cielab(lab1, lab3), 5.0, places=4)

    def test_cielab_roundtrip(self):
        # Create synthetic BGR image
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        bgr[:, :] = (34, 139, 34)  # Forest Green
        lab_std = bgr_to_standard_cielab(bgr)
        rec_bgr = standard_cielab_to_bgr(lab_std)
        # Allow +/- 4 pixel difference due to color space quantization
        max_diff = np.max(np.abs(bgr.astype(int) - rec_bgr.astype(int)))
        self.assertLessEqual(max_diff, 4)

    def test_target_lab_resolution(self):
        # Test known name
        target = self.wardrobe_lock.get_target_lab(["yellow", "navy blue"])
        self.assertIsNotNone(target)
        self.assertTrue(np.allclose(target, REFERENCE_LAB_COLORS["yellow"]))

        # Test bank override
        custom_bank_lab = np.array([60.0, 25.0, -10.0], dtype=np.float32)
        target = self.wardrobe_lock.get_target_lab(["yellow"], bank_wardrobe_lab=custom_bank_lab)
        self.assertTrue(np.allclose(target, custom_bank_lab))

    def test_bounded_repair_and_face_preservation(self):
        # Create a 200x200 canvas
        # Face at x=50, y=20, w=40, h=40 (Flesh tone)
        # Torso at x=30, y=70, w=80, h=90 (Wrong color: bright red)
        # Target color: "navy blue"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = (200, 200, 200)  # light gray background

        # Face region
        face_bgr = np.array([140, 180, 230], dtype=np.uint8)  # skin tone
        img[20:60, 50:90] = face_bgr
        original_face = img[20:60, 50:90].copy()

        # Torso region (Wrong color: red [0, 0, 220])
        img[70:160, 30:110] = (0, 0, 220)

        face_bbox = {"x": 50, "y": 20, "w": 40, "h": 40}
        target_navy = REFERENCE_LAB_COLORS["navy blue"]

        # Run lock_scene_wardrobe
        faces_info = [{"character_id": "char_test", "bbox": face_bbox}]
        wardrobes = {"char_test": {"clothing_colors": ["navy blue"]}}

        repaired, report = self.wardrobe_lock.lock_scene_wardrobe(
            image_bgr=img,
            faces_info=faces_info,
            character_wardrobes=wardrobes,
            blend_strength=0.75
        )

        # 1. Verify repair was triggered
        self.assertGreater(report["repairs_applied"], 0)
        detail = report["details"][0]
        self.assertTrue(detail["repaired"])
        self.assertLess(detail["final_delta_e"], detail["initial_delta_e"])

        # 2. STRICT IDENTITY PRESERVATION CHECK:
        # Face pixels must be 100% byte-identical
        repaired_face = repaired[20:60, 50:90]
        face_diff = np.max(np.abs(original_face.astype(int) - repaired_face.astype(int)))
        self.assertEqual(face_diff, 0, "Face pixels were modified during wardrobe repair!")


if __name__ == "__main__":
    unittest.main()
