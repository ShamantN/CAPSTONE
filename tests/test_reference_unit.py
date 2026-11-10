"""
Unit tests for Reference-Only Attention Injection (M8)
"""
import unittest
import numpy as np
from pathlib import Path
from consistency_v2.reference_unit import ReferenceOnlyUnitBuilder


class TestReferenceOnlyUnit(unittest.TestCase):
    def setUp(self):
        self.builder = ReferenceOnlyUnitBuilder(
            module="reference_only",
            default_weight=0.45,
            default_guidance_end=0.75,
            style_fidelity=0.50
        )
        self.sample_img = np.zeros((128, 128, 3), dtype=np.uint8)
        self.sample_img[:, :] = (120, 150, 180)

    def test_encode_image_b64(self):
        b64 = self.builder.encode_image_b64(self.sample_img)
        self.assertTrue(b64.startswith("data:image/png;base64,"))
        self.assertGreater(len(b64), 50)

    def test_build_unit_schema(self):
        unit = self.builder.build_unit(
            reference_image=self.sample_img,
            weight=0.50,
            guidance_end=0.70
        )
        self.assertTrue(unit["enabled"])
        self.assertEqual(unit["module"], "reference_only")
        self.assertEqual(unit["model"], "None")
        self.assertEqual(unit["weight"], 0.50)
        self.assertEqual(unit["guidance_end"], 0.70)
        self.assertEqual(unit["control_mode"], "Balanced")
        self.assertTrue(unit["input_image"].startswith("data:image/png;base64,"))

    def test_build_unit_with_region_mask(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[:, :64] = 255  # Left half
        unit = self.builder.build_unit(
            reference_image=self.sample_img,
            effective_region_mask=mask
        )
        self.assertIn("effective_region_mask", unit)
        self.assertTrue(unit["effective_region_mask"].startswith("data:image/png;base64,"))

    def test_build_scene_reference_units(self):
        mock_banks = {
            "char_a": {"anchor_path": None, "references": [{"image_path": self.sample_img}]}
        }
        mock_masks = {
            "char_a": np.zeros((128, 128), dtype=np.uint8)
        }
        units = self.builder.build_scene_reference_units(
            characters=["char_a", "non_existent"],
            identity_banks=mock_banks,
            layout_masks=mock_masks
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["module"], "reference_only")
        self.assertIn("effective_region_mask", units[0])


if __name__ == "__main__":
    unittest.main()
