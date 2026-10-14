"""
Unit tests for Method M4: Identity Units Done Right (IdentityUnitManager).
Uses Python standard library unittest.
"""
import unittest
from unittest.mock import MagicMock
from consistency_v2.identity_units import IdentityUnitConfig, IdentityUnitManager


class TestIdentityUnitManager(unittest.TestCase):

    def setUp(self):
        self.bank_mgr = MagicMock()
        self.bank_mgr.select_best_reference.return_value = {
            "path": "consistency_v2/banks/young_ebright/ref_00_young_ebright_anchor.png",
            "pose": {"yaw": 0.0, "pitch": 0.0}
        }

    def test_default_builds_ip_adapter_plus_face(self):
        cfg = IdentityUnitConfig(enable_ip_adapter_plus_face=True, enable_instantid=False)
        specs = IdentityUnitManager.build_identity_control_specs(
            character_ids=["young_ebright"],
            bank_manager=self.bank_mgr,
            target_yaw=0.0,
            config=cfg
        )
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.name, "ip_adapter_plus_face_young_ebright")
        self.assertEqual(spec.module, "CLIP-ViT-H (IPAdapter)")
        self.assertIn("ip-adapter-plus-face_sdxl_vit-h", spec.model)
        self.assertEqual(spec.weight, 0.60)
        self.assertEqual(spec.guidance_end, 0.85)

    def test_instantid_dual_units_generated(self):
        cfg = IdentityUnitConfig(enable_ip_adapter_plus_face=False, enable_instantid=True)
        specs = IdentityUnitManager.build_identity_control_specs(
            character_ids=["young_ebright"],
            bank_manager=self.bank_mgr,
            target_yaw=0.0,
            config=cfg
        )
        self.assertEqual(len(specs), 2)
        ip_unit = [s for s in specs if "instantid_ip" in s.name][0]
        kp_unit = [s for s in specs if "instantid_kp" in s.name][0]

        # Verify dual-unit specifications fixing D4
        self.assertEqual(ip_unit.module, "InsightFace (InstantID)")
        self.assertIn("ip-adapter_instant_id_sdxl", ip_unit.model)
        self.assertEqual(ip_unit.weight, 0.65)

        self.assertEqual(kp_unit.module, "instant_id_face_keypoints")
        self.assertIn("control_instant_id_sdxl", kp_unit.model)
        self.assertEqual(kp_unit.weight, 0.50)

    def test_combined_plus_face_and_instantid(self):
        cfg = IdentityUnitConfig(enable_ip_adapter_plus_face=True, enable_instantid=True)
        specs = IdentityUnitManager.build_identity_control_specs(
            character_ids=["young_ebright"],
            bank_manager=self.bank_mgr,
            target_yaw=0.0,
            config=cfg
        )
        self.assertEqual(len(specs), 3)

    def test_missing_character_handled_gracefully(self):
        self.bank_mgr.select_best_reference.return_value = None
        specs = IdentityUnitManager.build_identity_control_specs(
            character_ids=["unknown_char"],
            bank_manager=self.bank_mgr
        )
        self.assertEqual(specs, [])


if __name__ == "__main__":
    unittest.main()
