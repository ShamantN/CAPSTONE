"""
Unit tests for ConsistencyPipelineV2 integration
"""
import unittest
from consistency_v2.config import ConsistencyV2Config, RESERVED_SEEDS
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2


class TestPipelineV2Integration(unittest.TestCase):
    def setUp(self):
        self.config = ConsistencyV2Config(mode="default")
        self.pipeline = ConsistencyPipelineV2(config=self.config)

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline.bank_manager)
        self.assertIsNotNone(self.pipeline.layout_planner)
        self.assertIsNotNone(self.pipeline.face_detailer)
        self.assertIsNotNone(self.pipeline.wardrobe_lock)
        self.assertIsNotNone(self.pipeline.memory_manager)

    def test_reserved_seed_scope_guard(self):
        for seed in RESERVED_SEEDS:
            with self.assertRaises(ValueError):
                self.config.validate_seed(seed)

    def test_mode_presets(self):
        fast_cfg = ConsistencyV2Config(mode="fast")
        self.assertEqual(fast_cfg.steps_pass1, 14)
        self.assertEqual(fast_cfg.steps_pass2, 10)
        self.assertFalse(fast_cfg.enable_face_detailer)

        qual_cfg = ConsistencyV2Config(mode="quality")
        self.assertEqual(qual_cfg.steps_pass1, 25)
        self.assertEqual(qual_cfg.steps_pass2, 18)
        self.assertTrue(qual_cfg.enable_face_detailer)
        self.assertTrue(qual_cfg.enable_best_of_n)

    def test_config_hash(self):
        h1 = self.config.to_hash()
        self.assertEqual(len(h1), 12)
        # Modifying parameter must alter hash
        alt_cfg = ConsistencyV2Config(openpose_weight=0.85)
        h2 = alt_cfg.to_hash()
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
