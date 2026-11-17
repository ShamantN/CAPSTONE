"""
Unit tests for eval3 runner, anchors, and configuration management.
"""
import tempfile
import unittest
from pathlib import Path
from eval3.runner import Eval3Runner, CONFIG_PRESETS
from eval3.best_of_n import score_best_of_n_candidates
from eval3.anchors import select_best_anchor, CERTIFIED_ADMISSION_RULES


class TestEval3Runner(unittest.TestCase):
    def test_config_presets_hashes(self):
        """Pre-registered config presets C0, C1, C2 must have valid 12-char hashes."""
        runner = Eval3Runner(run_id="test_runner")
        for c_id in ["C0", "C1", "C2"]:
            cfg, cfg_hash = runner.get_config_preset(c_id)
            self.assertEqual(len(cfg_hash), 12)
            self.assertIn(c_id, CONFIG_PRESETS)

    def test_stage_d_hard_block(self):
        """Runner must raise ValueError if Stage D is requested without explicit permission."""
        runner = Eval3Runner(run_id="test_runner")
        with self.assertRaises(ValueError):
            runner.run_protocol(arm="naive", stage="D")
        with self.assertRaises(ValueError):
            runner.run_protocol(arm="v1", stage="D")

    def test_best_of_n_scoring_tiering(self):
        """Best of n ranking must prioritize valid person counts and sharpness."""
        cand_a = {
            "candidate_idx": 0,
            "yolo_persons_count": 1,
            "detected_faces_count": 1,
            "sharpness": 120.0,
            "buffalo_l_sim": 0.82
        }
        cand_b = {
            "candidate_idx": 1,
            "yolo_persons_count": 2,  # Extra person
            "detected_faces_count": 2,
            "sharpness": 200.0,
            "buffalo_l_sim": 0.85
        }
        cand_c = {
            "candidate_idx": 2,
            "yolo_persons_count": 1,
            "detected_faces_count": 1,
            "sharpness": 150.0,
            "buffalo_l_sim": 0.80
        }
        ranked = score_best_of_n_candidates([cand_a, cand_b, cand_c], expected_characters_count=1)
        self.assertEqual(len(ranked), 3)
        # cand_c should beat cand_a because it has higher sharpness and valid counts; cand_b should be penalized for extra person
        self.assertEqual(ranked[0]["candidate_idx"], 2)
        self.assertEqual(ranked[-1]["candidate_idx"], 1)

    def test_anchor_admission_gate_defaults(self):
        """Admission gate constants must conform to protocol specification."""
        self.assertAlmostEqual(CERTIFIED_ADMISSION_RULES["min_confidence"], 0.80)
        self.assertEqual(CERTIFIED_ADMISSION_RULES["min_face_width_px"], 80)
        self.assertAlmostEqual(CERTIFIED_ADMISSION_RULES["min_sharpness"], 60.0)


if __name__ == "__main__":
    unittest.main()
