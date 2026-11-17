"""
Unit tests for eval3 metrics, calibration, and analysis.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from eval3.metrics.calibration import (
    score_pair_with_calibration,
    compute_d_prime,
    compute_platt_parameters,
    save_calibration,
    load_calibration,
    DEFAULT_CALIBRATION,
)
from eval3.metrics.evaluator import Eval3Evaluator
from eval3.analysis import hierarchical_cluster_bootstrap, export_evaluation_results


class TestEval3Metrics(unittest.TestCase):
    def test_default_calibration_scoring(self):
        """Scoring with default calibration produces expected probabilities."""
        cos_map = {"Facenet512": 0.8, "SFace": 0.7, "GhostFaceNet": 0.75, "ArcFace": 0.85}
        score = score_pair_with_calibration(cos_map, DEFAULT_CALIBRATION)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertGreater(score, 0.7)

    def test_d_prime_computation(self):
        """d-prime calculates separation correctly."""
        gen = [0.8, 0.85, 0.9, 0.92]
        imp = [0.1, 0.15, 0.2, 0.25]
        dp = compute_d_prime(gen, imp)
        self.assertGreater(dp, 2.0)

    def test_metric_labels_completeness(self):
        """All primary and secondary metrics must be labeled IN-LOOP or OUT-OF-LOOP."""
        labels = Eval3Evaluator.METRIC_LOOP_LABELS
        self.assertEqual(labels["pm1_cross_scene_auc"], "OUT-OF-LOOP")
        self.assertEqual(labels["pm2_cross_scene_rank1"], "OUT-OF-LOOP")
        self.assertEqual(labels["pm3_scene_correctness_rate"], "OUT-OF-LOOP")
        self.assertEqual(labels["v1_2_ensemble_score"], "IN-LOOP")

    def test_hierarchical_bootstrap(self):
        """Hierarchical bootstrap returns mean, std, and 95% CIs."""
        mock_reps = [
            {"pm1_cross_scene_auc": 0.92, "pm2_cross_scene_rank1": 0.85, "pm3_scene_correctness_rate": 0.80, "scenes": [{"pm3_correct": True}, {"pm3_correct": False}]},
            {"pm1_cross_scene_auc": 0.94, "pm2_cross_scene_rank1": 0.88, "pm3_scene_correctness_rate": 0.90, "scenes": [{"pm3_correct": True}, {"pm3_correct": True}]},
            {"pm1_cross_scene_auc": 0.90, "pm2_cross_scene_rank1": 0.82, "pm3_scene_correctness_rate": 0.75, "scenes": [{"pm3_correct": True}, {"pm3_correct": False}]},
        ]
        stats = hierarchical_cluster_bootstrap(mock_reps, n_resamples=100)
        self.assertIn("pm1_cross_scene_auc", stats)
        pm1_stat = stats["pm1_cross_scene_auc"]
        self.assertIn("mean", pm1_stat)
        self.assertIn("ci_95_lower", pm1_stat)
        self.assertIn("ci_95_upper", pm1_stat)
        self.assertLessEqual(pm1_stat["ci_95_lower"], pm1_stat["ci_95_upper"])

    def test_export_evaluation_results(self):
        """Exporting writes CSVs and summary JSON correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            f_csv, s_csv, sum_json = export_evaluation_results(
                run_id="test_run",
                output_dir=out_path,
                replicate_audits=[{"pm1_cross_scene_auc": 0.9, "pm2_cross_scene_rank1": 0.8, "pm3_scene_correctness_rate": 0.8}],
                scene_records=[{"scene_idx": 1, "pm3_correct": True}],
                face_records=[{"face_idx": 0, "character_id": "test"}],
            )
            self.assertTrue(f_csv.exists())
            self.assertTrue(s_csv.exists())
            self.assertTrue(sum_json.exists())


if __name__ == "__main__":
    unittest.main()
