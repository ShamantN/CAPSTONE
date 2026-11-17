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

    def test_pair_formation_disjointness_regression(self):
        """Regression test for F3: genuine and impostor pairs must NEVER come from the same scene or image."""
        evaluator = Eval3Evaluator()
        # Mock audit with two characters in scene 1 and one in scene 2
        mock_audits = [
            {
                "scene_number": 1,
                "render_id": "rep_00_scene_01_seed_100",
                "expected_characters": ["char_a", "char_b"],
                "detected_faces": [
                    {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "embeddings": {m: [0.1]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}},
                    {"bbox": {"x": 100, "y": 10, "w": 50, "h": 50}, "embeddings": {m: [0.2]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}}
                ]
            },
            {
                "scene_number": 2,
                "render_id": "rep_00_scene_02_seed_101",
                "expected_characters": ["char_a"],
                "detected_faces": [
                    {"bbox": {"x": 50, "y": 50, "w": 60, "h": 60}, "embeddings": {m: [0.12]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}}
                ]
            }
        ]
        rep_res = evaluator.evaluate_replicate(
            replicate_idx=0,
            scene_audits=mock_audits,
            expected_chars=["char_a", "char_b"]
        )
        # Genuine pairs: char_a across scene 1 and scene 2 -> 1 pair
        self.assertEqual(rep_res["n_genuine_pairs"], 1)
        # Impostor pairs: char_a (scene 2) vs char_b (scene 1) -> 1 pair
        # char_a (scene 1) vs char_b (scene 1) MUST BE EXCLUDED because they share scene 1!
        self.assertEqual(rep_res["n_impostor_pairs"], 1)

    def test_pm2_ineligible_character_single_scene(self):
        """Regression test for G1: characters appearing in only 1 scene are ineligible for leave-one-scene-out."""
        evaluator = Eval3Evaluator()
        # char_a in scenes 1 and 2; char_b in scene 1 only
        mock_audits = [
            {
                "scene_number": 1,
                "render_id": "rep_00_scene_01",
                "expected_characters": ["char_a", "char_b"],
                "detected_faces": [
                    {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "embeddings": {m: [0.1]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}},
                    {"bbox": {"x": 100, "y": 10, "w": 50, "h": 50}, "embeddings": {m: [0.2]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}}
                ]
            },
            {
                "scene_number": 2,
                "render_id": "rep_00_scene_02",
                "expected_characters": ["char_a"],
                "detected_faces": [
                    {"bbox": {"x": 50, "y": 50, "w": 60, "h": 60}, "embeddings": {m: [0.12]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}}
                ]
            }
        ]
        rep_res = evaluator.evaluate_replicate(
            replicate_idx=0,
            scene_audits=mock_audits,
            expected_chars=["char_a", "char_b"]
        )
        self.assertEqual(rep_res["n_expected"], 3)
        self.assertEqual(rep_res["n_eligible"], 2)
        self.assertEqual(rep_res["n_ineligible"], 1)
        self.assertEqual(rep_res["eligible_characters"], ["char_a"])
        self.assertEqual(rep_res["ineligible_characters"], ["char_b"])
        self.assertEqual(rep_res["chance_level"], 0.5)

    def test_pm2_eligible_missed_face_intention_to_treat(self):
        """Regression test for G1: an eligible character with a missed face is penalized under intention-to-treat."""
        evaluator = Eval3Evaluator()
        # char_a expected in scenes 1 and 2, but detected ONLY in scene 1 (scene 2 face detection failed)
        mock_audits = [
            {
                "scene_number": 1,
                "render_id": "rep_00_scene_01",
                "expected_characters": ["char_a"],
                "detected_faces": [
                    {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "embeddings": {m: [0.1]*512 for m in ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]}}
                ]
            },
            {
                "scene_number": 2,
                "render_id": "rep_00_scene_02",
                "expected_characters": ["char_a"],
                "detected_faces": []  # Face detection failed
            }
        ]
        rep_res = evaluator.evaluate_replicate(
            replicate_idx=0,
            scene_audits=mock_audits,
            expected_chars=["char_a"]
        )
        self.assertEqual(rep_res["n_expected"], 2)
        self.assertEqual(rep_res["n_eligible"], 2)
        self.assertEqual(rep_res["n_ineligible"], 0)
        # Both probe instances fail: scene 2 probe is missing; scene 1 probe has no gallery face in other scenes
        self.assertEqual(rep_res["rank1_hits"], 0)
        self.assertEqual(rep_res["pm2_cross_scene_rank1"], 0.0)
        self.assertEqual(rep_res["mrr"], 0.0)

    def test_pm2_story_a_hand_computed_counts(self):
        """Regression test for G1: exact hand-computed counts for Story A scenes 1 to 5."""
        evaluator = Eval3Evaluator()
        # Story A layout:
        # Scene 1: young_ebright, mother
        # Scene 2: young_ebright
        # Scene 3: teen_ebright
        # Scene 4: teen_ebright
        # Scene 5: college_ebright, james_wong
        mock_audits = [
            {"scene_number": 1, "render_id": "s1", "expected_characters": ["young_ebright", "mother"], "detected_faces": []},
            {"scene_number": 2, "render_id": "s2", "expected_characters": ["young_ebright"], "detected_faces": []},
            {"scene_number": 3, "render_id": "s3", "expected_characters": ["teen_ebright"], "detected_faces": []},
            {"scene_number": 4, "render_id": "s4", "expected_characters": ["teen_ebright"], "detected_faces": []},
            {"scene_number": 5, "render_id": "s5", "expected_characters": ["college_ebright", "james_wong"], "detected_faces": []},
        ]
        story_chars = ["young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong"]
        rep_res = evaluator.evaluate_replicate(
            replicate_idx=0,
            scene_audits=mock_audits,
            expected_chars=story_chars
        )
        self.assertEqual(rep_res["n_expected"], 7)
        self.assertEqual(rep_res["n_eligible"], 4)
        self.assertEqual(rep_res["n_ineligible"], 3)
        self.assertEqual(rep_res["eligible_characters"], ["teen_ebright", "young_ebright"])
        self.assertEqual(rep_res["ineligible_characters"], ["college_ebright", "james_wong", "mother"])
        self.assertEqual(rep_res["chance_level"], 0.2000)
        self.assertEqual(rep_res["n_gallery_characters"], 5)


if __name__ == "__main__":
    unittest.main()
