"""
Unit tests for Best-of-N Candidate Selector and Diagnosis-Driven Healing v2 (M9)
"""
import unittest
import numpy as np
from consistency_v2.candidates import CandidateScreen, CandidateScore
from consistency_v2.healing_v2 import HealingControllerV2, HealingAction


class TestCandidatesAndHealing(unittest.TestCase):
    def setUp(self):
        self.screener = CandidateScreen(min_face_size_px=64, target_face_size_px=110)
        self.healer = HealingControllerV2(max_retries=2)
        self.sample_img = np.zeros((256, 256, 3), dtype=np.uint8)
        self.sample_img[:, :] = (128, 128, 128)

    def test_candidate_screen_heuristic(self):
        # Empty gray image has 0 faces
        res = self.screener.screen_candidate(
            candidate_id=1,
            image_bgr=self.sample_img,
            expected_faces=1,
            expected_positions=[{"position": "center"}]
        )
        self.assertEqual(res.candidate_id, 1)
        self.assertEqual(res.face_count, 0)
        self.assertIn("missing_faces_0_of_1", res.defect_flags)
        self.assertLess(res.total_score, 0.70)

    def test_candidate_selection_ranking(self):
        candidates = [(1, self.sample_img), (2, self.sample_img)]
        top = self.screener.select_best_candidates(
            candidates=candidates,
            expected_faces=0,
            expected_positions=[],
            top_k=1
        )
        self.assertEqual(len(top), 1)
        self.assertIsInstance(top[0], CandidateScore)

    def test_healing_diagnose_clone_bleed(self):
        mock_report = {
            "scene_audit": {"defects_detected": ["cloned_or_bleed_faces"]}
        }
        action = self.healer.diagnose_failure(mock_report, [])
        self.assertIsNotNone(action)
        self.assertEqual(action.remedy_id, "ISOLATE_REGIONAL_CONDITIONING")
        self.assertTrue(action.re_render_required)

    def test_healing_diagnose_wardrobe_discrepancy(self):
        mock_report = {
            "scene_audit": {"defects_detected": []},
            "binding_check": {
                "per_character": {
                    "richard_ebright": {"wardrobe_pass": False, "delta_e": 45.2, "spatial_pass": True}
                }
            }
        }
        action = self.healer.diagnose_failure(mock_report, [])
        self.assertIsNotNone(action)
        self.assertEqual(action.remedy_id, "WARDROBE_CIELAB_REPAIR")
        self.assertEqual(action.parameter_overrides["target_character"], "richard_ebright")

    def test_healing_diagnose_low_identity_small_face(self):
        mock_report = {
            "scene_audit": {"defects_detected": []},
            "binding_check": {"per_character": {}},
            "biometrics": {
                "per_character": {
                    "ebright_young": {"ensemble_mean": 0.52}
                }
            }
        }
        face_bboxes = [{"character_id": "ebright_young", "bbox": {"w": 72, "h": 72}}]
        action = self.healer.diagnose_failure(mock_report, face_bboxes)
        self.assertIsNotNone(action)
        self.assertEqual(action.remedy_id, "BOOST_FACE_DETAILER")

    def test_healing_diagnose_low_identity_large_face(self):
        mock_report = {
            "scene_audit": {"defects_detected": []},
            "binding_check": {"per_character": {}},
            "biometrics": {
                "per_character": {
                    "ebright_young": {"ensemble_mean": 0.52}
                }
            }
        }
        face_bboxes = [{"character_id": "ebright_young", "bbox": {"w": 130, "h": 130}}]
        action = self.healer.diagnose_failure(mock_report, face_bboxes)
        self.assertIsNotNone(action)
        self.assertEqual(action.remedy_id, "BOOST_IDENTITY_TRANSPLANT")

    def test_execute_healing_loop(self):
        mock_report = {
            "scene_audit": {"defects_detected": []},
            "binding_check": {
                "pass_rate": 0.40,
                "per_character": {
                    "ebright_adult": {"wardrobe_pass": False, "delta_e": 48.0, "spatial_pass": True}
                }
            },
            "biometrics": {"mean_ensemble": 0.65}
        }

        def mock_wardrobe_handler(img, params):
            # Simulate color fix
            return img.copy()

        healed_img, final_report, history = self.healer.execute_healing_loop(
            initial_image_bgr=self.sample_img,
            initial_report=mock_report,
            face_bboxes=[],
            wardrobe_handler=mock_wardrobe_handler
        )

        self.assertIsNotNone(healed_img)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0].diagnosis, "WARDROBE_CIELAB_REPAIR")


if __name__ == "__main__":
    unittest.main()
