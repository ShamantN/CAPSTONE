"""Canary unit tests for tools/score_human_blind_test.py.

Uses synthetic temporary sealed keys and predictions ONLY (never real TEST-2
seeds or real sealed keys) to verify 100%, 0%, near-chance, tie handling,
validation errors, and rejection of the unfilled human prediction template.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import score_human_blind_test as scorer


def _make_synthetic_keys(ab_count: int = 10, reid_count: int = 10):
    chars = list(scorer.VALID_CHARACTER_IDS)
    sealed_ab = {}
    for i in range(1, ab_count + 1):
        pid = f"pair_{i:02d}"
        v2_opt = "A" if i % 2 == 1 else "B"
        sealed_ab[pid] = {
            "scene_number": i,
            "seed": 9000 + i,
            "option_a": "v2" if v2_opt == "A" else "v1_baseline",
            "option_b": "v1_baseline" if v2_opt == "A" else "v2",
            "v2_choice": v2_opt,
        }

    sealed_reid = {}
    for i in range(1, reid_count + 1):
        cid = f"crop_{i:02d}"
        sealed_reid[cid] = {
            "crop_id": cid,
            "true_character_id": chars[(i - 1) % len(chars)],
            "source": f"synthetic_scene_{i}",
        }
    return sealed_ab, sealed_reid


def _write_case(tmp_dir: Path, sealed_ab, sealed_reid, pred_data):
    sealed_dir = tmp_dir / "sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    (sealed_dir / "sealed_ab_key.json").write_text(json.dumps(sealed_ab), encoding="utf-8")
    (sealed_dir / "sealed_reid_key.json").write_text(json.dumps(sealed_reid), encoding="utf-8")
    pred_path = tmp_dir / "preds.json"
    pred_path.write_text(json.dumps(pred_data), encoding="utf-8")
    return pred_path, sealed_dir


class TestScoreHumanBlindTest(unittest.TestCase):
    def test_100_percent_canary(self):
        sealed_ab, sealed_reid = _make_synthetic_keys(10, 10)
        pred_data = {
            "evaluator_name": "canary_100",
            "date": "2026-11-12",
            "ab_preference": {
                pid: {"selected_option": k["v2_choice"], "notes": ""}
                for pid, k in sealed_ab.items()
            },
            "blind_reid": {
                cid: {"predicted_character_id": k["true_character_id"]}
                for cid, k in sealed_reid.items()
            },
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pred_path, sealed_dir = _write_case(td_path, sealed_ab, sealed_reid, pred_data)
            report = scorer.score_human_blind_test(
                pred_path,
                sealed_dir,
                out_json=td_path / "out.json",
                out_md=td_path / "out.md",
            )
            self.assertEqual(report["blind_ab_study"]["v2_wins"], 10)
            self.assertEqual(report["blind_ab_study"]["v2_win_rate_decisive"], 1.0)
            self.assertEqual(report["blind_reid_study"]["correct_reid"], 10)
            self.assertEqual(report["blind_reid_study"]["reid_accuracy"], 1.0)
            self.assertTrue((td_path / "out.json").exists())
            self.assertTrue((td_path / "out.md").exists())

    def test_0_percent_canary(self):
        sealed_ab, sealed_reid = _make_synthetic_keys(10, 10)
        chars = list(scorer.VALID_CHARACTER_IDS)
        pred_data = {
            "ab_preference": {
                pid: {
                    "selected_option": "B" if k["v2_choice"] == "A" else "A",
                    "notes": "",
                }
                for pid, k in sealed_ab.items()
            },
            "blind_reid": {
                cid: {
                    "predicted_character_id": chars[(chars.index(k["true_character_id"]) + 1) % len(chars)]
                }
                for cid, k in sealed_reid.items()
            },
        }
        with tempfile.TemporaryDirectory() as td:
            pred_path, sealed_dir = _write_case(Path(td), sealed_ab, sealed_reid, pred_data)
            report = scorer.score_human_blind_test(pred_path, sealed_dir, out_json=None, out_md=None)
            self.assertEqual(report["blind_ab_study"]["v2_wins"], 0)
            self.assertEqual(report["blind_ab_study"]["v1_wins"], 10)
            self.assertEqual(report["blind_ab_study"]["v2_win_rate_decisive"], 0.0)
            self.assertEqual(report["blind_reid_study"]["correct_reid"], 0)
            self.assertEqual(report["blind_reid_study"]["reid_accuracy"], 0.0)

    def test_chance_and_ties_canary(self):
        sealed_ab, sealed_reid = _make_synthetic_keys(10, 10)
        chars = list(scorer.VALID_CHARACTER_IDS)
        # 4 v2 wins, 4 v1 wins, 2 ties -> 50% of 8 decisive; 2/10 Re-ID -> 20% chance
        ab_preds = {}
        for idx, (pid, k) in enumerate(sorted(sealed_ab.items())):
            wrong = "B" if k["v2_choice"] == "A" else "A"
            if idx < 4:
                choice = k["v2_choice"]
            elif idx < 8:
                choice = wrong
            else:
                choice = "tie"
            ab_preds[pid] = {"selected_option": choice, "notes": ""}

        reid_preds = {}
        for idx, (cid, k) in enumerate(sorted(sealed_reid.items())):
            wrong_char = chars[(chars.index(k["true_character_id"]) + 1) % len(chars)]
            reid_preds[cid] = {
                "predicted_character_id": k["true_character_id"] if idx < 2 else wrong_char
            }

        with tempfile.TemporaryDirectory() as td:
            pred_path, sealed_dir = _write_case(
                Path(td), sealed_ab, sealed_reid, {"ab_preference": ab_preds, "blind_reid": reid_preds}
            )
            report = scorer.score_human_blind_test(pred_path, sealed_dir, out_json=None, out_md=None)
            self.assertEqual(report["blind_ab_study"]["decisive_pairs"], 8)
            self.assertEqual(report["blind_ab_study"]["ties"], 2)
            self.assertEqual(report["blind_ab_study"]["v2_win_rate_decisive"], 0.5)
            self.assertEqual(report["blind_ab_study"]["binomial_p_value_vs_chance"], 1.0)
            self.assertEqual(report["blind_reid_study"]["reid_accuracy"], 0.2)
            self.assertEqual(report["blind_reid_study"]["binomial_p_value_vs_chance"], 1.0)

    def test_rejects_blank_and_invalid_values(self):
        sealed_ab, sealed_reid = _make_synthetic_keys(4, 4)
        blank_pred = {
            "ab_preference": {pid: {"selected_option": ""} for pid in sealed_ab},
            "blind_reid": {cid: {"predicted_character_id": ""} for cid in sealed_reid},
        }
        with tempfile.TemporaryDirectory() as td:
            pred_path, sealed_dir = _write_case(Path(td), sealed_ab, sealed_reid, blank_pred)
            with self.assertRaises(scorer.PendingHumanInputError):
                scorer.score_human_blind_test(pred_path, sealed_dir, out_json=None, out_md=None)

        bad_pred = {
            "ab_preference": {pid: {"selected_option": "Z"} for pid in sealed_ab},
            "blind_reid": {cid: {"predicted_character_id": "unknown_char"} for cid in sealed_reid},
        }
        with tempfile.TemporaryDirectory() as td:
            pred_path, sealed_dir = _write_case(Path(td), sealed_ab, sealed_reid, bad_pred)
            with self.assertRaises(scorer.PredictionValidationError):
                scorer.score_human_blind_test(pred_path, sealed_dir, out_json=None, out_md=None)

    def test_actual_unfilled_human_template_is_rejected_without_scoring(self):
        rc = scorer.main(["--predictions", str(scorer.DEFAULT_PREDICTIONS_PATH)])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
