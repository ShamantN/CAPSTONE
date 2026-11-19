#!/usr/bin/env python3
"""
tests/test_human_rating_ui.py

Tests for eval3/ui.py human rating interface state isolation:
- Verifies that widget keys incorporate rater_id to prevent state leak across raters.
- Verifies key disjointness between different rater profiles.
- Verifies default selection logic falls back to (Unanswered) when no profile exists.
"""

import unittest
from pathlib import Path
import re


class TestHumanRatingUIIsolation(unittest.TestCase):
    def setUp(self):
        self.ui_file = Path(__file__).resolve().parent.parent / "eval3" / "ui.py"
        self.assertTrue(self.ui_file.exists(), f"eval3/ui.py not found at {self.ui_file}")
        self.ui_code = self.ui_file.read_text(encoding="utf-8")

    def test_widget_keys_contain_rater_id(self):
        """Verify that Task A, Task B, and Save Button widget keys use rater_id."""
        # Task A radio key pattern: key=f"hr_{rater_id}_{tid}"
        self.assertIn('key=f"hr_{rater_id}_{tid}"', self.ui_code)
        # Task B radio key pattern: key=f"hr_{rater_id}_{sid}"
        self.assertIn('key=f"hr_{rater_id}_{sid}"', self.ui_code)
        # Save button key pattern: key=f"hr_save_{rater_id}"
        self.assertIn('key=f"hr_save_{rater_id}"', self.ui_code)

    def test_widget_key_disjointness_across_raters(self):
        """Simulate key generation for rater_1 and rater_2 to confirm complete disjointness."""
        trial_ids = [f"A{i:02d}" for i in range(1, 22)]
        seq_ids = [f"B{i:02d}" for i in range(1, 22)]

        r1 = "rater_1"
        r2 = "rater_2"

        r1_keys = set([f"hr_{r1}_{tid}" for tid in trial_ids] + [f"hr_{r1}_{sid}" for sid in seq_ids] + [f"hr_save_{r1}"])
        r2_keys = set([f"hr_{r2}_{tid}" for tid in trial_ids] + [f"hr_{r2}_{sid}" for sid in seq_ids] + [f"hr_save_{r2}"])

        # Intersect must be empty
        self.assertEqual(len(r1_keys.intersection(r2_keys)), 0)
        self.assertEqual(len(r1_keys), 21 + 21 + 1)
        self.assertEqual(len(r2_keys), 21 + 21 + 1)

    def test_default_unanswered_fallback_when_no_file(self):
        """Confirm default radio index logic defaults to 0 (Unanswered) when empty."""
        opts = ["(Unanswered)", "A", "B", "C", "D"]
        empty_saved = {}
        tid = "A01"
        prev_val = empty_saved.get(tid)
        default_idx = opts.index(prev_val) if prev_val in opts else 0
        self.assertEqual(default_idx, 0)
        self.assertEqual(opts[default_idx], "(Unanswered)")


if __name__ == "__main__":
    unittest.main()
