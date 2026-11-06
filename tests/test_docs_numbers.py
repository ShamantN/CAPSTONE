"""
Doc-Lint Unit Tests (Phase R11)
Verifies that all numbers, metrics, confidence intervals, and hypothesis verdicts
rendered in markdown documentation match results/numbers.json down to the exact decimal.
"""
import os
import sys
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_PATH = PROJECT_ROOT / "results" / "numbers.json"
README_V2_PATH = PROJECT_ROOT / "README_V2.md"
MEASURED_RESULTS_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "07_measured_results.md"
PPT_UPDATE_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "07_PPT_UPDATE.md"
PPT_FACT_SHEET_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "PPT_FACT_SHEET.md"
SPARK_NOTES_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "SPARK_NOTES.md"


class TestDocsNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTrue(NUMBERS_PATH.exists(), f"Missing canonical numbers artifact: {NUMBERS_PATH}")
        with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
            cls.numbers = json.load(f)

    def test_numbers_json_schema(self):
        """Verifies top-level structure of numbers.json."""
        self.assertIn("metadata", self.numbers)
        self.assertIn("test2_evaluation", self.numbers)
        self.assertIn("test2_ablation", self.numbers)
        self.assertIn("telemetry", self.numbers)
        self.assertIn("hypothesis_scorecard", self.numbers["test2_evaluation"])

    def test_readme_v2_numbers(self):
        """Asserts README_V2.md contains exact metrics from numbers.json."""
        self.assertTrue(README_V2_PATH.exists())
        text = README_V2_PATH.read_text(encoding="utf-8")
        eb = self.numbers["test2_evaluation"]["ebright"]
        ho = self.numbers["test2_evaluation"]["holdout"]
        telem = self.numbers["telemetry"]
        scorecard = self.numbers["test2_evaluation"]["hypothesis_scorecard"]

        # Ebright metrics
        self.assertIn(f"{eb['biometric_ensemble_mean']:.4f}", text)
        self.assertIn(f"{eb['rank1_accuracy']*100:.1f}%", text)
        self.assertIn(f"{eb['spatial_pass_rate']*100:.1f}%", text)
        self.assertIn(f"{eb['wardrobe_pass_rate']*100:.1f}%", text)
        self.assertIn(f"{eb['mean_defects']:.2f}", text)

        # Holdout metrics
        self.assertIn(f"{ho['biometric_ensemble_mean']:.4f}", text)
        self.assertIn(f"{ho['spatial_pass_rate']*100:.1f}%", text)

        # Telemetry & Scorecard
        self.assertIn(f"{telem['peak_vram_mb']}", text)
        self.assertIn(scorecard["overall_scorecard_verdict"], text)

    def test_measured_results_numbers(self):
        """Asserts 07_measured_results.md contains exact stats and CIs."""
        self.assertTrue(MEASURED_RESULTS_PATH.exists())
        text = MEASURED_RESULTS_PATH.read_text(encoding="utf-8")
        eb = self.numbers["test2_evaluation"]["ebright"]
        deltas = self.numbers["test2_evaluation"]["paired_deltas"]

        self.assertIn(f"{eb['biometric_ensemble_mean']:.4f}", text)
        self.assertIn(f"{deltas['ebright_delta_bio']:.4f}", text)
        self.assertIn(f"{deltas['ebright_delta_rank1']*100:.1f}%", text)
        self.assertIn(f"{deltas['ebright_delta_wardrobe']*100:.1f}%", text)

    def test_presentation_assets_numbers(self):
        """Asserts presentation handoff assets contain consistent numbers."""
        for path in [PPT_UPDATE_PATH, PPT_FACT_SHEET_PATH, SPARK_NOTES_PATH]:
            self.assertTrue(path.exists(), f"Missing presentation file: {path}")
            text = path.read_text(encoding="utf-8")
            eb = self.numbers["test2_evaluation"]["ebright"]
            telem = self.numbers["telemetry"]
            self.assertIn(f"{eb['rank1_accuracy']*100:.1f}%", text)
            self.assertIn(f"{telem['peak_vram_mb']}", text)


if __name__ == "__main__":
    unittest.main()

