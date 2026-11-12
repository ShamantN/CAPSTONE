"""Unit tests for tools/lint_claims.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import lint_claims


class TestLintClaims(unittest.TestCase):
    def test_repo_docs_pass_cleanly(self):
        scanned, issues = lint_claims.run_lint(PROJECT_ROOT, final_dir=None)
        self.assertGreaterEqual(scanned, 7)
        self.assertEqual(issues, [])

    def test_catches_unbacked_metric_and_banned_overclaims(self):
        backed = lint_claims.load_backed_numbers()
        with tempfile.TemporaryDirectory() as td:
            md = Path(td) / "bad_claim.md"
            md.write_text(
                "\n".join(
                    [
                        "We achieved 91.23% accuracy with 0.9987 similarity.",
                        "This proves our pipeline is state-of-the-art and solves character drift.",
                        "Ebright biometric mean improved significantly.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            issues = lint_claims.lint_markdown_file(md, backed)
            joined = "\n".join(issues)
            self.assertIn("91.23%", joined)
            self.assertIn("0.9987", joined)
            self.assertIn("state-of-the-art", joined)
            self.assertIn("proves", joined)
            self.assertIn("solves", joined)
            self.assertIn("significantly", joined)


if __name__ == "__main__":
    unittest.main()

