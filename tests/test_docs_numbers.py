"""
Doc-Lint Unit Tests (Phase R11 + Finalization Pass)
1. Parses EVERY numeric token in all rendered documentation files and fails if
   any token cannot be matched to a machine-generated number in results/numbers.json.
2. Enforces banned-phrase guard unless backed by an explicit True flag in
   results/numbers.json["claims_flags"].
"""
import json
import re
import unittest
from pathlib import Path
from typing import Any, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_PATH = PROJECT_ROOT / "results" / "numbers.json"

RENDERED_DOCS = [
    PROJECT_ROOT / "README_V2.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "07_measured_results.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "07_PPT_UPDATE.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "PPT_FACT_SHEET.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "SPARK_NOTES.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "PR19_FINAL_COMMENT.md",
    PROJECT_ROOT / "docs_generated" / "v2" / "PREREG_RECONCILIATION.md",
]


def _collect_json_numbers(obj: Any, values: Set[float], int_vals: Set[int], hex_tokens: Set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int):
        int_vals.add(obj)
        int_vals.add(abs(obj))
        values.add(float(obj))
        values.add(abs(float(obj)))
    elif isinstance(obj, float):
        values.add(obj)
        values.add(abs(obj))
        values.add(obj * 100.0)
        values.add(abs(obj * 100.0))
        values.add(-obj)
        values.add(-obj * 100.0)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_json_numbers(v, values, int_vals, hex_tokens)
    elif isinstance(obj, list):
        for v in obj:
            _collect_json_numbers(v, values, int_vals, hex_tokens)
    elif isinstance(obj, str):
        for m in re.findall(r"[0-9a-f]{7,}", obj):
            hex_tokens.add(m)


def _strip_structural_tokens(text: str, hex_tokens: Set[str]) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        # Strip markdown headings prefixes like "### 1.2 " and ordered list prefixes "1. "
        line = re.sub(r"^\s*#+\s*\d+(?:\.\d+)*\.?\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        # Strip links
        line = re.sub(r"file:///[^\s\)\]>]+", " ", line)
        line = re.sub(r"https?://[^\s\)\]>]+", " ", line)
        # Strip ISO dates & times
        line = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s*[+-]\d{4}|Z)?)?", " ", line)
        # Strip versions & IDs (keep pure numeric metrics intact!)
        line = re.sub(r"\bv\d+\.\d+(?:\.\d+)?(?:-rc\d+|-dev-rc\d+)?\b", " ", line)
        line = re.sub(r"\b(?:v1|v2|P[1-4]|M\d+|W\d+|R\d+|step=\d+|PW26_CGM_02|PR\s*#\d+|RTX\s+\d+)\b", " ", line)
        line = re.sub(r"\b(?:scene|Scene)\s*\d+\b", " ", line)
        line = re.sub(r"scene_\d+(?:_[a-z]+)?_seed_\d+(?:\.png)?", " ", line)
        line = re.sub(r"\b\d+(?=-character|-shot|-char|-arm|-step|-pid|-image)\b", " ", line, flags=re.I)
        line = re.sub(r"\b0\.\d+(?=\s*(?:\)|,|;|$))", lambda m: m.group(0), line)  # keep decimals!
        for hx in hex_tokens:
            if any(c in "abcdef" for c in hx):
                line = line.replace(hx, " ")
        # Strip backticked hex commit/SHA strings
        line = re.sub(r"`(?:SHA:\s*)?[0-9a-f]{7,64}(?:\.\.\.)?`", " ", line)
        line = re.sub(r"\b[0-9a-f]{7,64}\.\.\.\b", " ", line)
        # Strip inline code tokens like `+0530` or table alignment dashes
        line = re.sub(r"\s[+-]0530\b", " ", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


class TestDocsNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert NUMBERS_PATH.exists(), f"Missing canonical numbers artifact: {NUMBERS_PATH}"
        cls.numbers = json.loads(NUMBERS_PATH.read_text(encoding="utf-8"))
        cls.num_vals: Set[float] = set()
        cls.int_vals: Set[int] = set()
        cls.hex_tokens: Set[str] = set()
        _collect_json_numbers(cls.numbers, cls.num_vals, cls.int_vals, cls.hex_tokens)
        # Add universal small integers from section references / table row counts
        for d in range(0, 10):
            cls.int_vals.add(d)
            cls.num_vals.add(float(d))

    def _value_in_numbers_json(self, val: float, decimals: int) -> bool:
        # Match against numbers.json within rounding tolerance (half-ULP at reported decimal places)
        tol = max(0.5 * (10.0 ** (-decimals)) + 1e-6, 1e-5) if decimals > 0 else 0.51
        for ref in self.num_vals:
            if decimals == 0 and isinstance(ref, float):
                if abs(val - ref) <= 1e-6:
                    return True
            if abs(val - ref) <= tol or abs(-val - ref) <= tol:
                return True
        return False

    def test_numbers_json_schema(self):
        for k in ["metadata", "protocol_constants", "claims_flags", "test2_evaluation", "test2_ablation", "telemetry"]:
            self.assertIn(k, self.numbers)
        t2 = self.numbers["test2_evaluation"]
        for k in ["ebright", "holdout", "combined", "v1_baselines", "paired_deltas", "hypothesis_scorecard", "post_hoc_redefined_scorecard_rc2"]:
            self.assertIn(k, t2)

    def test_every_numeric_token_in_all_rendered_docs_backed_by_numbers_json(self):
        token_re = re.compile(r"(?<![A-Za-z0-9_])[+-]?\d+(?:\.\d+)?%?(?![A-Za-z0-9_])")
        for doc_path in RENDERED_DOCS:
            self.assertTrue(doc_path.exists(), f"Missing rendered doc: {doc_path}")
            raw_text = doc_path.read_text(encoding="utf-8")
            stripped = _strip_structural_tokens(raw_text, self.hex_tokens)

            unmatched: List[str] = []
            for m in token_re.finditer(stripped):
                tok = m.group(0)
                is_pct = tok.endswith("%")
                num_str = tok[:-1] if is_pct else tok
                if "." in num_str:
                    decimals = len(num_str.split(".")[1])
                    val = float(num_str)
                else:
                    decimals = 0
                    val = float(int(num_str))
                    if int(num_str) in self.int_vals or abs(int(num_str)) in self.int_vals:
                        continue
                if not self._value_in_numbers_json(val, decimals):
                    unmatched.append(tok)

            self.assertEqual(
                unmatched,
                [],
                f"{doc_path.name} contains numeric tokens not backed by results/numbers.json: {unmatched}",
            )

    def test_banned_overstated_phrases_require_true_claims_flags(self):
        flags = self.numbers["claims_flags"]
        phrase_flag_map = [
            ("zero memorization", "zero_memorization_proven"),
            ("zero identity collapse", "zero_identity_collapse_proven"),
            ("all 4 hypotheses", "all_4_hypotheses_met"),
            ("100% compliant", "vram_100pct_compliant"),
            ("statistically significant", "statistically_significant_ebright_bio"),
        ]
        for doc_path in RENDERED_DOCS:
            text_low = doc_path.read_text(encoding="utf-8").lower()
            for phrase, flag_key in phrase_flag_map:
                if phrase in text_low:
                    self.assertTrue(
                        flags.get(flag_key, False),
                        f"Forbidden unbacked phrase '{phrase}' found in {doc_path.name} while claims_flags['{flag_key}'] is False",
                    )

    def test_rendered_docs_key_anchors(self):
        eb = self.numbers["test2_evaluation"]["ebright"]
        ho = self.numbers["test2_evaluation"]["holdout"]
        telem = self.numbers["telemetry"]
        orig_sc = self.numbers["test2_evaluation"]["hypothesis_scorecard"]

        readme = (PROJECT_ROOT / "README_V2.md").read_text(encoding="utf-8")
        self.assertIn(f"{eb['biometric_ensemble_mean']:.4f}", readme)
        self.assertIn(f"{ho['biometric_ensemble_mean']:.4f}", readme)
        self.assertIn(f"{telem['peak_vram_mb']}", readme)
        self.assertIn("NOT MET", readme)

        reconcil = (PROJECT_ROOT / "docs_generated" / "v2" / "PREREG_RECONCILIATION.md").read_text(encoding="utf-8")
        for p in ["P1", "P2", "P3", "P4"]:
            self.assertIn(orig_sc[p]["status"], reconcil)


if __name__ == "__main__":
    unittest.main()
