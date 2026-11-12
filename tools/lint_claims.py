#!/usr/bin/env python3
"""Verify quantitative claims and wording constraints across MMCCP v2.0 docs.

Checks:
1. Every decimal float and percentage metric in repo claim docs and final
   presentation docs matches a backed number in ``results/numbers.json``,
   ``docs/demo/scorecard.csv``, ``consistency_v2/best_config.json``, or
   explicitly declared software/hardware spec tokens.
2. Forbidden overstatements (``proves``, ``state-of-the-art``, ``solves``,
   unbacked ``significantly``, ``all 4 hypotheses met``, ``zero memorization``,
   ``zero identity collapse``, ``100% VRAM compliant``) are rejected unless
   explicitly negated or cited inside an explicit prohibition literal.
3. ``docs/KNOWN_LIMITATIONS.md`` contains all 14 subsections ``(a)``..``(n)``.
4. ``results/numbers.json["claims_flags"]`` preserves locked negative flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NUMBERS_JSON = PROJECT_ROOT / "results" / "numbers.json"
DEFAULT_DEMO_CSV = PROJECT_ROOT / "docs" / "demo" / "scorecard.csv"
DEFAULT_BEST_CONFIG = PROJECT_ROOT / "consistency_v2" / "best_config.json"
DEFAULT_FINAL_DIR = Path(r"D:\5TH SEM\_final")

REPO_TARGET_DOCS = [
    "README.md",
    "README_V2.md",
    "CONTRIBUTING.md",
    "docs/00_MASTER_INDEX.md",
    "docs/KNOWN_LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/demo/METHODS_USED.md",
    "docs/maintainers/REPO_SETTINGS.md",
    "docs/run/06_HOW_IT_WORKS_FOR_BEGINNERS.md",
    "docs/run/HANDOFF_CHECKLIST.md",
    "docs/run/LICENSES_AND_MODELS.md",
    "docs/run/MAINTAINER_SETUP.md",
    "docs_generated/v2/README_V2.md",
    "docs_generated/v2/07_measured_results.md",
    "docs_generated/v2/07_PPT_UPDATE.md",
    "docs_generated/v2/PPT_FACT_SHEET.md",
    "docs_generated/v2/PR19_FINAL_COMMENT.md",
    "docs_generated/v2/PREREG_RECONCILIATION.md",
    "docs_generated/v2/SPARK_NOTES.md",
    "docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md",
]

FINAL_TARGET_DOCS = [
    "CLAIMS_AND_LIMITS.md",
    "SLIDE_FACTS.md",
    "QA_PREP.md",
]

# Non-metric version/spec/hardware numbers legitimately appearing in docs.
ALLOWED_SPEC_NUMBERS: Set[float] = {
    1.0, 1.2, 1.3, 1.5, 2.0, 2.1, 2.3, 3.0, 3.1, 3.10, 4.0, 5.5, 6.0, 7.4,
    8.0, 10.0, 12.1, 15.0, 16.0, 20.0, 24.0, 25.0, 28.0, 30.0, 32.0, 35.0,
    38.0, 40.0, 42.0, 45.0, 48.0, 50.0, 55.0, 60.0, 64.0, 65.0, 70.0, 75.0,
    80.0, 85.0, 90.0, 95.0, 99.0, 100.0, 128.0, 140.0, 512.0, 1024.0,
}

BANNED_SINGLE_WORDS = ("proves", "state-of-the-art", "solves")
BANNED_OVERCLAIMS = (
    "all 4 hypotheses met",
    "zero memorization",
    "zero identity collapse",
    "100% vram compliant",
    "100% compliant",
)

NEGATION_PREFIX_RE = re.compile(
    r"(?i)(?:\b(?:not|never|no|cannot|don't|do not|false|fail(?:s|ed)?|reject(?:s|ed)?|"
    r"unproven|avoid|banned|forbidden)\b|:\s*false\b)"
)
SIGNIFICANT_ALLOWED_RE = re.compile(
    r"(?i)(?:wardrobe|defect|rank-?1|p\s*=\s*0\.000183|p\s*=\s*0\.001453|p\s*=\s*0\.004556|p\s*=\s*0\.000877|image-level|image ci)"
)


def _collect_json_numbers(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        out.add(val)
        out.add(abs(val))
        out.add(round(val * 100.0, 6))
        out.add(round(abs(val) * 100.0, 6))
        out.add(round(val / 1024.0, 6))
        out.add(round(val / 1000.0, 6))
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_json_numbers(item, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_json_numbers(v, out)


def load_backed_numbers(
    numbers_json: Path = DEFAULT_NUMBERS_JSON,
    demo_csv: Path = DEFAULT_DEMO_CSV,
    best_config: Path = DEFAULT_BEST_CONFIG,
) -> Set[float]:
    backed: Set[float] = set(ALLOWED_SPEC_NUMBERS)
    if numbers_json.exists():
        data = json.loads(numbers_json.read_text(encoding="utf-8"))
        _collect_json_numbers(data, backed)
    if best_config.exists():
        cfg = json.loads(best_config.read_text(encoding="utf-8"))
        _collect_json_numbers(cfg, backed)
    if demo_csv.exists():
        with demo_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for val_str in row.values():
                    if val_str is None:
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    backed.add(val)
                    backed.add(abs(val))
                    backed.add(round(val * 100.0, 6))
    return backed


def _strip_non_prose_tokens(line: str) -> str:
    # Remove markdown links URLs, code spans containing hashes/paths/versions, ISO dates, timestamps
    cleaned = re.sub(r"https?://\S+", " ", line)
    cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{2}:\d{2}(?::\d{2})?\b", " ", cleaned)
    cleaned = re.sub(r"\bv?\d+\.\d+\.\d+(?:-[a-z0-9.]+)?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", " ", cleaned)
    # Remove section headers like "1.2 " or "### 3. " or "3.2)"
    cleaned = re.sub(r"^\s*(?:#+\s*)?\d+(?:\.\d+)*[.)]\s+", " ", cleaned)
    # Remove file paths with extensions or slashes
    cleaned = re.sub(r"[A-Za-z0-9_./\\-]+\.(?:py|md|json|csv|png|onnx|safetensors|ipynb|jsonl)\b", " ", cleaned)
    return cleaned


def _is_number_backed(token: str, is_pct: bool, backed: Set[float]) -> bool:
    raw = float(token.rstrip("%"))
    # Determine rounding tolerance from displayed precision
    if "." in token:
        decimals = len(token.rstrip("%").split(".")[1])
    else:
        decimals = 0
    tol = 0.5 * (10.0 ** (-decimals)) + 1e-6
    candidates = [raw]
    if is_pct:
        candidates.append(raw / 100.0)
    for cand in candidates:
        for b in backed:
            if abs(b - cand) <= tol or abs(abs(b) - cand) <= tol:
                return True
    return False


def lint_markdown_file(path: Path, backed: Set[float], check_numbers: bool = True) -> List[str]:
    issues: List[str] = []
    text = path.read_text(encoding="utf-8")
    in_fence = False

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # 1. Numeric token verification (decimals and percentages) on strict claim deliverables
        if check_numbers:
            scan_line = _strip_non_prose_tokens(line)
            for match in re.finditer(r"(?<![A-Za-z0-9_.])(\d+\.\d+%?|\d+%)(?![A-Za-z0-9_.])", scan_line):
                tok = match.group(1)
                is_pct = tok.endswith("%")
                if not _is_number_backed(tok, is_pct, backed):
                    issues.append(f"{path}:{lineno}: unbacked numeric token {tok!r} in line: {stripped[:120]}")

        # 2. Banned overstatement / wording checks (ignore backtick literals quoting forbidden terms)
        prose_no_ticks = re.sub(r"`[^`]*`", " ", line)
        lower = prose_no_ticks.lower()

        for word in BANNED_SINGLE_WORDS:
            for m in re.finditer(rf"\b{re.escape(word)}\b", lower):
                prefix = lower[max(0, m.start() - 45) : m.start()]
                if NEGATION_PREFIX_RE.search(prefix):
                    continue
                issues.append(f"{path}:{lineno}: banned word {word!r} in: {stripped[:120]}")

        for phrase in BANNED_OVERCLAIMS:
            for m in re.finditer(re.escape(phrase), lower):
                prefix = lower[max(0, m.start() - 45) : m.start()]
                suffix = lower[m.end() : min(len(lower), m.end() + 35)]
                if NEGATION_PREFIX_RE.search(prefix) or NEGATION_PREFIX_RE.search(suffix):
                    continue
                issues.append(f"{path}:{lineno}: banned overclaim {phrase!r} in: {stripped[:120]}")

        for m in re.finditer(r"\bsignificantly\b", lower):
            window = lower[max(0, m.start() - 60) : min(len(lower), m.end() + 80)]
            if NEGATION_PREFIX_RE.search(window):
                continue
            if not SIGNIFICANT_ALLOWED_RE.search(window):
                issues.append(
                    f"{path}:{lineno}: unbacked 'significantly' (only allowed for Wardrobe/Defects/Rank-1 image CI): {stripped[:120]}"
                )

    return issues


def check_structural_invariants(project_root: Path, numbers_json: Path) -> List[str]:
    issues: List[str] = []
    kl_path = project_root / "docs" / "KNOWN_LIMITATIONS.md"
    if not kl_path.exists():
        issues.append(f"Missing {kl_path}")
    else:
        kl_text = kl_path.read_text(encoding="utf-8")
        for letter in "abcdefghijklmn":
            if f"({letter})" not in kl_text:
                issues.append(f"{kl_path}: missing limitation subsection ({letter})")

    if numbers_json.exists():
        data = json.loads(numbers_json.read_text(encoding="utf-8"))
        flags = data.get("claims_flags", {})
        for must_false in (
            "zero_memorization_proven",
            "zero_identity_collapse_proven",
            "all_4_hypotheses_met",
            "vram_100pct_compliant",
            "human_blind_test_completed",
        ):
            if flags.get(must_false) is not False:
                issues.append(f"{numbers_json}: claims_flags.{must_false} must be False, got {flags.get(must_false)!r}")
    return issues


def run_lint(
    project_root: Path = PROJECT_ROOT,
    final_dir: Optional[Path] = DEFAULT_FINAL_DIR,
    extra_files: Optional[List[Path]] = None,
) -> Tuple[int, List[str]]:
    numbers_json = project_root / "results" / "numbers.json"
    backed = load_backed_numbers(
        numbers_json=numbers_json,
        demo_csv=project_root / "docs" / "demo" / "scorecard.csv",
        best_config=project_root / "consistency_v2" / "best_config.json",
    )

    strict_targets: List[Path] = []
    for rel in REPO_TARGET_DOCS:
        p = (project_root / rel).resolve()
        if p.exists():
            strict_targets.append(p)
    if final_dir is not None and final_dir.exists():
        for name in FINAL_TARGET_DOCS:
            p = (final_dir / name).resolve()
            if p.exists():
                strict_targets.append(p)
    if extra_files:
        for p in extra_files:
            rp = p.resolve()
            if rp.exists() and rp not in strict_targets:
                strict_targets.append(rp)

    strict_set = set(strict_targets)
    prose_targets: List[Path] = []
    for pat in ("docs/**/*.md", "docs_generated/v2/**/*.md"):
        for p in sorted(project_root.glob(pat)):
            if "retracted" in p.parts:
                continue
            rp = p.resolve()
            if rp not in strict_set:
                prose_targets.append(rp)
    ch_log = (project_root / "CHANGELOG.md").resolve()
    if ch_log.exists() and ch_log not in strict_set and ch_log not in prose_targets:
        prose_targets.append(ch_log)
    if final_dir is not None and final_dir.exists():
        for p in sorted(final_dir.glob("*.md")):
            rp = p.resolve()
            if rp not in strict_set and rp not in prose_targets:
                prose_targets.append(rp)

    issues: List[str] = check_structural_invariants(project_root, numbers_json)
    for target in strict_targets:
        issues.extend(lint_markdown_file(target, backed, check_numbers=True))
    for target in prose_targets:
        issues.extend(lint_markdown_file(target, backed, check_numbers=False))
    return len(strict_targets) + len(prose_targets), issues


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lint numeric claims and forbidden overstatements in MMCCP docs.")
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--no-final-dir", action="store_true")
    parser.add_argument("--files", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)

    final_dir = None if args.no_final_dir else args.final_dir
    scanned_count, issues = run_lint(PROJECT_ROOT, final_dir=final_dir, extra_files=args.files)
    print(f"[lint_claims] Scanned {scanned_count} documents; found {len(issues)} issue(s).")
    for issue in issues:
        print(f"  [FAIL] {issue}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

