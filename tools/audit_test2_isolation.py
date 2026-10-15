"""
Audit Test 2 and Reserved Seeds Isolation
Ensures that TEST-2 seeds, reserved seeds, and holdout story id do not leak into
any dev, tuning, ablation, or training scripts.
"""
import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_2_SEEDS = [1101, 1202, 1303, 1404, 1505]
RESERVED_SEEDS = [606, 707, 808]
HOLDOUT_STORY_ID = "holdout_expedition_aurora"

EXCLUDED_FILES = {
    PROJECT_ROOT / "consistency_v2" / "splits.py",
    PROJECT_ROOT / "consistency_v2" / "config.py",
    PROJECT_ROOT / "consistency_v2" / "test2_seal.json",
    PROJECT_ROOT / "tests" / "test_splits.py",
    PROJECT_ROOT / "tools" / "audit_test2_isolation.py",
    PROJECT_ROOT / "tools" / "claims_audit.py",
}


def audit_isolation():
    violations = []
    py_files = list(PROJECT_ROOT.glob("**/*.py"))

    for p in py_files:
        # Skip git, venvs, test worktrees
        if any(part in p.parts for part in [".git", "my_app_env", "node_modules", ".gemini", "worktrees"]):
            continue
        if p.resolve() in [f.resolve() for f in EXCLUDED_FILES]:
            continue

        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for seed in RESERVED_SEEDS:
            # Match whole number seed
            if re.search(r"\b" + str(seed) + r"\b", content):
                violations.append(f"RESERVED seed {seed} found in {p.relative_to(PROJECT_ROOT)}")

        for seed in TEST_2_SEEDS:
            if re.search(r"\b" + str(seed) + r"\b", content):
                violations.append(f"TEST-2 seed {seed} found in dev/tuning file: {p.relative_to(PROJECT_ROOT)}")

        if HOLDOUT_STORY_ID in content:
            violations.append(f"Holdout story ID '{HOLDOUT_STORY_ID}' found in {p.relative_to(PROJECT_ROOT)}")

    if violations:
        print(f"FAILED: {len(violations)} isolation violations found:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print("PASSED: 100% clean isolation. TEST-2 seeds, RESERVED seeds, and holdout story ID are completely sealed.")
        return True


if __name__ == "__main__":
    success = audit_isolation()
    sys.exit(0 if success else 1)
