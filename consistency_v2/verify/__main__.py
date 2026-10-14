"""
Module entrypoint for `python -m consistency_v2.verify`
"""
import sys
import argparse
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from consistency_v2.verify.cli import run_verification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMCCP v2 Verifier CLI")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing scene images to audit")
    parser.add_argument("--story", required=True, type=Path, help="Path to story spec JSON")
    parser.add_argument("--bank", required=True, type=Path, help="Directory containing character bank reference images")
    parser.add_argument("--out", required=True, type=Path, help="Output path for report.json")
    parser.add_argument("--baseline-report", required=False, type=Path, help="Optional baseline report for Wilcoxon test")
    args = parser.parse_args()

    run_verification(
        images_dir=args.images,
        story_path=args.story,
        bank_dir=args.bank,
        output_report_path=args.out,
        baseline_report_path=args.baseline_report
    )
