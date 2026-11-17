"""
MMCCP TEST-3 Evaluation Protocol: DEV-3 Controls Execution (tools/run_dev3_controls.py).

Executes all 5 pre-registered controls:
1. Positive control: same image after flip/crop/JPEG yields AUC >= 0.99.
2. Negative control: 1000 random label shuffles yields AUC within 0.50 +/- 0.05.
3. Blank/no-face image yields 0 detected faces.
4. Synthetic-data tests: perfect separation AUC 1.0, identical dist AUC ~0.50, bootstrap coverage >= 90%.
5. Diversity check: verifies that DEV-3 renders are not an artificial copy-paste set.

Saves results to results/eval3/dev3_controls_report.json.
"""
import os
import sys
import json
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval3.metrics.evaluator import Eval3Evaluator
from eval3.metrics.controls import (
    run_positive_control,
    run_negative_control,
    run_blank_image_control,
    run_synthetic_separation_tests,
    run_diversity_copy_paste_check,
)
from eval3.runner import EVAL_OUTPUT_BASE


def main():
    print("=" * 70)
    print("RUNNING DEV-3 CONTROLS (1 TO 5)")
    print("=" * 70)

    evaluator = Eval3Evaluator()

    # Load pilot results to obtain genuine and impostor scores for Control 2
    pilot_report_path = PROJECT_ROOT / "results" / "eval3" / "dev3_pilot_report.json"
    genuine_scores = []
    impostor_scores = []
    if pilot_report_path.exists():
        with open(pilot_report_path, "r", encoding="utf-8") as f:
            pilot_rep = json.load(f)
            # Use winner C0 scores if available
            c0_data = pilot_rep.get("candidates", {}).get("C0", {})
            genuine_scores = c0_data.get("raw_genuine_scores", [])
            impostor_scores = c0_data.get("raw_impostor_scores", [])

    # Find sample images for Control 1 and Control 5
    c0_dir = EVAL_OUTPUT_BASE / "dev3_pilot_C0"
    all_c0_pngs = sorted(list(c0_dir.glob("**/*.png")))
    if not all_c0_pngs:
        raise FileNotFoundError(f"No C0 renders found in {c0_dir}!")

    test_image_p = all_c0_pngs[0]
    test_img = cv2.imread(str(test_image_p))
    print(f"Using test image for positive control: {test_image_p.name}")

    results = {}

    # Control 1: Positive Control
    print("\n--- Running Control 1: Positive Control ---")
    c1_res = run_positive_control(test_img, evaluator)
    results["control_1_positive"] = c1_res
    print(f"  Result: {c1_res['measured_auc']:.4f} (PASS: {c1_res['passed']})")

    # Control 2: Negative Control (1000 shuffles)
    print("\n--- Running Control 2: Negative Control (1000 shuffles) ---")
    c2_res = run_negative_control(genuine_scores, impostor_scores, n_shuffles=1000, seed=42)
    results["control_2_negative"] = c2_res
    print(f"  Mean Shuffled AUC: {c2_res['mean_shuffled_auc']:.4f} (PASS: {c2_res['passed']})")

    # Control 3: Blank Image Detection
    print("\n--- Running Control 3: Blank Image Detection ---")
    c3_res = run_blank_image_control(evaluator.auditor)
    results["control_3_blank"] = c3_res
    print(f"  Faces Detected: {c3_res['faces_detected']} (PASS: {c3_res['passed']})")

    # Control 4: Synthetic-Data Calibration & Coverage
    print("\n--- Running Control 4: Synthetic Calibration & Bootstrap Coverage ---")
    c4_res = run_synthetic_separation_tests(n_trials=100, seed=42)
    results["control_4_synthetic"] = c4_res
    print(f"  Perfect Sep: {c4_res['perfect_separation_auc']} | Identical: {c4_res['identical_distributions_auc']} | Coverage: {c4_res['bootstrap_coverage_rate']:.2%} (PASS: {c4_res['passed']})")

    # Control 5: Diversity & Copy-Paste Guard
    print("\n--- Running Control 5: Diversity & Copy-Paste Guard ---")
    c5_res = run_diversity_copy_paste_check(all_c0_pngs[:10])
    results["control_5_diversity"] = c5_res
    print(f"  Mean Sim: {c5_res['mean_cross_scene_similarity']:.4f} | Max: {c5_res['max_cross_scene_similarity']:.4f} (PASS: {c5_res['passed']})")

    all_passed = all(r.get("passed", False) for r in results.values())
    report = {
        "all_controls_passed": all_passed,
        "controls": results
    }

    out_path = PROJECT_ROOT / "results" / "eval3" / "dev3_controls_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL DEV-3 CONTROLS PASSED: {all_passed}")
    print(f"Saved controls report to {out_path}")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
