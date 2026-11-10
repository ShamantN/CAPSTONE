# Phase R0: Reserved Seeds Verification

- **Date:** 2026-10-15
- **Reserved Seeds:** `{606, 707, 808}`
- **Requirement:** Confirm reserved seeds {606, 707, 808} appear nowhere in code, experiments, test configurations, storyboards, or test runs.

## Search Methodology:
1. Systematic regex search `\b(606|707|808)\b` across all files in repository (`.py`, `.json`, `.yaml`, `.yml`, `.sh`, `.bat`, `.ps1`), excluding git objects and virtual environments.
2. Verified matches:
   - `consistency_v2/config.py`: Explicit scope guard constant `RESERVED_SEEDS = {606, 707, 808}` preventing these seeds from being used in any pipeline run.
   - `results/baseline_v1/dev_full/report.json`: Floating point metric value `0.707` (centroid similarity).
   - `experiments/clean_run_ground_truth/anchors/ground_truth_anchors.json`: Substring within base64 encoded image data.

## Verdict:
**CLEAN**: Reserved seeds `{606, 707, 808}` have never been used in any rendering, tuning, test, or evaluation run. They remain strictly reserved and untainted for future verification.
