# CHANGELOG: Multi-Modal Character Consistency Pipeline (MMCCP)

This document tracks every file modification, architectural fix, bug resolution, and scoring verification performed during the execution and loop-engineering of the MMCCP project (PES University Capstone Phase 3, Project ID `PW26_CGM_02`).

## Change Log Entries Format
Each entry documents:
- **Timestamp**: ISO 8601 timestamp
- **File Modified**: Relative path to target file
- **Category**: BUG_FIX / REFACTOR / ENVIRONMENT / SCORING_CHANGE / TEST_HARNESS
- **Reason**: Root cause and motivation for the modification
- **Diff Summary**: Exact before/after code changes
- **Evidence / Verification**: Test or run demonstrating resolution with zero regressions

---

## Initial State Log
- **2026-10-14T11:03:00+05:30**: Initialized project run environment and audit suite. All existing source code unmodified.

---

## Session Modifications Log

### Entry 1: Bug Fix in Forge Script Payload Assembly
- **Timestamp:** 2026-10-14T11:15:20+05:30
- **File Modified:** [`generation/forge_client.py`](file:///d:/5TH%20SEM/HACKATHON/generation/forge_client.py)
- **Category:** BUG_FIX
- **Reason:** Stable Diffusion WebUI Forge requires exact lowercase matching for extension keys in `alwayson_scripts` (e.g. `dynamicthresholding (cfg-fix) integrated`). Mismatched casing resulted in silent omission of PAG and CFG-Fix.
- **Diff Summary:** Normalized script keys in `build_alwayson_payload()` to match WebUI Forge `/sdapi/v1/scripts` registry.
- **Evidence / Verification:** Smoke test `experiments/test_api_smoke.py` passed in 16.09s with all latent enhancers confirmed active.

### Entry 2: Font Asset Fallback & UI Stability in Streamlit
- **Timestamp:** 2026-10-14T11:25:40+05:30
- **File Modified:** [`app.py`](file:///d:/5TH%20SEM/HACKATHON/app.py)
- **Category:** BUG_FIX
- **Reason:** Hardcoded absolute path to `OpenDyslexic-Regular.ttf` caused font loading warnings when running across different environments.
- **Diff Summary:** Replaced hardcoded path with `Path(__file__).resolve().parent / "OpenDyslexic-Regular.ttf"` relative path resolver.
- **Evidence / Verification:** Streamlit app loaded cleanly on `http://localhost:8501` without console font warnings.

### Entry 3: Comprehensive Function Coverage in Reference Documentation
- **Timestamp:** 2026-10-14T12:28:40+05:30
- **File Modified:** [`docs_generated/05_function_by_function_reference.md`](file:///d:/5TH%20SEM/HACKATHON/docs_generated/05_function_by_function_reference.md)
- **Category:** REFACTOR
- **Reason:** User non-negotiable rule required an AST parse of every `.py` file to prove that every function, class, and method is documented with zero diff.
- **Diff Summary:** Appended Section 10 documenting all auxiliary diagnostic methods, test fixtures, and class `DeepFeatureExtractor`.
- **Evidence / Verification:** `scratch/verify_ast_coverage.py` executed with exit code 0, reporting `Total .py files audited: 59 | Files missing in 04: 0 | Classes missing in 05: 0 | Functions missing in 05: 0`.

### Entry 4: Systematic 10-Stage Ablation Harness
- **Timestamp:** 2026-10-14T12:15:10+05:30
- **File Modified:** [`scratch/run_systematic_ablation.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/run_systematic_ablation.py)
- **Category:** TEST_HARNESS
- **Reason:** Systematic ablation holding seed 42 and Scene 2 fixed across Plain SDXL, Demographic Locking, FreeU/PAG, Depth, Pass 2, ReActor, IP-Adapter, InstantID, FaceID Plus v2, and Full Pipeline.
- **Diff Summary:** Integrated `ControlSpec`, `ControlNetConditioner`, `RefinementController`, and quad-modal evaluation outputting to JSON and CSV.
- **Evidence / Verification:** Generated `docs_generated/raw/ablation_study_results.csv` and all 10 high-resolution comparison images in `docs_generated/raw/ablation_images/`.

### Entry 5: Red-Team Chromatic Inversion Experiment Harness
- **Timestamp:** 2026-10-14T11:45:00+05:30
- **File Modified:** [`scratch/test_inversion_experiment.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/test_inversion_experiment.py)
- **Category:** TEST_HARNESS
- **Reason:** Empirically cross-examine the claimed ArcFace inversion score (0.7410) vs reality using active certified anchor `young_ebright`.
- **Diff Summary:** Inverted RGB channels ($255-x$), computed ArcFace, Palette, and zero-shot VLM anomaly probe.
- **Evidence / Verification:** Proven ArcFace scores `0.9346` on inverted face, while VLM zero-shot CLIP flags the anomaly with 99.81% confidence.
