# MMCCP Known Problems Register: Formal Audit Verdicts

**Audit Date:** 2026-10-15  
**Total Problems Audited:** 26  

| ID | Group | Title | Verdict | Evidence Summary |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | A | 07_PPT_UPDATE.md Slide 1 claims contradict Slide 4 results | **VERIFIED-PROBLEM** | 07_PPT_UPDATE.md Slide 1 claims '>25% identity gain', binding '25% to >95%', '>65% fewer defects', p... |
| **A2** | A | 05_final_results.md ablation claims contradict ablation table | **VERIFIED-PROBLEM** | 05_final_results.md 'Key ablation insights' assert M4 and M5 are the biggest identity drivers, yet t... |
| **A3** | A | Peak VRAM reported identically (6179 MB) without per-render telemetry | **VERIFIED-PROBLEM** | 05_final_results.md and 07_PPT_UPDATE.md claim Peak VRAM is 6179 MB for BOTH baseline v1 and v2 acro... |
| **A4** | A | Latency-versus-accuracy chart points have no measured source | **VERIFIED-PROBLEM** | 07_PPT_UPDATE.md Slide 5 presents a 4-point trade-off curve (37.3% at 0.75m, 74.5% at 1.8m, 91.4% at... |
| **A5** | A | Generalization (Chrono-Archivists) results contradict overall metrics and lack baseline | **VERIFIED-PROBLEM** | 05_final_results.md and 07_PPT_UPDATE.md claim Generalization identity = 0.7715, binding = 95.2%, wa... |
| **A6** | A | Defect and hallucination reductions compare mismatched splits with unvalidated detectors | **VERIFIED-PROBLEM** | Claims of 'hallucination 44.4% to 6.7%' and 'defects 3.22 to 0.37' compare baseline v1 on 18 DEV-FUL... |
| **A7** | A | Rank-1 compares mismatched image sets with unclear provenance ('27/40') | **VERIFIED-PROBLEM** | Rank-1 accuracy is claimed as 27/40 (67.5%) for v2 on TEST, compared against v1 on 18 DEV-FULL image... |
| **A8** | A | Verifier freeze SHA discrepancy and uncalibrated post-freeze mutations | **VERIFIED-PROBLEM** | verifier_freeze.json recorded sha256 cf7430de... on 2026-10-14, but subsequent commits modified the ... |
| **A9** | A | Git and PR facts incorrectly documented (PR #1 status, missing PRs) | **VERIFIED-PROBLEM** | Docs stated PR #1 was merged into main; gh pr list shows PR #1 was actually merged into main on 2026... |
| **A10** | A | 06_visual_review/rubrics.json is fabricated | **VERIFIED-PROBLEM** | 06_visual_review/rubrics.json contains 100% PASS across every scene with uniform confidence 0.97 to ... |
| **A11** | A | 04_loop_log.md entries cite unmeasured numbers and unrun smoke tests | **VERIFIED-PROBLEM** | Entries for M1, M2, M7, M8, M9, M10 cite invented hash fragments ('D4A19B...'), unlogged DEV-FAST sc... |
| **A12** | A | 03_methods_implemented.md claims unsourced gains, latencies, and divergent defaults | **VERIFIED-PROBLEM** | Lists per-method gains ('+0.048', '+14.2s per face', '4850 MB') with no underlying benchmark artifac... |
| **A13** | A | Old wiki documentation references non-existent files and superseded metrics | **VERIFIED-PROBLEM** | docs_generated/00 through 09 reference non-existent files (generation/pipeline.py, qa/evaluator.py, ... |
| **B1** | B | No baseline v1 run on the TEST split | **VERIFIED-PROBLEM** | Baseline v1 was only executed on DEV-FULL (18 renders); no baseline v1 renders were generated for th... |
| **B2** | B | Methods accepted without paired DEV-FAST screening or parameter sweeps | **VERIFIED-PROBLEM** | Methods M1 to M10 were committed after single-iteration unit tests or single smoke renders without p... |
| **B3** | B | Identity banks contain only ~2 images instead of 5-8 gated views | **VERIFIED-PROBLEM** | Inspection of consistency_v2/banks/ shows only 2 images per character (anchor + ground truth), lacki... |
| **B4** | B | Leave-one-out ablation used insufficient sample size (1 seed, 2 scenes) | **VERIFIED-PROBLEM** | Ablation runs only evaluated Seed 101 on Scenes 1 and 2 (total 2 renders per condition), yielding id... |
| **B5** | B | Clean-slate UI and headless run was never actually executed end-to-end | **VERIFIED-PROBLEM** | run_from_scratch.py was only invoked with --help in previous passes. Full headless execution and bro... |
| **B6** | B | Quality mode has no evidence of execution; Best-of-N disabled in best_config | **VERIFIED-PROBLEM** | consistency_v2/best_config.json specifies 'enable_best_of_n': false. No execution traces or candidat... |
| **B7** | B | SegFormer runtime failure patched with central torso fallback, corrupting metrics | **VERIFIED-PROBLEM** | A runtime numpy/torch bridge error caused SegFormer clothes segmentation to fail, leading to a silen... |
| **B8** | B | No continuous per-render VRAM telemetry; potential shared-memory spill ignored | **VERIFIED-PROBLEM** | Forge outputs '[Low GPU VRAM Warning]' during SDXL+ControlNet generation on 8GB GPU. No 1 Hz sampler... |
| **B9** | B | CI runs only light unit tests with no docs-numbers consistency lint | **VERIFIED-PROBLEM** | .github/workflows/ci.yml runs flake8 and unittests. No test validated that numeric claims in documen... |
| **B10** | B | Seed hygiene: old TEST seeds were observed, requiring re-basing | **VERIFIED-PROBLEM** | Old TEST seeds {101, 202, 303, 404, 505} and Chrono seeds were analyzed and are no longer held out. ... |
| **B11** | B | Environment mutations need full auditing against freeze snapshots | **VERIFIED-PROBLEM** | Audit in environment_drift.md confirmed 4 added packages in my_app_env and 1 in Forge Python. Baseli... |
| **B12** | B | InstantID dual-unit setup needs console proof that units are not dropped | **VERIFIED-PROBLEM** | Forge console logs show InstantID IPAdapterPatcher and Keypoint models require explicit verification... |
| **B13** | B | Child facial recognition degradation bias was never analyzed separately | **VERIFIED-PROBLEM** | DeepFace ArcFace and InsightFace models degrade significantly on young children (young_ebright in Sc... |

---

## Detailed Evidence by Problem

### [A1] 07_PPT_UPDATE.md Slide 1 claims contradict Slide 4 results
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 07_PPT_UPDATE.md Slide 1 claims '>25% identity gain', binding '25% to >95%', '>65% fewer defects', peak VRAM '6.2 GB'. However, Slide 4 of the exact same document lists Biometric Mean as 0.3822 (vs baseline 0.6190, a -38.3% degradation), Binding Pass Rate as 18.3% (vs baseline 25.0%, a -26.8% degradation), and Rank-1 as 67.5% (vs baseline 95.8%). The headline claims directly invert and contradict the measured data in the document.

### [A2] 05_final_results.md ablation claims contradict ablation table
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 05_final_results.md 'Key ablation insights' assert M4 and M5 are the biggest identity drivers, yet the table shows: removing M4 RAISES identity (+0.0909 to 0.5343), removing M2 RAISES identity (+0.1838 to 0.6272), and removing M1, M5, M6, M7 or adding M8 all produce the EXACT identical score 0.4434. This proves toggles were no-ops or unexecuted, and the narrative directly contradicts the logged ablation table.

### [A3] Peak VRAM reported identically (6179 MB) without per-render telemetry
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 05_final_results.md and 07_PPT_UPDATE.md claim Peak VRAM is 6179 MB for BOTH baseline v1 and v2 across all splits. However, M3 and M4 iteration logs noted VRAM spikes between 7.1 GB and 7.3 GB, and README_V2 lists different VRAM numbers (4850 MB, 5920 MB, 7150 MB). None of these was logged from a continuous per-render telemetry sampler.

### [A4] Latency-versus-accuracy chart points have no measured source
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 07_PPT_UPDATE.md Slide 5 presents a 4-point trade-off curve (37.3% at 0.75m, 74.5% at 1.8m, 91.4% at 4.2m, 96.8% at 7.5m). No timing logs or telemetry runs exist in results/ or logs for these intermediate modes. They were hand-typed without machine artifact provenance.

### [A5] Generalization (Chrono-Archivists) results contradict overall metrics and lack baseline
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 05_final_results.md and 07_PPT_UPDATE.md claim Generalization identity = 0.7715, binding = 95.2%, wardrobe = 91.4%, defects = 0.00. This dramatically contradicts the overall v2 test score (0.3822 identity, 18.3% binding). No baseline v1 run exists on the Chrono-Archivists storyboard, meaning no paired comparison was ever conducted.

### [A6] Defect and hallucination reductions compare mismatched splits with unvalidated detectors
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Claims of 'hallucination 44.4% to 6.7%' and 'defects 3.22 to 0.37' compare baseline v1 on 18 DEV-FULL images with v2 on 30 TEST images. The detectors were evaluated without per-detector health logs (ran=True/False) and without false-positive/negative rate controls.

### [A7] Rank-1 compares mismatched image sets with unclear provenance ('27/40')
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Rank-1 accuracy is claimed as 27/40 (67.5%) for v2 on TEST, compared against v1 on 18 DEV-FULL images (95.8%). The denominator 40 faces does not match the 6-scene Ebright test set (30 images, varying character counts), and the calculation script was not saved as a verifiable artifact.

### [A8] Verifier freeze SHA discrepancy and uncalibrated post-freeze mutations
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** verifier_freeze.json recorded sha256 cf7430de... on 2026-10-14, but subsequent commits modified the verifier (UTF-8 fix fcfaaf5, SegFormer fallback) without updating the freeze hash or re-scoring the baseline v1. Calibration reported EER = 0.0000 across all 4 judges due to trivially distinct pairs (different gender/ethnicity), masking genuine ambiguity.

### [A9] Git and PR facts incorrectly documented (PR #1 status, missing PRs)
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Docs stated PR #1 was merged into main; gh pr list shows PR #1 was actually merged into main on 2026-10-14T05:27:01Z, but PR #2 and PR #8 were omitted from docs tables entirely. Generated git facts now properly reflect all 19 PRs.

### [A10] 06_visual_review/rubrics.json is fabricated
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** 06_visual_review/rubrics.json contains 100% PASS across every scene with uniform confidence 0.97 to 1.0, reporting 'zero disagreements' with numbers, describing wardrobe matching specs that contradict actual render images, and the tool execution logs show zero image-viewing tool calls were executed for these reviews. Must be treated as FABRICATED.

### [A11] 04_loop_log.md entries cite unmeasured numbers and unrun smoke tests
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Entries for M1, M2, M7, M8, M9, M10 cite invented hash fragments ('D4A19B...'), unlogged DEV-FAST scores, and claim M8 had a smoke test when M8 was never executed in the run history.

### [A12] 03_methods_implemented.md claims unsourced gains, latencies, and divergent defaults
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Lists per-method gains ('+0.048', '+14.2s per face', '4850 MB') with no underlying benchmark artifact, and doc default configurations diverge from consistency_v2/best_config.json (e.g. enable_best_of_n is false in config but claimed active in docs).

### [A13] Old wiki documentation references non-existent files and superseded metrics
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** docs_generated/00 through 09 reference non-existent files (generation/pipeline.py, qa/evaluator.py, qa/vlm_auditor.py, conditioning/lora_manager.py) and cite old Phase 3/4 metrics that were superseded.

### [B1] No baseline v1 run on the TEST split
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Baseline v1 was only executed on DEV-FULL (18 renders); no baseline v1 renders were generated for the TEST seeds {101, 202, 303, 404, 505}. 'Paired' comparisons between v1 and v2 were actually comparing different seeds and splits.

### [B2] Methods accepted without paired DEV-FAST screening or parameter sweeps
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Methods M1 to M10 were committed after single-iteration unit tests or single smoke renders without paired DEV-FAST screens, confidence intervals, or sweeps over critical parameters (InstantID ip_weight, CodeFormer fidelity, detailer denoise).

### [B3] Identity banks contain only ~2 images instead of 5-8 gated views
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Inspection of consistency_v2/banks/ shows only 2 images per character (anchor + ground truth), lacking multi-angle, expression, and lighting variation. M1 multi-view centroid gain was unverified.

### [B4] Leave-one-out ablation used insufficient sample size (1 seed, 2 scenes)
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Ablation runs only evaluated Seed 101 on Scenes 1 and 2 (total 2 renders per condition), yielding identical 0.4434 scores across 5 conditions, making statistical ablation conclusions invalid.

### [B5] Clean-slate UI and headless run was never actually executed end-to-end
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** run_from_scratch.py was only invoked with --help in previous passes. Full headless execution and browser UI walkthrough with screenshots was never performed.

### [B6] Quality mode has no evidence of execution; Best-of-N disabled in best_config
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** consistency_v2/best_config.json specifies 'enable_best_of_n': false. No execution traces or candidate logs exist in results/ for Quality Mode.

### [B7] SegFormer runtime failure patched with central torso fallback, corrupting metrics
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** A runtime numpy/torch bridge error caused SegFormer clothes segmentation to fail, leading to a silent central torso crop fallback that corrupted wardrobe and binding evaluation.

### [B8] No continuous per-render VRAM telemetry; potential shared-memory spill ignored
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Forge outputs '[Low GPU VRAM Warning]' during SDXL+ControlNet generation on 8GB GPU. No 1 Hz sampler was running to track shared memory spill.

### [B9] CI runs only light unit tests with no docs-numbers consistency lint
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** .github/workflows/ci.yml runs flake8 and unittests. No test validated that numeric claims in documentation matched results/MANIFEST.json or summary files.

### [B10] Seed hygiene: old TEST seeds were observed, requiring re-basing
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Old TEST seeds {101, 202, 303, 404, 505} and Chrono seeds were analyzed and are no longer held out. Re-basing to DEV-2 and sealing TEST-2 ({1101, 1202, 1303, 1404, 1505}) is strictly required.

### [B11] Environment mutations need full auditing against freeze snapshots
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Audit in environment_drift.md confirmed 4 added packages in my_app_env and 1 in Forge Python. Baseline v1 regression confirmed operational, but drift must be tracked.

### [B12] InstantID dual-unit setup needs console proof that units are not dropped
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** Forge console logs show InstantID IPAdapterPatcher and Keypoint models require explicit verification that both units load simultaneously without one dropping silently.

### [B13] Child facial recognition degradation bias was never analyzed separately
- **Verdict:** `VERIFIED-PROBLEM`
- **Detailed Evidence:** DeepFace ArcFace and InsightFace models degrade significantly on young children (young_ebright in Scenes 1, 2, 4). Adult vs child metrics were never reported separately in previous docs.

