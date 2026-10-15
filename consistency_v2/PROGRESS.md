# MMCCP Consistency v2: Master Progress & Execution Log

**Current Campaign:** REPAIR AND REAL-IMPROVEMENT PASS  
**Current Phase:** Phase R2 (Complete - Merging) / Phase R3 (Splits, Seal, Baselines)  
**Previous Phase:** Phase R2 (Repair the Measuring Tools - COMPLETE)  
**Active Integration Branch:** `consistency-v2`  
**Open PR:** [PR #19](https://github.com/ShamantN/CAPSTONE/pull/19) (`consistency-v2` into `main` - MUST STAY OPEN)  
**Last Tag Pushed:** `pre-repair-snapshot` (`80d3369`)  
**Hardware & Power:** Intel i7-12650H, 32GB RAM, RTX 3070 Ti Laptop GPU (8GB), AC Standby/Hibernate disabled (0s).  
**Telemetry:** 1 Hz `nvidia-smi` background sampler running in `results/telemetry/`.  

---

## 1. Branch and Commit Registry

| Branch | Base | Status | Last Pushed SHA | PR Link | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `main` | - | Protected | `cf114dc110060baf28dc3edda5e381af7fcbc2d4` | - | Production stable branch |
| `sync/local-baseline` | `main` | Open PR | `cf12f7f585af60c82c179672fafe6ef9292efb26` | [PR #1](https://github.com/ShamantN/CAPSTONE/pull/1) | Initial local baseline import & hygiene |
| `consistency-v2` | `sync/local-baseline` | Active | `912f74d` | - | Long-lived v2 integration branch |
| `feat/verifier` | `consistency-v2` | Merged (PR #3) | `96aa099` | [PR #3](https://github.com/ShamantN/CAPSTONE/pull/3) | Multi-modal verification suite |
| `fix/verifier-encoding` | `consistency-v2` | Merged (PR #4) | `85257ee` | [PR #4](https://github.com/ShamantN/CAPSTONE/pull/4) | Verifier Windows UTF-8 stdout encoding |
| `feat/m1-identity-bank` | `consistency-v2` | Merged (PR #5) | `31afaf8` | [PR #5](https://github.com/ShamantN/CAPSTONE/pull/5) | M1: Multi-reference identity bank and centroid embeddings |
| `feat/m2-layout-planner` | `consistency-v2` | Merged (PR #6) | `32eea5d` | [PR #6](https://github.com/ShamantN/CAPSTONE/pull/6) | M2: Programmatic layout planner, OpenPose skeleton & D1 fix |
| `feat/m3-regional-couple` | `consistency-v2` | Merged (PR #7) | `18c5190` | [PR #7](https://github.com/ShamantN/CAPSTONE/pull/7) | M3: True regional conditioning via Forge Couple driver |
| `feat/m4-identity-units` | `consistency-v2` | Merged (PR #9) | `96035a7` | [PR #9](https://github.com/ShamantN/CAPSTONE/pull/9) | M4: Dual-unit InstantID & IP-Adapter Plus-Face SDXL |
| `feat/m5-face-transplant` | `consistency-v2` | Merged (PR #10) | `fe592e7` | [PR #10](https://github.com/ShamantN/CAPSTONE/pull/10) | M5: Reliable identity transplant & CodeFormer enhancement |
| `feat/m6-face-detailer` | `consistency-v2` | Merged (PR #11) | `1bbe16b` | [PR #11](https://github.com/ShamantN/CAPSTONE/pull/11) | M6: Identity-conditioned face detailer & inpaint |
| `feat/m7-wardrobe-lock` | `consistency-v2` | Merged (PR #12) | `aabe03b` | [PR #12](https://github.com/ShamantN/CAPSTONE/pull/12) | M7: Wardrobe and palette lock via SegFormer |
| `feat/m8-reference-only` | `consistency-v2` | Merged (PR #13) | `18e76c5` | [PR #13](https://github.com/ShamantN/CAPSTONE/pull/13) | M8: Reference-only attention injection |
| `feat/m9-best-of-n-healing` | `consistency-v2` | Merged (PR #14) | `87c51e7` | [PR #14](https://github.com/ShamantN/CAPSTONE/pull/14) | M9: Best-of-N selection and diagnostic-driven healing |
| `feat/m10-identity-memory` | `consistency-v2` | Merged (PR #15) | `67dada5` | [PR #15](https://github.com/ShamantN/CAPSTONE/pull/15) | M10: Identity memory across scenes with drift guard |
| `feat/pipeline-v2-integration` | `consistency-v2` | Merged (PR #16) | `228f898` | [PR #16](https://github.com/ShamantN/CAPSTONE/pull/16) | Pipeline v2 integration & best_config.json |
| `feat/ui-v2` | `consistency-v2` | Merged (PR #17) | `29953c7` | [PR #17](https://github.com/ShamantN/CAPSTONE/pull/17) | UI v2 in app.py & CLI runners |
| `chore/r0-safety-setup` | `consistency-v2` | Merged (PR #20) | `33b497d` | [PR #20](https://github.com/ShamantN/CAPSTONE/pull/20) | R0 safety setup, telemetry, canary, seeds audit |
| `docs/audit-retractions` | `consistency-v2` | Merged (PR #21) | `a4cbac3` | [PR #21](https://github.com/ShamantN/CAPSTONE/pull/21) | R1 audit, quarantine unsupported docs, RETRACTIONS.md |
| `fix/verifier-v1.1` | `consistency-v2` | Active | Pending | [PR #22](https://github.com/ShamantN/CAPSTONE/pull/22) | R2 Verifier v1.1 freeze, controls, recalibration, rescore |

---

## 2. Phase Execution Checklist

### Phase 0: Setup, Repository Pre-Flight & Baseline
- [x] G1: Tooling check (`git`, `git-lfs`, `gh`), identity configuration, GitHub authentication.
- [x] G1: Case C repo initialization, baseline commit on `sync/local-baseline`, tagged `baseline-v1`.
- [x] G1: PR #1 opened to `main`, integration branch `consistency-v2` created and pushed.
- [x] G2: `.gitignore` and `.gitattributes` (LFS tracking for curated images) established.
- [x] G2: Secret scan clean, `.github/pull_request_template.md` & `.github/workflows/ci.yml` installed.
- [x] Hardware, power configuration, and Forge API endpoints verified and dumped.
- [x] Environments documented in `ENVIRONMENTS.md`; freeze files saved.
- [x] `hardcoded_cache.json` backed up outside the repository.
- [x] Image-viewing canary check (verify agent visual ability on normal vs inverted canary).
- [x] Run baseline v1 on DEV-FULL and TEST splits and cache in `results/baseline_v1/` (DEV-FULL 18/18 renders verified, report.json generated: ensemble mean 0.6190, Rank-1 95.8%, binding/wardrobe 25.0%, mean defects 3.22).

### Phase 1: Build & Calibrate Frozen Verification Suite (`feat/verifier`)
- [x] Implement Biometric Ensemble (DeepFace Facenet512, SFace, GhostFaceNet, ArcFace on CPU).
- [x] Calibration: genuine/impostor distributions, EER, and calibrated thresholds in `calibration.json`.
- [x] Binding check: SegFormer person/clothes segmentation + spatial layout assignment.
- [x] DINOv2 & CLIP-I appearance evaluation on person crops.
- [x] Scene defect detector & taxonomy evaluator (hallucination, limbs, count).
- [x] Visual Review tools (`make_contact_sheet.py`, `make_blind_sheet.py`, structured rubric JSON).
- [x] Canary tests on planted failures (catch rate = 100% on inverted canary).
- [x] Verifier freeze: record SHA256 in `verifier_freeze.json`, PR #3 squash-merge, tag `verifier-frozen`.
- [x] Verifier encoding fix for Windows CP1252 merged via PR #4 (`fcfaaf5`).

### Phase 2: Diagnose & Fix Existing Pipeline (`fix/d1-...` to `fix/d6-...`)
- [x] D1: ControlNet Depth/Pose in Pass 1 missing input image (hash & diff proof: A03 vs A04 byte-identical `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`).
- [x] D2: ReActor payload argument array mismatch derived from `script-info` (proved 30 vs 31 args mismatch, string dropdown collisions, and direct InsightFace/Inswapper loading on CPU).
- [x] D3: Replace weak `BREAK` with true spatial conditioning (`sd-forge-couple` cloned and verified).
- [x] D4: Download missing InstantID ControlNet weights (`ip-adapter_instant_id_sdxl.bin` 1.69GB downloaded to D: and symlinked to C: ControlNet models).
- [x] D5: Evaluator circularity and position-agnostic Hungarian matching resolved by Phase 1 verifier.
- [x] D6: Reconcile code vs documentation contradictions documented in `docs_generated/v2/claims_corrected.md` and `docs_generated/v2/diagnostics.md`.

### Phase 3: Implement Consistency Methods (M1 to M10 Only)
- [x] M1: Multi-reference Identity Bank with pose clustering (`consistency_v2/identity_bank.py`).
- [x] M2: Layout Planner with OpenPose skeleton & region masks (`consistency_v2/layout_planner.py`).
- [x] M3: True Regional Conditioning via Forge Couple extension (`consistency_v2/regional.py`).
- [x] M4: Dual-Unit InstantID & IP-Adapter Plus-Face SDXL tuning (`consistency_v2/identity_units.py`).
- [x] M5: Reliable Identity Transplant (ReActor script-info + Direct InsightFace blended swap; `consistency_v2/face_swap.py`).
- [x] M6: Identity-Conditioned Face Detailer (`consistency_v2/face_detailer.py`).
- [x] M7: Wardrobe and Palette Lock via SegFormer (`consistency_v2/wardrobe.py`).
- [x] M8: Reference-Only Attention Injection (`consistency_v2/reference_unit.py`).
- [x] M9: Best-of-N Selection & Diagnostic-Driven Healing (`consistency_v2/candidates.py`, `healing_v2.py`).
- [x] M10: Identity Memory Across Scenes with Drift Guard (`consistency_v2/identity_memory.py`).
- [x] Scope Guard: M11 (LoRA) OUT OF SCOPE (`use_character_lora=False` flag); seeds {606, 707, 808} reserved.

### Phase 4: Integration, UI v2 & Tuning
- [x] Integrate into `pipeline_v2.py` and `best_config.json`.
- [x] Wire v2 into `app.py` (Consistency v2 sidebar, Verifier tab with contact sheets).
- [x] Update `run_from_scratch.py` (`--pipeline v2`, `--mode fast|default|quality`).
- [x] Lightweight regression test suite in `tests/` (44/44 tests passing).

### Phase 5: Final Held-Out Evaluation (TEST Set)
- [x] Freeze code, verify git status is clean.
- [x] Run v2 default across TEST split (Ebright seeds {101, 202, 303, 404, 505} + Chrono-Archivists seeds {101, 202, 303}) completed.
- [x] Run v2 quality mode on seed 101.
- [x] Run leave-one-out ablation study across M1 to M8.
- [x] Compute master statistics (bootstrap 95% CIs, paired Wilcoxon signed-rank tests against baseline v1).
- [x] Visual review rubrics (VR-2) documented in `docs_generated/v2/06_visual_review/rubrics.json`.

### Phase 6: Documentation and Final PR
- [x] `00_progress_log.md`
- [x] `01_diagnostics.md` (D1 to D6 evidence)
- [x] `02_verifier_spec_and_calibration.md`
- [x] `03_methods_implemented.md` (M1 to M10 full specification)
- [x] `04_loop_log.md`
- [x] `06_visual_review/rubrics.json`
- [x] `07_PPT_UPDATE.md` (slide-by-slide text, tables, latency-vs-accuracy trade-off chart, M11 listed as optional future tier)
- [x] `README_V2.md` (one-command demo, modes, rollback, and "How to add M11 next" section)
- [x] `blockers.md`, `claims_corrected.md`, `git_setup.md`, `ENVIRONMENTS.md`, `CHANGELOG.md`
- [x] Tag `v2.0.0-rc1` and push tag to origin.
- [x] Open final PR from `consistency-v2` into `main`: [PR #19](https://github.com/ShamantN/CAPSTONE/pull/19) (left OPEN for user review).
