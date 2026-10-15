# MMCCP Repair & Real-Improvement Plan (R0 - R11)

> **Project:** Multi-Modal Character Consistency Pipeline (MMCCP) v2.0  
> **Capstone Project ID:** PW26_CGM_02 (PES University)  
> **Repository:** https://github.com/ShamantN/CAPSTONE.git  
> **Current Base Branch:** `consistency-v2`  
> **Tracking Pull Request:** [PR #19](https://github.com/ShamantN/CAPSTONE/pull/19) (Must remain OPEN)  
> **Status:** ACTIVE EXECUTION  

---

## Master Checklist

- [x] **R0: Orientation and Safety**
  - [x] G1 pre-flight checks (`git`, `git-lfs`, `gh auth status`, remote fetch, sync check)
  - [x] Push annotated tag `pre-repair-snapshot` at `consistency-v2` head (`80d3369`)
  - [x] Machine-generate `docs_generated/v2/repair/git_facts.md` via `tools/generate_git_facts.py`
  - [x] Environment audit (diff `pip freeze` vs baseline snapshots for `my_app_env` and Forge Python in `environment_drift.md`)
  - [x] Baseline v1 regression render check (Scene 1, seed 11 verified in `v1_regression_verification.md`)
  - [x] Start 1 Hz `nvidia-smi` background telemetry sampler (`results/telemetry/vram_<date>.csv`)
  - [x] Record original Windows AC sleep/hibernate power settings (`powercfg /q` in `power_settings_original.txt`) and set to 0 (never)
  - [x] Image-viewing canary test (view known image and inverted copy with image tool in `canary_verification.md`)
  - [x] Verify reserved seeds {606, 707, 808} are completely untouched (verified in `reserved_seeds_verification.md`)
  - [x] Update `consistency_v2/PROGRESS.md` and `REPAIR_PLAN.md`

- [x] **R1: Audit and Retract** (Branch: `docs/audit-retractions`)
  - [x] Write and run `tools/claims_audit.py` to produce `docs_generated/v2/repair/claims_ledger.csv`
  - [x] Verify and record verdicts for Problem Register items A1–A13 and B1–B13 in `problem_register.md`
  - [x] Find all dead/non-existent markdown path references (`broken_links.md`)
  - [x] Quarantine ungrounded docs into `docs_generated/v2/retracted/` with RETRACTED banner
  - [x] Add "UNDER REPAIR" banners to `README_V2.md` and top-level `README.md`
  - [x] Write `docs_generated/v2/RETRACTIONS.md`
  - [x] Post status comment on PR #19 and update body with audit notice
  - [x] PR and squash-merge `docs/audit-retractions` into `consistency-v2`

- [ ] **R2: Repair the Measuring Tools** (Branch: `fix/verifier-v1.1`)
  - [x] Root-cause fix for SegFormer / transformers / torch / numpy bridge
  - [x] Remove central-torso fallback; raise explicit `seg_failed` on failure
  - [x] Detector health monitoring (`ran=True/False` recorded per image across all 6 sub-detectors)
  - [x] Implement control test suite (`tests/verifier_controls/`) covering flip, inversion, wrong face, clone face, JPEG resave
  - [x] Rebuild calibration curves with challenging genuine/impostor pairs and child-specific genuine/impostor splits
  - [x] Calibrate wardrobe tolerance ($\Delta E$) on >=60 labeled clothing crops with holdout balanced accuracy
  - [x] Deterministic freeze script `tools/freeze_verifier.py` and `tools/check_freeze.py` updating `verifier_freeze.json` (v1.1)
  - [x] Write `docs_generated/v2/verifier_changelog.md`
  - [x] Re-score all historical renders (v1 baseline, v2 rc1, ablation) via `tools/rescore_all.py` into `results/rescored_v1.1/`
  - [ ] PR, merge into `consistency-v2`, tag `verifier-v1.1-frozen`, push tag

- [ ] **R3: Splits, Seal and Baselines** (Branch: `feat/splits-and-seal`)
  - [ ] Define splits in `consistency_v2/splits.py` (DEV-FAST, DEV-FULL, DEV-2, TEST-2, RESERVED) with access guard logging
  - [ ] Author `storyboards/holdout_v2.json` with 3 brand-new characters and 4 scenes (never rendered by v2)
  - [ ] Generate holdout anchors via standard certified anchor process
  - [ ] Write `consistency_v2/test2_seal.json` and tag `test2-sealed`
  - [ ] Run baseline v1 from git worktree at tag `baseline-v1` on DEV-2 and TEST-2, scored with Verifier v1.1

- [ ] **R4: Instrument the Pipeline** (Branches: `feat/stage-trace`, `fix/ablation-toggles`)
  - [ ] Add `trace=True` intermediate image/embedding saving with SHA256 and pixel-diff metrics
  - [ ] Add `StageNoOpError` assertion for any enabled stage that fails to change pixels
  - [ ] Fix ablation runner (`tools/ablate.py`) with flag propagation and diff assertions
  - [ ] Measure and document Forge diffusion nondeterminism across identical seeds

- [ ] **R5: Diagnose the Real Pipeline**
  - [ ] Execute v2 default on DEV-FAST with full stage tracing
  - [ ] Analyze per-stage identity scores, swap targeting, face detailer overwrites, and binding failures
  - [ ] Analyze adult vs child disparity
  - [ ] Write `docs_generated/v2/repair/diagnosis.md` with empirical evidence to direct R7 work streams

- [ ] **R6: Rebuild the Identity Bank & M12 Prep** (Branch: `feat/bank-views`)
  - [ ] Generate and gate 5 to 8 multi-pose/expression views per character into local identity bank
  - [ ] Measure M1 gain on DEV-FAST (single anchor vs centroid vs nearest-pose selection)
  - [ ] Prepare M12 Collage-Init implementation behind feature flag

- [ ] **R7: Identity and Binding Repair Loops** (Branches: `fix/identity-*`, `fix/binding-*`, `feat/m12-collage-init`)
  - [ ] W1: Stage execution order (lock identity last)
  - [ ] W2: Face swap fidelity (2x crop upscale, feathered paste-back, CodeFormer sweep)
  - [ ] W3: Layout & face scale enforcement (minimum face pixel constraints)
  - [ ] W4: Identity unit weight sweeps (InstantID & IP-Adapter Plus)
  - [ ] W5: Regional & wardrobe prompt binding + masked inpainting
  - [ ] W6: M12 Collage-Init evaluation
  - [ ] W7: Best-of-N candidate selection (M9) with CPU pre-screening
  - [ ] W8: Identity memory (M10) drift testing
  - [ ] W9: Reference-only attention (M8) sweeps
  - [ ] Update `best_config.json` via script from empirical wins
  - [ ] Execute leave-one-out ablation on DEV-2

- [ ] **R8: Real Visual Review** (Branch: `test/visual-review-v2`)
  - [ ] Contact sheet generator, blind review template, and merge tools
  - [ ] Planted failure canary verification (100% catch rate required)
  - [ ] Blind A/B preference testing (v1 vs v2) with binomial CI
  - [ ] Blind re-identification testing on anonymous face crops
  - [ ] Self-consistency testing on 12 key images
  - [ ] Local VLM second opinion (Qwen2-VL-2B or moondream2) with Cohen's kappa

- [ ] **R9: End-to-End Clean-Slate Run** (Branch: `test/e2e-clean-slate`)
  - [ ] Wipe project caches (keep weights/source safe)
  - [ ] Headless execution (`run_from_scratch.py --pipeline v2 --mode default|quality`, `--pipeline v1`)
  - [ ] Streamlit Studio UI walk-through via browser subagent with screenshots
  - [ ] Latency and VRAM telemetry extraction

- [ ] **R10: Final Sealed Evaluation** (TEST-2, Run ONCE)
  - [ ] Confirm preconditions and log split access
  - [ ] Run v2-default and v2-quality on TEST-2 (Ebright 30 renders + Holdout 12 renders)
  - [ ] Verifier v1.1 scoring, cluster-bootstrap 95% CIs, Wilcoxon tests, adult/child splits
  - [ ] Post-eval leave-one-out ablation on TEST-2 subset
  - [ ] Evaluate P1, P2, P3, P4 success criteria honestly (MET / NOT MET)

- [ ] **R11: Rebuild Documentation from Artifacts** (Branch: `docs/rebuild-from-artifacts`)
  - [ ] Run `tools/collect_numbers.py` to write `results/numbers.json`
  - [ ] Render docs from templates via `tools/render_docs.py`
  - [ ] Run `tests/test_docs_numbers.py` lint
  - [ ] Generate slide handoff: `07_PPT_UPDATE.md`, `PPT_FACT_SHEET.md`, `SPARK_NOTES.md`
  - [ ] Mark old docs SUPERSEDED
  - [ ] Update PR #19 body with machine-read facts and retractions
  - [ ] Tag `v2.0.0-rc2` (annotated, pushed)
  - [ ] Restore Windows power settings and stop telemetry sampler
