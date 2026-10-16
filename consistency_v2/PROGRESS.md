# MMCCP Consistency v2: Master Progress & Execution Log

**Current Campaign:** REPAIR AND REAL-IMPROVEMENT PASS (R0 - R11)  
**Current Phase:** Phase R6 (Rebuild Identity Bank & M1 Screening - IN PROGRESS)  
**Previous Phase:** Phase R5 (Pipeline Stage-Trace Diagnosis - MERGED, PR #25)  
**Active Integration Branch:** `consistency-v2` (`b55ab2b`)  
**Active Working Branch:** `feat/bank-views`  
**Tracking Pull Request:** [PR #19](https://github.com/ShamantN/CAPSTONE/pull/19) (`consistency-v2` into `main` - MUST STAY OPEN)  
**Hardware & Power:** Intel i7-12650H, 32GB RAM, RTX 3070 Ti Laptop GPU (8GB), AC Standby/Hibernate disabled (0s).  
**Telemetry:** 1 Hz `nvidia-smi` background sampler actively logging to `results/telemetry/vram_20261016_105701.csv`.  
**Verifier Freeze:** Frozen v1.1 canonical SHA `ee3b29d5a3511f1bad0efed114b079d4c4478beeeafd6c13b9053ebb99c9b8ac` (verified PASS).  
**Data Splits & Sealing:** TEST-2 SEALED (`consistency_v2/test2_seal.json`), RESERVED seeds {606, 707, 808} UNTOUCHED.

---

## 1. Machine-Verified Branch and PR Registry

| PR # | Branch | Base | Status | Merge Commit SHA | Merged At | Purpose |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **#19** | `consistency-v2` | `main` | **OPEN** | - | - | Master tracking PR into main (Must remain OPEN) |
| **#25** | `chore/diagnosis-r5` | `consistency-v2` | MERGED | `b291aa5` | 2026-10-15 15:12 UTC | R5 pipeline stage-trace diagnosis on DEV-FAST |
| **#24** | `feat/stage-trace` | `consistency-v2` | MERGED | `ad3f8bd` | 2026-10-15 10:39 UTC | R4 StageTracer instrumentation, assertions, bugfixes |
| **#23** | `feat/splits-and-seal` | `consistency-v2` | MERGED | `eed746f` | 2026-10-15 10:17 UTC | R3 split guards, TEST-2 sealing, holdout story, baseline v1 |
| **#22** | `fix/verifier-v1.1` | `consistency-v2` | MERGED | `ed0aac9` | 2026-10-15 06:47 UTC | R2 Verifier v1.1 freeze, controls, recalibration, rescore |
| **#21** | `docs/audit-retractions` | `consistency-v2` | MERGED | `a4cbac3` | 2026-10-15 05:36 UTC | R1 claims audit, quarantine ungrounded docs, RETRACTIONS.md |
| **#20** | `chore/r0-safety-setup` | `consistency-v2` | MERGED | `bf05162` | 2026-10-15 05:25 UTC | R0 safety setup, telemetry, canary, seeds audit |

### Historical v2-rc1 PRs (SUPERSEDED / AUDITED)
- PR #1 to #18 cover legacy v2-rc1 exploratory work. Historical metrics reported in rc1 were retracted in Phase R1 (see `docs_generated/v2/RETRACTIONS.md`).

---

## 2. Pause Gap and Environment Logging

1. **Pause Gap 1 (Context Compaction):**
   - **Interval:** 2026-10-15 16:13 IST to 2026-10-15 19:30 IST (~3h 17m).
   - **Resolution:** Re-verified environment, Forge status, power settings, and verifier freeze.
2. **Pause Gap 2 (Server Restart & Interruption):**
   - **Interval:** 2026-10-15 ~21:22 IST to 2026-10-16 10:51 IST (~13h 29m).
   - **Resolution:**
     - WebUI Forge restarted with `--api --xformers --cuda-malloc` on PID 10376 (`/sdapi/v1/scripts` returned 20 scripts; Forge Couple active; ControlNet returned 10 models including InstantID units).
     - 1 Hz VRAM sampler restarted to `results/telemetry/vram_20261016_105701.csv`.
     - Windows AC standby and hibernate verified and set to 0s (`never`).
     - Split access log isolated: `consistency_v2/splits.py` updated to support `SPLIT_ACCESS_LOG_PATH` override; `tests/test_splits.py` now logs to temp files, preserving real log integrity.
     - Split access log audited: 8 entries present (2 baseline v1 runs, 6 unit test runs; 0 unauthorized accesses).
     - Verifier freeze check verified passing (`ee3b29d5...`); hardened with stray file detection.
     - Test suite executed: 55/55 tests passing.

---

## 3. Repair Pass Execution Checklist (Current Campaign)

### Phase R0: Orientation and Safety Setup
- [x] G1 tooling check (`git`, `git-lfs`, `gh`), identity configured.
- [x] Tag `pre-repair-snapshot` pushed at `80d3369`.
- [x] Machine-generate `git_facts.md` and environment drift snapshots.
- [x] Baseline v1 regression render check on Scene 1 Seed 11.
- [x] 1 Hz `nvidia-smi` background sampler and power management set to 0.
- [x] Visual canary check and reserved seeds {606, 707, 808} audited clean.
- [x] Merged via PR #20 (`bf05162`).

### Phase R1: Audit and Retract
- [x] Machine claims audit ledger `docs_generated/v2/repair/claims_ledger.csv`.
- [x] Problem register items A1-A13 and B1-B13 evaluated in `problem_register.md`.
- [x] Quarantine ungrounded docs into `docs_generated/v2/retracted/` with RETRACTED banner.
- [x] Under-repair banners added to READMEs; status comment posted to PR #19.
- [x] Merged via PR #21 (`a4cbac3`).

### Phase R2: Repair the Measuring Tools (Verifier v1.1)
- [x] SegFormer bridge fix, detector health monitoring, and control test suite.
- [x] Recalibrate genuine/impostor distributions with child/adult splits.
- [x] Calibrate wardrobe tolerance on labeled crops.
- [x] Freeze verifier v1.1 (`ee3b29d5...`) and rescore historical runs.
- [x] Merged via PR #22 (`ed0aac9`), tagged `verifier-v1.1-frozen`.

### Phase R3: Splits, Seal and Baselines
- [x] Define immutable splits in `consistency_v2/splits.py` with access guard logging.
- [x] Author holdout storyboard `storyboards/holdout_v2.json` and generate certified anchors.
- [x] Seal TEST-2 (`consistency_v2/test2_seal.json`), tag `test2-sealed` (`eed746f`).
- [x] Run baseline v1 on DEV-2 and TEST-2, scored with Verifier v1.1 (72 renders).
- [x] Merged via PR #23 (`eed746f`).

### Phase R4: Instrument the Pipeline
- [x] Add StageTracer instrumentation with SHA256 and pixel diff assertions.
- [x] Add `StageNoOpError` assertion for inactive stages.
- [x] Repair ablation runner with flag propagation and diff assertions.
- [x] Measure Forge diffusion non-determinism across identical seeds.
- [x] Merged via PR #24 (`ad3f8bd`).

### Phase R5: Diagnose the Real Pipeline
- [x] Execute v2 default on DEV-FAST with full stage tracing.
- [x] Uncover empirical baseline v1 vs v2 performance: v1 = 0.2278, v2 = 0.1685.
- [x] Isolate M5 face swap as primary identity driver (+0.0307 jump over Pass 2).
- [x] Diagnose legacy v2-rc1 artifact: M5 was inactive in rc1 due to `'scene_layout' in locals()`.
- [x] Document diagnosis in `docs_generated/v2/repair/diagnosis.md`.
- [x] Merged via PR #25 (`b291aa5`).

### Phase R6: Rebuild Identity Bank & M1 Screening
- [x] Multi-angle reference views: generate and gate 7 views/character with 4-point gate.
- [x] Bank views architecture: implemented `IdentityBankManager` multi-view selection (`single_anchor`, `centroid`, `nearest_pose`).
- [x] M1 strategy screening on DEV-FAST: 36 renders completed across 3 arms. `nearest_pose` won with score 0.1789 (+0.0045 over single_anchor baseline 0.1744). Logged as `loop-R6-iter1`.
- [x] M12 decoupling: M12 Collage-Init flags placed behind default OFF in `config.py` and `pipeline_v2.py`. Implementation separated to `feat/m12-collage-init`.
- [ ] PR and squash-merge into `consistency-v2`.

### Phases R7 - R11: Upcoming Repair Work Streams
- [ ] R7: Identity and Binding Repair Loops (W1 - W9)
- [ ] R8: Real Visual Review (Contact sheets, blind review, canary verification)
- [ ] R9: End-to-End Clean-Slate Run (Headless execution and UI walkthrough)
- [ ] R10: Final Sealed Evaluation on TEST-2 (Single run, cluster bootstrap CIs)
- [ ] R11: Rebuild Documentation from Artifacts & Tag `v2.0.0-rc2`

---

## 4. SUPERSEDED Checklist: Historical v2-rc1 (Retracted)
> [!CAUTION]
> **SUPERSEDED: Claims and numbers below are from the retracted v2-rc1 pass and were measured on the uncalibrated old verifier. See `docs_generated/v2/RETRACTIONS.md` for details.**

- [x] Phase 0: Setup, Repository Pre-Flight & Baseline (rc1, retracted)
- [x] Phase 1: Build & Calibrate Verification Suite (rc1, superseded by Verifier v1.1)
- [x] Phase 2: Diagnose & Fix Existing Pipeline (rc1, superseded by R4/R5 stage-trace diagnosis)
- [x] Phase 3: Implement Consistency Methods M1-M10 (rc1, undergoing real tuning in R6-R7)
- [x] Phase 4: Integration, UI v2 & Tuning (rc1)
- [x] Phase 5: Held-Out Evaluation on TEST Split (rc1, retracted due to unsealed split access and old verifier)
- [x] Phase 6: Documentation and PR #19 opened (rc1, PR #19 kept open as master tracking PR)
