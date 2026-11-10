# MMCCP Consistency v2: Master Progress & Execution Log

**Current Campaign:** REPAIR AND REAL-IMPROVEMENT PASS (R0 - R11 + Finalization Part A)  
**Current Phase:** Finalization Pass Part A (`v2.0.0-rc2` Audit, Pre-Registration Reconciliation & Honest Closeout)  
**Previous Phase:** Phase R10–R11 (`7904ad5`, tagged `v2.0.0-rc2`)  
**Active Integration Branch:** `consistency-v2` (`7904ad5`)  
**Active Working Branch:** `docs/finalize-honest`  
**Tracking Pull Request:** [PR #19](https://github.com/ShamantN/CAPSTONE/pull/19) (`consistency-v2` into `main` - MUST STAY OPEN)  
**Hardware & Power:** Intel i7-12650H, 32GB RAM, RTX 3070 Ti Laptop GPU (8GB), AC Standby/Hibernate restored to originals (`900s` / `10800s`).  
**Telemetry:** 1 Hz `nvidia-smi` telemetry archives in `results/telemetry/` (sampler stopped; clean closeout).  
**Verifier Freeze:** Frozen v1.2 canonical SHA `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6` (verified PASS).  
**Data Splits & Sealing:** TEST-2 SEALED (`consistency_v2/test2_seal.json`), RESERVED seeds {606, 707, 808} UNTOUCHED.

---

## 0. Canonical Truth Table under Frozen Verifier v1.2

Source file: `results/rescored_v1.2/rescored_summary.json` (Rescored on 2026-10-18)

| Result Set | Images | Faces | Bio Ensemble Mean (95% CI) | Rank-1 Re-ID | Spatial Pass | Wardrobe Pass | Id@Pos Pass | Mean Defects | Adult Bio | Teen Bio | Child Bio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_v1_dev_full** | 18 | 24 | **0.3097** [0.2855, 0.3369] | 79.2% | 100.0% | 25.0% | 0.0% | 2.94 | 0.3006 | 0.2809 | 0.3568 |
| **baseline_v1_dev2** | 30 | 40 | **0.2876** [0.2687, 0.3049] | 67.5% | 97.5% | 32.5% | 0.0% | 3.33 | 0.3078 | 0.2745 | 0.2604 |
| **v2_rc1_dev2** | 30 | 40 | **0.2125** [0.1717, 0.2556] | 22.5% | 70.0% | 22.5% | 2.5% | 3.23 | 0.1560 | 0.2906 | 0.2475 |
| **v2_rc1_generalization** (VOID) | 15 | 21 | **0.0000** [0.0000, 0.0000] | 0.0% | 57.1% | 47.6% | 0.0% | 1.93 | 0.0000 | 0.0000 | 0.0000 |
| **v2_rc1_ablation** (VOID) | 16 | 24 | **0.1103** [0.0599, 0.1607] | 0.0% | 41.7% | 12.5% | 0.0% | 1.94 | 0.0349 | 0.0000 | 0.1479 |
| **baseline_v1_test2_ebright** | 30 | 40 | **0.3091** [0.2940, 0.3242] | 65.0% | 100.0% | 45.0% | 0.0% | 3.07 | 0.3290 | 0.2778 | 0.3006 |
| **baseline_v1_test2_holdout** | 12 | 24 | **0.2758** [0.2220, 0.3272] | 62.5% | 79.2% | 41.7% | 0.0% | 1.92 | 0.2100 | 0.0000 | 0.3854 |

*Note: rc1 generalization and ablation sets are VOID (historical flawed runs). Baseline TEST-2 rescoring authorized under verifier change protocol with logged access.*

---

## 0.1 Step 4 M1 Strategy Resolution (Acceptance Rule Re-check)
- Empirical paired cluster-bootstrap (10,000 resamples clustered by DEV-FAST scenes 1, 2, 4, 5) evaluated on `results/m1_screening/m1_screening_summary.json`:
  - `nearest_pose` vs `single_anchor`: Delta Mean = **+0.00458**, 95% Cluster-Bootstrap CI = **[0.0000, 0.01375]**.
  - Delta fails the strict acceptance rule because the 95% CI lower bound does not rule out zero ($\le 0.0000$).
  - Per protocol: defaulted to the simplest non-inferior arm (`single_anchor`).
  - Executed `tools/update_best_config_m1.py` to set `"bank_selection_metric": "single_anchor"` in `consistency_v2/best_config.json`.

---

## 1. Machine-Verified Branch and PR Registry

| PR # | Branch | Base | Status | Merge Commit SHA | Merged At | Purpose |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **#19** | `consistency-v2` | `main` | **OPEN** | - | - | Master tracking PR into main (Must remain OPEN) |
| **#31** | `fix/openpose-payload-fix` | `consistency-v2` | **MERGED** | `acf1370` | 2026-10-21 | OpenPose payload keys (image/input_image), Forge options API, memory bounding |
| **-** | `feat/r7-dev2-ablation` | `consistency-v2` | **COMMITTED** | `b13142e` | 2026-11-02 | R7 repair loops, locked DEV-2 config, 7-arm LOO DEV-2 ablation |
| **-** | `test/visual-review-v2` | `consistency-v2` | **MERGED** | `ee4eb69` | 2026-11-02 | R8 visual review assets, automated blind study & VLM second opinion (superseded/remediated in `7873074`) |
| **-** | `test/e2e-clean-slate` | `consistency-v2` | **MERGED** | `1d00ac4` | 2026-11-02 | R9 clean-slate zero-cache runs & Streamlit UI verification |
| **-** | `consistency-v2` | `consistency-v2` | **COMMITTED** | `96ff419` | 2026-11-05 | R7-R9 forensic remediation (`7873074`), bank parity (`68653e6`), wardrobe priority fix (`96ff419`) |
| **-** | `consistency-v2` (`v2.0.0-rc2`) | `consistency-v2` | **COMMITTED** | `7904ad5` | 2026-11-06 | R10 sealed TEST-2 evaluation (30 Ebright + 12 Holdout) & R11 docs rebuild |
| **#30** | `fix/identity-layout-scale` | `consistency-v2` | MERGED | `72769ef` | 2026-10-18 07:49 UTC | Restore layout planner to baseline and document loop-W3-iter1-face-scale rejection |
| **#29** | `chore/update-progress` | `consistency-v2` | MERGED | `3e07680` | 2026-10-18 07:25 UTC | Record PR #28 in master registry and update progress |
| **#28** | `fix/verifier-v1.2` | `consistency-v2` | MERGED | `6c4573b` | 2026-10-18 07:24 UTC | Verifier v1.2 upgrade, controls, persistent worker, historical rescore, M1 default |
| **#27** | `feat/m12-collage-init` | `consistency-v2` | MERGED | `87c3589` | 2026-10-16 06:55 UTC | M12 collage-init module and unit tests behind default-OFF flag |
| **#26** | `feat/bank-views` | `consistency-v2` | MERGED | `5258633` | 2026-10-16 06:52 UTC | Rebuild identity banks, screen M1 strategies, harden split logging |
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
     - WebUI Forge restarted with `--api --xformers --cuda-malloc` on PID 10376.
     - Split access log isolated; verifier freeze check verified passing; 55/55 tests passing.
3. **Pause Gap 3 (Server Restart & Weekend Pause):**
   - **Interval:** 2026-10-16 ~12:53 IST to 2026-10-18 ~10:50 IST (~45h 57m).
   - **Resolution:**
     - Background jobs and render task 6572 were cancelled by server restart. 7/12 renders produced for `loop-W3-iter1-face-scale` were recorded as cut off / incomplete (no summary JSON produced).
     - WebUI Forge restarted with `--api --xformers --cuda-malloc`; verified 20 scripts, Forge Couple, and 10 ControlNet models (InstantID units active).
     - New 1 Hz VRAM sampler actively logging to `results/telemetry/vram_20261018_105634.csv`.
     - Windows AC standby and hibernate verified 0s (`never`).
     - Split access log verified (8 entries: 2 baseline v1 runs, 6 unit test runs; 0 unauthorized accesses). `ALLOW_TEST2` and `ALLOW_RESERVED` confirmed unset.
     - Verifier freeze v1.1 verified passing (`ee3b29d5...`); 55/55 non-GPU unit tests pass.

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
- [x] M12 decoupling: M12 Collage-Init flags placed behind default OFF in `config.py` and `pipeline_v2.py`. Implementation merged via PR #27.
- [x] Merged via PR #26 (`5258633`) and PR #27 (`87c3589`).

### Phase R7: Identity and Binding Repair Loops (COMPLETED)
- [x] Verifier Metric Audit (Gate before further tuning): Audit binding saturation (0%), multi-face assignment (Scene 6 graduation tiny faces), prop defect saturation, and 3-way age group disaggregation.
- [x] Verifier v1.2 Formal Upgrade: Calibrated thresholds, spatial prominence hero matching, small-crop ArcFace guard, CLIP prop calibration, persistent IPC vision worker, 15 control tests passing, frozen at SHA `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6`.
- [x] Canonical Historical Rescore: All 7 baseline and rc1 sets rescored with Verifier v1.2, logged to `results/rescored_v1.2/` and `results/split_access_log.jsonl`.
- [x] M1 Strategy Selection Re-check: Cluster-bootstrap CI [0.0000, 0.01375] fails acceptance rule; defaulted to `single_anchor` in `best_config.json` via script.
- [x] `loop-W3-iter1-face-scale`: REJECTED (Delta = -0.0092, 95% CI [-0.0171, 0.0000], positive pair rate = 0.0%). Contact sheets revealed oversized boxes caused two-shot character fusion into a single chimeric body. Baseline restored in PR #30.
- [x] `loop-W6-iter1-collage-init`: REJECTED (Delta = -0.0055, 95% CI [-0.0209, +0.0100], positive pair rate = 22.2%). Contact sheets revealed collage heads floating in empty space because Pass 1 draft collapsed into a single central person. Diagnosed root cause: OpenPose unit used key `input_image` instead of `image` and omitted hash in model string `openpose_xl`, causing Forge to silently ignore pose conditioning.
- [x] W4 (`acf1370`, PR #31): OpenPose payload keys fixed (`image` + `input_image`), Forge options API & memory bounding; post-`#31` W1–W4 repair loops executed (`3c955df`..`2adf480`) and locked into `consistency_v2/best_config.json`.
- [x] Bank Parity, Wardrobe Priority Fix & DEV-2 Re-evaluation: Remediated in `7873074`, `68653e6`, and `96ff419` (30-render DEV-2 bundle & 7-arm LOO DEV-2 ablation completed with synced banks).
- [x] R8: Automated Visual Review (Contact sheets, automated blind study, planted canaries, local VLM second opinion — **automated, not human**; remediated in `7873074`; human blind test prepared in `HUMAN_BLIND_TEST.md`).
- [x] R9: End-to-End Clean-Slate Run (Headless execution & Streamlit UI walkthrough completed in `1d00ac4` / `96ff419`).
- [x] R10: Final Sealed Evaluation on TEST-2: 30 Ebright + 12 Holdout renders + 7-arm TEST-2 LOO subset completed strictly one-shot (`7904ad5`). Canonical reconciliation of original vs post-hoc P1–P4 criteria documented in `docs_generated/v2/PREREG_RECONCILIATION.md`.
- [x] R11: Rebuild Documentation from Artifacts: `results/numbers.json` gathered, documentation rendered via `tools/render_docs.py`, doc-lint tests verified passing, tagged `v2.0.0-rc2` (`7904ad5`), and undergoing honest closeout reconciliation on `docs/finalize-honest`.

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
