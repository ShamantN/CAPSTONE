# Multi-Modal Character Consistency Pipeline (MMCCP) v2.0

**PES University Capstone Project | Project ID:** `PW26_CGM_02`  
**Reconciled Closeout (`v2.0.0-rc3` Candidate) — Honest Artifact-Backed State**

> [!NOTE]
> Documentation reconciled from `v2.0.0-rc2`: earlier `rc2` summaries omitted pre-registration shifts, mixed defect units (`per-scene` vs `per-face-instance`), overstated Holdout performance, and omitted whole-system VRAM excursions above `7.4 GB`. All metrics below are generated strictly from [`results/numbers.json`](results/numbers.json).

### Branch, Tag, Reproduction & Non-Repo Assets
- **Recommended Branch / Tag:** Use `main` (post-merge), `finalize/v2.0.0` (closeout PR #37), or tag `v2.0.0-rc3` (`consistency-v2` release candidate snapshot).
- **Run From Scratch (New Machine, Linux & Google Colab):** Start at [docs/run/00_START_HERE.md](docs/run/00_START_HERE.md), run [tools/verify_setup.py](tools/verify_setup.py), or open the Colab notebook at [notebooks/MMCCP_Colab.ipynb](notebooks/MMCCP_Colab.ipynb).
- **Reproduce Verification, Statistics & Docs (CPU / No GPU Required):**
  ```powershell
  my_app_env\Scripts\python.exe tools\check_freeze.py
  my_app_env\Scripts\python.exe tools\audit_test2_isolation.py
  my_app_env\Scripts\python.exe tools\compute_test2_statistics.py
  my_app_env\Scripts\python.exe tools\collect_numbers.py
  my_app_env\Scripts\python.exe tools\render_docs.py
  my_app_env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
  my_app_env\Scripts\python.exe -m unittest -v tests.verifier_controls.test_controls
  ```
- **Reproduce Canonical Banks & DEV Renders (Local Forge WebUI + GPU Required):**
  ```powershell
  my_app_env\Scripts\python.exe tools\sync_canonical_banks.py
  my_app_env\Scripts\python.exe tools\render_dev2_bundle.py
  my_app_env\Scripts\python.exe tools\score_ablation_dev2.py
  ```
  *(Sealed `TEST-2` is burned after one-shot evaluation: never re-run with `ALLOW_TEST2`, and keep quarantined seeds `{606, 707, 808}` untouched.)*
- **What Is NOT in the Repository & Where It Lives:**
  - **Model Weights (`*.safetensors`, `*.onnx`, `*.pt`, `*.bin`):** SDXL base, ControlNet OpenPose XL, IP-Adapter Plus Face, InsightFace `inswapper_128`, SegFormer, DeepFace, and YOLO weights live locally in `C:\Stable-Diffusion\` and workstation HuggingFace / InsightFace caches (gitignored).
  - **Active Character Banks & Embedding Caches:** `consistency_v2/banks/` and `hardcoded_cache.json` are gitignored runtime artifacts regenerated via `tools/sync_canonical_banks.py` from ground-truth anchors; workstation archives live outside git in `D:\5TH SEM\HACKATHON_RC2_BACKUP_FOR_LIVE_DEMO\` and `D:\5TH SEM\_demo_state_archive\`.
  - **Generated Renders & Visual Review PNGs:** `results/test2_ebright/renders/`, `results/test2_holdout/renders/`, `results/bundle_dev2/renders/`, `results/ablation_dev2/`, `results/ablation_test2/`, `docs_generated/v2/visual_review/contact_sheets/`, `human_blind_ab/*.png`, and `human_blind_reid_crops/*.png` are excluded via `.gitignore`. All quantitative verification JSON/CSV artifacts (`results/numbers.json`, `results/**/report*.json`, `results/**/*.csv`, `docs_generated/v2/*.md`) ARE tracked in git; full render archives should be shared via a GitHub Release asset or Google Drive archive rather than committed.

### Verifier Freeze & Protocol Constants
- **Frozen Verifier:** `v1.2` (`SHA: 3fa69071f888643b...`)
- **Bootstrap:** `10000` iterations (fixed seed `42`), reporting **both** image-level and scene-cluster 95% CIs.
- **Quarantined Seeds:** `{606, 707, 808}` remained strictly untouched.

---

## 1. Sealed TEST-2 Summary (v2 vs Baseline v1)

Evaluated one-shot on **TEST-2** across `42` renders (`30` Ebright / `40` faces; `12` Holdout / `24` faces; `64` total face slots):

| Split / System | Renders | Face Slots | Bio Mean | Rank-1 (`n/N`) | Spatial (`n/N`) | Wardrobe (`n/N`) | Defects/Scene (Primary) | Defects/Instance-Wt (Secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ebright TEST-2 (v2)** | 30 | 40 | **0.3147** | **80.0%** (32/40) | **97.5%** (39/40) | **80.0%** (32/40) | **1.50** | **1.525** |
| *Ebright TEST-2 (v1 Baseline)* | 30 | 40 | *0.3091* | *65.0%* (26/40) | *100.0%* (40/40) | *45.0%* (18/40) | *3.07* | *2.575* |
| **Ebright Paired Delta (`v2 - v1`)** | - | - | **+0.0056** | **+15.0%** | **-2.5%** | **+35.0%** | **-1.57** | **-1.05** |
| **Holdout TEST-2 (v2)** | 12 | 24 | **0.2596** | **54.2%** (13/24) | **87.5%** (21/24) | **37.5%** (9/24) | **1.25** | **1.417** |
| *Holdout TEST-2 (v1 Baseline)* | 12 | 24 | *0.2758* | *62.5%* (15/24) | *79.2%* (19/24) | *41.7%* (10/24) | *1.92* | *1.792* |
| **Holdout Paired Delta (`v2 - v1`)** | - | - | **-0.0162** | **-8.3%** | **+8.3%** | **-4.2%** | **-0.67** | **-0.375** |
| **Combined TEST-2 (v2 vs v1)** | 42 | 64 | **0.2940** vs *0.2966* | **70.31%** vs *64.06%* | **93.75%** vs *92.19%* | **64.06%** vs *43.75%* | **1.43** vs *2.74* | **1.484** vs *2.281* |

### Paired Inference & Key Limitations
- **Ebright:** Identity shows parity with Baseline v1 (`Delta Bio = +0.0056`, scene-cluster 95% CI `[-0.0149, +0.0317]`, image 95% CI `[-0.0124, +0.0244]`, Wilcoxon `W=343.0`, `p=0.3754`); **Rank-1** improved (`+15.0%`, image 95% CI `[+2.4%, +28.6%]` excludes 0, whereas 6-cluster CI `[+0.0%, +33.3%]` includes 0, `p=0.0578`); **Wardrobe** improved (`+35.0%`, cluster CI `[+20.0%, +48.6%]` and image CI `[+22.5%, +47.6%]` both exclude 0, `W=0.0`, `p=0.000183`); **Defects per scene** dropped `51.1%` (`3.07 -> 1.50`, `Delta = -1.57`, cluster CI `[-3.20, -0.13]` and image CI `[-2.37, -0.77]` both exclude 0, `W=39.5`, `p=0.001453`).
- **Holdout:** MMCCP v2 was **not better than Baseline v1** (`Bio Delta = -0.0162`, `Rank-1 Delta = -8.3%`, `Wardrobe Delta = -4.2%`, `Defects/Scene Delta = -0.67`, all CIs include 0). Spatial pass (`87.5%`, `21/24`) fell below `90.0%` because Scene 4 (`3` expected characters) rendered only `2` characters on all `3` seeds.
- **Unmeasurable Combined Binding Gate:** `identity_at_position_pass_rate` and `combined_binding_pass_rate` are **`0.0%`** (`0/40` Ebright, `0/24` Holdout) for **both** Baseline v1 and MMCCP v2 due to strict multi-judge ensemble calibration, making combined binding unmeasurable under Verifier v1.2.
- **Partial Scoring Circularity:** Verifier anchor references are identical to M4/M5 source anchors; InsightFace similarity reflects swap alignment more strongly than independent judges (Section 6 of [`07_measured_results.md`](docs_generated/v2/07_measured_results.md)).

---

## 2. Original Pre-Registered Scorecard (`P1`–`P4`)

Full forensic comparison in [`PREREG_RECONCILIATION.md`](docs_generated/v2/PREREG_RECONCILIATION.md):
- **Overall Original Verdict:** **`0/4` strict all-split original rules satisfied** (`0/4` `MET` across all required TEST-2 splits; `P2` and `P3` pass on Ebright & Combined only while failing Holdout).
1. **`P1 (Original): Delta Re-ID (Biometric Ensemble & Rank-1) > 0 with 95% CI lower bound > 0` — [`NOT MET`]** (`Ebright Biometric Ensemble Delta = +0.0056 (cluster 95% CI [-0.0149, 0.0317], image 95% CI [-0.0124, 0.0244], Wilcoxon p=0.3754); Holdout Delta = -0.0162 (cluster CI [-0.092, 0.0661]); Combined Delta = -0.0026 (cluster CI [-0.039, 0.033]). Lower bound <= 0 across splits.`)
2. **`P2 (Original): Spatial assignment pass rate >= 90.0%` — [`NOT MET`]** (`Ebright Spatial Pass = 97.5% (39/40, >=90% MET) and Combined = 93.75% (60/64, >=90%), but Holdout Spatial Pass = 87.5% (21/24, <90% NOT MET due to missing 3rd character on all 3 Scene 4 renders).`)
3. **`P3 (Original): Wardrobe Delta-E pass rate >= 60.0%` — [`NOT MET`]** (`Ebright Wardrobe Pass = 80.0% (32/40, >=60% MET) and Combined = 64.06% (41/64, >=60%), but Holdout Wardrobe Pass = 37.5% (9/24, <60% NOT MET vs v1 41.7%).`)
4. **`P4 (Original): Engineering (Peak VRAM <= 7.4 GB, 0 OOMs, 0 extra YOLO persons)` — [`NOT MET`]** (`Whole-system Peak VRAM = 7984 MiB (7.797 GiB > 7400 MB / 7577.6 MiB target; 52s > 7400 MiB; 29s > 7577.6 MiB), while process-attributable delta (max - 973 MiB idle) = 7011 MiB (6.847 GiB), 0 OOMs, and extra_person_detector.py (conf=0.40) flagged 1/42 extra YOLO persons (v2 Ebright scene_5_seed_1202.png; Verifier YOLO also flagged scene_2_seed_1303.png).`)

---

## 3. Leave-One-Out Ablation (`6` Scenes, `8` Faces, TEST-2 Seed `1101` — Post-Hoc Exploratory Ablation After Unsealing)

> [!WARNING]
> **Provenance (`TEST-2` Ablation):** Post-hoc exploratory ablation on `TEST-2` after unsealing (`tools/run_sealed_test2.py`, PID `1608`, `2026-11-06T05:45:11Z`, scored `13:38:16 +0530` prior to inspection); **not used to tune `v2.0`** (`consistency_v2/best_config.json` unchanged since `2026-11-03`).  
> **Noise Band & Balanced M4 Readout:** Deltas smaller than `~0.03` on a single seed (`n=8` faces) lie within single-seed diffusion noise. While `no_m4_ip_adapter` edged out `full_v2` on 4-judge biometric ensemble on `DEV-2` (`0.3423` vs `0.3323`) and `TEST-2` (`0.3373` vs `0.3269`), removing M4 sharply dropped `Facenet512` (`0.7087 -> 0.6218` DEV-2; `0.7000 -> 0.6060` TEST-2) and `DINOv2` (`0.5878 -> 0.4935` DEV-2; `0.6069 -> 0.4943` TEST-2) and increased DEV-2 defects (`1.17 -> 1.50`), making M4 a **candidate for a multi-seed re-test in `v2.1`** rather than an unconditional disable.

| Arm | Bio Mean | Rank-1 | Spatial | Wardrobe | Defects/Scene | `Delta Bio vs Full` | Mean Pixel Diff vs Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_v2` | 0.3269 | 87.5% | 100.0% | 87.5% | 1.50 | +0.0000 | 0.00 px |
| `no_m2_layout` | 0.3167 | 50.0% | 100.0% | 62.5% | 1.83 | -0.0102 | 66.48 px |
| `no_m3_couple` | 0.3064 | 87.5% | 100.0% | 62.5% | 1.50 | -0.0205 | 19.85 px |
| `no_m4_ip_adapter` | 0.3373 | 87.5% | 100.0% | 87.5% | 1.50 | +0.0104 | 65.33 px |
| `no_m5_swap` | 0.2954 | 62.5% | 100.0% | 87.5% | 1.50 | -0.0315 | 0.79 px |
| `no_m6_detailer` | 0.3181 | 87.5% | 100.0% | 87.5% | 1.33 | -0.0088 | 0.44 px |
| `no_m7_wardrobe` | 0.3227 | 87.5% | 100.0% | 62.5% | 1.33 | -0.0042 | 2.59 px |

---

## 4. Telemetry, Extra-Person & Visual Review Status
- **Sealed TEST-2 VRAM (`8112` 1 Hz samples):** Whole-system peak = **`7984` MiB** (`7.797` GiB / `7.984` GB; exceeds `7.4 GB` ceiling: `52`s > `7400` MiB, `29`s > `7577.6` MiB across `36` episodes). Idle-desktop pre-render baseline = `973` MiB (`32` samples), giving `7011` MiB (`6.847` GiB) process-attributable delta (`0` OOMs).
- **Combined 9 Telemetry CSVs (`9` files, `157357` samples):** Overall peak `8005` MiB, `p99 = 7442.0` MiB, `1838`s > `7400` MiB.
- **Extra YOLO Person Detector (`conf=0.40`):** Flagged `1/42` v2 TEST-2 renders (`scene_5_seed_1202.png`: `3` vs `2` expected) vs `2/42` in v1 (`scene_1_seed_1202.png` + `scene_2_mentorship_seed_1202.png`).
- **Visual Review:** Prior DEV-2 blind A/B (`10/12` = `83.33%`), Re-ID (`20/20`), canary (`3/3`), and VLM checks were **automated, not human**. Human TEST-2 Ebright blind evaluation protocol is staged in [`HUMAN_BLIND_TEST.md`](docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md) (status: `PENDING_HUMAN_INPUT`).
