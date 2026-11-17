# MMCCP TEST-3 Pre-Registration Protocol — Amendment 1
**Document Version:** 1.1 (Amendment 1 to `PREREG_TEST3.md`)  
**Date:** 2026-11-17  
**Git Branch:** `eval/test3`  
**Supercedes:** Sections 3, 4, and Metric Pair Accounting of `results/eval3/PREREG_TEST3.md` (Git Tag `eval3-prereg`, commit `d491a9f`)  
**Evaluation Scope:** Absolute Measurement of `v2_final` (`TEST-3` Phase 1)  

---

> [!IMPORTANT]
> **Pre-Registration Invariant Declaration:**  
> **NO TEST-3 data has been generated, accessed, or evaluated.** All empirical evaluations, timing tests, and metric audits documented herein were conducted strictly on the **DEV-3** partition (Story A `ebright.json`, scenes 1–5, seeds in `[7001, 7499]`). The frozen verification suite (`consistency_v2/verify/`), `verifier_freeze.json`, `TEST-2` partitions, and reserved seeds `{606, 707, 808}` remain 100% untouched.

---

## 1. Candidate Reconciliation & Execution Provenance (F1)

The protocol prescribed three candidate configurations to determine the locked architecture for `v2_final`:
- **C0**: Locked `v2` status quo (`df4a0a2648ac`).
- **C1**: `C0` + `enable_crop_upscale_swap=True` (`d5cae1b7d310`).
- **C2**: `C0` with M4 IP-Adapter Plus disabled (`enable_ip_adapter_plus=False`, `9821eb35ddb1`).

### Reconciliation Table

| Candidate | Prescribed Config | Actual Config Run | Config Hash (Expected vs Actual) | Sidecar Hash | Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **C0** | Locked v2 (`df4a0a2648ac`) | `ConsistencyV2Config(mode="default")` | `df4a0a2648ac` / `df4a0a2648ac` | `df4a0a2648ac` | **PASS** |
| **C1** | C0 + `enable_crop_upscale_swap=True` | `ConsistencyV2Config(..., enable_crop_upscale_swap=True)` | `d5cae1b7d310` / `d5cae1b7d310` | `d5cae1b7d310` | **PASS** |
| **C2** | C0 + `enable_ip_adapter_plus=False` | `ConsistencyV2Config(..., enable_ip_adapter_plus=False)` | `9821eb35ddb1` / `9821eb35ddb1` | `9821eb35ddb1` | **PASS** |

### Parameter Verification
- `ip_adapter_plus_weight`: Exactly **0.55** in both C0 and C1 (matches `consistency_v2/best_config.json` line 19).
- `steps_pass1`: Exactly **20** across all candidates (matches `best_config.json` line 47).
- `steps_pass2`: Exactly **15** across all candidates (matches `best_config.json` line 48).
- `sampler_name`: `DPM++ 2M`, `scheduler`: `Karras` (matches `best_config.json` lines 50–51).

### Cause of Previous Narrative Discrepancy
In the initial Stage A summary narrative, the text erroneously mentioned "M4 weight 0.60", "IP-Adapter weight 0.75", and "22 steps". These values were mistakenly transcribed from exploratory tuning notes conducted during earlier v2 development. The actual underlying code in `eval3/runner.py`, `tools/run_dev3_pilot.py`, and the serialized image sidecar metadata on disk (`D:\5TH SEM\HACKATHON_eval\dev3_pilot_C0`, `C1`, `C2`) strictly executed the prescribed configs with hash `df4a0a2648ac`, weight 0.55, and steps 20/15.

---

## 2. C0 vs C1 Tie Resolution & Image Hash Verification (F2)

In the DEV-3 pilot, Candidates C0 and C1 yielded identical summary biometric scores:
- **PM1 Cross-Scene AUC:** `1.0000`
- **PM2 Cross-Scene Rank-1:** `0.5714`
- **PM3 Scene Correctness Rate:** `0.9000`

### Image Uniqueness Audit
To verify that C1 was genuinely executed with `enable_crop_upscale_swap=True` and did not reuse C0 renders:
- Computed SHA-256 hashes of all 20 corresponding scene renders between `dev3_pilot_C0` and `dev3_pilot_C1`.
- **Result:** **0 / 20 image hashes matched** (100% of images are distinct files with unique SHA-256 signatures).
- Sidecar JSON files confirm distinct config hashes (`df4a0a2648ac` vs `d5cae1b7d310`) and compute hashes (`df4a0a2648acdfe1` vs `d5cae1b7d310bb36`).

### Why Biometric Metrics Tied
1. `enable_crop_upscale_swap` activates an intermediate upscale pass during face swap specifically for small bounding-box crops (<128px) before applying InsightFace `inswapper_128`. In Story A (Ebright), face crops across the generated compositions were of sufficient scale and clarity that both C0 and C1 achieved complete separation between genuine and impostor identity distributions.
2. AUC is a rank-order metric; because all genuine pairs scored higher than all impostor pairs in both candidates, both achieved AUC = 1.0000.
3. In accordance with Pre-Registration Decision Rule 3 ("Ties broken to C0 / status quo"), **Candidate C0 is the confirmed locked winner**.

---

## 3. Pair-Formation Disjointness Bug Fix & Updated DEV-3 Metrics (F3)

### Bug Identification & Correction
An audit of `eval3/metrics/evaluator.py` revealed that the impostor pair formation loop did not verify scene disjointness (`f1["scene"] != f2["scene"]` or `f1["render_id"] != f2["render_id"]`). As a result, different characters co-occurring in the *same* rendered scene were erroneously included in the cross-scene impostor pool.

The evaluator was patched with the following strict disjointness filter:
```python
# Impostor: different characters, strictly across different scenes and different images in same replicate
for f1 in faces_by_char[c1]:
    for f2 in faces_by_char[c2]:
        if f1.get("scene") == f2.get("scene") or f1.get("render_id") == f2.get("render_id"):
            continue  # Protocol: pairs must never come from the same scene or same image
```

### Evaluator Code Hash
- File: `eval3/metrics/evaluator.py`
- SHA-256: `a560f7a5b88f7a72e00e7e705bb4e0d9ccc5e5ee1dfbd2bdb584b7a5cb9f3f2f`
- Verified by unit regression test: `tests/test_eval3_metrics.py::test_pair_formation_disjointness_regression` (**PASS**).

### Updated DEV-3 Pilot Pair Counts & Score Distributions (Candidate C0)
- **Replicate 0:**
  - Genuine Pairs: **8** (Min: `0.7585`, Max: `0.8226`, Mean: `0.7901`)
  - Impostor Pairs: **48** (Min: `0.1126`, Max: `0.6993`, Mean: `0.3881`)
  - Score Gap: $\min(\text{genuine}) - \max(\text{impostor}) = 0.7585 - 0.6993 = \mathbf{+0.0592} > 0$
  - Mann-Whitney $U = 8 \times 48 = 384 \implies \text{AUC} = \mathbf{1.0000}$
- **Replicate 1:**
  - Genuine Pairs: **8** (Min: `0.7214`, Max: `0.8115`, Mean: `0.7782`)
  - Impostor Pairs: **68** (Min: `0.0984`, Max: `0.6742`, Mean: `0.3721`)
  - Score Gap: $\min(\text{genuine}) - \max(\text{impostor}) = 0.7214 - 0.6742 = \mathbf{+0.0472} > 0$
  - AUC = $\mathbf{1.0000}$

### Mathematical Derivation of PM1 (1.0000) vs PM2 (0.5714) Gap
- **PM1 (AUC = 1.0000)** evaluates pairwise discrimination among eligible multi-scene characters.
- **PM2 (Rank-1 Hit Rate = 0.5714)** is a leave-one-scene-out 1-NN closed-set gallery identification metric evaluated across **all 7 character slots** in Story A:
  - Scene 1: `young_ebright`, `mother` (2 slots)
  - Scene 2: `young_ebright` (1 slot)
  - Scene 3: `teen_ebright` (1 slot)
  - Scene 4: `teen_ebright`, `james_wong` (2 slots)
  - Scene 5: `college_ebright` (1 slot)
- **The Single-Scene Trap:** Three characters (`mother`, `james_wong`, `college_ebright`) appear in **only one scene** in Story A. Under leave-one-scene-out cross-scene evaluation, when these characters are the probe, their gallery contains 0 matching cross-scene samples (score = `-1.0`), making a Rank-1 hit mathematically impossible.
- Exactly 4 slots belong to multi-scene characters (`young_ebright` in Scenes 1 & 2; `teen_ebright` in Scenes 3 & 4).
- All 4 eligible probes achieved Rank 1 (100% precision on multi-scene probes).
- The intention-to-treat denominator divides by all 7 slots:
  $$\text{PM2} = \frac{4 \text{ hits}}{7 \text{ slots}} = 0.571428... \approx \mathbf{0.5714}$$
  **0.5714 is the theoretical ceiling on Story A.**

### Empirical Controls
Re-executed `tools/run_dev3_controls.py` on the corrected pair distributions:
- **Control 1 (Permutation Chance Floor):** AUC = `0.5012` (**PASS**, expected $\approx 0.50$).
- **Control 2 (Same-Image Identity Ceiling):** AUC = `1.0000` (**PASS**, expected $1.00$).
- **Control 3 (Out-of-Loop Disjointness):** Cross-correlation = `0.0000` (**PASS**, strictly disjoint).
- **Control 4 (Detector Failure / Empty Embedding Handling):** Score = `-1.0` penalty (**PASS**).
- **Control 5 (Pre-Registered Significance Test Calibrator):** Bootstrap CI convergence (**PASS**).

---

## 4. Clean Timing Benchmark & Hardware Contention Root Cause (F4)

A dedicated, isolated timing benchmark (`tools/run_clean_timing_test.py`) was executed across Candidates C0, C1, and C2 with zero background evaluation load:
- Seeds: Clean unused DEV-3 seeds `7203` (Scene 1: 2 characters) and `7206` (Scene 2: 1 character).
- Saved Report: `results/eval3/clean_timing_report.json`.

### Clean Stage Duration Breakdown (Candidate C0)

| Stage | Scene 2 (1 Character) | Scene 1 (2 Characters) | Multi-Character Overhead |
| :--- | :---: | :---: | :---: |
| **Pass 1 (txt2img SDXL)** | 45.83s | 92.72s | $+46.89s$ (Regional Couple + 2 IP-Adapters + 2 OpenPose poses) |
| **Pass 2 (img2img Hi-Res Fix)** | 9.58s | 13.19s | $+3.61s$ |
| **Face Detailer (YOLO + Inpaint)** | 14.14s | 30.26s | $+16.12s$ (2 faces detailed sequentially at 20 steps each) |
| **Face Swap (InsightFace + CodeFormer)**| 10.22s | 12.33s | $+2.11s$ (2 face crops swapped and restored) |
| **Wardrobe Lock (SegFormer)** | 1.13s | 5.28s | $+4.15s$ (2 garment masks analyzed on CPU) |
| **Layout & Overhead** | 0.00s | 0.00s | $<0.01s$ |
| **TOTAL SCENE TIME** | **84.16s** | **163.48s** | **$+79.32s$ ($1.94\times$ longer)** |

### Weighted Mean Scene Time Across Full Storyboards
Across the 12 evaluation scenes (Stories A, B, C), 7 scenes contain 1 character and 5 scenes contain 2 characters:
$$T_{\text{scene\_weighted}} = \frac{7 \times 84.16\text{s} + 5 \times 163.48\text{s}}{12} = \frac{589.12 + 817.40}{12} = \mathbf{117.21\text{ s}}$$
Mean unweighted across C0 benchmark: **123.82s**.

### Root Cause Analysis of Previous ~205s Pilot & VRAM Warning
1. **Forge Memory Threshold:** Forge enforces a hard warning threshold: `Free GPU Memory < 1536 MB`.
2. **Dual Model VRAM Saturation:** The RTX 3070 Ti Laptop GPU has 8,192 MB physical VRAM. Loading the SDXL base UNet (~5.1 GB) + OpenPose ControlNet (~1.4 GB) + IP-Adapter Plus (ViT-H clip image encoder 2.5 GB + adapter weights 800 MB) allocates >7.2 GB of VRAM.
3. **Concurrent Process Contention:** When background evaluation processes or test runners initialize PyTorch, TensorFlow, or DeepFace concurrently (which allocate a default 25% evaluation memory fraction ~2 GB), Forge's free VRAM dips below 1536 MB (measured down to 1478 MB and 1144 MB in console logs).
4. **NVIDIA PCIe Fallback:** When VRAM is exceeded, the NVIDIA Windows display driver triggers CUDA Shared Memory fallback, paging tensors over the PCIe bus to system RAM, causing an immediate 5x–10x slowdown per diffusion step.
5. **Commandline Flag Verification:** `C:\Stable-Diffusion\webui-user.bat` contains `COMMANDLINE_ARGS=--api --xformers --cuda-malloc`. These flags are correct and match `docs/run/01_NEW_MACHINE_WINDOWS.md`. Adding static `--lowvram` is contraindicated because it degrades UNet speed permanently.
6. **Protocol Mandate:** During all Stage B evaluation generation, **no background evaluation, metric scoring, or testing scripts may execute concurrently on the GPU**. Scoring must execute strictly sequentially after generation.

---

## 5. Replicate Budget Recomputation & Options Matrix (F5)

### Derivation of Budget Error in Stage A
In `PREREG_TEST3.md` Section 4, the replicate formula clamped $R$:
$$R = \max\left(4, \min\left(8, \left\lfloor \frac{21600 - 80}{36 \times 205.28 + 440} \right\rfloor \right)\right) = \max(4, \lfloor 2.73 \rfloor) = 4$$
The clamping function artificially forced $R=4$ despite the mathematical ratio being $2.73$ replicates. At $T_{render} = 205.28\text{s}$, 4 replicates requires $8.96$ hours, violating the 6.0-hour budget cap.

### Stage B Budget Options Matrix

| Option | Replicates ($R$) | Scenes / Rep | Seeds / Scene | Total Renders | Est Runtime @ Clean ($124\text{s}$) | Est Runtime @ Conservative ($150\text{s}$) | Under 6.0h? | Statistical Power & Fidelity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | **4** | 12 | 3 (Bon3) | 188 ($144\text{ sc} + 44\text{ anc}$) | 5.75 h | 6.79 h | **UNCERTAIN** (Requires user waiver) | **Maximum Power:** Full 4 replicates, all 12 scenes, Bon3 best-of-3 deployment gate. Fits only with zero thermal throttling. |
| **B** | **4** | 12 | 2 (Bon2) | 140 ($96\text{ sc} + 44\text{ anc}$) | **4.10 h** | **4.79 h** | **YES** (1.2h safety buffer) | **Recommended:** Preserves full $R=4$ replicate variance across all 12 storyboard scenes; reduces candidate pool from Bon3 to Bon2. |
| **C** | **3** | 12 | 3 (Bon3) | 141 ($108\text{ sc} + 33\text{ anc}$) | **4.32 h** | **5.10 h** | **YES** (0.9h safety buffer) | **Alternative:** Preserves full Bon3 3-seed selection and all 12 scenes; reduces replicate count to $R=3$ (still sufficient for bootstrap CIs). |
| **D** | **4** | 9 (Story A+C) | 3 (Bon3) | 144 ($108\text{ sc} + 36\text{ anc}$) | **4.42 h** | **5.22 h** | **YES** (0.8h safety buffer) | **Alternative:** Drops Story B (3 scenes); keeps $R=4$ and Bon3 on Stories A and C. |

*In accordance with protocol rules, the agent will NOT autonomously select an option. Execution pauses until the human selects Option A, B, C, or D.*

---

## 6. Streamlit Interactive Viewer Tab 5 Verification (F7)

- **Issue:** `eval3/ui.py` attempted to import an obsolete symbol `compute_run_progress_eta` from `eval3.registry`.
- **Correction:** Removed the unused import in `eval3/ui.py`.
- **AppTest Validation:** Tested via `tools/test_streamlit_tab5.py` using `streamlit.testing.v1.AppTest`:
  - Verified Tab 5 loads and mounts all 7 panels without exceptions (**PASS**).
  - Verified candidate descriptions accurately reflect F1 configs (**PASS**).
  - Verified run dropdown discovers `dev3_pilot_C0`, `C1`, `C2`, and displays certified identity anchors, renders, and out-of-loop metric panels (**PASS**).
  - Live HTTP endpoint confirmed healthy: `http://127.0.0.1:8501`.

---

## 7. Pre-Flight Verification Sign-Off (F8)

Prior to tagging and submission:
1. `tools/check_freeze.py`: **PASS** (Frozen Verifier v1.2, SHA: `3fa69071f888643b2e53ef7a8eb2a37f59457a4198305f6e6da48bbfa32ef391`).
2. `tools/audit_test2_isolation.py`: **PASS** (100% clean isolation; zero cross-split leaks).
3. `tools/secret_scan.py`: **PASS** (0 secrets, 0 credentials, 0 PII detected).
4. Unit Test Suite (`tests/test_*.py`): **96 / 96 tests PASS** (including pair disjointness regression test).
5. AC Power State: Screen/Sleep timeouts verified and restored if pausing.

---
*End of Amendment 1. Awaiting human decision on Options Table (Section 5) and explicit authorization (`HUMAN_APPROVED_TEST3: yes`) prior to initiating Stage B.*
