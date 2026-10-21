# Corrected DEV-FAST Comparison: Baseline v1 vs v2@72769ef vs Bundle (iter3)

> **Source File:** `results/corrected_dev_fast_comparison.json`  
> **Dataset:** DEV-FAST (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37 | 12 images, 18 face instances)  
> **Canonical Verifier:** Frozen Verifier v1.2 (`3fa69071f888643b...`)  

---

## 1. Summary Comparison Table

| Metric | Baseline v1 | v2@72769ef (Broken v2) | Bundle (iter3) | Bundle vs v1 $\Delta$ |
| :--- | :---: | :---: | :---: | :---: |
| **Biometric Ensemble Mean** | **0.3142** | 0.2081 | **0.3277** | **+0.0135** |
| **Image Bootstrap 95% CI** | - | - | - | `[-0.0728, +0.0513]` |
| **Scene Cluster 95% CI** | - | - | - | `[-0.0896, +0.0635]` |
| **Presence Rate** | 100.0% (18/18) | 66.7% (12/18) | 94.4% (17/18) | -5.6% |
| **Identity-Given-Presence** | 0.3142 | 0.3122 | 0.3470 | +0.0328 |
| **Rank-1 Discrimination** | 77.8% | 27.8% | 88.9% | +11.1% |
| **Discrimination Margin** | +0.0253 | -0.3583 | +0.0044 | -0.0209 |
| **Wardrobe Pass Rate** | 27.8% | 11.1% | 44.4% | +16.7% |
| **Mean Face Width** | 233.1 px | 140.2 px | 128.0 px | -105.1 px |
| **Structural Defects** | 1 | 6 | 1 | +0 |
| - Missing Characters | 0 | 6 | 1 | +1 |
| - Extra People (YOLO) | 0 | 0 | 0 | 0 |
| - Inversions | 1 | 0 | 0 | 0 |
| - Melted Faces | 0 | 0 | 0 | 0 |
| **Prop Defects** | 28 | 37 | 18 | -10 |

---

## 2. Age Subgroup Breakdown

| Age Group | Baseline v1 | v2@72769ef | Bundle (iter3) | Bundle vs v1 $\Delta$ | Bundle Width vs v1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Child** (n=6) | **0.3568** | 0.3210 | **0.2937** | **-0.0631** | 160.2 vs 284.5 px |
| **Teen** (n=3) | **0.2886** | 0.3276 | **0.2513** | **-0.0373** | 171.3 vs 388.7 px |
| **Adult** (n=9) | **0.2943** | 0.0929 | **0.3759** | **+0.0816** | 92.1 vs 147.0 px |

*Takeaway:* Bundle gains substantially on Adult (+0.0816) due to layout anchoring preventing collapse in Scene 1 & 5, but Child loses -0.0631 and Teen loses -0.0373 due to smaller face framing (125.7 px vs 196.2 px on Child; 178.7 px vs 359.7 px on Teen).

---

## 3. Face-Size-Matched Bins

| Face Size Bin | Baseline v1 Bio (n) | Bundle (iter3) Bio (n) | $\Delta$ |
| :--- | :---: | :---: | :---: |
| **Small (<100px)** | 0.3093 (n=1) | 0.2597 (n=4) | -0.0496 |
| **Medium (100-180px)** | 0.2756 (n=9) | 0.3515 (n=12) | +0.0759 |
| **Large (>180px)** | 0.3582 (n=8) | 0.3208 (n=2) | -0.0374 |

---

## 4. Telemetry Audit (6476 vs 7805 MB Claim Reconciliation)

- **Claimed Peak:** 6,476 MB
- **Summary Recorded Peak:** 7,805 MB
- **Actual Exact Window Peak:** **8,004 MB** (at `14:19:17 UTC` during Scene 1 Seed 11 Pass 1)
- **Idle Baseline:** 854 MB (desktop cold), 2,226 MB (Forge idle with loaded weights)
- **Spill Evidence:** During Pass 1 of two-shot scenes, allocated VRAM spiked to 8,004 MB (97.7% of 8,192 MB capacity), triggering PyTorch CUDA memory thrashing and latency spikes up to 186.8s per scene (vs 114.4s on single-character scenes).

---

## 5. Effective Configuration Audit

- **iter3 Effective Config:** `ConsistencyV2Config(mode='fast', steps_pass1=14, steps_pass2=10, cfg_scale=5.0, codeformer_fidelity=0.85, enable_face_detailer=False, enable_dual_instantid=False)`
- **Config Hash:** `51a25afa0e338c5c` (`51a25afa0e33`)
- **Finding:** Both broken v2 (`single_anchor`) and `iter3` were rendered under `mode='fast'`, whereas `best_config.json` was stored under `mode='default'`.
