# MMCCP TEST-3 Pre-Registration Protocol — Amendment 2
**Document Version:** 1.2 (Amendment 2 to `PREREG_TEST3.md`)  
**Date:** 2026-11-17  
**Git Branch:** `eval/test3`  
**Supercedes:** Sections 2, 3, and 5 of `results/eval3/PREREG_TEST3_AMENDMENT_1.md` and `results/eval3/PREREG_TEST3.md`  
**Evaluation Scope:** Absolute Measurement of `v2_final` (`TEST-3` Phase 1)  

---

> [!IMPORTANT]
> **Pre-Registration Invariant Declaration:**  
> **NO TEST-3 data has been generated, accessed, or evaluated.** All empirical evaluations, calibrations, timing benchmarks, and metric audits documented herein were conducted strictly on the **DEV-3** partition (Story A `ebright.json`, scenes 1–5, seeds in `[7001, 7499]`). The frozen verification suite (`consistency_v2/verify/`), `verifier_freeze.json`, `TEST-2` partitions, and reserved seeds `{606, 707, 808}` remain 100% untouched.

---

## 1. PM2 Eligibility Amendment & Cross-Scene Metrics (G1)

### Metric Logic & Justification
In Story A (`storyboards/ebright.json`), three characters (`mother`, `james_wong`, `college_ebright`) appear in **only a single scene** across the storyboard:
- `mother`: Scene 1 only
- `james_wong`: Scene 5 only
- `college_ebright`: Scene 5 only

Under the definition of leave-one-scene-out cross-scene identification, a probe face in scene $s$ is compared against a gallery formed from appearances of candidates in *other* scenes ($s' \neq s$). For single-scene characters, no cross-scene gallery representation can exist by definition. 

**Amendment:** The denominator for PM2 is amended to **ELIGIBLE probes**: faces of characters who appear in **at least 2 different scenes** of the same replicate.
- This rule is chosen strictly from **metric logic** (preventing impossible test cases), not from empirical results.
- Within eligible probes, **intention-to-treat is strictly preserved**: any undetected or missing face is counted as incorrect (penalized with rank $N_{\text{gallery}} + 1$ and reciprocal rank $0.0$).

### Probe Accounting & Chance Floor
- **Story A Scene Structure (Scenes 1–5):**
  - Scene 1: `young_ebright`, `mother` (2 slots)
  - Scene 2: `young_ebright` (1 slot)
  - Scene 3: `teen_ebright` (1 slot)
  - Scene 4: `teen_ebright` (1 slot)
  - Scene 5: `college_ebright`, `james_wong` (2 slots)
- **Accounting per Replicate:**
  - $n_{\text{expected}} = 7$ total character slots
  - $n_{\text{eligible}} = 4$ probes (`young_ebright` in Scenes 1 & 2 $\implies 2$; `teen_ebright` in Scenes 3 & 4 $\implies 2$)
  - $n_{\text{ineligible}} = 3$ probes (`mother`, `college_ebright`, `james_wong`)
- **Chance Level:**
  - Full closed-set gallery size: $N_{\text{gallery}} = 5$ characters
  - Chance floor: $\frac{1}{N_{\text{gallery}}} = \frac{1}{5} = \mathbf{0.2000}$

### DEV-3 Pilot Rescoring (New vs Old PM2)

| Candidate | Config Hash | PM1 AUC | PM2 Rank-1 (Eligible) | PM2 Old (All Slots) | PM3 Correctness | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | `df4a0a2648ac` | **1.0000** | **1.0000** ($8/8$ probes) | `0.5714` ($8/14$ slots) | `0.9000` | **Winner (Status Quo)** |
| **C1** | `d5cae1b7d310` | **1.0000** | **1.0000** ($8/8$ probes) | `0.5714` ($8/14$ slots) | `0.9000` | Tied C0; breaks to C0 |
| **C2** | `9821eb35ddb1` | `0.9384` | `0.9375` ($7.5/8$ probes) | `0.5357` ($7.5/14$ slots) | `0.9000` | Defeated |

### Application Across All Evaluation Metrics
- **Rank-3 Accuracy, MRR, and CMC Curve:** Evaluated over eligible probes ($n_{\text{eligible}}$), with missing/undetected faces scored as 0.0 under intention-to-treat.
- **Per-Character & Per-Age Breakdowns:** Single-scene characters are explicitly labeled `N/A (ineligible)` rather than false `0.0`.
- **Genuine Pair Counts & ROC AUC (PM1):** Single-scene characters naturally yield 0 cross-scene genuine pairs; impostor pairs remain disjoint across different scenes and characters. No denominator distortion.
- **Face Recall & PM3 (Scene Correctness):** Operate on all expected face slots ($n_{\text{expected}}$) and all scenes ($n_{\text{scenes}}$); intention-to-treat is maintained.

### Evaluator Code Hash & Regression Testing
- File: `eval3/metrics/evaluator.py`
- **SHA-256:** `6cc0b42652cd02ccf03ad821309633cf1e8b11cf284dd5dddab9e338e9a77524`
- Tested by 3 dedicated regression tests in `tests/test_eval3_metrics.py`:
  1. `test_pm2_ineligible_character_single_scene`: PASS
  2. `test_pm2_eligible_missed_face_intention_to_treat`: PASS
  3. `test_pm2_story_a_hand_computed_counts`: PASS

---

## 2. Out-of-Loop Calibration Refit (G2)

### Verification of Prior Calibration
An audit confirmed that `results/eval3/calibration_eval3.json` in commit `d491a9f` contained initial toy/sample counts ($n_{\text{genuine}}=5, n_{\text{impostor}}=5$) and predated Amendment 1's disjoint pair filter.

### Refit with Strict Disjoint Pairs
Calibration was refitted using Candidate C0 renders across Replicate 0 and Replicate 1 on the strictly disjoint pair set:
- **Genuine Pairs:** **16 pairs** (same character, different scenes)
- **Impostor Pairs:** **116 pairs** (different characters, different scenes, different images)

### Refitted Logistic & Separation Parameters

| Judge Model | Mean Genuine | Std Genuine | Mean Impostor | Std Impostor | Logistic $w$ | Logistic $b$ | FAR 5% Threshold | $d'$ Separation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Facenet512** | 0.8168 | 0.0525 | 0.3841 | 0.1972 | 21.3705 | -16.0376 | 0.6798 | 2.9985 |
| **SFace** | 0.7623 | 0.0693 | 0.3168 | 0.2055 | 17.1122 | -12.1868 | 0.6546 | 2.9048 |
| **GhostFaceNet** | 0.7508 | 0.0725 | 0.3376 | 0.1510 | 18.3043 | -12.2855 | 0.5993 | 3.4894 |
| **ArcFace** | 0.7522 | 0.0939 | 0.3526 | 0.1811 | 13.4494 | -9.7150 | 0.6575 | 2.7700 |

- File: `results/eval3/calibration_eval3.json`
- **SHA-256:** `5adaa6fa2234bf702d609a86677130595a222a2751a78e5066244f517f17b0f2`

### Verification of the 5 Empirical Controls (`tools/run_dev3_controls.py`)
1. **Control 1 (Positive Control):** AUC = **1.0000** (**PASS**, threshold $\ge 0.99$).
2. **Control 2 (Negative Control — 1000 Shuffles):** Mean Shuffled AUC = **0.4979** (**PASS**, range $0.50 \pm 0.05$).
3. **Control 3 (Blank Image Detection):** **0** faces detected (**PASS**).
4. **Control 4 (Synthetic Calibration & Coverage):** Perfect Sep AUC = **1.0**, Identical AUC = **0.4516**, 95% Bootstrap Coverage = **91.00%** (**PASS**, threshold $\ge 90\%$).
5. **Control 5 (Diversity & Copy-Paste Guard):** Mean Similarity = **0.6716**, Max Similarity = **0.7690** (**PASS**, max $< 0.85$).

---

## 3. Empirical Time Plan & Option A Exact Arithmetic (G3)

### Empirically Measured Parameters
All pipeline stages were measured from disk timestamps and live benchmark runs with zero GPU contention:
1. **Scene Generation Time ($T_{\text{scene}}$):**
   - 1-character scene: $84.16\text{s}$
   - 2-character scene: $163.48\text{s}$
   - Weighted mean across 12 scenes (7 single, 5 double): **$117.21\text{ seconds}$** per render.
2. **Anchor Generation Time ($T_{\text{anchor}}$):**
   - Measured across 5 characters (20 candidate PNGs + 4-point gate audit + selection): $723\text{ seconds}$ total.
   - Mean per character: **$144.60\text{ seconds}$** ($36.15\text{s}$ per candidate txt2img).
3. **Evaluation Scoring Time ($T_{\text{score}}$):**
   - Live benchmark on Replicate 0 (10 scenes): $431.625\text{s}$ total.
   - One-time vision worker cold-start: **$85.0\text{ seconds}$**.
   - Steady-state audit time per scene image: **$34.69\text{ seconds}$** (YOLO + 4 DeepFace judges + SegFormer + DINOv2 + CLIP).
   - Replicate evaluation matrix calculation: **$0.034\text{ seconds}$** per replicate.
4. **Human-Rating Pack Preparation:**
   - Script extraction and HTML compilation: **$60.0\text{ seconds}$**.

### Exact Arithmetic for Option A ($R = 4$ Replicates)
- **Scene Renders:** 4 replicates $\times$ 12 scenes $\times$ 3 seeds = **144 renders**.
- **Anchor Characters:** 4 replicates $\times$ 11 characters = **44 characters** (176 candidate renders).

$$\begin{aligned}
T_{\text{gen}} &= 144 \times 117.21\text{s} = 16,878.24\text{ s} && (4.688\text{ h}) \\
T_{\text{anchor}} &= 44 \times 144.60\text{s} = 6,362.40\text{ s} && (1.767\text{ h}) \\
T_{\text{scoring}} &= 85.0\text{s} + (144 \times 34.69\text{s}) + (4 \times 0.034\text{s}) = 5,080.50\text{ s} && (1.411\text{ h}) \\
T_{\text{human\_pack}} &= 60.00\text{ s} && (0.017\text{ h}) \\
\hline
\mathbf{T_{\text{total}}} &= \mathbf{28,381.14\text{ s}} = \mathbf{7.884\text{ hours}} && (\approx \mathbf{7.88\text{ hours}})
\end{aligned}$$

Option A executes within the pre-declared **8.0-hour** budget cap.

### Baselines (Stage D: `naive` and `v1` on same 4 replicates)
1. **1 Seed per Scene (48 renders per arm = 96 total renders):**
   - Generation: $48 \times 22.0\text{s} + 48 \times 45.0\text{s} = 1,056\text{s} + 2,160\text{s} = 3,216.0\text{s}$ ($0.893\text{h}$)
   - Scoring: $96 \times 34.69\text{s} = 3,330.24\text{s}$ ($0.925\text{h}$)
   - **Total Stage D (1 seed):** $6,546.24\text{s} = \mathbf{1.818\text{ hours}} \approx \mathbf{1.82\text{ hours}}$
2. **3 Seeds per Scene (144 renders per arm = 288 total renders):**
   - Generation: $144 \times 22.0\text{s} + 144 \times 45.0\text{s} = 3,168\text{s} + 6,480\text{s} = 9,648.0\text{s}$ ($2.680\text{h}$)
   - Scoring: $288 \times 34.69\text{s} = 9,990.72\text{s}$ ($2.775\text{h}$)
   - **Total Stage D (3 seeds):** $19,638.72\text{s} = \mathbf{5.455\text{ hours}} \approx \mathbf{5.46\text{ hours}}$

---

## 4. Pre-Declared Stopping Rule (G4)

To prevent budget overruns or incomplete evaluations:
1. Replicates are executed sequentially in strict replicate index order: $\text{rep\_00} \to \text{rep\_01} \to \text{rep\_02} \to \text{rep\_03}$.
2. If interrupted by the human operator, or if cumulative execution time reaches **8.0 hours**, the run terminates immediately **AFTER the last complete replicate**.
3. A minimum of **$R \ge 3$ complete replicates** is required for primary statistical reporting and bootstrap confidence intervals.
4. If a replicate is partially completed when the cap is reached, all partial data is archived into `results/eval3/partial/`, reported separately for transparency, and **strictly excluded from the primary statistical analysis**.

---

## 5. Honest Correction of Narrative Transcription Error (G5)

In the initial Stage A summary report, the descriptive text mistakenly stated:
- Candidate C0 as having "M4 weight 0.60"
- Candidate C1 as having "IP-Adapter weight 0.75"
- Candidate C2 as having "22 steps"

**Statement of Fact:** These erroneous descriptions were the agent's own **narrative transcription error** when synthesizing textual notes from exploratory v2 tuning phases. The human operator provided no incorrect notes and bore no responsibility for this error.

The underlying execution code (`eval3/runner.py`, `tools/run_dev3_pilot.py`), configuration files (`consistency_v2/best_config.json`), and disk metadata (`D:\5TH SEM\HACKATHON_eval\dev3_pilot_C0`, `C1`, `C2`) always executed the exact prescribed parameters:
- IP-Adapter Plus face weight: Exactly **0.55**
- Pass 1 steps: Exactly **20**
- Pass 2 steps: Exactly **15**
- Sampler / Scheduler: `DPM++ 2M Karras`
- Candidate C0 config hash: `df4a0a2648ac` (100% match with locked v2)

---

## 6. Pre-Flight Verification Sign-Off Table (G7)

| Verification Gate | Command | Result | Details |
| :--- | :--- | :---: | :--- |
| **Verifier Code Freeze** | `python tools/check_freeze.py` | **PASS** | v1.2, SHA: `3fa69071f888643b...` |
| **TEST-2 Data Isolation** | `python tools/audit_test2_isolation.py` | **PASS** | 100% clean isolation; zero leaks |
| **Secret & Credential Scan** | `python tools/secret_scan.py` | **PASS** | 0 secrets, 0 credentials, 0 PII |
| **Unit Test Suite** | `python -m unittest discover tests` | **PASS** | 99 / 99 tests PASS |
| **Streamlit AppTest** | `python tools/test_streamlit_tab5.py` | **PASS** | Tab 5 loads with zero exceptions |
| **Pre-Registration Tag** | `git tag -a eval3-prereg-a2` | **LOCKED** | Ready for push |

---
*End of Amendment 2. Execution remains strictly paused awaiting human authorization (`HUMAN_APPROVED_TEST3: yes`).*
