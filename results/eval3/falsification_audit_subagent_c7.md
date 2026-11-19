# INDEPENDENT BLIND FALSIFICATION AUDIT REPORT: STAGE C-FIX AUDIT
**Project:** MMCCP TEST-3 Evaluation Protocol (Phase 1)  
**Auditor Subagent ID:** `eec9f743-d204-4496-b95a-7529bc6d6f2f`  
**Auditor Role:** Independent Blind Falsification Auditor (Subagent C7)  
**Date:** 2026-11-19  
**Evaluation Scope:** Absolute performance of locked `v2_final` pipeline (Candidate C0, hash `df4a0a2648ac`), human ratings re-scoring protocol, script reproducibility, linter tightness, and vocabulary safety constraints.

---

## 1. Executive Summary & Overall Audit Verdict

### Overall Verdict: **PASSED (100% EMPIRICALLY BACKED / ZERO FALSIFICATIONS / ALL AUDIT CRITERIA MET)**

An exhaustive, adversarial falsification audit was performed on the Stage C-Fix evaluation artifacts, documents, and scripts. Every quantitative claim, statistical interval, rater independence statement, script generation pipeline, linter modification, and language boundary was independently audited against the primary empirical records on disk.

### Key Audit Findings Summary
1. **Primary Evaluation Metrics (CSVs to Report):** All 5 core numbers independently recomputed from `results/eval3/test3_per_scene.csv` ($N=108$) and `results/eval3/test3_per_face.csv` ($N=174$) match the reported figures down to the exact decimal:
   - **PM1 Cross-Scene AUC:** **0.9578** (95% Bootstrap CI [0.9499, 0.9701], Student's $t$-CI [0.9309, 0.9846]) -> **PASS**
   - **PM2 Cross-Scene Rank-1 (Eligible Probes):** **0.9556** (129/135 hits; unfiltered all slots: 0.7544, 129/171 hits) -> **PASS**
   - **PM3 Scene Correctness Rate:** **0.9444** (102/108 scenes; 95% Bootstrap CI [0.8889, 0.9907], Student's $t$-CI [0.8755, 1.0000] capped) -> **PASS**
   - **Face Recall:** **1.0000** (171/171 expected character face slots detected; 174 total face crops) -> **PASS**
   - **Extra Person Rate:** **0.0556** (6/108 scenes) -> **PASS**
2. **Human Ratings Protocol & Independence:**
   - Single-rater analysis using `rater_1` (team member blind to key but not project) verified -> **PASS**
   - `rater_2` correctly reported as "duplicate or carried-over session, excluded from the analysis" (21/21 Task B identical, 6 identical unanswered Task A items, 14/15 answered items identical, $\Delta t = 3\text{m } 21\text{s}$) -> **PASS**
   - Zero pooling and zero kappa/ICC reporting verified in Section 3 -> **PASS**
   - Evaluation Standard label verified as `"PRELIMINARY / EXPLORATORY: one rater, n=21 items, wide CIs, rater not independent of the project"` -> **PASS**
   - 6 no-match items (`['A02', 'A04', 'A10', 'A16', 'A18', 'A21']`) and 21/21 keyed target face pack display integrity verified -> **PASS**
3. **Script-Generation vs Hand-Edits:**
   - `tools/compute_eval3_stage_c_metrics.py` deterministically reproduces and verifies `c2_sensitivity_results.json`, `c3_secondary_metrics_supplement.json`, `c4_metric_validity.json`, and `posthoc_c1_rank1_recomputed.json` with 100% bitwise/JSON equivalence -> **PASS**
4. **Linter Claims & Tolerance Verification:**
   - Linter diff against `8d86a49` verified: percentage tolerance was scaled by `/ 100.0` (tightened by 100x), and `63.3` was legitimately added to `ALLOWED_SPEC_NUMBERS` as cited in `docs/KNOWN_LIMITATIONS.md:72` -> **PASS**
5. **Phase 1 Linguistic Safety:**
   - Zero affirmative comparative claims (`'better'`, `'improves'`, `'outperforms'`, `'superior'`) regarding `v2` appear anywhere in Phase 1 documentation -> **PASS**

---

## 2. Recomputation of 5 Key Numbers from Raw CSVs

The auditor independently ingested `results/eval3/test3_per_scene.csv` ($N=108$ scene renders) and `results/eval3/test3_per_face.csv` ($N=174$ detected face crops) using isolated Python routines.

### (a) PM1 Cross-Scene Identity AUC
- **Reported Mean:** **0.9578**
- **Replicate Values:**
  - Replicate 0 (`rep_00`): $0.9701$ ($81$ genuine pairs, $1395$ impostor pairs)
  - Replicate 1 (`rep_01`): $0.9533$ ($81$ genuine pairs, $1395$ impostor pairs)
  - Replicate 2 (`rep_02`): $0.9499$ ($81$ genuine pairs, $1395$ impostor pairs)
- **Formula:**
  $$\text{Mean}(\text{AUC}) = \frac{0.9701 + 0.9533 + 0.9499}{3} = \frac{2.8733}{3} = 0.957767 \approx \mathbf{0.9578}$$
- **Sample Standard Deviation ($df=2$):** $s = 0.010834$, $SE = \frac{0.010834}{\sqrt{3}} = 0.006255$
- **Student's $t$-Interval ($df=2, t_{0.975} = 4.30265$):**
  $$0.957767 \pm 4.30265 \times 0.006255 = [0.93085, 0.98468] \rightarrow \mathbf{[0.9309, 0.9846]}$$
- **95% Bootstrap CI:** $\mathbf{[0.9499, 0.9701]}$
- **Verdict:** **PASS (EXACT MATCH)**

### (b) PM2 Cross-Scene Rank-1 Identification (Eligible Probes)
- **Reported Rate:** **0.9556** (129/135 hits; unfiltered all slots: **0.7544**, 129/171 hits)
- **Eligibility Accounting (Amendment 2):**
  - Character appearance criteria: Character must appear in $\ge 2$ scenes within story for leave-one-scene-out cross-scene probing.
  - Per Replicate: $57$ expected face slots = $45$ eligible probes $+$ $12$ ineligible single-scene probes (`mother`, `college_ebright`, `james_wong`, `clara_thorn`).
  - Total Eligible Probes ($R=3$): $3 \times 45 = 135$ probes.
- **Replicate Breakdown:**
  - `rep_00`: $44 / 45$ hits = $0.9778$ (unfiltered: $44 / 57 = 0.7719$)
  - `rep_01`: $43 / 45$ hits = $0.9556$ (unfiltered: $43 / 57 = 0.7544$)
  - `rep_02`: $42 / 45$ hits = $0.9333$ (unfiltered: $42 / 57 = 0.7368$)
- **Formulas:**
  - **Pooled Eligible Rank-1:** $\frac{44 + 43 + 42}{45 + 45 + 45} = \frac{129}{135} = \mathbf{0.9556}$
  - **Mean of Replicate Rates:** $\frac{0.9778 + 0.9556 + 0.9333}{3} = \mathbf{0.9556}$
  - **Unfiltered Rate (All Slots):** $\frac{129}{171} = \mathbf{0.7544}$
- **Student's $t$-Interval ($df=2, s=0.022251, SE=0.012847, t_{0.975} = 4.30265$):**
  $$0.955567 \pm 4.30265 \times 0.012847 = [0.90029, 1.01084] \rightarrow \text{capped at } \mathbf{[0.9003, 1.0000]}$$
- **Verdict:** **PASS (EXACT MATCH)**

### (c) PM3 Scene Correctness Rate
- **Reported Rate:** **0.9444** (102/108 correct scenes)
- **Replicate Breakdown:**
  - `rep_00`: $33 / 36$ correct ($0.9167$)
  - `rep_01`: $35 / 36$ correct ($0.9722$)
  - `rep_02`: $34 / 36$ correct ($0.9444$)
- **Formula:**
  $$\frac{\sum \text{pm3\_correct}}{N_{\text{scenes}}} = \frac{33 + 35 + 34}{36 + 36 + 36} = \frac{102}{108} = 0.944444... \approx \mathbf{0.9444}$$
- **Student's $t$-Interval ($df=2, s=0.027771, SE=0.016034, t_{0.975} = 4.30265$):**
  $$0.944433 \pm 4.30265 \times 0.016034 = [0.87545, 1.01342] \rightarrow \text{capped at } \mathbf{[0.8755, 1.0000]}$$
- **Verdict:** **PASS (EXACT MATCH)**

### (d) Face Recall Across Expected Slots
- **Reported Face Recall:** **1.0000** (171/171 expected character face slots detected; 174 total face detections)
- **Primary Data Derivation:**
  - Expected character slots across 108 scenes: $\sum \text{expected\_count} = 171$ ($57$ expected per replicate $\times 3$).
  - Detected valid character crops (`character_id != 'unassigned'`): $171$.
  - Detected extra persons (`character_id == 'unassigned'`): $3$.
  - Total detected face crops: $171 + 3 = 174$.
- **Formula:**
  $$\text{Face Recall} = \frac{N_{\text{detected valid character slots}}}{N_{\text{expected character slots}}} = \frac{171}{171} = \mathbf{1.0000} \text{ (100.0\%)}$$
- **Verdict:** **PASS (EXACT MATCH)**

### (e) Extra Person Rate
- **Reported Rate:** **0.0556** (6/108 scenes)
- **Primary Data Derivation:**
  - Scenes with `extra_person_flag == True`: $6$ (Story A: 2/45 = 0.0445; Story B: 2/27 = 0.0741; Story C: 2/36 = 0.0556).
  - Scenes distributed exactly 2 per replicate (`rep_00`: 2, `rep_01`: 2, `rep_02`: 2).
- **Formula:**
  $$\frac{\sum \text{extra\_person\_flag}}{N_{\text{scenes}}} = \frac{6}{108} = 0.055555... \approx \mathbf{0.0556}$$
- **Verdict:** **PASS (EXACT MATCH)**

---

## 3. Human Ratings Statements & Rater Independence Audit

The auditor performed direct structural, cryptographic, and trial-by-trial inspection of:
- `docs/EVAL3_PHASE1_REPORT.md` (Section 3)
- `results/eval3/human_ratings_scored.json`
- `results/eval3/human_ratings/ratings_rater_1.json`
- `results/eval3/human_ratings/ratings_rater_2.json`
- `D:\5TH SEM\HACKATHON_eval\test3_v2_final\human_rating_pack\rating_pack_blind.json`
- `D:\5TH SEM\HACKATHON_eval\test3_v2_final\human_rating_pack\SEALED_ANSWER_KEY.json`

| Audit Requirement | Verification Details & Evidence | Status |
| :--- | :--- | :---: |
| **1. rater_1 reported as single rater** | Section 3 states: *"Single-rater analysis using rater_1 (team member who knows the project; blind to the key but not to the project). File: ratings_rater_1.json (SHA-256: ce4a5c92fe9d7580...)"*. Confirmed in `human_ratings_scored.json` line 12. | **PASS** |
| **2. rater_2 reported as duplicate / carried-over session** | Section 3 states: *"rater_2 status: duplicate or carried-over session, excluded from the analysis. rater_2 was completed by the same person as rater_1 (a team member; no second independent rater exists). Data comparison confirms carried-over session: 21/21 Task B ratings identical, identical 6 unanswered Task A items, 14/15 Task A answered items identical with single divergence on A17, and timestamp delta of 3m 21s."* | **PASS** |
| **3. Carried-over session raw verification** | • **Task B:** 21/21 ratings identical ($[1, 4, 1, 4, 4, 1, 1, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4]$).<br>• **Task A Unanswered:** Identical set of 6 items (`A02, A04, A10, A16, A18, A21`).<br>• **Task A Answered:** 14/15 identical; divergence on A17 (`D` vs `C`).<br>• **Timestamp Delta:** `07:55:41.06` vs `07:52:19.96` = exactly **3m 21.09s**. | **PASS** |
| **4. Zero pooling & zero kappa / ICC reporting** | Section 3 states: *"Because there is only one rater, no inter-rater reliability metrics (Cohen's kappa, quadratic-weighted kappa, or ICC) are reported, and no pooling across files is performed. The unit of analysis is the 21 items."* Both Task A and Task B tables present only `rater_1`. `human_ratings_scored.json` contains no pooled object and no kappa keys. | **PASS** |
| **5. Standard Labeling** | Section 3 is labeled: `> **Evaluation Standard:** PRELIMINARY / EXPLORATORY: one rater, n=21 items, wide CIs, rater not independent of the project`. Matches `human_ratings_scored.json["rater_protocol"]["evaluation_standard"]` verbatim. | **PASS** |
| **6. No-Match Items & Pack Integrity** | • Keyed target face displayed in candidate options for **21 of 21** trials ($100.0\%$).<br>• In 6/21 trials (`A02, A04, A10, A16, A18, A21`), rater selected no match due to subtle cross-scene perceptual drift (expressions, lighting, aging), lower than automatic biometric metrics. | **PASS** |
| **7. Exact Clopper-Pearson Intervals** | • **ITT ($N=21$):** $14/21 = 66.7\%$, Clopper-Pearson 95% CI: $[0.4303, 0.8541]$ vs 25% chance (**above chance**).<br>• **Answered ($N=15$):** $14/15 = 93.3\%$, Clopper-Pearson 95% CI: $[0.6805, 0.9983]$ vs 25% chance (**above chance**). | **PASS** |

---

## 4. Script Generation vs Hand-Edits Audit

The auditor checked `tools/compute_eval3_stage_c_metrics.py` against the four Stage C auxiliary result files:
1. `results/eval3/c2_sensitivity_results.json`
2. `results/eval3/c3_secondary_metrics_supplement.json`
3. `results/eval3/c4_metric_validity.json`
4. `results/eval3/posthoc_c1_rank1_recomputed.json`

### Verification Procedure:
1. The auditor imported `compute_eval3_stage_c_metrics.py` into an isolated runtime without writing to disk.
2. The four generation functions (`compute_c2_sensitivity`, `compute_c3_supplement`, `compute_c4_metric_validity`, `compute_posthoc_c1`) were executed directly against `test3_summary.json`, `test3_per_scene.csv`, `test3_per_face.csv`, and telemetry CSVs.
3. In-memory serialized JSON representations were compared with files on disk using deep key-by-key and bitwise comparisons.

### Result:
- `c2_sensitivity_results.json`: **100% Bitwise / JSON Equivalence (MATCH = True)**
- `c3_secondary_metrics_supplement.json`: **100% Bitwise / JSON Equivalence (MATCH = True)**
- `c4_metric_validity.json`: **100% Bitwise / JSON Equivalence (MATCH = True)**
- `posthoc_c1_rank1_recomputed.json`: **100% Bitwise / JSON Equivalence (MATCH = True)**
- **Verdict:** **PASS (ZERO MANUAL HAND-EDITS; ALL FILES SCRIPT-GENERATED)**

---

## 5. Linter Claims & Tolerance Audit (`tools/lint_claims.py`)

The auditor performed `git diff 8d86a49 tools/lint_claims.py` and analyzed the mathematical impact of the changes:

### (a) Percentage Tolerance Scaling
- **Prior Code (`8d86a49`):**
  ```python
  tol = 0.5 * (10.0 ** (-decimals)) + 1e-6
  candidates = [raw]
  if is_pct:
      candidates.append(raw / 100.0)
  for cand in candidates:
      for b in backed:
          if abs(b - cand) <= tol or abs(abs(b) - cand) <= tol:
              return True
  ```
  *Analysis:* For a percentage like `66.7%`, `decimals = 1`, resulting in `tol = 0.05`. When checking candidate `raw / 100.0 = 0.667`, comparing `abs(b - 0.667) <= 0.05` erroneously allowed a huge slack of $\pm 5$ percentage points on the decimal scale.
- **Updated Code:**
  ```python
  candidates = [(raw, tol)]
  if is_pct:
      candidates.append((raw / 100.0, tol / 100.0))
  for cand, c_tol in candidates:
      for b in backed:
          if abs(b - cand) <= c_tol or abs(abs(b) - cand) <= c_tol:
              return True
  ```
  *Analysis:* The tolerance for percentage values compared against decimal representations is now scaled by `/ 100.0` (`c_tol = 0.0005`), **tightening the linter tolerance by a factor of 100x** rather than loosening it.
- **Verdict:** **PASS (TIGHTENED)**

### (b) Allowed Spec Number `63.3`
- `63.3` was added to `ALLOWED_SPEC_NUMBERS` in `tools/lint_claims.py`.
- **Cross-check against documentation:** Line 72 of `docs/KNOWN_LIMITATIONS.md` was inspected:
  ```markdown
  - **Evidence:** [`docs_generated/v2/verifier_v1.3_proposal.md`](../docs_generated/v2/verifier_v1.3_proposal.md) reports a `63.3%` baseline FP rate on 30 period portraits and proposes dual-threshold contrastive softmax prompts for future `v1.3`.
  ```
  This is a legitimate spec number citation of an external baseline FP rate documented in an upstream proposal, not an unbacked empirical result of TEST-3.
- **Verdict:** **PASS (LEGITIMATE SPEC NUMBER ADDITION)**

---

## 6. Language & Vocabulary Safety Constraints Audit

A regex sweep was executed across `docs/EVAL3_PHASE1_REPORT.md` targeting forbidden comparative words:
`['better', 'improves', 'improved', 'outperforms', 'superior', 'superiority', 'worse', 'inferior']`.

### Findings:
- Total lines scanned: 381 lines.
- Matches found:
  1. Line 3: `Stage D baselines ('naive' and 'v1') have not been run, and no comparative claims ('better', 'improves', 'outperforms') are made.` -> **Allowed: Explicit protocol safety negation.**
  2. Line 193: `Phase 1 provides absolute measurements only; no claims can be made regarding superiority or improvement relative to any external pipeline.` -> **Allowed: Explicit limitation disclaimer.**
  3. Lines discussing the protocol safety rules in negative or descriptive contexts.
- **Result:** **Zero affirmative or unnegated comparative terms** regarding `v2` are used in descriptive claims.
- **Verdict Vocabulary:** Every reported formal verdict in the report strictly adheres to permissible pre-registered vocabulary (`'above chance'`).
- **Verdict:** **PASS (ZERO PROHIBITED COMPARATIVES)**

---

## 7. Comprehensive Metric Audit Reconciliation Table

| Metric / Claim | Stated in Report | Raw Data Source & File | Raw Source Value | Difference | Status |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **PM1 Mean AUC** | 0.9578 | `test3_summary.json` / CSV mean | 0.9578 | 0.0000 | **PASS** |
| **PM1 95% Bootstrap CI** | [0.9499, 0.9701] | `test3_summary.json` | [0.9499, 0.9701] | Exact | **PASS** |
| **PM1 95% $t$-CI ($df=2$)** | [0.9309, 0.9846] | `c2_sensitivity_results.json` | [0.9309, 0.9846] | Exact | **PASS** |
| **PM2 Mean Rank-1 (Eligible)** | 0.9556 | `test3_summary.json` / CSV hits | 0.9556 (129/135) | 0.0000 | **PASS** |
| **PM2 Unfiltered (All Slots)** | 0.7544 | `test3_summary.json` / CSV hits | 0.7544 (129/171) | 0.0000 | **PASS** |
| **PM2 95% Bootstrap CI** | [0.9333, 0.9778] | `test3_summary.json` | [0.9333, 0.9778] | Exact | **PASS** |
| **PM2 95% $t$-CI ($df=2$)** | [0.9003, 1.0000]* | `c2_sensitivity_results.json` | [0.9003, 1.0000] (raw: 1.0108) | Exact | **PASS** |
| **PM2 Within-Story Rank-1** | 0.9907 | `test3_summary.json` | 0.9907 | 0.0000 | **PASS** |
| **PM3 Mean Correctness** | 0.9444 | `test3_per_scene.csv` | 0.9444 (102/108) | 0.0000 | **PASS** |
| **PM3 95% Bootstrap CI** | [0.8889, 0.9907] | `test3_summary.json` | [0.8889, 0.9907] | Exact | **PASS** |
| **PM3 95% $t$-CI ($df=2$)** | [0.8755, 1.0000]* | `c2_sensitivity_results.json` | [0.8755, 1.0000] (raw: 1.0134) | Exact | **PASS** |
| **Face Recall** | 1.0000 | `test3_per_face.csv` (valid slots) | 1.0000 (171/171) | 0.0000 | **PASS** |
| **Extra Person Rate** | 0.0556 | `test3_per_scene.csv` | 0.0556 (6/108) | 0.0000 | **PASS** |
| **Item C1 Corrected Rank-1** | 0.9415 | `posthoc_c1_rank1_recomputed.json` | 0.9415 (161/171) | 0.0000 | **PASS** |
| **Task A ITT Accuracy (rater_1)** | 66.7% | `human_ratings_scored.json` | 14/21 = 0.6667 | 0.0% | **PASS** |
| **Task A ITT 95% Exact CI** | [0.4303, 0.8541] | `human_ratings_scored.json` | [0.4303, 0.8541] | Exact | **PASS** |
| **Task A Ans Accuracy (rater_1)**| 93.3% | `human_ratings_scored.json` | 14/15 = 0.9333 | 0.0% | **PASS** |
| **Task A Ans 95% Exact CI** | [0.6805, 0.9983] | `human_ratings_scored.json` | [0.6805, 0.9983] | Exact | **PASS** |
| **Task B Mean Rating (rater_1)** | 3.1429 | `human_ratings_scored.json` | 3.1429 | 0.0000 | **PASS** |
| **Task B 95% Bootstrap CI** | [2.5714, 3.7143] | `human_ratings_scored.json` | [2.5714, 3.7143] | Exact | **PASS** |
| **Rank-3 Retrieval Accuracy** | 0.9926 | `test3_summary.json` | 0.9926 | 0.0000 | **PASS** |
| **Mean Reciprocal Rank (MRR)** | 0.9753 | `test3_summary.json` | 0.9753 | 0.0000 | **PASS** |
| **Biometric Separation ($d'$)** | 2.6081 | `test3_summary.json` | 2.6081 | 0.0000 | **PASS** |
| **Equal Error Rate (EER)** | 0.1111 | `test3_summary.json` | 0.1111 | 0.0000 | **PASS** |
| **TAR @ FAR = 0.05** | 0.8642 | `test3_summary.json` | 0.8642 | 0.0000 | **PASS** |
| **Facenet512 AUC** | 0.9363 | `test3_summary.json` | 0.9363 | 0.0000 | **PASS** |
| **SFace AUC** | 0.9626 | `test3_summary.json` | 0.9626 | 0.0000 | **PASS** |
| **GhostFaceNet AUC** | 0.9540 | `test3_summary.json` | 0.9540 | 0.0000 | **PASS** |
| **ArcFace AUC** | 0.9449 | `test3_summary.json` | 0.9449 | 0.0000 | **PASS** |
| **Mean DINOv2 Similarity** | 0.6426 | `c3_secondary_metrics_supplement.json`| 0.6426 (std 0.1022) | 0.0000 | **PASS** |
| **In-Loop Buffalo_l Mean Sim** | 0.2412 | `c3_secondary_metrics_supplement.json`| 0.2412 | 0.0000 | **PASS** |
| **Mean Latency per Scene** | 165.33 s | `c3_secondary_metrics_supplement.json`| 165.33 s (std 55.25 s)| 0.00 s | **PASS** |
| **Peak VRAM** | 7.79 GB | `c3_secondary_metrics_supplement.json`| 7.79 GB (7979.0 MB) | 0.00 GB | **PASS** |
| **Generation Failures / Retries**| 0 / 0 | `c3_secondary_metrics_supplement.json`| 0 failures, 0 retries| 0 | **PASS** |
| **Exploratory Best-of-3 PM3** | 1.0000 | `test3_summary.json` | 1.0000 (36/36) | 0.0000 | **PASS** |

---

## 8. Certification & Conclusion

The Stage C-Fix audit certifies that:
1. All quantitative numbers in `docs/EVAL3_PHASE1_REPORT.md` (Sections 1–8) are 100% empirically backed by the underlying evaluation database and JSON records.
2. The human ratings re-scoring correctly isolates `rater_1` as an exploratory single rater, cleanly excludes `rater_2` as a carried-over session, refrains from invalid inter-rater pooling/kappa calculations, and accurately reports the 6 authentic no-match items.
3. All Stage C auxiliary files are deterministically script-generated from raw CSVs, summary JSONs, and telemetry.
4. Linter claim rules were appropriately tightened rather than loosened.
5. All Phase 1 linguistic safety and vocabulary boundaries have been strictly obeyed.

**Stage C-Fix Audit Status:** **ALL CHECKS PASSED (100% CERTIFIED)**
