# MMCCP TEST-3 Pre-Registration Protocol — Clarification 1
**Document Version:** 1.2.1 (Operational Clarification prior to Stage B)  
**Date:** 2026-11-18  
**Git Branch:** `eval/test3`  
**Parent Documents:** `PREREG_TEST3.md` (`eval3-prereg`), `PREREG_TEST3_AMENDMENT_1.md` (`eval3-prereg-a1`), `PREREG_TEST3_AMENDMENT_2.md` (`eval3-prereg-a2`)  
**Type:** Operational Clarification (NOT a design change)

---

> [!IMPORTANT]
> **Pre-Start Invariant Declaration:**  
> **NO TEST-3 renders have been generated at the time of this commit.** All frozen verifier files, evaluator code, calibration parameters, selected configuration, and storyboards are verified unchanged from tag `eval3-prereg-a2`.

---

## 1. Scope of the 8.0-Hour Budget Cap & Stopping Rule Clarification

1. **Generation-Only Budget Cap:** The **8.0-hour** cap (`BUDGET_HOURS = 8.0`, $28,800$ seconds) applies strictly to **ANCHOR GENERATION + SCENE GENERATION** wall-clock time.
2. **Post-Generation Phases Excluded from Generation Cap:** Evaluation scoring (`B2`), visual QA (`B3`), and human-rating pack construction (`B4`) execute sequentially **after** all GPU generation completes (with zero GPU contention) and are excluded from the 8.0-hour generation cap.
3. **Stopping Rule (Unchanged):**
   - Replicates execute sequentially in strict index order: `rep_00`, `rep_01`, `rep_02`, `rep_03`.
   - If stopped by the operator or if cumulative generation time reaches **8.0 hours**, generation terminates **only after the last COMPLETE replicate**.
   - A minimum of **$R \ge 3$ complete replicates** is required for the primary statistical analysis.
   - Any partial replicate is archived in `results/eval3/partial/` and strictly excluded from the primary analysis (reported separately for transparency).
4. **Stage D Exclusion:** Stage D baselines (`naive` and `v1`) are **NOT** part of Stage B and remain hard-blocked until explicitly authorized.
5. **3-Character Stress Scene (`Story B Scene 4`):** Executed last within each replicate **only** if the 8.0-hour generation cap allows, and reported strictly separately from the 12 primary supported scenes.

---

## 2. Pre-Start Artifact Integrity Verification (Tag `eval3-prereg-a2`)

| Artifact | SHA-256 | Status |
| :--- | :--- | :---: |
| `eval3/metrics/evaluator.py` | `6cc0b42652cd02ccf03ad821309633cf1e8b11cf284dd5dddab9e338e9a77524` | **VERIFIED** (matches `AMENDMENT_2`) |
| `results/eval3/calibration_eval3.json` | `5adaa6fa2234bf702d609a86677130595a222a2751a78e5066244f517f17b0f2` | **VERIFIED** (matches `AMENDMENT_2`) |
| `results/eval3/selected_config.json` | `2ce233846a41c06f193fb760112ff924dde9d1740caca0da492975cdac17084f` (`config_hash`: `df4a0a2648ac`) | **VERIFIED** |
| `storyboards/eval3_story_c.json` | `e89b6156544f2472c0259d15acbed27403e94d02c797c1870bef84e686e223ef` | **VERIFIED** |
| `consistency_v2/verifier_freeze.json` | v1.2 combined SHA `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6` | **VERIFIED** |

---

## 3. Operational Conventions Locked Prior to First TEST-3 Render

1. **Deterministic Seed Formulas (`eval3/splits.py`, unit-tested by `tests/test_eval3_splits.py`):**
   - TEST-3 Anchors: `8001 + (replicate_idx * 88) + (char_idx * 8) + candidate_idx` $\in [8001, 8450]$
   - TEST-3 v2 Scenes: `8451 + (replicate_idx * 45) + (story_idx * 15) + (scene_idx * 3) + seed_idx` $\in [8451, 8700]$
   - Zero collisions with `DEV-FAST`, `DEV-2`, `TEST-2`, `RESERVED`, `DEMO`, `DEV-3`, or `STAGE-D`.
2. **Anchor Candidate Synthesis (`eval3/anchors.py`):**
   - Each character's primary `age_state` is supplied to `build_anchor_prompt` (so child, teen, adult, and elder characters synthesize at their specified storyboard ages).
   - Generates 4 initial candidates (`k = 0..3`); if 0 pass the 4-point admission gate, extends sequentially (`k = 4..7`, up to 8 candidates total) until a candidate certifies or 8 candidates are generated. Every generated candidate is saved on disk.
   - Intra-replicate separation check runs across all selected anchors in each replicate (`mean_4judge_cosine < 0.42`).
3. **PM2 Tie-Handling Convention:**
   - Gallery ranking uses Python's Timsort descending by calibrated ensemble score (`sorted(gallery_scores.items(), key=lambda x: x[1], reverse=True)`), preserving insertion order of `expected_chars` on exact ties; missing probe faces or empty other-scene galleries (`score == -1.0`) are scored as incorrect (`rank = N_gallery + 1`, reciprocal rank `0.0`).
4. **Bootstrap RNG Seed:**
   - Hierarchical cluster bootstrap runs with `n_resamples = 10000` and fixed `seed = 42` (`eval3/analysis.py`).

---
*End of Clarification 1. Committed and pushed prior to the first TEST-3 render.*
