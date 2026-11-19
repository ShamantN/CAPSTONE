# INDEPENDENT ADVERSARIAL FALSIFICATION AUDIT REPORT: ITEMS R1 THROUGH R5

## Auditor Metadata
- **Date:** 2026-11-19
- **Auditor Role:** Independent Adversarial Falsification Auditor
- **Subagent Conversation ID:** fc3b9dcb-0e5d-438f-9493-0a090d3723d8
- **Caller Agent ID:** 7707bba8-129e-45a9-b207-9486516b4557
- **Target Repository:** D:\5TH SEM\HACKATHON
- **Target Evaluation Documents:** 
  1. `docs/EVAL3_PHASE1_REPORT.md`
  2. `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`
  3. `tools/generate_phase1_report.py`
  4. Storyboards & Preregistration Artefacts (`storyboards/*.json`, `results/eval3/PREREG_TEST3*.md`, `results/eval3/c3_secondary_metrics_supplement.json`)

---

## Executive Summary
An independent, adversarial, strict falsification audit was conducted on items R1 through R5 across the repository code, storyboards, preregistration protocols, and generated evaluation reports. All audited files were examined directly on disk and verified against source artifacts, git history, and runtime scripts.

**Overall Summary of Findings:**
- **R1 (Unsupported Statements):** The speculative claim attributing 6 unanswered ratings to "perceptual drift" was completely eradicated from `docs/EVAL3_PHASE1_REPORT.md` and `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md` (0 occurrences). It was replaced at both locations (lines 77 and 337) with the verbatim required statement.
- **R2 (Audit Label & Independence Notice):** Section 9 contains the required disclaimer notice in both documents. The original audit file on disk (`results/eval3/falsification_audit_subagent_c7.md`) has `git status` clean and `git diff` empty (0 modifications).
- **R3 (Stress Scenes):** Verified against raw storyboard files that `Story B Scene 4` is a verified 3-character scene, whereas `Story A Scene 6` (1 character) and `Story C Scene 5` (non-existent; story has 4 scenes total) are correctly denoted as "identifiers unverified". Both Section 4 and Section 8 document this accurately.
- **R4 (Number Definitions):** Clear distinctions are established between production latency (165.33 s) and benchmark latency (117.21 s), as well as face-weighted ensemble score (0.7075) and scene-averaged score (0.7147). Peak VRAM (7979.0 MB) is explicitly defined as whole-system memory sampled via `nvidia-smi`, and claims of "below capacity" and "zero CUDA OOMs" have been removed.
- **R5 (PM3 Wording):** PM3's nominal chance floor is documented as 0.0000, establishing that "above chance" is uninformative; PM3 is presented as an absolute rate of 0.9444 with bootstrap CI [0.8889, 0.9907] and capped t-interval [0.8755, 1.0000]; and "above chance" is retained strictly as a direct quote of the pre-registered decision rule.

**Overall Audit Verdict: PASS (All items R1 through R5 strictly verified and passed).**

---

## Per-Item Detailed Audit (R1 through R5)

### ITEM R1: UNSUPPORTED STATEMENTS
- **Requirement:**
  1. Inspect `docs/EVAL3_PHASE1_REPORT.md`, `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`, and `tools/generate_phase1_report.py`.
  2. Check that the sentence attributing the 6 unanswered human ratings to subtle cross-scene perceptual drift has been removed from both generated files (near lines 75 and 329) and replaced with:
     `"the single rater reported that none of the four displayed options matched the reference face; the keyed face was displayed in 21 of 21 items; no cause is recorded."`
  3. Check if "perceptual drift" appears anywhere in `docs/EVAL3_PHASE1_REPORT.md` or `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`.
- **Textual & File Evidence:**
  - Automated string search for `"perceptual drift"` (case-insensitive):
    - `docs/EVAL3_PHASE1_REPORT.md`: **0 occurrences**.
    - `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`: **0 occurrences**.
  - Verification of replacement text:
    `"the single rater reported that none of the four displayed options matched the reference face; the keyed face was displayed in 21 of 21 items; no cause is recorded."`
    - Present in `docs/EVAL3_PHASE1_REPORT.md` at:
      - Line 77 (under *Automatic Metrics versus Human Judgement*):
        `"This human accuracy of 66.7% (14/21) is lower than the automatic metrics suggest (PM2 Rank-1 of 95.6% on eligible probes, 75.4% unfiltered). For the 6 unanswered items (28.6% of trials), the single rater reported that none of the four displayed options matched the reference face; the keyed face was displayed in 21 of 21 items; no cause is recorded."`
      - Line 337 (under *Section 9: Falsification Audit Report Reconciliation Table, Item 6*):
        `"• In 6/21 trials (A02, A04, A10, A16, A18, A21), the single rater reported that none of the four displayed options matched the reference face; the keyed face was displayed in 21 of 21 items; no cause is recorded."`
    - Present in `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md` at:
      - Line 77 and Line 337 verbatim.
  - Generator script (`tools/generate_phase1_report.py`):
    - Line 159 defines the replacement string in the core narrative generation.
    - Lines 308-311 dynamically sanitize subagent report embeds, replacing any prior instance of `"rater selected no match due to subtle cross-scene perceptual drift (expressions, lighting, aging), lower than automatic biometric metrics."` with the replacement text.
- **Verdict: PASS**

---

### ITEM R2: AUDIT LABEL & INDEPENDENCE NOTICE
- **Requirement:**
  1. Inspect Section 9 of `docs/EVAL3_PHASE1_REPORT.md` and `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`.
  2. Check that the generator includes a clear notice stating:
     `"this file was edited by the main agent after the subagent returned, and its independence is NOT established."`
  3. Check git status and git diff on `results/eval3/falsification_audit_subagent_c7.md`: verify that the audit file itself was NOT edited on disk.
- **Textual & File Evidence:**
  - In `docs/EVAL3_PHASE1_REPORT.md` (lines 302-306) and `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md` (lines 302-306):
    ```markdown
    ## 9. Falsification Audit Report (Subagent C7)

    > [!WARNING]
    > **Audit Independence Notice:** This file was edited by the main agent after the subagent returned, and its independence is NOT established. It is preserved here for historical transparency regarding Stage C-Fix checks.

    # FALSIFICATION AUDIT REPORT: STAGE C-FIX AUDIT (Independence NOT Established)
    ```
    The required wording is prominently displayed in callout formatting.
  - In `tools/generate_phase1_report.py`:
    - Lines 304-305 inject the notice:
      `md.append("> **Audit Independence Notice:** This file was edited by the main agent after the subagent returned, and its independence is NOT established. It is preserved here for historical transparency regarding Stage C-Fix checks.")`
    - Line 313 replaces the title with `# FALSIFICATION AUDIT REPORT: STAGE C-FIX AUDIT (Independence NOT Established)`.
  - Git status and git diff inspection of `results/eval3/falsification_audit_subagent_c7.md`:
    - `git status --porcelain results/eval3/falsification_audit_subagent_c7.md` -> Empty output (clean).
    - `git diff HEAD -- results/eval3/falsification_audit_subagent_c7.md` -> Empty output (0 diff lines).
    - Working directory file matches commit `e6abd12b8aabe408f81433e6bce0f09e417d2264` exactly.
    - Verified: The underlying audit file on disk was NOT modified.
- **Verdict: PASS**

---

### ITEM R3: STRESS SCENES VERIFICATION
- **Requirement:**
  1. Check `storyboards/ebright.json`, `storyboards/holdout_v2.json`, `storyboards/eval3_story_c.json`, `results/eval3/PREREG_TEST3.md`, `results/eval3/PREREG_TEST3_AMENDMENT_1.md`, `results/eval3/PREREG_TEST3_AMENDMENT_2.md`, `results/eval3/PREREG_TEST3_CLARIFICATION_1.md:line_26`, and `results/eval3/c3_secondary_metrics_supplement.json`.
  2. Verify whether `Story B Scene 4` is verified as a 3-character stress scene, and whether `Story A Scene 6` and `Story C Scene 5` are "identifiers unverified" because Story A Scene 6 has 1 character and Story C has only 4 scenes total.
  3. Check that Section 4 and Section 8 in the report state this clearly.
- **Textual & File Evidence:**
  - Storyboard ground-truth checks:
    - `storyboards/ebright.json` (Story A): Total 6 scenes. Scene 6 (`scene_6`) specifies `characters: ['college_ebright']` (exactly 1 character).
    - `storyboards/holdout_v2.json` (Story B): Total 4 scenes. Scene 4 (`scene_4_expedition`) specifies `characters: ['clara_thorn', 'elena_child', 'dr_marcus_vance']` (exactly 3 characters -> verified 3-character stress scene).
    - `storyboards/eval3_story_c.json` (Story C): Total 4 scenes (`scene_1_porch`, `scene_2_workshop`, `scene_3_kitchen`, `scene_4_living_room`). Scene 5 does NOT exist.
  - Preregistration & Supplement checks:
    - `results/eval3/PREREG_TEST3_CLARIFICATION_1.md:line_26`:
      `"5. **3-Character Stress Scene (`Story B Scene 4`):** Executed last within each replicate **only** if the 8.0-hour generation cap allows, and reported strictly separately from the 12 primary supported scenes."`
    - `results/eval3/c3_secondary_metrics_supplement.json`:
      `"not_computed_metrics": { "stress_scene": { "status": "NOT COMPUTED", "reason": "3-character stress scenes (Story A Scene 6, Story B Scene 4, Story C Scene 5) were pre-registered to run after primary replicates; execution was halted at 8.029h by the pre-declared 8.0h cap stopping rule." } }`
  - Report Documentation checks (in `docs/EVAL3_PHASE1_REPORT.md` and `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`):
    - **Section 4 (Secondary Pre-Registered Metrics & Performance Telemetry, line 102):**
      `"3. **3-Character Stress Scene:** NOT COMPUTED (OMITTED). Story B Scene 4 is verified as a 3-character stress scene containing characters Clara Thorn, Elena Child, and Dr. Marcus Vance (storyboards/holdout_v2.json and results/eval3/PREREG_TEST3_CLARIFICATION_1.md:line_26). However, other identifiers cited in prior reports (Story A Scene 6 and Story C Scene 5) are identifiers unverified: storyboards/ebright.json Scene 6 specifies only 1 character (college_ebright), and storyboards/eval3_story_c.json contains only 4 scenes total (no Scene 5 exists). Sources read: storyboards/ebright.json, storyboards/holdout_v2.json, storyboards/eval3_story_c.json, results/eval3/PREREG_TEST3.md, results/eval3/PREREG_TEST3_AMENDMENT_1.md, results/eval3/PREREG_TEST3_AMENDMENT_2.md, results/eval3/PREREG_TEST3_CLARIFICATION_1.md:line_26, and results/eval3/c3_secondary_metrics_supplement.json. Pre-registered stress scenes were scheduled to run only if the 8.0-hour cap permitted; generation halted at 8.029h mid-rep_03, so zero stress scenes were rendered."`
    - **Section 8 (Explicit Threats to Validity & Protocol Limitations, item 6, line 297):**
      `"6. **Omitted Stress Scenes:** The 3-character stress scenes were omitted after generation halted at the 8.0-hour cap. Only Story B Scene 4 is verified as a 3-character stress scene in storyboard files; identifiers Story A Scene 6 and Story C Scene 5 are identifiers unverified. Performance on crowded multi-character scenes remains uncharacterized."`
- **Verdict: PASS**

---

### ITEM R4: NUMBER DEFINITIONS & BOUNDARY CORRECTIONS
- **Requirement:**
  In `docs/EVAL3_PHASE1_REPORT.md` and `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`:
  1. Check that mean render time is distinguished between Stage B production run (165.33 s, source: `c3_secondary_metrics_supplement.json`) and clean timing benchmark (117.21 s, source: `dev3_pilot_report.json` and `PREREG_TEST3_AMENDMENT_1.md:138`).
  2. Check that in-loop ensemble score weighted per face (0.7075, source: `test3_summary.json`) is distinguished from averaged per scene (0.7147, source: `test3_per_scene.csv`).
  3. Check that `peak_vram_process_delta_mb` / peak VRAM is defined as whole-system GPU memory sampled at ~1 Hz via nvidia-smi query memory_used_mb peaking at 7979.0 MB, not a process delta.
  4. Check that claims of memory below capacity and zero CUDA OOMs have been removed.
- **Textual & File Evidence:**
  - Render Time Distinction:
    - Line 106: `- **Latency per Scene (Stage B Production Run):** Mean = **165.33 s** (std = 55.25 s, median = 151.31 s, min = 82.95 s, max = 304.88 s). Definition: empirical wall-clock render time per scene across all 108 production scene renders with concurrent system processes and live telemetry logging; source: results/eval3/c3_secondary_metrics_supplement.json#latency_per_scene_seconds.`
    - Line 107: `- **Latency per Scene (Clean Timing Benchmark):** Benchmark Mean = **117.21 s**. Definition: per-scene render time measured under isolated, dedicated benchmark conditions across 12 scenes without background load; source: dev3_pilot_report.json#timing.mean_seconds_per_scene_render and results/eval3/PREREG_TEST3_AMENDMENT_1.md:line_138.`
  - In-Loop Ensemble Score Distinction:
    - Line 95: `- **In-Loop Ensemble Score (Weighted per Face):** Mean = **0.7075**. Definition: composite in-loop quality and identity score averaged across all 171 individual character face crops; source: results/eval3/test3_summary.json#in_loop_metrics_v1_2.mean_ensemble_score.`
    - Line 96: `- **In-Loop Ensemble Score (Averaged per Scene):** Mean = **0.7147**. Definition: unweighted mean of per-scene average in-loop ensemble scores across all 108 generated scenes; source: results/eval3/test3_per_scene.csv column mean_v1_2_ensemble_score_in_loop.`
  - Peak VRAM & OOM Removal:
    - Line 110: `- **VRAM Footprint:** Peak = **7.79 GB** (7979.0 MB), Mean = 4789.0 MB on NVIDIA GeForce RTX 3070 Ti Laptop GPU (8,192 MB physical VRAM; source: results/eval3/c3_secondary_metrics_supplement.json#peak_vram). Definition: peak_vram_process_delta_mb / peak VRAM measures whole-system GPU memory usage sampled at ~1 Hz via nvidia-smi query memory_used_mb in results/telemetry/vram_20261118_121015.csv, not a process-isolated delta. Total GPU memory utilization was recorded by nvidia-smi. Any claim that memory usage remained safely below capacity or that there were zero CUDA OOM errors is removed, as whole-system memory peaked at 7979.0 MB of 8192 MB capacity and process-level memory was not separately isolated.`
    - Verified across the entire file: zero instances claiming "safely below capacity", zero instances claiming "zero CUDA OOMs".
- **Verdict: PASS**

---

### ITEM R5: PM3 WORDING & DECISION RULE FIDELITY
- **Requirement:**
  In Table 1, Key Statistical Takeaways, and footnote:
  1. Check that PM3's nominal chance floor is stated as 0.0000, rendering "above chance" uninformative.
  2. Check that PM3 is presented as an absolute rate (0.9444, 102/108 scenes) with its 95% bootstrap CI [0.8889, 0.9907] and capped t-interval [0.8755, 1.0000].
  3. Check that "above chance" is retained strictly as a direct quote of the pre-registered decision rule.
- **Textual & File Evidence:**
  - Table 1 (lines 14-20):
    - Row for PM3:
      `| **PM3: Scene Correctness Rate** | OUT-OF-LOOP | **0.9444** | [0.8889, 0.9907] | [0.8755, 1.0000]* | 0.0000 | 1.0000 | **"above chance" (registered rule)†** |`
    - Chance floor explicitly given as `0.0000`. Registered verdict explicitly rendered with quotation marks and caveat dagger: `**"above chance" (registered rule)†**`.
  - Table 1 Footnotes (lines 22-25):
    - `† *Note on PM3 chance floor and verdict: PM3 measures scene correctness rate with a pre-registered chance floor of 0.0000. Because the chance floor is 0, testing whether a non-negative rate is "above chance" is uninformative. PM3 is therefore primarily interpreted as an absolute rate of 0.9444 (102/108 scenes) with 95% bootstrap CI [0.8889, 0.9907] and capped Student's t-interval [0.8755, 1.0000]; the phrase "above chance" is retained strictly as a direct quote of the pre-registered decision rule (PREREG_TEST3.md Section 9.2).*`
  - Key Statistical Takeaways (lines 27-33):
    - Item 3:
      `3. **PM3 (Scene Correctness Rate):** Achieved an absolute rate of **0.9444** (102/108 correct scenes; 95% bootstrap CI [0.8889, 0.9907], $t$-interval [0.8755, 1.0000]). Because PM3's nominal pre-registered chance floor is 0.0000, describing it as "above chance" is uninformative; "above chance" is quoted strictly per the registered decision rule (PREREG_TEST3.md Section 9.2). Across all 108 renders, DeepFace face recall was 1.0000 (171/171 expected character face slots detected; 174 total faces detected including 3 extra persons), and extra person rate was 0.0556 (6/108 scenes).`
- **Verdict: PASS**

---

## Overall Audit Verdict

| Audit Item | Focus Area | Status | Result |
| :---: | :--- | :---: | :---: |
| **R1** | Unsupported Statements (Human ratings & Perceptual Drift) | **VERIFIED** | **PASS** |
| **R2** | Audit Label & Independence Disclaimer (`falsification_audit_subagent_c7.md`) | **VERIFIED** | **PASS** |
| **R3** | Stress Scenes Ground-Truth Reconciliation (Story A, B, C) | **VERIFIED** | **PASS** |
| **R4** | Number Definitions & Boundary Corrections (Latency, VRAM, In-Loop Score) | **VERIFIED** | **PASS** |
| **R5** | PM3 Chance Floor & Registered Wording Fidelity | **VERIFIED** | **PASS** |

### **OVERALL AUDIT VERDICT: PASS**
All items R1 through R5 are empirically verified, conform strictly to the preregistration rules and ground-truth storyboards, and are correctly reflected across `docs/EVAL3_PHASE1_REPORT.md`, `D:\5TH SEM\_final\EVAL3_PHASE1_SUMMARY.md`, and `tools/generate_phase1_report.py`. Zero falsifications or unverified discrepancies remain.
