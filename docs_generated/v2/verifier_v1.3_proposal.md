# Proposal for Verifier v1.3 (Tools / Evaluation Protocol Enhancement)

> **Document Status:** PROPOSAL / SHADOW EVALUATION PROTOCOL (Freeze v1.2 Untouched)  
> **Target Verifier Version:** v1.3-proposal  
> **Current Frozen Verifier:** v1.2 Canonical SHA `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6` (STRICTLY FROZEN & UNMODIFIED)  
> **Scope:** Defines evaluation improvements for shadow reporting in `tools/` without altering frozen verifier code.

---

## Executive Summary

External reviewer technical analysis of the MMCCP v2 handoff pack identified three critical measurement gaps in Verifier v1.2:
1. **Tautological Spatial Positioning:** `assign_faces_to_characters` matches detected faces to character slots based on horizontal coordinate prominence ($r_x$). Consequently, `verify_spatial_position` tests the assigned face against the very zone used to assign it, resulting in a 100% pass rate even if characters' physical faces are deliberately swapped.
2. **Pathological 0% Identity-at-Position:** In Verifier v1.2 `cli.py`, `id_pos_pass` requires `bio_res["ensemble_passed"] == True`, demanding that $\ge 3$ of the 4 judge models pass their individual absolute cosine similarity thresholds. Because SFace (mean 0.11 vs threshold 0.24), GhostFaceNet (mean 0.02 vs threshold 0.18), and ArcFace (mean 0.13 vs threshold 0.25) almost never pass on diffusion crops, `ensemble_passed` is False on 100% of images, collapsing `identity_at_position_pass` and `combined_binding_pass` to 0.0% across all baseline and intervention runs.
3. **Blindness to Hallucinated Figures:** `scene_auditor.py` evaluated extra people using `num_faces > expected_faces`. Hallucinated headless figures, turned-away bodies, or masked dolls (such as the third figure in Scene 1 Seed 11) have no detected face and were completely unflagged.
4. **Hyperactive Zero-Shot Prop False Positives:** CLIP open-vocabulary probes fire spurious defect violations on clean period imagery (e.g. "beard on child" on chin shadows, "magic sphere" on laboratory glassware).

This document formalizes the **Verifier v1.3 Proposal** to address these issues symmetrically across v1 and v2, enabling honest shadow reporting alongside frozen v1.2.

---

## 1. Specification of Proposed Changes

### 1.1 Threshold-Free Identity-at-Position

#### Problem in v1.2
Absolute cosine thresholds calibrated on pristine high-res crops fail on generated diffusion images due to checkpoint stylistic divergence, even when identity is unambiguously discriminable.

#### Proposed v1.3 Formalization
Identity-at-position is reformulated as a relative discrimination metric combined with physical spatial compliance:

$$\text{IdAtPosPass}(C_k) = \mathbb{I}\left[ \text{Rank1Correct}(C_k) \land (\text{Margin}(C_k) \ge \tau_{\text{margin}}) \land \text{SpatialZonePass}(C_k) \right]$$

Where:
- $\text{Rank1Correct}(C_k)$: The candidate face crop assigned to character $C_k$ achieves the highest ensemble cosine similarity to character $C_k$'s anchor compared to all other character anchors in the story:
  $$\arg\max_{c \in \mathcal{C}} \text{Sim}(e_{\text{face}}, e_{\text{anchor}}(c)) = C_k$$
- $\text{Margin}(C_k)$: The discrimination margin between the correct character anchor and the closest impostor anchor:
  $$\text{Margin}(C_k) = \text{Sim}(e_{\text{face}}, e_{\text{anchor}}(C_k)) - \max_{c \ne C_k} \text{Sim}(e_{\text{face}}, e_{\text{anchor}}(c))$$
- $\tau_{\text{margin}}$: Calibrated minimum separation margin (set to $+0.020$ based on held-out negative splits).
- $\text{SpatialZonePass}(C_k)$: Independent physical zone validation ($r_x \le 0.48$ for left, $r_x \ge 0.52$ for right, $0.28 \le r_x \le 0.72$ for center).

---

### 1.2 Non-Tautological Face Assignment & Spatial Verification

#### Problem in v1.2
```python
# Tautological: faces assigned by zone, then checked for zone
assigned_faces = assign_faces_to_characters(detected_faces, expected_chars, positions)
spatial_pass = verify_spatial_position(assigned_faces[cid], expected_pos) # ALWAYS TRUE
```

#### Proposed v1.3 Formalization
Face-to-character assignment must decouple **biometric identity matching** from **spatial position evaluation**:
1. **Bipartite Identity-First Matching:** Calculate the full $M \times N$ similarity matrix between all $M$ detected faces and all $N$ expected character anchors. Solve maximum weight bipartite matching (Hungarian algorithm) using biometric ensemble similarity as the edge weight.
2. **Spatial Position Test on Identity-Matched Faces:** Once Face $i$ is assigned to Character $C_k$ based strictly on *who looks most like $C_k$*, evaluate whether Face $i$'s bounding box center $r_x$ satisfies Character $C_k$'s expected position zone.
3. **Empirical Verification:** When faces are swapped in `tools/test_spatial_tautology.py`, the Hungarian matcher assigns the right face to Ebright and the left face to Mother. `verify_spatial_position` then correctly evaluates:
   - Ebright (on right): expected left $\rightarrow$ **FAIL** (detected swap!)
   - Mother (on left): expected right $\rightarrow$ **FAIL** (detected swap!)
   This completely resolves Defect D1 tautology.

---

### 1.3 Structural Extra-Person Detection

#### Problem in v1.2
Verifier checked `len(detected_faces) > expected_faces`. Hallucinations without full faces were invisible.

#### Proposed v1.3 Formalization
Cross-check YOLO person detections ($C=0$) against expected characters and face detections:

$$\text{ExtraPersonsCount} = \max\left(0, N_{\text{person}}^{\text{YOLO}} - N_{\text{expected}}\right)$$
$$\text{ExtraUnfacedCount} = \max\left(0, N_{\text{person}}^{\text{YOLO}} - N_{\text{faces}}^{\text{RetinaFace}}\right)$$

Rules:
1. If $N_{\text{person}}^{\text{YOLO}} > N_{\text{expected}}$, flag structural defect `extra_people` with count $N_{\text{person}}^{\text{YOLO}} - N_{\text{expected}}$.
2. If $N_{\text{person}}^{\text{YOLO}} > N_{\text{faces}}^{\text{RetinaFace}}$ and $N_{\text{person}}^{\text{YOLO}} > N_{\text{expected}}$, flag structural defect `hallucinated_unfaced_figure` (Severity: HIGH).
3. Evaluated on Scene 1 Seed 11: caught with confidence $0.841$ at coordinates $[694, 1, 1040, 504]$.

---

### 1.4 Prop Probe Redesign (Eliminating Spurious False Positives)

#### Problem in v1.2
Single positive text prompt zero-shot CLIP queries trigger high false positive rates:
- Clean child portraits fire "beard on child" due to chin shadows.
- Scientific glass jars fire "magic sphere".
- Measured FP rate across 30 labeled ground truth historical portraits: **63.3%**.

#### Proposed v1.3 Formalization
1. **Contrastive Prompt Pairs:** Replace open-ended queries with normalized softmax contrast pairs:
   $$P(\text{beard}) = \frac{\exp(\text{CLIP}(\text{crop}, \text{"a child with an adult beard and mustache"}) / \tau)}{\exp(\text{CLIP}(\text{crop}, \text{"a clean-shaven child with smooth skin"}) / \tau) + \exp(\text{CLIP}(\text{crop}, \text{"a child with an adult beard and mustache"}) / \tau)}$$
2. **Dual-Threshold Confirmation:** Require $P(\text{defect}) \ge 0.75$ and absolute cosine similarity $\ge 0.26$.
3. **Calibration Requirement:** Validated against 30 clean labeled period reference images to ensure $\text{FPR} \le 3.3\%$ (at most 1 false alarm per 30 images).

---

## 2. Shadow Evaluation Tool Architecture

All proposed improvements are implemented strictly in `tools/` as shadow analysis scripts:
- `tools/extra_person_detector.py`: Structural YOLO person cross-checker.
- `tools/test_spatial_tautology.py`: Empirical verification of spatial swap detection.
- `tools/cross_scene_coherence.py`: Cross-scene genuine-to-impostor ratio evaluator.
- `tools/swap_sanity_checker.py`: InsightFace native vs independent judge correlation tester.
- `tools/shadow_eval_v1_3.py`: Full shadow scoring of DEV-2 under both v1.2 and proposed v1.3 definitions.

**Frozen Verifier v1.2 SHA `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6` remains completely frozen and unmodified.**
Both v1.2 numbers and proposed v1.3 numbers are presented side-by-side in all final reporting.
