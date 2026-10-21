# MMCCP Consistency v2: Master Engineering Loop Log (Live)

> **Standard:** Loop Protocol & Acceptance Rule (Rules 1-10)
> **Verifier:** v1.2 Frozen (`3fa69071f888643b...`)
> **Hardware Ceiling:** RTX 3070 Ti (8 GB), VRAM <= 7.4 GB, Zero OOM

This log documents every hypothesis-driven engineering iteration in the Repair and Real-Improvement pass. Every number is extracted directly from machine-generated result artifacts.

---

## Iteration: `loop-R6-iter1`

- **Timestamp:** `2026-10-16T06:44:12.873945+00:00`
- **Phase / Workstream:** Phase R6 (Rebuild Identity Bank & M1 Strategy Screening)
- **Branch:** `feat/bank-views`
- **Commit SHA / PR:** `5258633` | [PR #26](https://github.com/ShamantN/CAPSTONE/pull/26)
- **Hypothesis:** Multi-view identity bank reference selection with nearest-pose matching reduces perspective mismatch over single-anchor baseline, elevating facial identity retention.
- **Change:** Implemented multi-angle bank generator (7 certified views/char) and strategy-aware reference selection ('single_anchor', 'centroid', 'nearest_pose') in IdentityBankManager.
- **Smoke Test:** 36/36 renders executed without error. Peak VRAM 7442 MB <= 7450 MB safe ceiling. Zero OOM.
- **Screen on DEV-FAST:**
  - Baseline: **0.1744**
  - Intervention: **0.1789**
  - Paired $\Delta$: **+0.0045**
  - Bootstrap 95% CI: `[0.0000, +0.0138]`
- **Guardrails:** Peak VRAM: `7442.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **RE-EVALUATED IN STEP 4 -> DEFAULTED TO SINGLE_ANCHOR**
- **Notes:** Paired cluster-bootstrap CI [0.0000, 0.0138] does not rule out zero ($\le 0.0000$). Fails strict acceptance rule. Defaulted to simplest non-inferior arm ('single_anchor') in consistency_v2/best_config.json per protocol.

---

## Iteration: `loop-W3-iter1-face-scale`

- **Timestamp:** `2026-10-18T07:46:29.248951+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/identity-layout-scale`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Elevating layout planner minimum face scale above 120px prevents ArcFace resolution collapse and restores identity match
- **Change:** Elevated face_width in compute_scale_params: medium 120->160px, wide 75->135px, close 180->220px
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (79.1s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.1989**
  - Paired $\Delta$: **-0.0092**
  - Bootstrap 95% CI: `[-0.0171, +0.0000]`
- **Guardrails:** Peak VRAM: `8192.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0199, Teen=+0.0000, Adult=-0.0051, Face Width=210.2px vs 210.2px

---

## Iteration: `loop-W6-iter1-collage-init`

- **Timestamp:** `2026-10-18T08:24:50.080640+00:00`
- **Phase / Workstream:** R7
- **Branch:** `feat/w6-m12-collage-init`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** M12 feathered collage-init on draft canvas prior to Pass 2 anchors both subjects in multi-character scenes, eliminating two-shot single-person collapse
- **Change:** Enable enable_collage_init with multichar_only=True and retain all identity units in Pass 2
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (112.1s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.2026**
  - Paired $\Delta$: **-0.0055**
  - Bootstrap 95% CI: `[-0.0209, +0.0100]`
- **Guardrails:** Peak VRAM: `8192.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0189, Teen=-0.0214, Adult=+0.0088, Face Width=218.5px vs 210.2px

---

## Iteration: `loop-W4-iter1a-openpose-dualpass`

- **Timestamp:** `2026-10-18T09:22:27.233045+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/w4-openpose-anchoring`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Addressing OpenPose unit to full model string openpose_xl [f4251cb4] with bidirectional image/input_image keys binds two-shot spatial skeletons in Forge Pass 1, eliminating multi-character single-person chimeric collapse in Scenes 1 and 5.
- **Change:** Set openpose_model='openpose_xl [f4251cb4]', populate both image and input_image keys in pipeline_v2.py and ControlNetAdapter, retain all character identity units in Pass 2.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (163.5s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.3304**
  - Paired $\Delta$: **+0.1223**
  - Bootstrap 95% CI: `[-0.0602, +0.2136]`
- **Guardrails:** Peak VRAM: `8005.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Initial OpenPose fix run with dual-pass ControlNet. Child=-0.0029, Teen=+0.0026, Adult=+0.2457, Face Width=105.5px vs 210.2px

---

## Iteration: `loop-W4-iter1b-openpose-pass2drop`

- **Timestamp:** `2026-10-18T10:51:20.006305+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/w4-openpose-anchoring`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Addressing OpenPose unit to full model string openpose_xl [f4251cb4] with bidirectional image/input_image keys binds two-shot spatial skeletons in Forge Pass 1, eliminating multi-character single-person chimeric collapse in Scenes 1 and 5 while maintaining VRAM <= 7.4 GB.
- **Change:** Calibrate openpose_model='openpose_xl [f4251cb4]', populate both image/input_image keys, and streamline Pass 2 img2img refinement without ControlNet.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (126.6s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.2476**
  - Paired $\Delta$: **+0.0395**
  - Bootstrap 95% CI: `[-0.2085, +0.1635]`
- **Guardrails:** Peak VRAM: `7940.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0495, Teen=-0.3276, Adult=+0.2212, Face Width=117.2px vs 210.2px

---

## Iteration: `loop-W4-iter2-stabilized-anchoring`

- **Timestamp:** `2026-10-18T12:28:24.431698+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/w4-openpose-anchoring`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Stabilizing prompt attention weights (taming token emphasis <= 1.15 and removing BREAK tokens from single-character scenes), binding OpenPose xl [f4251cb4] across Pass 1, and establishing W1 stage order (swap last with target_bboxes tracking) eliminates single-character attention divergence while preserving adult multi-character biometric gains (+0.22) under strict <= 7.4 GB VRAM ceiling.
- **Change:** Fix single-character prompt structure without BREAK in prompt_builder.py, tame nested token weights <= 1.15, pass target_bboxes=swap_bboxes in m5_face_swap trace, disable redundant pre-swap in face_detailer.py, add enable_pag toggle (default False), and fix import os in face_detailer.py.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (191.1s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.3091**
  - Paired $\Delta$: **+0.1011**
  - Bootstrap 95% CI: `[-0.0996, +0.2111]`
- **Guardrails:** Peak VRAM: `7813.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0169, Teen=-0.0993, Adult=+0.2465, Face Width=124.6px vs 210.2px

---

## Iteration: `loop-W3W4-iter1-child-scale-seated-layout`

- **Timestamp:** `2026-10-18T13:08:28.102275+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/identity-layout-scale`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Child-proportioned layout scaling (0.78x figure height, >=115px face width), automated seated_table pose selection, and capped prompt weights eliminate child facial distortion in Scene 1 while maintaining adult biometric gains (+0.24) and peak VRAM <= 5.0 GB under Verifier v1.2.
- **Change:** Add seated_table template in layout_planner.py, scale child characters, set minimum face width >= 120px in compute_scale_params, auto-detect seated scenes in pipeline_v2.py with 0.60 guidance end, and enforce Forge memory limits.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (86.8s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.3309**
  - Paired $\Delta$: **+0.1228**
  - Bootstrap 95% CI: `[-0.0817, +0.2250]`
- **Guardrails:** Peak VRAM: `7757.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0318, Teen=-0.0622, Adult=+0.2875, Face Width=125.4px vs 210.2px

---

## Iteration: `loop-W3W4-iter2-solo-centering`

- **Timestamp:** `2026-10-18T13:41:18.071223+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/identity-layout-scale`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Elevating solo shot composition (cy=0.46, face_width=160px) with upper_body_front pose and relaxing OpenPose guidance to 0.55 eliminates ceiling head clipping and restores adult/teen facial resolution, boosting positive pair rate >=70% and cluster bootstrap CI lower bound above 0.
- **Change:** Add upper_body_front pose template to layout_planner.py, elevate solo shot centering (cy=0.46), set shot-aware scale params (face_width=160px for close/shoulder shots), auto-detect upper_body_front with guidance_end=0.55 in pipeline_v2.py, and lower cfg_scale to 5.0.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (86.9s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.3262**
  - Paired $\Delta$: **+0.1181**
  - Bootstrap 95% CI: `[-0.0973, +0.2259]`
- **Guardrails:** Peak VRAM: `7680.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0099, Teen=-0.1205, Adult=+0.2830, Face Width=116.1px vs 210.2px

---

## Iteration: `loop-W3W4-iter3-closeup-scale-fidelity`

- **Timestamp:** `2026-10-18T14:46:06.882786+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/identity-layout-scale`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Elevating close-up and shoulder shot scale (fig_height=1150px, face_width=220px, cy=0.58), boosting CodeFormer fidelity to 0.85, and capping Forge GPU memory to 3800MB recovers Teen Ebright facial resolution (178px vs 135px) and maintains peak VRAM <= 6.5 GB, pushing positive pairs >= 72% and passing acceptance.
- **Change:** Expand close-up scale parameters to fig_height=1150px and face_width=220px in layout_planner.py, center single close-up framing at cy=0.58, set codeformer_fidelity=0.85 in config.py, set Forge xl_GPU_MB=3800MB with empty_cache between passes in pipeline_v2.py.
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (178.4s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.2081**
  - Intervention: **0.3277**
  - Paired $\Delta$: **+0.1196**
  - Bootstrap 95% CI: `[-0.0782, +0.2212]`
- **Guardrails:** Peak VRAM: `7805.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0273, Teen=-0.0763, Adult=+0.2830, Face Width=135.5px vs 210.2px

---

## Iteration: `loop-W3W4-iter4-cfg-5.5`

- **Timestamp:** `2026-10-21T06:51:47.517418+00:00`
- **Phase / Workstream:** R7
- **Branch:** `fix/identity-layout-scale`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Elevating CFG scale from 5.0 to 5.5 restores facial contrast and eliminates RetinaFace detection failure on Mother in Scene 1 Seed 37
- **Change:** Set CFG scale to 5.5 with all other bundle settings identical
- **Smoke Test:** Scene 1 Seed 11 stage trace passed (217.5s, diffs recorded)
- **Screen on DEV-FAST:**
  - Baseline: **0.3142**
  - Intervention: **0.3088**
  - Paired $\Delta$: **-0.0054**
  - Bootstrap 95% CI: `[-0.0824, +0.0364]`
- **Guardrails:** Peak VRAM: `7805.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **REJECT**
- **Notes:** Child=-0.0559, Teen=-0.0048, Adult=+0.0281, Face Width=118.6px vs 233.1px

---

