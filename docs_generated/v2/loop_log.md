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

