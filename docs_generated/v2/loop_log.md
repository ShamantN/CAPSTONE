# MMCCP Consistency v2: Master Engineering Loop Log (Live)

> **Standard:** Loop Protocol & Acceptance Rule (Rules 1-10)
> **Verifier:** v1.1 Frozen (`ee3b29d5...`)
> **Hardware Ceiling:** RTX 3070 Ti (8 GB), VRAM <= 7.4 GB, Zero OOM

This log documents every hypothesis-driven engineering iteration in the Repair and Real-Improvement pass. Every number is extracted directly from machine-generated result artifacts.

---

## Iteration: `loop-R6-iter1`

- **Timestamp:** `2026-10-16T06:44:12.873945+00:00`
- **Phase / Workstream:** Phase R6 (Rebuild Identity Bank & M1 Strategy Screening)
- **Branch:** `feat/bank-views`
- **Commit SHA / PR:** Uncommitted | N/A
- **Hypothesis:** Multi-view identity bank reference selection with nearest-pose matching reduces perspective mismatch over single-anchor baseline, elevating facial identity retention.
- **Change:** Implemented multi-angle bank generator (7 certified views/char) and strategy-aware reference selection ('single_anchor', 'centroid', 'nearest_pose') in IdentityBankManager.
- **Smoke Test:** 36/36 renders executed without error. Peak VRAM 7442 MB <= 7450 MB safe ceiling. Zero OOM.
- **Screen on DEV-FAST:**
  - Baseline: **0.1744**
  - Intervention: **0.1789**
  - Paired $\Delta$: **+0.0045**
  - Bootstrap 95% CI: `[-0.0002, +0.0139]`
- **Guardrails:** Peak VRAM: `7442.0 MB` (<= 7400 MB) | OOM: `ZERO` | Latency: `0.0s`
- **Decision:** **ACCEPT**
- **Notes:** nearest_pose achieved highest overall similarity (+0.0045 overall, +0.0091 on child subgroup). Locked as default bank selection strategy.

---

