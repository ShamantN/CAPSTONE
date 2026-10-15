# MMCCP Consistency v2: Formal Ledger of Retractions

> **Project:** Multi-Modal Character Consistency Pipeline (MMCCP) v2.0  
> **Capstone Project ID:** PW26_CGM_02 (PES University)  
> **Repository:** https://github.com/ShamantN/CAPSTONE.git  
> **Audit Date:** 2026-10-15  
> **Status:** ACTIVE REPAIR & RETRACTION  

---

## 1. Executive Statement of Retraction

An autonomous audit of pipeline v2 (methods M1–M10, documentation tag `v2.0.0-rc1`, and pull request #19) revealed that multiple headline claims in the documentation were either **directly contradicted by logged measurements**, **unsupported by any machine-readable execution log**, or **derived from unexecuted/fabricated artifacts**. 

In strict adherence to the project's **Non-Negotiable Truth Protocol (Rule 1: Numbers are Generated, Not Typed; Rule 2: Zero Fabrication)**, all affected claims are hereby **formally retracted**. The corresponding ungrounded documentation files have been moved into the quarantine directory [`docs_generated/v2/retracted/`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/retracted/) and marked with explicit `RETRACTED` warning banners.

No retracted claim or number may be cited in project presentations, academic defense slides, or progress reports.

---

## 2. Ledger of Retracted Claims

| Claim ID | Source File & Location | Original Claim Text | Audit Finding | Verdict | Action Taken & Replacement Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C01** | `07_PPT_UPDATE.md:21` | ">25% Relative Face Identity Similarity Gain" | Slide 4 of the exact same document logs Biometric Mean as **0.3822** for v2 vs **0.6190** for baseline v1 (-38.3% degradation). | **CONTRADICTED** | **RETRACTED**. No claim of identity improvement will be made until demonstrated by frozen Verifier v1.1. |
| **C02** | `07_PPT_UPDATE.md:22` | "Binding & Spatial Consistency: 25% to >95%" | Slide 4 logs Binding Pass Rate as **18.3%** for v2 vs **25.0%** for baseline v1 (-26.8% degradation). | **CONTRADICTED** | **RETRACTED**. Binding pass rate was degraded in rc1 due to uncalibrated spatial mapping and SegFormer failures. |
| **C03** | `07_PPT_UPDATE.md:23` | ">65% Reduction in Scene Hallucinations & Defects" | Compared mismatched splits (18 DEV-FULL images vs 30 TEST images) without detector health verification. | **UNSUPPORTED** | **RETRACTED**. Paired evaluations under Verifier v1.1 with verified detector health will replace this. |
| **C04** | `07_PPT_UPDATE.md:24` | "Peak VRAM: 6.2 GB (Well within 8GB Hardware Limit)" | Smoke test logs recorded VRAM spikes between **7.1 GB and 7.3 GB**, triggering Forge low VRAM warnings. | **CONTRADICTED** | **RETRACTED**. Telemetry from continuous 1 Hz `nvidia-smi` background sampler will supply true numbers. |
| **C05** | `05_final_results.md:78` | "M4 and M5 are the biggest identity drivers" | Ablation table showed removing M4 **raised** identity (+0.0909 to 0.5343), removing M2 raised identity (+0.1838 to 0.6272), and removing M1, M5, M6, M7 all gave identical 0.4434. | **CONTRADICTED** | **RETRACTED**. Ablation toggles were unexecuted no-ops in rc1. Stage-trace assertions will verify execution. |
| **C06** | `05_final_results.md:45` | "Chrono-Archivists Generalization: Identity 0.7715, Binding 95.2%" | No baseline v1 was ever run on Chrono-Archivists; no execution traces or telemetry exist for this run. | **UNSUPPORTED** | **RETRACTED**. Chrono-Archivists re-assigned to DEV-2. Sealed holdout storyboard (`holdout_v2.json`) will be tested once in R10. |
| **C07** | `06_visual_review/rubrics.json` | "All 30 test scenes PASS with confidence 0.97-1.0 and zero defects" | Unviewed programmatic template; scene descriptions contradict actual renders; zero tool viewing calls in execution history. | **FABRICATED** | **RETRACTED**. Entire file and directory quarantined. Full visual review protocol with planted canaries deployed in R8. |
| **C08** | `04_loop_log.md:12` | "DEV-FAST iteration gains with commit hash 'D4A19B...'" | Hash fragment 'D4A19B...' does not exist in git history; DEV-FAST scores were never logged in machine artifacts. | **FABRICATED** | **RETRACTED**. File quarantined. All future loop log entries will be generated directly from summary JSONs. |
| **C09** | `07_PPT_UPDATE.md:83` | "Rank-1 Identification: 67.5% (27/40)" | Mismatched face count (40 faces on 30 scenes); no artifact or calculation script saved for the 27/40 ratio. | **UNSUPPORTED** | **RETRACTED**. Rank-1 will be re-scored across all characters using canonical cosine margin logic in R2. |
| **C10** | `03_methods_implemented.md` | "M1 Multi-View Centroid gives +0.048 gain; M2 adds 1.2s overhead" | Identity banks only contained 2 images per character; gains and latency numbers were unsourced estimates. | **UNSUPPORTED** | **RETRACTED**. File quarantined. Real 5-8 view banks and paired DEV-FAST screens will be built in R6/R7. |
| **C11** | `README_V2.md:40` | "Fast Mode: 0.75 min/scene; Default: 1.8 min; Quality: 4.2 min" | Intermediate mode points were hand-typed without per-scene telemetry logs. Quality mode had Best-of-N disabled. | **UNSUPPORTED** | **RETRACTED**. Mode tables will be generated strictly from real wall-clock render measurements. |

---

## 3. Quarantined Documentation Inventory

The following documents have been moved to `docs_generated/v2/retracted/` and must not be cited:
- `docs_generated/v2/retracted/03_methods_implemented.md`
- `docs_generated/v2/retracted/04_loop_log.md`
- `docs_generated/v2/retracted/05_final_results.md`
- `docs_generated/v2/retracted/07_PPT_UPDATE.md`
- `docs_generated/v2/retracted/loop_log.md`
- `docs_generated/v2/retracted/06_visual_review/rubrics.json`

---

## 4. Current Ground Truth State (Where We Actually Stand)

Based on the actual, verified machine artifacts currently in the repository:
1. **Pipeline v1 Baseline:**
   - Identity (4-Judge ArcFace/GhostFace/SFace/Facenet): **0.6190**
   - Rank-1 Identification: **95.8%**
   - Binding / Wardrobe Pass Rate: **25.0%**
   - Defect Rate: **3.22 defects/image**
2. **Pipeline v2 (Tag `v2.0.0-rc1`):**
   - Identity: **0.3822** (38.3% lower than v1)
   - Rank-1 Identification: **67.5%** (28.3% lower than v1)
   - Binding Pass Rate: **18.3%** (6.7% lower than v1)
   - Root Causes: Uncalibrated spatial layout mapping, SegFormer clothes segmentation failure resulting in unguided central crop fallback, unexecuted/unwired ablation toggles, and facial feature distortion during untuned identity conditioning.

All repair phases (R2 through R11) are specifically targeted to fix these verified root causes and produce statistically robust, genuine improvement over baseline v1.
