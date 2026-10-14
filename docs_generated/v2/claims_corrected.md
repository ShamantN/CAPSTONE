# MMCCP Consistency v2: Corrected Claims & Architectural Truths

**Project:** Multi-Modal Character Consistency Pipeline (MMCCP)  
**Capstone Project ID:** PW26_CGM_02  
**PES University Phase 3**  
**Date:** October 14, 2026  
**Auditor:** Autonomous Senior Generative-AI & QA Lead  

---

## Executive Summary

During Phase 0 and Phase 2 pre-flight verification, an exhaustive cross-reference between the original documentation (`docs/`, `docs_generated/00`–`09`) and the actual executable source code (`generation/`, `conditioning/`, `identity/`, `story_engine/`, and Forge runtime logs) was conducted.

Several substantial contradictions, architectural overstatements, and silent execution defects were discovered. In accordance with **Rule 1 (Truth Over Polish)**, this document records the ground-truth state of the codebase, refuting invalid claims with empirical, byte-level and AST-verified evidence.

---

## 1. Discrepancy Matrix: Documentation vs. Code Execution

| # | Topic | Documentation Claim (v1 Docs) | Ground-Truth Code / Runtime Reality | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | **Diffusion Sampler & Scheduler** | Stated as "Euler a" or "DPM++ 2M SDE Karras" in various sections. | Strictly hardcoded to `"DPM++ 2M"` with `"Karras"` scheduler in both Pass 1 and Pass 2. | `generation_orchestrator.py` L119–120, L157–158; Forge API payload logs. |
| **C2** | **Diffusion Step Counts** | Claimed as 30 steps (Pass 1) + 20 steps (Pass 2) in narrative reports. | Code defaults to `steps_pass1=20` and `steps_pass2=15`. Fast mode uses 14 + 10. | `run_from_scratch.py` L42–43; `app.py` UI sliders. |
| **C3** | **Medium-Shot Denoising Strength** | Claimed as `0.45` or `0.35–0.45` across deep-dive documentation. | In `scene_planner.py`, medium-shot denoising is calculated as `round(0.48 * 1.05, 3) = 0.504` (~0.50). | `scene_planner.py` L45–46. |
| **C4** | **Pass 1 ControlNet Depth Execution (D1)** | Claimed that ControlNet Depth Anything v2 was active and shaping Pass 1 geometry. | `create_depth_spec()` called with `depth_image_b64=None`. Forge dropped the unit. Outputs A03 and A04 are byte-identical. | Image SHA256 `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2` identical for both stages. |
| **C5** | **ReActor Face Restoration (D2)** | Claimed ReActor systematically restored facial fidelity across all characters. | Argument array in `refinement.py` had 30 items instead of 31, and passed integer `0` for string dropdowns. ArcFace dropped from 0.5048 to 0.4784. | Forge `/sdapi/v1/script-info` dump vs `refinement.py` L21–52; ablation CSV line 6 vs 7. |
| **C6** | **Spatial Isolation / Regional Conditioning (D3)** | Claimed "Regional spatial conditioning isolates attention across character columns". | Only concatenated character prompts with SD `BREAK` keyword. No spatial cross-attention masks applied. Bleed occurred in Scene 1. | `regional_conditioner.py` L18–20; contact sheet `scene_1_seed_11_contact.png` showing adult female face on Young Ebright. |
| **C7** | **InstantID Implementation (D4)** | Claimed "Full InstantID dual-stream conditioning active". | Only landmark keypoint unit was loaded (`instantid_xl.safetensors`). Essential IP-Adapter embedding unit (`ip-adapter.bin`) was missing from disk. | `C:\Stable-Diffusion\webui\models\ControlNet` listing; `identity/instantid.py` L35–45. |
| **C8** | **Evaluation Objectivity (D5a)** | Claimed independent biometric certification of generated faces. | Evaluator relied on Buffalo_L / InsightFace embeddings—the exact same model family used in conditioning/swapping. Circular confirmation bias. | `metrics.py` vs `consistency_v2/verify/biometric_ensemble.py` (introducing Facenet512, SFace, GhostFaceNet judges). |
| **C9** | **Multi-Character Binding (D5b)** | Claimed Hungarian matching proved character identity preservation. | Hungarian matching operated purely on biometric distance matrix, completely ignoring spatial coordinates and wardrobe colors. | `metrics.py` L180–220; Scene 1 smoke render passed Hungarian matching while Young Ebright had adult woman's body and blue dress. |

---

## 2. Incontrovertible Empirical Evidence

### 2.1 The ControlNet Inactivity Proof (Defect D1)
In `scratch/run_systematic_ablation.py`, stage `A03_latent_enhancers` and stage `A04_controlnet_depth` were generated sequentially:
```powershell
Get-FileHash "docs_generated\raw\ablation_images\A03_latent_enhancers.png", "docs_generated\raw\ablation_images\A04_controlnet_depth.png"
```
**Result:**
- `A03_latent_enhancers.png`: `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`
- `A04_controlnet_depth.png`:    `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`

Because the payload constructor in `conditioning/controlnet_conditioner.py` passed `image_b64=None`, Forge discarded the ControlNet unit during `txt2img`, generating the exact same image down to the individual pixel.

### 2.2 The ReActor Degradation Proof (Defect D2)
In `docs_generated/raw/ablation_study_results.csv`:
- `A05_pass2_adaptive_img2img` (without ReActor): ArcFace = **0.5048**
- `A06_reactor_codeformer` (with ReActor): ArcFace = **0.4784** (-0.0264 regression!)

An inspection of Forge's `/sdapi/v1/script-info` confirmed:
- Expected argument count: **31**
- `refinement.py` argument count: **30** (missing argument 30: `"tab_single"`)
- Expected argument types:
  - Arg 13 (`Console Log Level`): Choice string `['No log', 'Minimum', 'Default']` (Code passed `0`)
  - Arg 14 (`Gender Detection Source`): Choice string `['No', 'Female Only', 'Male Only']` (Code passed `0`)
  - Arg 15 (`Gender Detection Target`): Choice string `['No', 'Female Only', 'Male Only']` (Code passed `0`)
  - Arg 22 (`Select Source`): Choice string `['Image(s)', 'Face Model', 'Folder']` (Code passed `0`)

Because Gradio / WebUI scripts unpack positional arguments, passing type-mismatched and truncated argument tuples caused ReActor to either misroute execution or produce suboptimal face swaps.

---

## 3. Required Pipeline v2 Corrections

1. **True ControlNet Pre-conditioning (M2):** Layout Planner generates synthesized OpenPose skeleton images and spatial region masks with explicit coordinates so ControlNet receives valid, non-null control images.
2. **Schema-Compliant ReActor & Direct InsightFace Swap (M5):** Dynamically inspect `/sdapi/v1/script-info` to construct the exact 31-argument array with valid string choice tokens, and provide a direct InsightFace/CodeFormer fallback (`consistency_v2/face_swap.py`).
3. **True Regional Conditioning via Forge Couple (M3):** Install and drive the `sd-forge-couple` extension using API region coordinates, confining character attention to explicit bounding masks.
4. **Complete InstantID Dual-Unit Conditioning (M4):** Procure `ip-adapter.bin` from `InstantX/InstantID` to pair the cross-attention IP-Adapter unit with the keypoint ControlNet unit.
5. **Frozen Multi-Modal Verifier (Phase 1):** Evaluate identity using an ensemble of 4 non-conditioning models (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`), enforce horizontal spatial binding, and verify wardrobe color in CIELAB space.
