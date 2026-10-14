# MMCCP Consistency v2: Pipeline Diagnostics & Defect Proofs (D1–D6)

**Project:** Multi-Modal Character Consistency Pipeline (MMCCP)  
**Capstone Project ID:** PW26_CGM_02  
**Target:** 8 GB VRAM SDXL Character Consistency Architecture  
**Date:** October 14, 2026  
**Auditor:** Autonomous Senior Generative-AI & QA Lead  

---

## Executive Summary

Phase 2 focuses on rigorously diagnosing, mathematically proving, and empirically refuting or confirming the six suspected defects (D1 through D6) present in the baseline v1 architecture.

Every defect analyzed below contains:
1. **The Theoretical Failure Mode:** Root-cause explanation based on Stable Diffusion WebUI Forge source mechanics and SDXL attention behavior.
2. **The Incontrovertible Empirical Proof:** Exact file paths, AST analysis, SHA-256 byte hashes, or API payload telemetry dumps.
3. **The Proposed v2 Remediation:** Architectural design to eliminate the defect in Consistency v2.

---

## Defect D1: Pass 1 ControlNet Inactivity & Null Inputs

### 1. Hypothesis & Mechanism
In baseline v1, documentation claimed that ControlNet Depth Anything v2 (`depth_xl`) and OpenPose (`openpose_xl`) were actively guiding the structural draft generation during Pass 1 (`txt2img`). 

However, in Stable Diffusion WebUI Forge:
- For an `alwayson_scripts["controlnet"]` unit to execute during `txt2img`, an `input_image` (base64 string) must be supplied to the preprocessor module (e.g. `depth_anything_v2` or `dw_openpose_full`).
- If `"image": None` is supplied in `txt2img`, ControlNet has no reference image from which to extract depth maps or human skeleton keypoints. Consequently, Forge silently drops the unit from the conditioning graph and executes plain unconditioned diffusion.

### 2. Empirical Verification
- **Code Inspection:**
  In `generation/generation_orchestrator.py` (lines 86–94):
  ```python
  if enable_openpose:
      op_spec = ControlNetConditioner.create_openpose_spec()
      if op_spec:
          control_specs.append(op_spec)
  if enable_depth:
      depth_spec = ControlNetConditioner.create_depth_spec()
      if depth_spec:
          control_specs.append(depth_spec)
  ```
  In `conditioning/controlnet_conditioner.py` (lines 44, 60):
  ```python
  @classmethod
  def create_openpose_spec(cls, pose_image_b64: Optional[str] = None, weight: float = 0.70):
      ...
  @classmethod
  def create_depth_spec(cls, depth_image_b64: Optional[str] = None, weight: float = 0.65):
      ...
  ```
  Both functions were called with zero arguments, setting `image_b64=None`.

- **Ablation SHA-256 Byte Collision:**
  In the historical 10-stage systematic ablation on Scene 2 (seed 42):
  - Stage `A03_latent_enhancers`: Latent enhancers only.
  - Stage `A04_controlnet_depth`: Latent enhancers + Pass 1 ControlNet Depth.
  
  Cryptographic hash comparison:
  ```powershell
  Get-FileHash "docs_generated\raw\ablation_images\A03_latent_enhancers.png", "docs_generated\raw\ablation_images\A04_controlnet_depth.png"
  ```
  **Result:**
  - `A03_latent_enhancers.png`: `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`
  - `A04_controlnet_depth.png`:    `32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`

  The output PNG files are **100% byte-identical**, proving with absolute certainty that ControlNet Depth had zero effect in Pass 1.

### 3. Verdict & Remediation (M2)
- **Verdict:** **CONFIRMED DEFECT.** Pass 1 ControlNet was completely inert.
- **Remediation:** Implement `LayoutPlanner` (M2) to programmatically render OpenPose COCO-18 human skeletons with exact facial keypoints and spatial region masks. ControlNet receives pre-drawn control images with preprocessor `none`, guaranteeing direct geometric enforcement.

---

## Defect D2: ReActor Argument Array Mismatch & Identity Regression

### 1. Hypothesis & Mechanism
Baseline v1 attempted to apply ReActor face swapping during Pass 2 (`img2img`). However, empirical results showed that enabling ReActor caused ArcFace identity similarity to *decrease* from `0.5048` (Stage A05) down to `0.4784` (Stage A06).

Gradio / Forge extensions unpack script arguments positionally (`*args`). Any discrepancy in argument length, ordering, or type definitions (such as passing integer `0` where a string dropdown choice is expected) causes the script to abort or misroute parameters.

### 2. Empirical Verification
- **Forge Script-Info Schema vs Code Payload:**
  Forge's `/sdapi/v1/script-info` returns an array of **31 arguments** for `reactor`:
  ```json
  [
    {"label": "Single Source Image", "default": null},
    {"label": "ReActor", "default": false},
    {"label": "Comma separated face number(s); Example: 0,2,1", "default": "0"},
    {"label": "Comma separated face number(s); Example: 1,0,2", "default": "0"},
    {"label": "Model", "choices": ["inswapper_128.onnx"]},
    {"label": "Restore Face", "choices": ["None", "CodeFormer", "GFPGAN"]},
    {"label": "Restore Face Visibility", "default": 1.0},
    {"label": "1. Restore Face -> 2. Upscale", "default": true},
    {"label": "Upscaler", "choices": [...]},
    {"label": "Scale by", "default": 1.0},
    {"label": "Upscaler Visibility", "default": 1.0},
    {"label": "Swap in source image", "default": false},
    {"label": "Swap in generated image", "default": true},
    {"label": "Console Log Level", "choices": ["No log", "Minimum", "Default"]},
    {"label": "Gender Detection (Source)", "choices": ["No", "Female Only", "Male Only"]},
    {"label": "Gender Detection (Target)", "choices": ["No", "Female Only", "Male Only"]},
    {"label": "Save Original", "default": false},
    {"label": "CodeFormer Weight (Fidelity)", "default": 0.5},
    {"label": "Source Image Hash Check", "default": true},
    {"label": "Target Image Hash Check", "default": false},
    {"label": "Execution Provider", "choices": ["CPU", "CUDA"]},
    {"label": "Face Mask Correction", "default": false},
    {"label": "Select Source", "choices": ["Image(s)", "Face Model", "Folder"]},
    {"label": "Choose Face Model", "choices": ["None"]},
    {"label": "Source Folder", "default": ""},
    {"label": "Multiple Source Images", "default": null},
    {"label": "Random Image", "default": false},
    {"label": "Force Upscale", "default": false},
    {"label": "Threshold", "default": 0.5},
    {"label": "Max Faces", "default": 0},
    {"label": "", "default": "tab_single"}
  ]
  ```
  In `conditioning/refinement.py` (lines 21–52):
  - Total arguments passed: **30** (missing argument 30: `"tab_single"`).
  - Type collisions:
    - Arg 13 passed integer `0` instead of string `"Minimum"`.
    - Arg 14 passed integer `0` instead of string `"No"`.
    - Arg 15 passed integer `0` instead of string `"No"`.
    - Arg 22 passed integer `0` instead of string `"Image(s)"`.
    - Arg 23 passed empty string `""` instead of `None`.

  These positional misalignments prevented ReActor from reliably executing face swaps.

### 3. Verdict & Remediation (M5)
- **Verdict:** **CONFIRMED DEFECT.** ReActor payload was malformed and mismatched against Forge's API contract.
- **Remediation:**
  1. Build a dynamic, schema-validated ReActor payload builder (`consistency_v2/face_swap.py`) that strictly adheres to the 31-argument contract.
  2. Implement direct, deterministic InsightFace face swapping (`M5b`) using `inswapper_128.onnx` executed via Forge's Python environment with blended centroid embeddings, completely decoupling face restoration from WebUI extension fragility.

---

## Defect D3: Token Chunk Bleed via SD `BREAK` Syntax

### 1. Hypothesis & Mechanism
Baseline v1 claimed to implement "Regional Column Conditioning" to isolate multiple characters appearing in the same storyboard scene (e.g., Young Ebright and Mother in Scene 1).

In reality, `conditioning/regional_conditioner.py` only joined character prompts with the standard Stable Diffusion keyword `" BREAK "`.
- In Stable Diffusion (AUTOMATIC1111/Forge), `BREAK` does **not** perform spatial attention masking.
- `BREAK` merely pads the current 77-token CLIP embedding chunk with blank tokens and starts a new 77-token chunk.
- Cross-attention layers in the UNet still attend to ALL tokens globally across all spatial latents $(H \times W)$.
- Consequently, attributes from character A (e.g., Mother: blonde hair, yellow blouse) bleed directly into the latents of character B (e.g., Young Ebright: brown hair, striped polo).

### 2. Empirical Verification
- **Code Inspection:**
  In `conditioning/regional_conditioner.py`:
  ```python
  @staticmethod
  def format_column_prompt(prompts_per_column: List[str]) -> str:
      return " BREAK ".join(prompts_per_column)
  ```
  The returned string was simply passed to `txt2img_payload["prompt"]`. The column metadata from `get_regional_matrix()` was completely unused by any extension or script.

- **Scene 1 Smoke Contact Sheet Evidence:**
  Inspection of `docs_generated/v2/visual_review/contact_sheets/scene_1_seed_11_contact.png`:
  - Expected: Left = Young Ebright (boy, short brown hair, striped polo), Right = Mother (adult woman, blonde hair, yellow blouse).
  - Actual Render:
    - Left figure rendered as an adult female face wearing a blue floral blouse ($\Delta E^* = 60.6$ wardrobe error, catastrophic binding failure).
    - Right figure rendered with young blonde hair in a yellow shirt ($\Delta E^* = 20.7$).
  - Prompt tokens completely cross-contaminated both spatial regions.

### 3. Verdict & Remediation (M3)
- **Verdict:** **CONFIRMED DEFECT.** `BREAK` token splitting provides zero spatial isolation.
- **Remediation:** Install and drive `sd-forge-couple` extension (`M3`) to partition cross-attention maps into strictly bounded spatial sub-regions with independent positive prompts, combined with M2 spatial layout masks.

---

## Defect D4: InstantID Missing Weights & Single-Unit Configuration

### 1. Hypothesis & Mechanism
InstantID requires two distinct conditioning streams to achieve identity preservation:
1. **IP-Adapter Facial Embedding Unit:** Uses the InsightFace 512-dim embedding to guide the UNet cross-attention layers (`InsightFace (InstantID)` module with `ip-adapter.bin`).
2. **ControlNet Landmark Keypoint Unit:** Uses 5 facial keypoints to constrain spatial facial geometry (`instant_id_face_keypoints` module with `control_instant_id_sdxl.safetensors`).

If only the ControlNet keypoint model is used, the diffusion model receives facial pose guidance but zero biometric identity features.

### 2. Empirical Verification
- **Disk Model Inventory:**
  In `C:\Stable-Diffusion\webui\models\ControlNet`:
  - `instantid_xl.safetensors` (2.5 GB) was present.
  - `ip-adapter.bin` (~1.69 GB) was **completely absent**.
- **Code Inspection:**
  In `identity/instantid.py`:
  Only one ControlSpec was generated (`module="instant_id_face_keypoints"`, `model="instantid_xl [c5c25a50]"`). The IP-Adapter embedding unit was never initialized.

### 3. Verdict & Remediation (M4)
- **Verdict:** **CONFIRMED DEFECT.** InstantID was running with only half of its required dual-unit architecture.
- **Remediation:** Procure `ip-adapter.bin` from `InstantX/InstantID` via Hugging Face Hub into `C:\Stable-Diffusion\webui\models\ControlNet\ip-adapter_instant_id_sdxl.bin`. In `consistency_v2/identity_units.py`, instantiate both units in tandem.

---

## Defect D5: Evaluator Circularity & Binding Blindness

### 1. Hypothesis & Mechanism
The baseline v1 evaluation suite had two structural flaws:
1. **Model Circularity (D5a):** Evaluator used InsightFace Buffalo_L embeddings to score similarity to anchors that were themselves conditioned and swapped using InsightFace Buffalo_L.
2. **Spatial Binding Blindness (D5b):** Hungarian matching matched detected faces to character specifications purely by minimizing the biometric distance matrix. It had zero awareness of whether a face was located on the left or right, or whether the body wearing the face had the correct clothing.

### 2. Empirical Verification
- In Scene 1 smoke render: Hungarian matching matched the left face to Young Ebright with biometric score `0.4631`, declaring the scene partially valid—even though the person was visibly an adult woman wearing a blue blouse.
- In `metrics.py`: The evaluation code computed `np.linalg.norm(emb1 - emb2)` with zero spatial coordinate verification or semantic body segmentation.

### 3. Verdict & Remediation (Phase 1)
- **Verdict:** **CONFIRMED DEFECT.**
- **Remediation:** Addressed and frozen in Phase 1 (`verifier-frozen` tag):
  - Biometric Ensemble of 4 independent CPU judges (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`).
  - Spatial binding enforcement verifying detected face centroids against storyboard layout zones.
  - Semantic person and clothing segmentation via SegFormer with CIELAB color matching ($\Delta E^* \le 38.0$).

---

## Defect D6: Documentation Contradictions & Hyperbole

### 1. Hypothesis & Mechanism
Significant divergences between reported parameters and runtime values existed across `docs/` and `docs_generated/`.

### 2. Empirical Verification
- Documented in detail in `docs_generated/v2/claims_corrected.md`.
- Primary discrepancies verified:
  - Sampler: DPM++ 2M Karras (code) vs Euler a / SDE (docs).
  - Steps: 20+15 (code) vs 30+20 (docs).
  - Medium-shot denoise: 0.50 (code) vs 0.45 (docs).
  - CodeFormer weight: 0.80 (code) vs 0.60 (docs).

### 3. Verdict & Remediation
- **Verdict:** **CONFIRMED DEFECT.**
- **Remediation:** Corrected in `claims_corrected.md`. All future reports must derive strictly from code execution and verified telemetry.
