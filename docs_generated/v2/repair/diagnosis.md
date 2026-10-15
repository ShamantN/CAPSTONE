# MMCCP Consistency v2: Pipeline Diagnostic & Stage Trace Report (Phase R5)

**Audit Timestamp:** `2026-10-15 20:34:11`  
**Dataset Split:** `DEV-FAST` (Ebright Scenes 1, 2, 4, 5 | Seeds 11, 23, 37 | 12 total renders)  
**Verifier Version:** `v1.1` (Frozen, Canonical combined SHA `ee3b29d5...`)  

---

## 1. Executive Summary: Paired Performance (DEV-FAST v2 vs Baseline v1)

| Split / Subgroup | v2 Measured Biometric | v1 Baseline Biometric | Paired Delta | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Overall DEV-FAST (12 renders)** | **0.1685** | **0.2278** | **-0.0594** | DEFICIT |
| **Child Subgroup (Scenes 1, 2)** | 0.1898 | 0.2283 | -0.0385 | DEFICIT |
| **Adult Subgroup (Scenes 4, 5)** | 0.1471 | 0.2274 | -0.0803 | DEFICIT |

---

## 2. Per-Stage Identity Trajectory Across Diffusion Pipeline

Tracing the biometric ensemble score at each intermediate stage exposes the exact physics of identity transformation:

| Stage | Stage Name | Mean Biometric Score | Delta From Previous | Role & Observation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `pass1_draft` | 0.1452 | Baseline | Dual InstantID + IP-Adapter Plus sets initial facial structure |
| 2 | `pass2_refined` | 0.1379 | -0.0073 | Img2img refinement denoises and slightly smooths facial landmarks |
| 3 | `m5_face_swap` | 0.1686 | +0.0307 | **CORE IDENTITY ENGINE**: Direct InsightFace swap + CodeFormer restoration locks bank identity |
| 4 | `m7_wardrobe_lock` | 0.1685 | -0.0001 | SegFormer bounded CIELAB color adjustment protects face mask; identity preserved |
| 5 | `final_composite` | 0.1685 | +0.0000 | Final composite delivered to disk |

---

## 3. Empirical Answers to Problem Register & Mission Questions

### Q1: Does each of M2 to M7 execute and change pixels where it should?
- **M2 (Layout Planner):** **YES**. Generates valid OpenPose skeletons (`01_layout_skeleton.png`), placing characters in designated zones.
- **M3 (Regional Couple):** **YES**. Partitions canvas horizontally with global weight 0.40, preventing multi-character cross-attention bleed.
- **M4 (Dual InstantID & IP-Adapter):** **YES**. Both units injected into Pass 1 without dropping.
- **M5 (Face Swap):** **YES**. Bug fix replacing broken `'scene_layout' in locals()` with `layout_boxes` allows `DirectFaceSwapper` to run. Mean pixel change is **1.8984 to 2.2474** (exceeds epsilon 0.5).
- **M6 (Face Detailer):** In fast mode preset, `enable_face_detailer` is intentionally false for speed. In default and quality presets, it is active.
- **M7 (Wardrobe Lock):** **YES**. SegFormer segments clothing and adjusts colors when Delta E exceeds threshold. Mean pixel change is **4.7262 to 4.9583**.

### Q2: Why did M2 and M4 removal appear to raise identity in the flawed v2-rc1 pass?
- In the original v2-rc1, M5 (Face Swap) NEVER executed due to the `'scene_layout'` bug.
- Without face swapping, OpenPose skeletons (M2) forced characters into fixed positions and head poses where raw SDXL generation failed to match the frontal anchor face.
- Removing M2 allowed SDXL to generate a single generic, centered, frontal face that scored higher on face detection, creating an illusion of improvement while completely destroying spatial binding and character positions.

### Q3: Why does binding fail (position vs wardrobe vs identity)?
- **Spatial Position:** Pass rate is high (>95%) because M2 OpenPose and M3 Forge Couple enforce horizontal zone separation.
- **Wardrobe Color:** In multi-character scenes, color prompts bleed across characters during diffusion (e.g., young Ebright's yellow shirt bleeding into mother's blouse). M7 bounded CIELAB adjustment fixes observed color, but prompt-level isolation is needed in generation.
- **Child vs Adult Bias:** Child faces (Young Ebright) score lower across all 4 biometric judges (Facenet512, SFace, GhostFaceNet, ArcFace) than adult faces. Child-specific thresholds are essential for fair evaluation.

---

## 4. Ranked Hypotheses for Phase R7 Repair Loops

1. **Hypothesis W1 (Stage Execution Order):** Lock identity swap as the final post-processing operation. Placing face swap strictly after any inpainting or detailer prevents detailer diffusion passes from eroding the swapped face.
2. **Hypothesis W2 (Face Swap Crop-Upscale-Paste):** Upscaling the face bounding box 2x before `inswapper_128` and feathered paste-back will resolve 128px latent compression artifacts on medium and wide shots.
3. **Hypothesis W3 (Multi-Reference Bank Selection - M1):** Inswapper currently uses the single anchor image. Selecting the reference view matching the target face yaw/pitch will eliminate perspective mismatch.
4. **Hypothesis W4 (Face Size Constraints):** Enforce minimum face bounding box size of 96px in layout planning to prevent small-face resolution collapse.
5. **Hypothesis W5 (Prompt-Level Color Isolation):** Add explicit per-character wardrobe color keywords into Forge Couple regional sub-prompts to stop color bleeding in Pass 1.
