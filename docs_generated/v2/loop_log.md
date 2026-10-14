# MMCCP Consistency v2: Engineering Loop Log

This document records every hypothesis-driven iteration across methods M1 through M10 following the strict non-negotiable Loop Protocol.

---

## Iteration: loop-M1-iter1
- **Phase:** Phase 3 (M1: Identity Bank)
- **Branch:** `feat/m1-identity-bank`
- **Target Failure Mode:** Single-anchor overfitting and lack of multi-view pose embeddings causing viewpoint bias and identity collapse.
- **Hypothesis:** Multi-reference identity banks with 4-point admission gate (single face, min size 96px, sharpness, genuine cosine similarity) and L2-normalized centroid embeddings will establish robust reference identity representations across varying viewpoints.
- **Change:** Implemented `consistency_v2/identity_bank.py` with `IdentityBankManager`, `ConsistencyV2Config` with seed scope guard (blocking {606, 707, 808}), and seeded all 5 project characters (`young_ebright`, `teen_ebright`, `college_ebright`, `mother`, `james_wong`).
- **Test:** Unit tests in `tests/test_identity_bank.py` verifying admission gate, pose clustering, CIELAB extraction, and bank centroid persistence.
- **Results:** 6/6 tests pass (0.07s). All 5 character banks initialized with verified embeddings.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR #5 (commit `31afaf8`).

---

## Iteration: loop-M2-iter1
- **Phase:** Phase 3 (M2: Layout Planner & D1 Fix)
- **Branch:** `feat/m2-layout-planner`
- **Target Failure Mode:** D1 (ControlNet units inactive due to null input images) and cross-scene character position drift.
- **Hypothesis:** Programmatic layout planning from storyboard shot types and character positions will generate deterministic spatial bounding boxes, minimum guaranteed face scale ($\ge 96$ px for medium shots), and valid OpenPose COCO-18 skeletons, fixing D1 and enforcing character position binding.
- **Change:** Implemented `consistency_v2/layout_planner.py` with `LayoutPlanner`, `CharacterLayoutBox`, and COCO-18 skeleton rendering. Updated `conditioning/controlnet_conditioner.py` to require valid input images and support pre-rendered skeleton control images with `module="None"`.
- **Test:** Unit tests in `tests/test_layout_planner.py` verifying non-overlapping regions, scale parameters, and skeleton synthesis.
- **Results:** 3/3 tests pass (0.013s). Proved skeleton synthesis places distinct character bodies at planned horizontal positions.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR #6 (commit `32eea5d`).

---

## Iteration: loop-M3-iter1
- **Phase:** Phase 3 (M3: True Regional Conditioning)
- **Branch:** `feat/m3-regional-couple`
- **Target Failure Mode:** D3 (Weak `BREAK` token failing to isolate cross-attention, causing multi-character phenotype and wardrobe bleed).
- **Hypothesis:** Utilizing the Forge Couple extension with horizontal column tiling and background first-line blending will spatially partition self-attention during U-Net denoising, eliminating cross-character prompt bleed while maintaining scene background cohesion.
- **Change:**
  1. Installed `sd-forge-couple` (commit `c7884e81623d7fcbf4c92e3aea14ced8b5b6aa74`) in Forge WebUI extensions.
  2. Implemented `consistency_v2/regional.py` with `RegionalCoupleDriver` and `RegionalCoupleConfig`, formatting 17-element Forge Couple API payloads with horizontal ordering and background blending.
  3. Fixed capitalized `"None"` preprocessor module in `conditioning/controlnet_conditioner.py` to prevent Forge ControlNet `AttributeError`.
  4. Added unit tests in `tests/test_regional.py`.
- **Screen on DEV-FAST:**
  - Evaluated on Scene 1 (two characters: young Ebright on left, mother on right) with seed 11.
  - Runtime: 27.62s. Peak VRAM: 7310 MB ($\le 7.4$ GB budget). Zero OOM.
  - Visual inspection of `scratch/test_m2_m3_combined.png` demonstrates complete attention isolation: young boy on left in white t-shirt and denim shorts; mother on right in yellow floral dress. Zero wardrobe or demographic bleed.
- **Test:** 4/4 unit tests in `tests/test_regional.py` pass (0.000s). All 13 suite tests pass.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR #7 (commit `18c5190`).

---

## Iteration: loop-M4-iter1
- **Phase:** Phase 3 (M4: Identity Units Done Right)
- **Branch:** `feat/m4-identity-units`
- **Target Failure Mode:** D4 (Incomplete InstantID dual-unit architecture) and weak facial identity transfer under single-anchor setups.
- **Hypothesis:** Employing IP-Adapter Plus-Face SDXL (`ip-adapter-plus-face_sdxl_vit-h [368cf551]`) driven by Identity Bank reference images, combined with calibrated dual-unit InstantID (`InsightFace (InstantID)` + `instant_id_face_keypoints`), will significantly elevate facial biometric similarity without background copying or pose deformation.
- **Change:**
  1. Implemented `consistency_v2/identity_units.py` with `IdentityUnitManager` and `IdentityUnitConfig`.
  2. Supported automatic viewpoint-aligned reference image selection from `IdentityBankManager`.
  3. Built dual-unit InstantID configuration fixing D4 (`ip-adapter_instant_id_sdxl.bin` + `control_instant_id_sdxl.safetensors`).
  4. Added unit tests in `tests/test_identity_units.py`.
- **Screen on DEV-FAST:**
  - Evaluated on Young Ebright reading scene portrait (`scratch/test_plus_face_output.png`).
  - Runtime: 26.17s. Peak VRAM: ~7.1 GB ($\le 7.4$ GB budget). Zero OOM.
  - Biometric Ensemble similarity to bank centroid: **0.7390** (+19.4% relative gain over baseline v1 DEV-FULL mean of 0.6190).
- **Test:** 4/4 unit tests in `tests/test_identity_units.py` pass (0.47s). All 17 suite tests pass.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR #9 (commit `96035a7`).

---

## Iteration: loop-M5-iter1
- **Phase:** Phase 3 (M5: Reliable Identity Transplant)
- **Branch:** `feat/m5-face-transplant`
- **Target Failure Mode:** D2 (ReActor payload argument array mismatch, dropping face swaps and reducing ArcFace from 0.5048 to 0.4784).
- **Hypothesis:**
  1. Reconstructing the 31-argument schema for ReActor (correct string dropdown values `"Minimum"`, `"No"`, `"Image(s)"`, and mandatory 31st arg `"tab_single"`) fixes the Forge API contract.
  2. Direct InsightFace `inswapper_128.onnx` execution on CPU in Forge's python bound to `LayoutPlanner` spatial zones, paired with CodeFormer face restoration via Forge Extras API, reliably transplants identity without VRAM overhead.
- **Change:**
  1. Implemented `consistency_v2/face_swap.py` with `ReActorPayloadBuilder` and `DirectFaceSwapper`.
  2. Built spatial zone-aware face assignment (binding detected target faces to character anchors based on layout coordinates).
  3. Integrated CodeFormer restoration via `/sdapi/v1/extra-single-image`.
  4. Added unit tests in `tests/test_face_swap.py`.
- **Screen on DEV-FAST:**
  - Evaluated on Scene 1 two-character render (`scratch/test_swapped_restored.png`).
  - Runtime: ~8s for CPU swap + 2s for CodeFormer. Zero GPU VRAM allocated.
  - Frozen Biometric Ensemble scores: Young Ebright = **0.6218**, Mother = **0.7888** (mean across faces = **0.7053**).
  - Seam and artifact inspection: Zero boundary edges, zero color shift.
- **Test:** 2/2 unit tests in `tests/test_face_swap.py` pass. All 19 suite tests pass.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR #10 (commit `fe592e7`).

---

## Iteration: loop-M6-iter1
- **Phase:** Phase 3 (M6: Identity-Conditioned Face Detailer)
- **Branch:** `feat/m6-face-detailer`
- **Target Failure Mode:** Small faces (<120 px across) in medium and wide shots suffer from latent resolution starvation in SDXL's 128x128 latent space, degrading facial feature fidelity.
- **Hypothesis:** Localized high-resolution inpainting with Gaussian-blurred boundary masks (`inpaint_full_res=True`, padding 48px, denoise 0.35, 20 steps) conditioned on character identity units (IP-Adapter Plus-Face) upscales and refines the facial region, with post-detail M5 identity transplant and CodeFormer restoration providing high-fidelity facial locking without edge seams.
- **Change:**
  1. Implemented `consistency_v2/face_detailer.py` with `FaceDetailer` and `FaceDetailerConfig`.
  2. Integrated spatial layout matching (`match_face_to_layout_box`) to bind detected faces to intended characters.
  3. Built elliptical face and hair mask generator with configurable Gaussian blur (radius 21).
  4. Coupled post-detail identity transplant via `DirectFaceSwapper`.
  5. Added unit tests in `tests/test_face_detailer.py`.
- **Screen on DEV-FAST:**
  - Evaluated on Scene 1 Young Ebright face crop (106 px width).
  - Runtime: 20.95s for inpainting pass.
  - Face sharpness: 368.73 Laplacian variance (sharp, no blurriness).
  - Seamless boundary blending verified visually (`scratch/test_detailed_output.png`).
- **Test:** 3/3 unit tests in `tests/test_face_detailer.py` pass (0.014s). All 22 suite tests pass.
- **Decision:** ACCEPT. Merged into `consistency-v2` via PR.
