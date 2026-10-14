# PHASE 1 — SOTA FEATURE STATUS MATRIX & RUNTIME MAP

| Method | Code | Model | Extension | Imported | Instantiated | Runtime Called | Reaches Forge | Generates Image | Measured Effect | Visually Verified | Production |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Generic IP-Adapter** | `identity/ipadapter.py` | `ip-adapter_xl.safetensors` | `sd-webui-controlnet` / built-in | Yes | Yes | Yes | Yes | Yes | ArcFace: 0.5203, CLIP-I: 0.8112 | Yes | Yes |
| **FaceID Plus V2** | `identity/faceid.py` | `ip-adapter-faceid-plusv2_sdxl.bin` | `sd-webui-controlnet` / built-in | Yes | Yes | Yes (Patched numpy) | Yes | Yes | ArcFace biometric lock + CLIP-H | Yes | Evaluated |
| **InstantID** | `identity/instantid.py` | `instantid_xl.safetensors` | `sd-webui-controlnet` / built-in | Yes | Yes | Yes (Patched keypoints) | Yes | Yes | Facial keypoint ControlNet guidance | Yes | Evaluated |
| **OpenPose** | `conditioning/controlnet_conditioner.py` | `openpose_xl.safetensors` | `sd-webui-controlnet` | Yes | Yes | Yes | Yes | Yes | Pose structural constraint | Yes | Yes (Adaptive) |
| **Depth** | `conditioning/controlnet_conditioner.py` | `depth_xl.safetensors` | `sd-webui-controlnet` | Yes | Yes | Yes | Yes | Yes | Depth map 3D scene geometry | Yes | Evaluated |
| **Dynamic LoRA** | `conditioning/lora_registry.py` | `sdxl_lightning_8step_lora.safetensors` | Built-in extra networks | Yes | Yes | Yes | Yes | Yes | Prompt injection at runtime | Yes | Evaluated |
| **Regional BREAK** | `story_engine/prompt_builder.py` | SDXL U-Net Native | Built-in syntax | Yes | Yes | Yes | Yes | Yes | Left/Right column prompt isolation | Yes | Yes |
| **Regional Prompter** | N/A (External git blocked) | None on disk | External extension | No | No | No | No | No | Documented alternative: BREAK | N/A | No (Alternative used) |
| **ADetailer** | `conditioning/refinement.py` | None on disk | External extension | No | No | No | No | No | Documented alternative: ReActor+CodeFormer | N/A | No (Alternative used) |
| **ReActor** | `conditioning/refinement.py` | `inswapper_128.onnx` | `sd-webui-reactor` | Yes | Yes | Yes | Yes | Yes | Identity face swap & alignment | Yes | Yes |
| **CodeFormer** | `conditioning/refinement.py` | `codeformer-v0.1.0.pth` | ReActor extension | Yes | Yes | Yes | Yes | Yes | High-frequency cranial texture recovery | Yes | Yes |
| **FreeU** | `generation/generation_orchestrator.py` | SDXL U-Net Native | `sd_forge_freeu` | Yes | Yes | Yes | Yes | Yes | Backbone suppression & skip boost | Yes | Yes |
| **PAG** | `generation/generation_orchestrator.py` | SDXL U-Net Native | `sd_forge_perturbed_attention` | Yes | Yes | Yes | Yes | Yes | Perturbed Attention Guidance | Yes | Yes |
| **CFG-Fix** | `generation/generation_orchestrator.py` | SDXL U-Net Native | `sd_forge_dynamic_thresholding` | Yes | Yes | Yes | Yes | Yes | Dynamic clamping prevents saturation | Yes | Yes |
| **Adaptive Denoising** | `story_engine/scene_planner.py` | SDXL Img2Img | Native scheduler | Yes | Yes | Yes | Yes | Yes | Denoising 0.33 (close) to 0.60 (wide) | Yes | Yes |
| **Visual DNA** | `story_engine/character_registry.py` | Canonical JSON Cache | Native Python engine | Yes | Yes | Yes | Yes | Yes | Canonical source reference embedding | Yes | Yes |
| **Character Registry** | `story_engine/character_registry.py` | `hardcoded_cache.json` | Native Python engine | Yes | Yes | Yes | Yes | Yes | Age-state demographic token resolution | Yes | Yes |
| **CrossFrameConditioner**| `conditioning/crossframe_conditioner.py`| Frame history tracking | Native Python engine | Yes | Yes | Yes | Yes | Yes | Frame history state retention | Yes | Yes |
| **StoryDiffusion / StyleAlign**| `conditioning/crossframe_conditioner.py`| SDXL U-Net Native | `sd_forge_stylealign` | Yes | Yes | Yes | Yes | Yes | Cross-attention query/key sharing | Yes | Evaluated |
| **ArcFace** | `metrics.py` | `buffalo_l/w600k_r50.onnx` | InsightFace | Yes | Yes | Yes | Yes | Yes | 512-D L2 normalized cosine similarity | Yes | Yes (QA) |
| **RetinaFace** | `metrics.py` | `scrfd_10g_bnkps.onnx` | InsightFace | Yes | Yes | Yes | Yes | Yes | High-precision face bounding boxes | Yes | Yes (QA) |
| **CLIP-I / CLIP-Text** | `qa/programmatic_auditor.py` | `clip-vit-base-patch32` | Transformers | Yes | Yes | Yes | Yes | Yes | Semantic image & zero-shot validation | Yes | Yes (QA) |
| **Automated VLM QA** | `qa/vlm_visual_auditor.py` | Multi-Modal Visual QA | Python Vision Auditor | Yes | Yes | Yes | Yes | Yes | Structured visual inspection JSON | Yes | Yes (QA) |
| **HealingController** | `qa/healing_controller.py` | Orchestrator + Auditor | Autonomous Loop | Yes | Yes | Yes | Yes | Yes | Retries with targeted parameter tuning | Yes | Yes |
