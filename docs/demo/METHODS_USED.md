# MMCCP v2 Browser Showcase — Methods & Configuration Used

> **DEMO ONLY — NOT A HELD-OUT EVALUATION (`TEST-2` UNTOUCHED)**  
> All anchors and scenes in this showcase run were generated from scratch on DEV seeds (`4101–4199` for anchors, `4201–4299` for scenes) after archiving pre-existing local runtime state. Held-out `TEST-2`, quarantined seeds `{606, 707, 808}`, `storyboards/holdout_v2.json`, and Frozen Verifier `v1.2` were left untouched. Personal workstation paths have been omitted for public portability.

## Execution Environment
- **Base Checkpoint:** `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors [c9e3e68f89]`
- **Forge Flags:** `--api --xformers --cuda-malloc` (NVIDIA RTX 3070 Ti Laptop GPU, `8192` MiB VRAM)
- **Active `ConsistencyV2Config` Hash:** `73d4c8d9a3d8` *(selected after Scene 1 seed `4211` A/B check confirmed `enable_crop_upscale_swap=True` eliminated swap seams)*

## Method Matrix (`M1 – M11` & Auxiliary Controls)

| Method / Setting | Status | Parameter Values | Rationale / Note |
| :--- | :--- | :--- | :--- |
| **M1: Identity Bank** | **ON** | `enable_identity_bank=True`, `bank_selection_metric="single_anchor"`, `1` curated studio anchor / char | Initialized from scratch via `anchors/final/*.png` after visual selection + 4-point gate. |
| **M2: Layout Planner (OpenPose + Boxes)** | **ON** | `enable_layout_planner=True`, `openpose_model="openpose_xl [f4251cb4]"`, `openpose_weight=0.70`, `openpose_guidance_end=0.75` | Enforces distinct left/right multi-character silhouettes and framing. |
| **M3: Regional Conditioning (Forge Couple)** | **ON** | `enable_regional_couple=True`, `regional_global_weight=0.40` | Isolates per-character prompts spatially in 2-character scenes. |
| **M4: Dual InstantID** | **OFF** | `enable_dual_instantid=False` | Disabled per locked v2 preset to prevent spatial contention with OpenPose. |
| **M4: IP-Adapter Plus-Face SDXL** | **ON** | `enable_ip_adapter_plus=True`, `ip-adapter-plus-face_sdxl_vit-h [368cf551]`, `ip_adapter_plus_weight=0.55` | Injects studio anchor facial geometry into Pass 1 diffusion. |
| **M5: Reliable Face Transplant** | **ON** | `enable_reliable_face_swap=True`, `swap_backend="direct_insightface"`, `inswapper_128.onnx`, `codeformer_fidelity=0.85`, `enable_crop_upscale_swap=True` | Padded `2x` Lanczos crop alignment + `inswapper_128` + CodeFormer blend. |
| **M6: Identity Face Detailer (ADetailer)** | **ON** | `enable_face_detailer=True`, `face_detailer_denoise=0.28`, `face_detailer_steps=20`, `face_detailer_padding=48` | Refines skin texture and eye sharpness after face swap. |
| **M7: Wardrobe & Palette Lock (CIELAB)** | **ON** | `enable_wardrobe_lock=True`, `wardrobe_delta_e_threshold=38.0`, `wardrobe_repair_denoise=0.45`, **`blend_strength=0.70` (code-locked)** | Repaints drifting garment colors; `blend_strength=0.70` is hardcoded in `consistency_v2/pipeline_v2.py`. |
| **M8: Reference-Only Attention** | **OFF** | `enable_reference_attention=False` | Disabled to avoid white-studio background bleed and extra VRAM overhead. |
| **M9: Best-of-N Candidate Selection** | **OFF in config / External `N=3`** | `enable_best_of_n=False` in `ConsistencyV2Config`; helper script rendered `N=3` DEV seeds/scene | `ConsistencyPipelineV2.generate_scene()` does not execute internal multi-candidate loops. |
| **M10: Identity Memory (Drift Guard)** | **OFF** | `enable_identity_memory=False` | Disabled during demo showcase so scene crops do not mutate studio anchors. |
| **M11: Character LoRA Training** | **OFF** | `use_character_lora=False` | Out of scope for zero-cache inference. |
| **M12: Collage-Init** | **OFF** | `enable_collage_init=False` | Disabled in default showcase configuration. |
| **Sampling / CFG / Aux** | **ON** | `mode="default"`, `steps_pass1=20`, `steps_pass2=15`, `cfg_scale=5.5`, `DPM++ 2M Karras`, `enable_pag=False`, `enable_freeu=True`, `enable_dynamic_thresholding=True` | Standard default sampling schedule. |

## Scope Exclusions
- **Excluded Scene:** Scene 6 (`Graduation`) is excluded to keep the demo scoped to `1–2` character panels without static graduation gown wardrobe conflicts.
- **No `v1` Comparison:** Only `v2` outputs are included (`DEMO ONLY`).

