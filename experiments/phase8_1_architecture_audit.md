# Phase 8.1 Architecture and Codebase Audit

**Audit Date:** 2026-08-25  
**Architect:** Principal Generative AI Systems Architect  
**Objective:** Establish what to preserve, refactor, decouple, and build for the generalized character-consistency engine.

---

## 1. Inventory of Current Modules

| Module / Artifact | Current Responsibilities | Status & Architectural Verdict |
| :--- | :--- | :--- |
| `metrics.py` | Image decoding, DeepFace / ArcFace embeddings, SSIM, palette Bhattacharyya distance, SD API wrappers, hardcoded Ebright constants. | **Preserve & Refactor:** Retain robust CV/biometric utility functions; extract hardcoded story constants (`HARDCODED_CHARACTERS`, `HARDCODED_SCENES`) into `storyboards/ebright.json`. |
| `app.py` | Streamlit user interface, regional prompt building, two-pass generation pipeline, synthetic fallback preview, UI metrics render. | **Refactor:** Remove duplicate internal generation pipeline; refactor to call the canonical `GenerationOrchestrator`. Keep Streamlit UI controls. |
| `experiments/memory_utils.py` | Host RAM telemetry via `GlobalMemoryStatusEx`, PyTorch VRAM tracking, `vram_guard()`, `cleanup_memory()`. | **Preserve as Core:** Reusable exception-safe memory management module. |
| `experiments/controlnet_adapter.py` | `ControlSpec` dataclass, mapping to Forge ControlNet API unit dictionaries. | **Preserve & Extend:** Move to `conditioning/controlnet_conditioner.py` and expand with OpenPose and Depth support. |
| `experiments/sota_generator.py` | Standalone generation script with IP-Adapter, FreeU, PAG, CFG-Fix, and ReActor. | **Refactor into Canonical Engine:** Generalize into `generation/generation_orchestrator.py` decoupled from story data. |
| `experiments/sota_qa_auditor.py` | 3-Pillar QA auditor (ArcFace, spatial coordinates, CLIP prop classification). | **Refactor into Schema-Driven Auditor:** Move to `qa/programmatic_auditor.py` where tests are driven by `scene.qa_requirements` rather than hardcoded scene IDs. |
| `download_sota_models.py` | SHA256 verification and discovery of 10 core model assets. | **Preserve:** Keep as automated provisioning entry point. |

---

## 2. Identified Architectural Deficiencies & Redundancies

1. **Parallel Generation Pipelines:**
   - `app.py` (`render_phase3_scene`) and `experiments/sota_generator.py` implement nearly identical two-pass SDXL diffusion logic independently.
   - *Action:* Consolidate into a single canonical engine: `generation/generation_orchestrator.py` and `generation/forge_client.py`.

2. **Hardcoded Story Coupling:**
   - Story characters (Richard Ebright, Mother, James Wong), clothing items (yellow polo, green plaid), environments, and props are hardcoded directly into Python lists and conditional branches in `metrics.py` and `app.py`.
   - *Action:* Move all story content into standalone JSON files (`storyboards/ebright.json`, `storyboards/generalization_benchmark.json`).

3. **Missing Conditioning Abstractions:**
   - Identity conditioning lacks an interface abstraction (`IdentityConditioner`) for swapping between InstantID, FaceID Plus v2, and IP-Adapter SDXL.
   - OpenPose is on disk but was omitted from the runtime payload.
   - Regional conditioning, LoRA management, and cross-frame attention hooks lack dedicated manager classes.

---

## 3. Target Modular Architecture Plan

```text
story_engine/
    schema.py               <- Typed dataclasses (StorySpec, CharacterSpec, AgeState, SceneSpec, PropSpec)
    story_schema.json       <- JSON Schema definition
    story_analyzer.py       <- LLM / regex story parser extracting structured story specs from raw text
    character_registry.py   <- Manages Visual DNA, canonical anchors, and age-state embeddings
    scene_planner.py        <- Resolves cameras, lighting, poses, and conditioning requirements
    prompt_builder.py       <- Builds positive/negative/regional prompts dynamically from specs

identity/
    identity_conditioner.py <- Abstract base class for identity conditioning
    ipadapter.py            <- IP-Adapter SDXL cross-attention implementation
    instantid.py            <- InstantID adapter with keypoint conditioning
    faceid.py               <- FaceID Plus v2 adapter with embedding injection

conditioning/
    controlnet_conditioner.py <- OpenPose, Depth, and preprocessor pipeline
    regional_conditioner.py   <- Column/grid spatial latent isolation & BREAK tokenization
    lora_registry.py          <- Dynamic LoRA management and weighting
    refinement.py             <- Secondary high-resolution facial restoration (ReActor / CodeFormer / Inpaint)
    crossframe_conditioner.py <- Cross-frame anchor and attention consistency hooks

generation/
    forge_client.py           <- Robust HTTP client for SD Forge WebUI API
    generation_orchestrator.py<- Canonical single-point execution pipeline

qa/
    programmatic_auditor.py   <- Story-driven automated QA matrix (ArcFace, CLIP, SSIM, Palette)
    vlm_auditor.py            <- Multimodal visual QA checklist and verification engine
    healing_controller.py     <- Autonomous failure classifier and parameter tuner (max 3 retries)

storyboards/
    ebright.json                  <- Decoupled Benchmark A story data
    generalization_benchmark.json <- Independent Benchmark B story data
```
