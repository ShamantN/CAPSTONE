# Complete Project Implementation Inventory
*Forensic read-only audit of the current codebase capabilities.*

This document catalogs every feature, method, and system observed in the repository as of the latest read-only audit. It distinguishes between active logic currently executing in the `GenerationOrchestrator` pipeline, inactive/fallback code, and components that exhibit runtime failures.

## 1. Entry Points & UI

*   **Gradio Web UI (`app.py`) - Active**
    Main user interface driving generation, auditing, and config management.
*   **Benchmark Runner (`app.py`) - Active**
    Executes batch generation tests for quantitative evaluation.

## 2. Story & Schema Management

*   **Generalized Story Schema (`story_engine/schema.py`) - Active**
    Typed dataclasses (`StorySpec`, `CharacterSpec`, `SceneSpec`, `AgeState`) that act as the single source of truth for narrative inputs.
*   **LLM Story Analyzer (`story_engine/story_analyzer.py`) - Active**
    Calls the Gemini 1.5 Flash API to parse arbitrary text into structured `StorySpec` JSONs.
*   **Heuristic NLP Parser (`story_engine/story_analyzer.py`) - Active (Fallback)**
    Rule-based regex parser that extracts scenes and characters if the LLM API is unavailable.
*   **Character Registry (`story_engine/character_registry.py`) - Active**
    In-memory and on-disk storage of canonical character face anchors (base64) and calculated embeddings.

## 3. Planning & Prompt Engineering

*   **Scene Planner (`story_engine/scene_planner.py`) - Active**
    Transforms a `SceneSpec` into a `SceneExecutionPlan`. Computes the optimal denoising schedule based on `shot_type` (e.g., Close-up vs Wide) and resolves identity anchors.
*   **Prompt Builder (`story_engine/prompt_builder.py`) - Active**
    Constructs the final text prompts. Injects demographic details and formats spatial columns using `BREAK` tokens for regional conditioning.
*   **Regional Conditioner (`conditioning/regional_conditioner.py`) - Active**
    Handles metadata generation for regional spatial column isolation.
*   **Cross-Frame Conditioner (`conditioning/crossframe_conditioner.py`) - Inactive**
    Code exists to record frame history and track state, but it is not actively driving decisions in the `GenerationOrchestrator`.

## 4. Identity Adapters & Conditioners

*   **IP-Adapter SDXL (`identity/ipadapter.py`) - Active**
    Primary SOTA adapter for character injection utilizing `CLIP-ViT-bigG`.
*   **FaceID Plus v2 (`identity/faceid.py`) - Active**
    Available native FaceID integration that can be selected as the identity engine.
*   **InstantID SDXL (`identity/instantid.py`) - Inactive (Runtime Failure)**
    Class and logic exist, but relies on a Forge API preprocessor unpack that currently triggers dict errors at runtime. Acts only as a fallback.
*   **ControlNet Conditioner (`conditioning/controlnet_conditioner.py`) - Active**
    Dynamically creates payloads for OpenPose and Depth maps, injecting them based on `HealingController` directives or user overrides.
*   **LoRA Registry (`conditioning/lora_registry.py`) - Active**
    Scans the local Forge `models/Lora` directory and dynamically enables recognized LoRAs in the prompt.
*   **Refinement Controller (`conditioning/refinement.py`) - Active**
    Builds the deterministic payload for `ReActor + CodeFormer` high-res face restoration used during Pass 2.

## 5. Execution Pipeline

*   **Forge API Client (`generation/forge_client.py`) - Active**
    Provides `txt2img` and `img2img` integration over HTTP with the SD Forge backend.
*   **Generation Orchestrator (`generation/generation_orchestrator.py`) - Active**
    The canonical engine orchestrating the 2-pass deterministic generation. Uses Native U-Net modifications (FreeU, PAG, CFG-Fix), executes Pass 1 (txt2img), and applies Pass 2 (img2img) with adaptive denoising.
*   **VRAM Telemetry Guard (`experiments/memory_utils.py`) - Active**
    Monitors VRAM usage natively and orchestrates pre-generation/post-generation cleanup.

## 6. QA & Self-Healing

*   **Programmatic Auditor (`qa/programmatic_auditor.py`) - Active**
    Evaluates spatial constraints (RetinaFace count), biometric similarity (ArcFace thresholds), and prop presence (CLIP Zero-Shot classification) against the `SceneSpec`.
*   **Healing Controller (`qa/healing_controller.py`) - Active**
    Diagnoses QA failures. If a scene fails spatial checks, it dynamically enables OpenPose and boosts CFG; if biometric checks fail, it alters prompt weighting and triggers a retry via `GenerationOrchestrator`.
*   **Metrics Engine (`metrics.py`) - Active**
    Provides the core computer vision implementations and logic leveraged by the auditor.
