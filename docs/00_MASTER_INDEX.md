# MMCCP Technical Documentation Wiki: Master Index & System Directory

Welcome to the definitive engineering manual and forensic architectural documentation for the **Multi-Modal Character Consistency Pipeline (MMCCP)** (Project: *Novel to Video Generation (NVG)*, Capstone Phase 3, PES University).

This documentation suite serves as an exhaustive, textbook-grade technical reference covering the underlying mathematical foundations, neural conditioning architectures, control flow paths, API protocols, and line-by-line implementation details of every module in the repository.

---

## 1. Documentation Tracker & Navigation

The table below tracks the status of all documentation chapters and file-by-file forensic breakdowns. Each link leads to a standalone, zero-omission Markdown manual.

### Core Architectural Manuals

| Status | Chapter File | Title & Core Subject Matter |
| :---: | :--- | :--- |
| [x] | [`00_MASTER_INDEX.md`](00_MASTER_INDEX.md) | **Master Index & Repository Map:** Complete file index, role directory, and multi-turn documentation tracker. |
| [x] | [`01_SYSTEM_ARCHITECTURE_AND_FLOW.md`](01_SYSTEM_ARCHITECTURE_AND_FLOW.md) | **System Architecture & Execution Flow:** High-level and low-level system designs, dual-pass lifecycle, data flow diagrams, and state management. |
| [ ] | [`02_CHARACTER_CONSISTENCY_DEEP_DIVE.md`](02_CHARACTER_CONSISTENCY_DEEP_DIVE.md) | **Character Consistency Deep Dive:** Theoretical foundations, line-by-line code traces, and engineering trade-offs of all fully implemented consistency methods. |

---

### File-by-File Forensic Reference (`/docs/files/`)

#### 1. Core Production & Orchestration Root
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`app_py_explained.md`](files/app_py_explained.md) | [`app.py`](../app.py) | Streamlit interactive UI, regional prompt compilation, camera-distance adaptive denoise scheduler, localized face inpainting. |
| [ ] | [`metrics_py_explained.md`](files/metrics_py_explained.md) | [`metrics.py`](../metrics.py) | Quad-modal evaluation engine (ArcFace, CLIP-I, SSIM, HSV Palette), RetinaFace detection, Hungarian Kuhn-Munkres bipartite matching, VRAM guard. |
| [ ] | [`run_from_scratch_py_explained.md`](files/run_from_scratch_py_explained.md) | [`run_from_scratch.py`](../run_from_scratch.py) | Headless CLI production runner, zero-anchor cache wiping, canonical anchor bootstrapping, 6-scene storyboard generation. |
| [ ] | [`hardcoded_cache_json_explained.md`](files/hardcoded_cache_json_explained.md) | [`hardcoded_cache.json`](../hardcoded_cache.json) | Persistent JSON store of base64-encoded visual DNA canonical reference portraits for verified characters. |
| [ ] | [`requirements_txt_explained.md`](files/requirements_txt_explained.md) | [`requirements.txt`](../requirements.txt) | Explicit Python package dependencies, pinning deep learning and image processing frameworks. |

#### 2. Narrative Engine (`story_engine/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`schema_py_explained.md`](files/story_engine/schema_py_explained.md) | [`story_engine/schema.py`](../story_engine/schema.py) | Typed dataclasses defining universal data contracts (`StorySpec`, `SceneSpec`, `CharacterSpec`, `AgeState`, `PropSpec`). |
| [ ] | [`character_registry_py_explained.md`](files/story_engine/character_registry_py_explained.md) | [`story_engine/character_registry.py`](../story_engine/character_registry.py) | Central registry managing character visual DNA, age-state anchors, cache persistence, and demographic prompt assembly. |
| [ ] | [`prompt_builder_py_explained.md`](files/story_engine/prompt_builder_py_explained.md) | [`story_engine/prompt_builder.py`](../story_engine/prompt_builder.py) | Dynamic prompt compiler injecting demographic locking, token isolation, and `BREAK` syntax. |
| [ ] | [`scene_planner_py_explained.md`](files/story_engine/scene_planner_py_explained.md) | [`story_engine/scene_planner.py`](../story_engine/scene_planner.py) | Transforms abstract scene descriptions into concrete `SceneExecutionPlan` instances with adaptive denoise scheduling. |
| [ ] | [`story_analyzer_py_explained.md`](files/story_engine/story_analyzer_py_explained.md) | [`story_engine/story_analyzer.py`](../story_engine/story_analyzer.py) | Narrative NLP parser extracting characters, scene actions, locations, and shot types from raw prose. |
| [ ] | [`story_schema_json_explained.md`](files/story_engine/story_schema_json_explained.md) | [`story_engine/story_schema.json`](../story_engine/story_schema.json) | Formal JSON Schema specifying constraints for valid StorySpec documents. |

#### 3. Diffusion Generation Subsystem (`generation/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`forge_client_py_explained.md`](files/generation/forge_client_py_explained.md) | [`generation/forge_client.py`](../generation/forge_client.py) | Resilient HTTP REST client communicating with Stable Diffusion WebUI Forge (`/sdapi/v1/txt2img`, `/sdapi/v1/img2img`). |
| [ ] | [`generation_orchestrator_py_explained.md`](files/generation/generation_orchestrator_py_explained.md) | [`generation/generation_orchestrator.py`](../generation/generation_orchestrator.py) | Master diffusion orchestrator: Pass 1 draft, Pass 2 adaptive img2img, FreeU, PAG, CFG-Fix, ReActor, and CodeFormer. |

#### 4. Spatial & Latent Conditioning Subsystem (`conditioning/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`controlnet_conditioner_py_explained.md`](files/conditioning/controlnet_conditioner_py_explained.md) | [`conditioning/controlnet_conditioner.py`](../conditioning/controlnet_conditioner.py) | Multi-unit ControlNet payload generator for OpenPose, Depth (Depth Anything v2), and IP-Adapter. |
| [ ] | [`regional_conditioner_py_explained.md`](files/conditioning/regional_conditioner_py_explained.md) | [`conditioning/regional_conditioner.py`](../conditioning/regional_conditioner.py) | Spatial column partitioning, attention mask slicing, and multi-subject prompt chunking. |
| [ ] | [`refinement_py_explained.md`](files/conditioning/refinement_py_explained.md) | [`conditioning/refinement.py`](../conditioning/refinement.py) | Deterministic 30-parameter ReActor face-swap payload builder with CodeFormer restoration. |
| [ ] | [`lora_registry_py_explained.md`](files/conditioning/lora_registry_py_explained.md) | [`conditioning/lora_registry.py`](../conditioning/lora_registry.py) | Manages character-specific and style LoRA weights, trigger tokens, and weight scheduling. |
| [ ] | [`crossframe_conditioner_py_explained.md`](files/conditioning/crossframe_conditioner_py_explained.md) | [`conditioning/crossframe_conditioner.py`](../conditioning/crossframe_conditioner.py) | Inter-scene latent continuity management and visual context propagation across frames. |

#### 5. Identity Conditioning Subsystem (`identity/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`identity_conditioner_py_explained.md`](files/identity/identity_conditioner_py_explained.md) | [`identity/identity_conditioner.py`](../identity/identity_conditioner.py) | Abstract base class (`IdentityConditioner`) establishing unified interface for all facial adapters. |
| [ ] | [`ipadapter_py_explained.md`](files/identity/ipadapter_py_explained.md) | [`identity/ipadapter.py`](../identity/ipadapter.py) | Concrete IP-Adapter SDXL implementation with OpenCLIP-ViT-bigG decoupled cross-attention guidance. |
| [ ] | [`instantid_py_explained.md`](files/identity/instantid_py_explained.md) | [`identity/instantid.py`](../identity/instantid.py) | Concrete InstantID SDXL implementation using InsightFace facial landmarks and keypoints. |
| [ ] | [`faceid_py_explained.md`](files/identity/faceid_py_explained.md) | [`identity/faceid.py`](../identity/faceid.py) | Concrete FaceID Plus v2 SDXL adapter utilizing face recognition embeddings and companion LoRA. |

#### 6. Quality Assurance & Auto-Healing Subsystem (`qa/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`test_anchor_manager_py_explained.md`](files/qa/test_anchor_manager_py_explained.md) | [`qa/test_anchor_manager.py`](../qa/test_anchor_manager.py) | 4-point programmatic anchor validation gate, isolated test runtime environments, canonical caching. |
| [ ] | [`programmatic_auditor_py_explained.md`](files/qa/programmatic_auditor_py_explained.md) | [`qa/programmatic_auditor.py`](../qa/programmatic_auditor.py) | StorySpec rule engine evaluating face counts, bounding box geometry, and zero-shot CLIP crop attributes. |
| [ ] | [`vlm_visual_auditor_py_explained.md`](files/qa/vlm_visual_auditor_py_explained.md) | [`qa/vlm_visual_auditor.py`](../qa/vlm_visual_auditor.py) | Multimodal visual auditor using Salesforce BLIP captioning and OpenAI CLIP zero-shot anomaly probes. |
| [ ] | [`healing_controller_py_explained.md`](files/qa/healing_controller_py_explained.md) | [`qa/healing_controller.py`](../qa/healing_controller.py) | Autonomous closed-loop feedback controller diagnosing QA failures and triggering self-healing retries. |

#### 7. Storyboards & Production Specifications (`storyboards/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`ebright_json_explained.md`](files/storyboards/ebright_json_explained.md) | [`storyboards/ebright.json`](../storyboards/ebright.json) | Canonical 6-scene production storyboard for "The Making of a Scientist" (Richard Ebright). |
| [ ] | [`generalization_benchmark_json_explained.md`](files/storyboards/generalization_benchmark_json_explained.md) | [`storyboards/generalization_benchmark.json`](../storyboards/generalization_benchmark.json) | Multi-character generalization benchmark storyboard ("Chrono-Archivists"). |

#### 8. Experimental Suites & Diagnostic Utilities (`experiments/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`memory_utils_py_explained.md`](files/experiments/memory_utils_py_explained.md) | [`experiments/memory_utils.py`](../experiments/memory_utils.py) | Windows GlobalMemoryStatusEx bindings, PyTorch CUDA memory tracking, context-managed `vram_guard`. |
| [ ] | [`controlnet_adapter_py_explained.md`](files/experiments/controlnet_adapter_py_explained.md) | [`experiments/controlnet_adapter.py`](../experiments/controlnet_adapter.py) | ControlSpec dataclass and Forge ControlNet REST payload formatting adapter. |
| [ ] | [`sota_generator_py_explained.md`](files/experiments/sota_generator_py_explained.md) | [`experiments/sota_generator.py`](../experiments/sota_generator.py) | Scripted two-pass SOTA generation runner for batch verification. |
| [ ] | [`sota_qa_auditor_py_explained.md`](files/experiments/sota_qa_auditor_py_explained.md) | [`experiments/sota_qa_auditor.py`](../experiments/sota_qa_auditor.py) | Standalone evaluation script applying quad-modal metrics to generated SOTA scenes. |
| [ ] | [`benchmark_native_sota_py_explained.md`](files/experiments/benchmark_native_sota_py_explained.md) | [`experiments/benchmark_native_sota.py`](../experiments/benchmark_native_sota.py) | Automated harness benchmarking SOTA pipeline across multiple narrative scenes. |
| [ ] | [`benchmark_identity_spatial_methods_py_explained.md`](files/experiments/benchmark_identity_spatial_methods_py_explained.md) | [`experiments/benchmark_identity_spatial_methods.py`](../experiments/benchmark_identity_spatial_methods.py) | Comparative benchmark comparing IP-Adapter, InstantID, and FaceID with/without spatial guidance. |
| [ ] | [`phase8_4_execution_suite_py_explained.md`](files/experiments/phase8_4_execution_suite_py_explained.md) | [`experiments/phase8_4_execution_suite.py`](../experiments/phase8_4_execution_suite.py) | Full regression test suite covering InstantID, FaceID, ControlNet, and multi-character runs. |
| [ ] | [`phase8_4_stress_and_ablation_suite_py_explained.md`](files/experiments/phase8_4_stress_and_ablation_suite_py_explained.md) | [`experiments/phase8_4_stress_and_ablation_suite.py`](../experiments/phase8_4_stress_and_ablation_suite.py) | Comprehensive stress suite: LoRA sweeps, refinement ablations, position swaps, seed invariance. |
| [ ] | [`phase9_runner_py_explained.md`](files/experiments/phase9_runner_py_explained.md) | [`experiments/phase9_runner.py`](../experiments/phase9_runner.py) | Master Phase 9 benchmark harness executing ablations, sweeps, and generating master JSON reports. |
| [ ] | [`phase9_stress_suite_py_explained.md`](files/experiments/phase9_stress_suite_py_explained.md) | [`experiments/phase9_stress_suite.py`](../experiments/phase9_stress_suite.py) | Extreme condition testing: age progression, viewpoint variations, dramatic lighting, multi-character swaps. |

#### 9. Diagnostic Scripts & Scratch Utilities (`scratch/`)
| Status | Target Documentation File | Source Code Path | Primary Responsibility |
| :---: | :--- | :--- | :--- |
| [ ] | [`scratch_scripts_explained.md`](files/scratch/scratch_scripts_explained.md) | [`scratch/*.py`](../scratch/) | Comprehensive breakdown of diagnostic tools: anchor inspection, prompt inspection, ReActor payload verification, GPU throughput benchmarks. |

---

## 2. Complete Repository File Map

```
d:/5TH SEM/HACKATHON/
│
├── app.py                                   # Main interactive Streamlit application & microservice UI
├── metrics.py                               # Quad-modal biometric & visual evaluation engine
├── run_from_scratch.py                      # Headless CLI entry point for clean-slate execution
├── hardcoded_cache.json                     # Ground-truth canonical character visual DNA cache
├── requirements.txt                         # Python runtime dependency specifications
├── OpenDyslexic-Regular.ttf                 # Font asset for educational UI rendering
│
├── story_engine/                            # Narrative ingestion, prompt synthesis & planning
│   ├── schema.py                            # Typed dataclasses for Story, Scene, Character, Prop
│   ├── character_registry.py                # Character visual DNA & demographic anchor management
│   ├── prompt_builder.py                    # Regional prompt compiler & demographic token injector
│   ├── scene_planner.py                     # SceneExecutionPlan generator & adaptive denoise scheduler
│   ├── story_analyzer.py                    # Prose text NLP analyzer & character co-occurrence mapper
│   └── story_schema.json                    # JSON Schema definition for StorySpec validation
│
├── generation/                              # Diffusion backend communication & orchestration
│   ├── forge_client.py                      # Low-level REST client for Stable Diffusion WebUI Forge
│   └── generation_orchestrator.py           # Canonical two-pass coarse-to-fine diffusion coordinator
│
├── conditioning/                            # Latent & spatial conditioning modules
│   ├── controlnet_conditioner.py            # Multi-unit ControlNet manager (Depth, OpenPose)
│   ├── regional_conditioner.py              # Spatial column partitioning & prompt slicing
│   ├── refinement.py                        # ReActor latent face-swap & CodeFormer restoration payload builder
│   ├── lora_registry.py                     # LoRA adapter management & weight scheduling
│   └── crossframe_conditioner.py            # Cross-frame latent propagation & temporal context passing
│
├── identity/                                # Facial biometric & identity conditioning adapters
│   ├── identity_conditioner.py              # Abstract base class for all identity adapters
│   ├── ipadapter.py                         # IP-Adapter SDXL (OpenCLIP-ViT-bigG decoupled cross-attention)
│   ├── instantid.py                         # InstantID SDXL (InsightFace keypoint landmark projection)
│   └── faceid.py                            # FaceID Plus v2 SDXL (InsightFace identity embedding + LoRA)
│
├── qa/                                      # Automated quality assurance & self-healing
│   ├── test_anchor_manager.py               # 4-point programmatic anchor validation gate
│   ├── programmatic_auditor.py              # Story-driven rule auditor (face count, bbox, CLIP crop probe)
│   ├── vlm_visual_auditor.py                # Salesforce BLIP captioning & CLIP chromatic anomaly auditor
│   └── healing_controller.py                # Closed-loop diagnosis & autonomous healing retry controller
│
├── storyboards/                             # Structured story specifications (JSON)
│   ├── ebright.json                         # Canonical 6-scene storyboard: "The Making of a Scientist"
│   └── generalization_benchmark.json        # 4-scene multi-character benchmark: "Chrono-Archivists"
│
├── experiments/                             # Validation harnesses, benchmarks & test suites
│   ├── memory_utils.py                      # Windows RAM/VRAM telemetry & context-managed vram_guard
│   ├── controlnet_adapter.py                # ControlSpec dataclass & Forge payload adapter
│   ├── sota_generator.py                    # Scripted SOTA generator for automated validation
│   ├── sota_qa_auditor.py                   # Automated evaluation script for SOTA scene outputs
│   ├── benchmark_native_sota.py             # SOTA pipeline benchmark runner across scenes
│   ├── benchmark_identity_spatial_methods.py# Multi-method identity/spatial comparison harness
│   ├── evaluate_clean_run.py                # Clean run verification & scorecard generation
│   ├── heal_sota_scene5.py                  # Specialized diagnostic script for Scene 5 healing
│   ├── phase8_4_execution_suite.py          # Regression harness for Phase 8.4 features
│   ├── phase8_4_stress_and_ablation_suite.py# Stress harness (LoRA sweeps, spatial ablations, position swaps)
│   ├── phase9_anchor_quality_audit.py       # Programmatic validation of all cached visual anchors
│   ├── phase9_audit_collector.py            # Hashes and collects all experiment artifacts into manifests
│   ├── phase9_runner.py                     # Comprehensive Phase 9 evaluation harness
│   ├── phase9_sequence_audit.py             # Audit script validating full narrative sequence continuity
│   ├── phase9_stress_suite.py               # Stress harness (age progression, environmental perturbations)
│   ├── run_ipadapter_fixed_scene2.py        # Isolated test for IP-Adapter Scene 2 rendering
│   ├── scratch_inspect_audit.py             # Comprehensive system audit script
│   ├── test_api_smoke.py                    # Smoke test validating Forge REST API availability
│   ├── test_instantid_runtime.py            # Runtime verification of InstantID preprocessors
│   ├── test_ipadapter_clean.py              # Isolated test of IP-Adapter cross-attention payload
│   └── phase9_master_completion_report.md   # Comprehensive empirical milestone report
│
├── scratch/                                 # Exploratory utilities, test fixtures & benchmark helpers
│   ├── gen_new_teen_anchor.py               # Script generating fresh teenage anchor
│   ├── generate_consistent_teen_anchor.py   # Specialized teenage anchor generator
│   ├── inspect_anchors.py                   # Inspection utility for cached anchors
│   ├── inspect_prompts.py                   # Prompt syntax inspection utility
│   ├── inspect_reactor_args.py              # ReActor argument array validator
│   ├── regenerate_visual_dna_anchors.py     # Batch anchor regeneration script
│   ├── test_face_detect.py                  # Standalone face detection diagnostic
│   ├── test_fixed_faces.py                  # Verification of face alignment
│   ├── test_gpu_throughput_benchmark.py     # GPU generation throughput and latency benchmark
│   └── update_teen_cache.py                 # Manual cache injection helper
│
└── edu_outputs/                             # Default output directory for generated imagery & reports
    └── images/                              # Scene PNG renders and companion metadata JSON files
```

---

## 3. High-Level System Architecture Summary

The MMCCP pipeline implements a **decoupled coarse-to-fine generative architecture with closed-loop multi-modal quality verification**. Rather than attempting to force character identity, environmental composition, and high-frequency textures in a single diffusion step, the system partitions generation into distinct physical stages:

```
+----------------------------------------------------------------------------------------------------+
|                                    MMCCP END-TO-END PIPELINE                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Prose / Storyboard JSON]                                                                        |
|            │                                                                                       |
|            ▼                                                                                       |
|  ┌──────────────────┐                                                                              |
|  │ Story Engine     │ ──> CharacterRegistry & 4-Point Anchor Certification Gate                   |
|  │ & Scene Planner  │ ──> Regional Prompt Builder (SDXL BREAK Token Segmentation)                  |
|  └──────────────────┘                                                                              |
|            │                                                                                       |
|            ▼                                                                                       |
|  ┌──────────────────┐                                                                              |
|  │ Stage 1:         │ ──> SDXL txt2img (Euler a, 30 steps, 1024x1024)                              |
|  │ Compositional    │ ──> ControlNet Depth (0.70) + OpenPose (0.80)                                 |
|  │ Draft Pass       │ ──> Latent Hooks: FreeU (1.3, 1.4, 0.9, 0.2) + PAG (3.0) + CFG-Fix            |
|  └──────────────────┘                                                                              |
|            │                                                                                       |
|            ▼ [Draft Canvas Base64]                                                                 |
|  ┌──────────────────┐                                                                              |
|  │ Stage 2:         │ ──> Camera-Distance Adaptive Denoise Scheduling (0.38 - 0.62)                |
|  │ Biometric        │ ──> Strip ControlNet Structural Locks (Unlocks Pixel Restructuring)          |
|  │ Refinement Pass  │ ──> ReActor Face Transfer (InsightFace AntelopeV2) + CodeFormer (0.80)       |
|  └──────────────────┘                                                                              |
|            │                                                                                       |
|            ▼ [Refined Scene Canvas]                                                                |
|  ┌──────────────────┐                                                                              |
|  │ Stage 3:         │ ──> RetinaFace Multi-Face Localization                                       |
|  │ Closed-Loop QA   │ ──> Kuhn-Munkres Hungarian Bipartite Assignment Matrix                       |
|  │ & Auto-Healing   │ ──> Quad-Modal Scoring: ArcFace (45%) + CLIP-I (25%) + SSIM (15%) + Pal (15%)|
|  └──────────────────┘ ──> Autonomous Healing Loop (Retries <= 3 with Pose Perturbation)            |
|            │                                                                                       |
|            ▼                                                                                       |
|  [Verified Consistent Output Image & Audit Report JSON]                                            |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Hardware Constraints & Memory Topography

MMCCP was engineered from the ground up to operate within the strict hardware constraints of a **consumer-grade laptop GPU**:

- **Physical GPU:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8,192 MB GDDR6 VRAM, 140W TGP).
- **Host System:** Intel Core i7-12650H (10 Cores / 16 Threads), 32 GB DDR4 RAM, Windows 11 Home.
- **Inference Server:** Stable Diffusion WebUI Forge (Direct PyTorch 2.3.1+cu121, xFormers, `--cuda-malloc`, `--api`).

To prevent fatal `torch.cuda.OutOfMemoryError` exceptions when running multi-gigabyte foundation models (SDXL Base 1.0 = 6.6 GB fp16) alongside ControlNet units (Depth = 2.5 GB, OpenPose = 2.5 GB) and deep vision models (InsightFace AntelopeV2, CLIP ViT-B/32, BLIP), MMCCP enforces:

1. **Staged Memory Lifecycles:** ControlNet units are loaded transiently in Pass 1 and completely unloaded before Pass 2.
2. **CPU Pinning for Evaluators:** ArcFace, CLIP-I, BLIP, and structural SSIM evaluators are strictly pinned to host CPU/RAM (`device = "cpu"`), preserving 100% of GPU memory for diffusion latents.
3. **Explicit Garbage Collection:** Between every pass, `cleanup_memory()` invokes `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`.
4. **Active VRAM Telemetry:** Peak VRAM is instrumented via Win32 `GlobalMemoryStatusEx` and PyTorch memory counters, maintaining a safe plateau at **6.2 GB VRAM** (a 1.8 GB safety margin).

---

## 5. Next Steps

Proceed to [`01_SYSTEM_ARCHITECTURE_AND_FLOW.md`](01_SYSTEM_ARCHITECTURE_AND_FLOW.md) for a comprehensive deep dive into the high-level and low-level system architecture, component contracts, and end-to-end execution flow.
