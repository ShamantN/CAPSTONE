# MMCCP File-by-File Technical Reference

- **Project:** Multi-Modal Character Consistency Pipeline (MMCCP)
- **Document:** 04_file_by_file_reference.md
- **Repository Path:** `d:/5TH SEM/HACKATHON`
- **Scope:** Complete architectural inventory of every file in the repository (source code, schemas, storyboards, experiments, scratch scripts, and configuration artifacts).

---

## Group 1: Core Production Root & Orchestration

### 1. `app.py`
- **File Path:** [`app.py`](file:///d:/5TH%20SEM/HACKATHON/app.py) | **LOC:** 902
- **Purpose:** Primary interactive web application microservice built on Streamlit. Houses the three-tab operator dashboard, prompt compilation logic, camera-distance adaptive denoise scheduler, localized high-res face inpainting, and UI session state management.
- **Inputs:** User UI interactions (model selection, art style, evaluation weights, prompt overrides, image file uploads).
- **Outputs:** Interactive web UI on `http://localhost:8501`, rendered scene panels in `edu_outputs/images/scene_*.png`, and real-time consistency scorecards.
- **Key Dependencies:** `streamlit`, `cv2`, `numpy`, `PIL.Image`, `requests`, [`metrics.py`](file:///d:/5TH%20SEM/HACKATHON/metrics.py), [`generation/generation_orchestrator.py`](file:///d:/5TH%20SEM/HACKATHON/generation/generation_orchestrator.py), [`story_engine/schema.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/schema.py).
- **Callers / Importers:** Streamlit CLI (`streamlit run app.py`), [`run_from_scratch.py`](file:///d:/5TH%20SEM/HACKATHON/run_from_scratch.py) (imports `render_phase3_scene`), diagnostic scripts.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO PRODUCTION MAIN PATH**.
- **Notable Design Decisions / Smells:** Dual-path rendering: if SD Forge is offline, falls back to `_synthesize_benchmark_scene_cv()` synthetic mockups to allow offline mathematical benchmarking. Seed was historically unconstrained (`-1`) in UI calls.

---

### 2. `metrics.py`
- **File Path:** [`metrics.py`](file:///d:/5TH%20SEM/HACKATHON/metrics.py) | **LOC:** 2,029
- **Purpose:** Master quad-modal evaluation engine and biometrics subsystem. Computes 512-D ArcFace cosine similarity via DeepFace/InsightFace, CLIP-I semantic image embeddings via OpenAI ViT-B/32, structural pose SSIM via scikit-image, and 3D HSV/Lab Bhattacharyya color palette overlap. Solves multi-character bipartite assignment via the Hungarian Kuhn-Munkres algorithm.
- **Inputs:** Canonical anchor image inputs (base64, PIL, numpy array), synthesized scene image, character ID, bounding box annotations.
- **Outputs:** `ConsistencyReport` dataclass instances, assignment matrices, similarity scores $[0.0, 1.0]$, and JSON diagnostics.
- **Key Dependencies:** `deepface` (RetinaFace, ArcFace), `tf_keras`, `torch`, `torchvision`, `cv2`, `scipy.optimize.linear_sum_assignment`, `skimage.metrics.structural_similarity`.
- **Callers / Importers:** Imported by almost all modules: `app.py`, `run_from_scratch.py`, `generation_orchestrator.py`, `qa/*`, `experiments/*`.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO PRODUCTION MAIN PATH**.
- **Notable Design Decisions / Smells:** Dynamically links Forge's embedded PyTorch site-packages (`C:\Stable-Diffusion\system\python\Lib\site-packages`) to enable seamless CUDA GPU acceleration inside standard virtual environments. Contains aggressive 25% VRAM cap to protect SD Forge headroom.

---

### 3. `run_from_scratch.py`
- **File Path:** [`run_from_scratch.py`](file:///d:/5TH%20SEM/HACKATHON/run_from_scratch.py) | **LOC:** 162
- **Purpose:** Headless batch runner and CI/CD automation script. Wipes existing runtime caches (`hardcoded_cache.json`), bootstraps 5 fresh character anchors from pure prompt specifications using SDXL txt2img, renders the 6-scene storyboard sequentially, and serializes evaluation scores.
- **Inputs:** Narrative definitions in `metrics.HARDCODED_SCENES` and `metrics.HARDCODED_CHARACTERS`.
- **Outputs:** Fresh anchors in `experiments/fresh_run_from_scratch/anchors/`, rendered panels in `edu_outputs/images/scene_*.png`, and console output logs.
- **Key Dependencies:** [`metrics.py`](file:///d:/5TH%20SEM/HACKATHON/metrics.py), [`app.py`](file:///d:/5TH%20SEM/HACKATHON/app.py) (`render_phase3_scene`), `cv2`, `numpy`.
- **Callers / Importers:** CLI entry point (`python run_from_scratch.py`).
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO PRODUCTION MAIN PATH**.
- **Notable Design Decisions / Smells:** Relied on `app.render_phase3_scene` rather than importing `GenerationOrchestrator` directly; did not serialize summary JSON report until patched in Issue-01.

---

### 4. `hardcoded_cache.json`
- **File Path:** [`hardcoded_cache.json`](file:///d:/5TH%20SEM/HACKATHON/hardcoded_cache.json)
- **Purpose:** Persistent JSON store of base64-encoded visual DNA canonical reference portraits for verified characters (`young_ebright`, `teen_ebright`, `college_ebright`, `mother`, `james_wong`).
- **Structure:** Dictionary mapping `character_id` (string) to Base64 PNG string (1.0 to 1.7 MB per character).
- **Callers / Importers:** Read by `CharacterRegistry`, `metrics.py`, and `app.py`.
- **Pipeline Role:** **PRODUCTION CACHE STORE**.

---

### 5. `requirements.txt`
- **File Path:** [`requirements.txt`](file:///d:/5TH%20SEM/HACKATHON/requirements.txt) | **LOC:** 36
- **Purpose:** Explicit declaration of Python dependencies pinning versions for Streamlit, OpenCV, deepface, TensorFlow, PyTorch, SciPy, and Pydantic.
- **Pipeline Role:** **ENVIRONMENT SPECIFICATION**.

---

## Group 2: Narrative Engine (`story_engine/`)

### 6. `story_engine/schema.py`
- **File Path:** [`story_engine/schema.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/schema.py) | **LOC:** 170
- **Purpose:** Domain data model defining typed dataclasses for the narrative architecture: `StorySpec`, `SceneSpec`, `CharacterSpec`, `AgeState`, and `PropSpec`.
- **Inputs / Outputs:** JSON serialization and deserialization via `to_dict()`, `from_dict()`, `load_json()`, `save_json()`.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 7. `story_engine/character_registry.py`
- **File Path:** [`story_engine/character_registry.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/character_registry.py) | **LOC:** 112
- **Purpose:** Central repository for character visual DNA, age-state anchors, and demographic prompt assembly. Synchronizes runtime memory with `hardcoded_cache.json`.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 8. `story_engine/prompt_builder.py`
- **File Path:** [`story_engine/prompt_builder.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/prompt_builder.py) | **LOC:** 129
- **Purpose:** Dynamic prompt compiler implementing demographic phenotype locking and spatial `BREAK` syntax for cross-attention isolation in multi-character scenes.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 9. `story_engine/scene_planner.py`
- **File Path:** [`story_engine/scene_planner.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/scene_planner.py) | **LOC:** 113
- **Purpose:** Translates high-level narrative `SceneSpec` instances into low-level `SceneExecutionPlan` configurations, scheduling camera-distance adaptive denoising strengths and multi-character ReActor composite layouts.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 10. `story_engine/story_analyzer.py`
- **File Path:** [`story_engine/story_analyzer.py`](file:///d:/5TH%20SEM/HACKATHON/story_engine/story_analyzer.py) | **LOC:** 159
- **Purpose:** Natural language story parser leveraging Google Gemini LLM API (`google-generativeai`) to convert unstructured prose into structured `StorySpec` documents.
- **Pipeline Role:** **IMPLEMENTED BUT ONLY IN EXPERIMENTS / PRE-PROCESSING**. (Offline narrative ingest).

---

### 11. `story_engine/story_schema.json`
- **File Path:** [`story_engine/story_schema.json`](file:///d:/5TH%20SEM/HACKATHON/story_engine/story_schema.json) | **LOC:** 125
- **Purpose:** JSON Schema specification defining validation constraints for valid `StorySpec` JSON documents.
- **Pipeline Role:** **SCHEMA VALIDATION CONTRACT**.

---

## Group 3: Diffusion Generation Subsystem (`generation/`)

### 12. `generation/forge_client.py`
- **File Path:** [`generation/forge_client.py`](file:///d:/5TH%20SEM/HACKATHON/generation/forge_client.py) | **LOC:** 67
- **Purpose:** Low-level HTTP REST interface to Stable Diffusion WebUI Forge. Handles network retries, timeout management, Base64 image decoding, and error diagnostics for `/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, and `/sdapi/v1/options`.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 13. `generation/generation_orchestrator.py`
- **File Path:** [`generation/generation_orchestrator.py`](file:///d:/5TH%20SEM/HACKATHON/generation/generation_orchestrator.py) | **LOC:** 232
- **Purpose:** Master two-pass diffusion orchestrator. Executes Pass 1 structural draft (with FreeU, PAG, CFG-Fix, and ControlNet Depth/Pose) followed by Pass 2 adaptive img2img refinement (with stripped spatial ControlNet units, ReActor face swap, and CodeFormer restoration).
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

## Group 4: Spatial & Latent Conditioning Subsystem (`conditioning/`)

### 14. `conditioning/controlnet_conditioner.py`
- **File Path:** [`conditioning/controlnet_conditioner.py`](file:///d:/5TH%20SEM/HACKATHON/conditioning/controlnet_conditioner.py) | **LOC:** 79
- **Purpose:** Builds Forge ControlNet REST payloads for multi-unit conditioning across OpenPose (`thibaud_xl_openpose`), Depth (`depth_xl` / Depth Anything v2), and IP-Adapter.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 15. `conditioning/regional_conditioner.py`
- **File Path:** [`conditioning/regional_conditioner.py`](file:///d:/5TH%20SEM/HACKATHON/conditioning/regional_conditioner.py) | **LOC:** 127
- **Purpose:** Implements spatial column partitioning and attention mask slicing, compiling regional prompts across horizontal latent columns.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 16. `conditioning/refinement.py`
- **File Path:** [`conditioning/refinement.py`](file:///d:/5TH%20SEM/HACKATHON/conditioning/refinement.py) | **LOC:** 53
- **Purpose:** Formats the 30-argument deterministic ReActor face-swap payload for Forge's `/sdapi/v1/img2img` endpoint, configuring CodeFormer restoration at weight 0.80 and target face routing indices.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 17. `conditioning/lora_registry.py`
- **File Path:** [`conditioning/lora_registry.py`](file:///d:/5TH%20SEM/HACKATHON/conditioning/lora_registry.py) | **LOC:** 29
- **Purpose:** Registry managing character-specific and style LoRA weights, trigger tokens, and weight multipliers.
- **Pipeline Role:** **IMPLEMENTED BUT ONLY IN EXPERIMENTS / OPTIONAL ACCELERATION** (SDXL Lightning LoRA).

---

### 18. `conditioning/crossframe_conditioner.py`
- **File Path:** [`conditioning/crossframe_conditioner.py`](file:///d:/5TH%20SEM/HACKATHON/conditioning/crossframe_conditioner.py) | **LOC:** 28
- **Purpose:** Inter-scene latent context and visual continuity manager.
- **Pipeline Role:** **PARTIAL / EXPERIMENT STUB**. (True Cross-Attention KV-sharing across separate HTTP calls is unsupported in Forge REST API; substituted by CharacterRegistry Visual DNA anchors).

---

## Group 5: Identity Conditioning Subsystem (`identity/`)

### 19. `identity/identity_conditioner.py`
- **File Path:** [`identity/identity_conditioner.py`](file:///d:/5TH%20SEM/HACKATHON/identity/identity_conditioner.py) | **LOC:** 30
- **Purpose:** Abstract base class (`IdentityConditioner`) establishing unified interfaces (`build_control_specs`, `get_adapter_name`) across facial conditioning mechanisms.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH**.

---

### 20. `identity/ipadapter.py`
- **File Path:** [`identity/ipadapter.py`](file:///d:/5TH%20SEM/HACKATHON/identity/ipadapter.py) | **LOC:** 61
- **Purpose:** Concrete implementation of IP-Adapter SDXL (`ip-adapter_xl.safetensors`), projecting OpenCLIP-ViT-bigG image embeddings into decoupled cross-attention layers.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PATH** (Primary production identity engine).

---

### 21. `identity/instantid.py`
- **File Path:** [`identity/instantid.py`](file:///d:/5TH%20SEM/HACKATHON/identity/instantid.py) | **LOC:** 45
- **Purpose:** Concrete implementation of InstantID SDXL (`instantid_xl.safetensors`), injecting InsightFace facial landmark keypoints.
- **Pipeline Role:** **FULLY IMPLEMENTED (SELECTABLE IN ORCHESTRATOR)**.

---

### 22. `identity/faceid.py`
- **File Path:** [`identity/faceid.py`](file:///d:/5TH%20SEM/HACKATHON/identity/faceid.py) | **LOC:** 43
- **Purpose:** Concrete implementation of FaceID Plus v2 SDXL (`ip-adapter-faceid-plusv2_sdxl.bin`), utilizing 512-D face recognition embeddings and companion LoRA.
- **Pipeline Role:** **FULLY IMPLEMENTED (SELECTABLE IN ORCHESTRATOR)**.

---

## Group 6: Quality Assurance & Auto-Healing Subsystem (`qa/`)

### 23. `qa/test_anchor_manager.py`
- **File Path:** [`qa/test_anchor_manager.py`](file:///d:/5TH%20SEM/HACKATHON/qa/test_anchor_manager.py) | **LOC:** 143
- **Purpose:** Manages canonical anchor repository initialization and isolated temporary test runtime directories for regression suites.
- **Pipeline Role:** **FULLY IMPLEMENTED (TEST SUITE HARNESS)**.

---

### 24. `qa/programmatic_auditor.py`
- **File Path:** [`qa/programmatic_auditor.py`](file:///d:/5TH%20SEM/HACKATHON/qa/programmatic_auditor.py) | **LOC:** 214
- **Purpose:** Rule engine evaluating facial presence, bounding box geometry, aspect ratios, and zero-shot CLIP crop attributes against `StorySpec` constraints.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO QA SUITE**.

---

### 25. `qa/vlm_visual_auditor.py`
- **File Path:** [`qa/vlm_visual_auditor.py`](file:///d:/5TH%20SEM/HACKATHON/qa/vlm_visual_auditor.py) | **LOC:** 349
- **Purpose:** Multimodal visual auditor running Salesforce BLIP deep image captioning and OpenAI CLIP zero-shot anomaly probes pinned to CPU. Detects chromatic inversions, solarization, and anatomical distortions.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO QA SUITE**.

---

### 26. `qa/healing_controller.py`
- **File Path:** [`qa/healing_controller.py`](file:///d:/5TH%20SEM/HACKATHON/qa/healing_controller.py) | **LOC:** 116
- **Purpose:** Closed-loop feedback controller diagnosing QA failures and triggering autonomous retries with dynamic pose perturbation ($\Delta_{\text{seed}} = +10$) and denoise adjustment.
- **Pipeline Role:** **FULLY IMPLEMENTED AND WIRED INTO MAIN PIPELINE**.

---

## Group 7: Storyboards (`storyboards/`)

### 27. `storyboards/ebright.json`
- **File Path:** [`storyboards/ebright.json`](file:///d:/5TH%20SEM/HACKATHON/storyboards/ebright.json) | **Lines:** 296
- **Purpose:** Canonical 6-scene production storyboard for "The Making of a Scientist" (Richard Ebright). Defines character profiles, age states (child, teenager, college, graduation), required props, forbidden props, and negative prompt additions.
- **Pipeline Role:** **PRIMARY PRODUCTION STORYBOARD**.

---

### 28. `storyboards/generalization_benchmark.json`
- **File Path:** [`storyboards/generalization_benchmark.json`](file:///d:/5TH%20SEM/HACKATHON/storyboards/generalization_benchmark.json) | **Lines:** 226
- **Purpose:** Multi-character cross-domain benchmark storyboard ("The Chrono-Archivists: The Quest for the Solarium Astrolabe"). Tests generalization across Hispanic, African-American, and Japanese characters in a Victorian steampunk aesthetic.
- **Pipeline Role:** **GENERALIZATION BENCHMARK STORYBOARD**.

---

## Group 8: Experimental Suites & Diagnostics (`experiments/`)

### 29. `experiments/memory_utils.py`
- **File Path:** [`experiments/memory_utils.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/memory_utils.py) | **LOC:** 105
- **Purpose:** Windows `GlobalMemoryStatusEx` ctypes bindings, PyTorch CUDA telemetry tracker, and context-managed `vram_guard`.
- **Pipeline Role:** **CORE UTILITY (PRODUCTION & EXPERIMENTS)**.

---

### 30. `experiments/controlnet_adapter.py`
- **File Path:** [`experiments/controlnet_adapter.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/controlnet_adapter.py) | **LOC:** 82
- **Purpose:** `ControlSpec` dataclass and Forge ControlNet REST payload formatter adapter.
- **Pipeline Role:** **UTILITY / ADAPTER**.

---

### 31. `experiments/sota_generator.py`
- **File Path:** [`experiments/sota_generator.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/sota_generator.py) | **LOC:** 275
- **Purpose:** Standalone batch renderer executing two-pass SOTA generation across scenes with metadata dumps.
- **Pipeline Role:** **EXPERIMENT RUNNER**.

---

### 32. `experiments/sota_qa_auditor.py`
- **File Path:** [`experiments/sota_qa_auditor.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/sota_qa_auditor.py) | **LOC:** 285
- **Purpose:** Standalone evaluation harness applying quad-modal metrics to generated scene panels.
- **Pipeline Role:** **EXPERIMENT AUDITOR**.

---

### 33. `experiments/benchmark_native_sota.py`
- **File Path:** [`experiments/benchmark_native_sota.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/benchmark_native_sota.py) | **LOC:** 152
- **Purpose:** Benchmarks rendered panels across ArcFace, CLIP-I, SSIM, and Palette metrics, saving `experiments/benchmark_results.json`.
- **Pipeline Role:** **EXPERIMENT BENCHMARK**.

---

### 34. `experiments/benchmark_identity_spatial_methods.py`
- **File Path:** [`experiments/benchmark_identity_spatial_methods.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/benchmark_identity_spatial_methods.py) | **LOC:** 113
- **Purpose:** Comparative benchmark evaluating IP-Adapter, InstantID, and FaceID with and without spatial depth/pose guidance on Scene 2.
- **Pipeline Role:** **EXPERIMENT BENCHMARK**.

---

### 35. `experiments/evaluate_clean_run.py`
- **File Path:** [`experiments/evaluate_clean_run.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/evaluate_clean_run.py) | **LOC:** 130
- **Purpose:** Evaluates clean-slate rendered scenes against ground truth anchors and generates `experiments/clean_user_generation_final_report.json`.
- **Pipeline Role:** **CLEAN SLATE EVALUATOR**.

---

### 36. `experiments/heal_sota_scene5.py`
- **File Path:** [`experiments/heal_sota_scene5.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/heal_sota_scene5.py) | **LOC:** 39
- **Purpose:** Isolated diagnostic script testing closed-loop healing on Scene 5 (College Ebright + James Wong molecular model interaction).
- **Pipeline Role:** **DIAGNOSTIC / HEALING TEST**.

---

### 37. `experiments/phase8_4_execution_suite.py`
- **File Path:** [`experiments/phase8_4_execution_suite.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase8_4_execution_suite.py) | **LOC:** 279
- **Purpose:** Master regression test suite executing InstantID, FaceID, spatial depth/pose, and multi-character story sequences across isolated runtime caches.
- **Pipeline Role:** **REGRESSION TEST SUITE**.

---

### 38. `experiments/phase8_4_stress_and_ablation_suite.py`
- **File Path:** [`experiments/phase8_4_stress_and_ablation_suite.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase8_4_stress_and_ablation_suite.py) | **LOC:** 304
- **Purpose:** Comprehensive stress test harness executing LoRA sweeps (0.0 to 1.0), refinement ablations, position swaps, 5-seed robustness tests, and red-team fault injections.
- **Pipeline Role:** **STRESS AND RED-TEAM SUITE**.

---

### 39. `experiments/phase9_anchor_quality_audit.py`
- **File Path:** [`experiments/phase9_anchor_quality_audit.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase9_anchor_quality_audit.py) | **LOC:** 123
- **Purpose:** Audits canonical anchor source portraits for resolution ($\ge 512\times 512$), single-face presence ($k=1$), bounding box ratio ($4\% \le \text{Area} \le 90\%$), and 512-D embedding validity.
- **Pipeline Role:** **ANCHOR CERTIFICATION GATE**.

---

### 40. `experiments/phase9_audit_collector.py`
- **File Path:** [`experiments/phase9_audit_collector.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase9_audit_collector.py) | **LOC:** 92
- **Purpose:** Aggregates audit logs across sub-suites into unified reporting manifests.
- **Pipeline Role:** **AUDIT AGGREGATOR**.

---

### 41. `experiments/phase9_runner.py`
- **File Path:** [`experiments/phase9_runner.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase9_runner.py) | **LOC:** 420
- **Purpose:** Master benchmark runner for Phase 9 validation, executing A/B adapter comparisons, spatial ablations, LoRA sweeps, and generating master benchmark summaries.
- **Pipeline Role:** **MASTER BENCHMARK HARNESS**.

---

### 42. `experiments/phase9_sequence_audit.py`
- **File Path:** [`experiments/phase9_sequence_audit.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase9_sequence_audit.py) | **LOC:** 160
- **Purpose:** Computes inter-scene identity drift and pairwise cross-scene cosine similarity matrices across sequential narrative panels.
- **Pipeline Role:** **SEQUENCE AUDITOR**.

---

### 43. `experiments/phase9_stress_suite.py`
- **File Path:** [`experiments/phase9_stress_suite.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/phase9_stress_suite.py) | **LOC:** 241
- **Purpose:** Stress-tests pipeline against extreme narrative conditions: 3-tier age progressions, viewpoint variations, dramatic lighting shifts, and character spatial permutations.
- **Pipeline Role:** **STRESS HARNESS**.

---

### 44. `experiments/run_ipadapter_fixed_scene2.py`
- **File Path:** [`experiments/run_ipadapter_fixed_scene2.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/run_ipadapter_fixed_scene2.py) | **LOC:** 71
- **Purpose:** Diagnostic runner verifying IP-Adapter cross-attention fix on Scene 2.
- **Pipeline Role:** **DIAGNOSTIC SCRIPT**.

---

### 45. `experiments/scratch_inspect_audit.py`
- **File Path:** [`experiments/scratch_inspect_audit.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/scratch_inspect_audit.py) | **LOC:** 36
- **Purpose:** Inspection script printing summary metrics from saved JSON reports.
- **Pipeline Role:** **DIAGNOSTIC SCRIPT**.

---

### 46. `experiments/test_api_smoke.py`
- **File Path:** [`experiments/test_api_smoke.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/test_api_smoke.py) | **LOC:** 52
- **Purpose:** Validates Forge REST connectivity, queries active checkpoint, and performs minimal 5-step txt2img smoke test.
- **Pipeline Role:** **PRE-FLIGHT SMOKE TEST**.

---

### 47. `experiments/test_instantid_runtime.py`
- **File Path:** [`experiments/test_instantid_runtime.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/test_instantid_runtime.py) | **LOC:** 22
- **Purpose:** Diagnostic validating InstantID ControlNet preprocessor availability.
- **Pipeline Role:** **DIAGNOSTIC TEST**.

---

### 48. `experiments/test_ipadapter_clean.py`
- **File Path:** [`experiments/test_ipadapter_clean.py`](file:///d:/5TH%20SEM/HACKATHON/experiments/test_ipadapter_clean.py) | **LOC:** 173
- **Purpose:** Comprehensive test harness executing isolated IP-Adapter runs with VRAM tracking.
- **Pipeline Role:** **DIAGNOSTIC / BENCHMARK**.

---

## Group 9: Diagnostic Scripts & Scratch Utilities (`scratch/`)

### 49. `scratch/gen_new_teen_anchor.py` & `scratch/generate_consistent_teen_anchor.py`
- **File Paths:** [`scratch/gen_new_teen_anchor.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/gen_new_teen_anchor.py) | [`scratch/generate_consistent_teen_anchor.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/generate_consistent_teen_anchor.py) | **LOC:** 36 each
- **Purpose:** Standalone anchor generation scripts utilizing refined demographic prompts for Teenage Ebright.
- **Pipeline Role:** **SCRATCH UTILITY**.

---

### 50. `scratch/inspect_anchors.py`, `inspect_prompts.py`, `inspect_reactor_args.py`
- **File Paths:** [`scratch/inspect_anchors.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/inspect_anchors.py) (20 LOC), [`scratch/inspect_prompts.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/inspect_prompts.py) (13 LOC), [`scratch/inspect_reactor_args.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/inspect_reactor_args.py) (10 LOC)
- **Purpose:** Quick developer utilities to inspect base64 images, compiled scene prompts, and ReActor argument schemas.
- **Pipeline Role:** **SCRATCH DIAGNOSTICS**.

---

### 51. `scratch/regenerate_visual_dna_anchors.py`
- **File Path:** [`scratch/regenerate_visual_dna_anchors.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/regenerate_visual_dna_anchors.py) | **LOC:** 91
- **Purpose:** Scripted utility to regenerate all 5 visual DNA anchors via SD Forge and save them as individual PNGs and combined JSON.
- **Pipeline Role:** **ANCHOR REGENERATION UTILITY**.

---

### 52. `scratch/test_face_detect.py` & `scratch/test_fixed_faces.py`
- **File Paths:** [`scratch/test_face_detect.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/test_face_detect.py) (46 LOC), [`scratch/test_fixed_faces.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/test_fixed_faces.py) (39 LOC)
- **Purpose:** RetinaFace face detection diagnostics and post-processing verification on Scenes 3 and 5.
- **Pipeline Role:** **SCRATCH DIAGNOSTICS**.

---

### 53. `scratch/test_gpu_throughput_benchmark.py`
- **File Path:** [`scratch/test_gpu_throughput_benchmark.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/test_gpu_throughput_benchmark.py) | **LOC:** 171
- **Purpose:** High-throughput GPU benchmarking utility measuring latency, step iterations per second, and VRAM stability across multiple scene renders.
- **Pipeline Role:** **THROUGHPUT BENCHMARK UTILITY**.

---

### 54. `scratch/update_teen_cache.py`
- **File Path:** [`scratch/update_teen_cache.py`](file:///d:/5TH%20SEM/HACKATHON/scratch/update_teen_cache.py) | **LOC:** 24
- **Purpose:** Utility script to update the `teen_ebright` entry in `hardcoded_cache.json` with an improved reference portrait.
- **Pipeline Role:** **CACHE UTILITY**.

---

### 55. New Session Verification & Diagnostic Scripts (`scratch/`)
- **`scratch/find_caches.py` (52 LOC):** Forensic scanner searching all source files for cache paths, output directories, `st.cache`, and temporary files.
- **`scratch/vram_logger.py` (28 LOC):** Asynchronous telemetry daemon querying `nvidia-smi` every 2 seconds into `docs_generated/raw/vram_telemetry.csv`.
- **`scratch/test_inversion_experiment.py` (95 LOC):** Reproduces the chromatic inversion experiment, proving ArcFace invariance and zero-shot VLM chromatic anomaly detection.
- **`scratch/test_certified_anchors.py` (90 LOC):** Evaluates active anchors against the 4-point certification gate.
- **`scratch/run_systematic_ablation.py` (290 LOC):** 10-stage systematic ablation study runner with fixed seed=42 and quad-modal evaluation.
- **`scratch/ast_catalog.py` (65 LOC):** AST parser extracting every class, method, function, line number, and docstring across all 59 Python files.
