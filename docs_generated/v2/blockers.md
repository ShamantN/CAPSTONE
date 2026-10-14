# MMCCP Consistency v2: Blockers, Technical Challenges & Documented Fallbacks

This document tracks all environmental challenges, software dependency hurdles, hardware constraints, and documented fallbacks encountered during the implementation and evaluation of the Multi-Modal Character Consistency Pipeline v2 (MMCCP). It serves as formal compliance with **RULE 9 (Documented Fallback Protocol)** and **RULE 10 (Honest Reporting)**.

---

## 1. Documented Fallbacks & Deviations (Rule 9 Compliance)

### 1.1 M11 (Per-Character LoRA Training)
- **Status:** **DELIBERATELY OUT OF SCOPE FOR THIS RUN** (per explicit prompt instruction).
- **Scope Justification:** M11 requires heavy training runtimes, dataset curation pipelines (`kohya-ss/sd-scripts`), and distinct VRAM allocation that exceeds rapid iteration on an 8 GB mobile GPU while maintaining active Forge server processes.
- **Architectural Preservation:** A clean extension point has been retained in `conditioning/lora_registry.py` and `consistency_v2/config.py` (`use_character_lora: bool = False`). Seeds **{606, 707, 808}** are strictly reserved and held out for future evaluation.

### 1.2 ComfyUI / PuLID / StoryDiffusion Native Stretch Items
- **Status:** **OUT OF SCOPE FOR THIS RUN**.
- **Scope Justification:** The project execution environment is built around Stable Diffusion WebUI Forge on Windows 11. ComfyUI custom node graphs require divergent API wrappers and separate execution servers.
- **Fallback Implemented:** Attention sharing was approximated via M8 Reference-Only ControlNet units, and identity conditioning was realized via M4 Dual InstantID and M5 Direct InsightFace Inswapper within Forge.

### 1.3 Local VLM Visual Auditor
- **Status:** Evaluated and substituted with deterministic CLIP zero-shot and SegFormer probes.
- **Root Cause:** Heavy multi-modal VLMs (e.g., Qwen2-VL, Moondream) allocate 3.5 to 5.0 GB of VRAM during model loading, creating direct VRAM contention with SDXL and ControlNet units on an 8 GB GPU.
- **Rule 9 Resolution:** Auditing visual attributes (clothing color, hair color, person count) runs deterministically on CPU using `mattmdjaga/segformer_b2_clothes` (CIELAB $\Delta E$) and ultralytics YOLOv8n, avoiding GPU memory spikes entirely.

---

## 2. Environmental & Runtime Constraints

### 2.1 GPU Memory Management (8 GB VRAM Budget)
- **Hardware:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8192 MB physical VRAM).
- **Safe Peak Ceiling:** 7400 MB.
- **Engineering Countermeasure:**
  1. WebUI Forge launched with `--api --xformers --cuda-malloc`.
  2. All evaluator models (DeepFace 4-model biometric ensemble, SegFormer segmentation, DINOv2, YOLOv8n) run strictly on CPU (`device="cpu"`).
  3. Direct InsightFace `inswapper_128.onnx` executed via subprocess in Forge's embedded Python strictly on CPU in ~8 seconds, consuming 0 MB GPU VRAM.
  4. Real-time background telemetry daemon (`scratch/vram_logger.py`) continuously verified that peak VRAM during diffusion remained below 6380 MB.

### 2.2 Windows 11 Power Management
- **Issue:** Windows 11 default AC power plans put the laptop into modern standby or hibernation after 15–30 minutes of terminal inactivity, suspending background tasks.
- **Resolution:** Executed `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0` at Phase 0 pre-flight. Original settings recorded for restoration at job completion.

### 2.3 Python Encoding on Windows (CP1252 vs UTF-8)
- **Issue:** Windows PowerShell defaults to code page 1252. Printing UTF-8 characters (e.g. mathematical symbols, emojis, formatted logs) throws `UnicodeEncodeError`.
- **Resolution:** Wrapped `sys.stdout` and `sys.stderr` with `io.TextIOWrapper(..., encoding="utf-8", errors="replace")` across all runner scripts and test entry points. Merged via PR #4 (`fcfaaf5`).
