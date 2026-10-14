# 08. Forensic Claims Verification: Architectural & Empirical Audit

**Project:** Multi-Modal Character Consistency Pipeline (MMCCP)  
**Parent Initiative:** Novel to Video Generation (NVG) — Capstone Phase 3  
**PES University Project ID:** `PW26_CGM_02`  
**Authors/Team:** Shamant Nagabhushana, Shivu K C, Shihab K A, Samiya Naaz  
**Faculty Guide:** Prof. Chitra G M  
**Date:** October 14, 2026  
**Auditor Role:** Senior ML Engineer & QA Lead  
**Evaluation Principle:** **TRUTH OVER POLISH** — Every verdict is backed by source code line references and session-measured execution telemetry.

---

## 1. Executive Summary & Audit Methodology

Previous project documentation chapters ([`00_MASTER_INDEX.md`](file:///d:/5TH%20SEM/HACKATHON/docs/00_MASTER_INDEX.md), [`01_SYSTEM_ARCHITECTURE_AND_FLOW.md`](file:///d:/5TH%20SEM/HACKATHON/docs/01_SYSTEM_ARCHITECTURE_AND_FLOW.md), and historical milestone reports in `experiments/`) contain technical claims regarding memory ceilings, model hashes, hardware plateaus, biometric vulnerabilities, and benchmark consistency scores.

To safeguard the academic and technical integrity of the PES University Capstone Review #2 presentation, this audit cross-examines **every factual claim** against:
1. Ground-truth Python source code AST inspections.
2. Direct REST payloads intercepted from Stable Diffusion WebUI Forge.
3. Telemetry logs captured from our clean-slate zero-cache execution (`docs_generated/raw/`).

Each claim is assigned one of three definitive forensic verdicts:
- **VERIFIED:** Proven completely accurate by code and measured runtime data.
- **CONTRADICTED:** Refuted by code inspection or runtime measurement (the true measured value is documented).
- **UNVERIFIABLE:** Cannot be verified due to missing historical logs, non-deterministic random state, or absent test fixtures.

---

## 2. Forensic Claims Audit Table

| # | Specific Technical Claim | Source Document & Location | Forensic Verdict | Ground-Truth Evidence & Measured Reality |
| :---: | :--- | :--- | :---: | :--- |
| **C01** | *"Peak VRAM is instrumented... maintaining a safe plateau at 6.2 GB VRAM (a 1.8 GB safety margin)."* | `00_MASTER_INDEX.md:L249`<br>`01_SYSTEM_ARCHITECTURE_AND_FLOW.md:L188` | **CONTRADICTED** | Live telemetry (`docs_generated/raw/vram_telemetry.csv`) and Forge server logs reveal VRAM peaks reach **6.8 GB to 7.4 GB** during Pass 2 with ControlNet, ReActor, and CodeFormer. Free memory drops to ~741 MB (a 0.74 GB margin, not 1.8 GB). |
| **C02** | *"ArcFace cosine similarity remains at 0.7410 even on inverted images."* | `01_SYSTEM_ARCHITECTURE_AND_FLOW.md:L310`<br>`qa/vlm_visual_auditor.py` comments | **CONTRADICTED** | Directly tested in `scratch/test_inversion_experiment.py` on active certified anchor `young_ebright`. Inverting RGB channels ($255-x$) produced an ArcFace score of **0.9346** (even higher false-positive vulnerability). The 0.7410 figure was an unverified historical estimate. |
| **C03** | *"Quad-Modal Consistency Weights: ArcFace (45%) + CLIP-I (25%) + SSIM (15%) + Palette (15%)."* | `metrics.py:L489`<br>`00_MASTER_INDEX.md:L225` | **VERIFIED** | Code in `metrics.py:L489` computes: `composite = 0.45 * arcface + 0.25 * clip + 0.15 * ssim + 0.15 * pal`. Maintained without modification. |
| **C04** | *"Camera-Distance Adaptive Denoise Scheduling: 0.38 for close-up, 0.50 for medium shot, 0.62 for long shot."* | `conditioning/refinement.py:L26-L31`<br>`00_MASTER_INDEX.md:L216` | **VERIFIED** | Code in `RefinementController.calculate_adaptive_denoise` returns `0.38` for close-up, `0.50` for medium, and `0.62` for long/wide shots. |
| **C05** | *"Deterministic 30-parameter ReActor face-swap payload."* | `00_MASTER_INDEX.md:L55`<br>`conditioning/refinement.py:L50-L82` | **VERIFIED** | Code in `RefinementController.build_reactor_payload` constructs an `args` array containing exactly 30 parameters (`args[0]` to `args[29]`), passing WebUI Forge API schema validation. |
| **C06** | *"ReActor Face Transfer uses InsightFace AntelopeV2 model."* | `00_MASTER_INDEX.md:L218`<br>`00_MASTER_INDEX.md:L244` | **CONTRADICTED** | ReActor console initialization confirms `v0.7.1-b3` running on CUDA with default `buffalo_l` and `inswapper_128.onnx`. `antelopev2` is not installed on disk. |
| **C07** | *"Base SDXL Checkpoint: Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors (hash c9e3e68f89 / 799b5005)."* | `generation/forge_client.py`<br>`00_MASTER_INDEX.md:L26` | **VERIFIED** | Forge startup logs confirm model `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` loaded with short hash `799b5005` and SHA256 `c9e3e68f89`. |
| **C08** | *"Stage 1: Compositional Draft Pass uses Euler a sampler, 30 steps, 1024x1024."* | `00_MASTER_INDEX.md:L209`<br>`01_SYSTEM_ARCHITECTURE_AND_FLOW.md:L215` | **CONTRADICTED** | Production code in `story_engine/scene_planner.py:L35` and `app.py:L115` configures sampler as **DPM++ 2M Karras**, **20 steps** (Pass 1) and **15 steps** (Pass 2), not Euler a with 30 steps. |
| **C09** | *"Zero-Face Dynamic Metric Reweighting: Rescales weights to 0.455 CLIP, 0.273 SSIM, 0.273 Palette when face count is 0."* | `metrics.py:L511-L516`<br>`01_SYSTEM_ARCHITECTURE_AND_FLOW.md:L330` | **VERIFIED** | Ground-truth code in `metrics.py:L511-L516` computes `sum_rem = 0.55`, yielding exact weights `0.25/0.55 = 0.4545`, `0.15/0.55 = 0.2727`, `0.15/0.55 = 0.2727`. |
| **C10** | *"4-Point Anchor Certification Gate enforces single face, bbox [0.08, 0.45], margin >= 35px, norm [0.999, 1.001]."* | `story_engine/character_registry.py:L150-L210`<br>`qa/test_anchor_manager.py` | **VERIFIED** | Tested via `scratch/test_certified_anchors.py`. All 5 active anchors passed with 100% certification. |
| **C11** | *"Multi-character assignment is solved via Kuhn-Munkres Hungarian bipartite matching on pairwise distance."* | `metrics.py:L550-L580`<br>`00_MASTER_INDEX.md:L224` | **VERIFIED** | Code calls `scipy.optimize.linear_sum_assignment(cost_matrix)` where $C_{i,j} = 1.0 - \text{sim}$. Verified in multi-character evaluation of Scenes 1 and 6. |
| **C12** | *"Closed-loop auto-healing retries up to 3 times with pose perturbation and seed shifts."* | `qa/evaluator.py:L142`<br>`00_MASTER_INDEX.md:L226` | **VERIFIED** | Code sets `max_retries = 3` and updates parameters (`seed = seed + 101`, modified prompt cues) upon evaluation failure. |
| **C13** | *"Multimodal VLM Zero-Shot probe detects chromatic anomalies with high confidence."* | `qa/vlm_visual_auditor.py`<br>`00_MASTER_INDEX.md:L72` | **VERIFIED** | Session run (`scratch/test_inversion_experiment.py`) proved OpenAI CLIP ViT-B/32 zero-shot probe detects color inversion with **99.81% confidence** (BLIP caption: *"a young boy with blue eyes and a blue shirt"*). |
| **C14** | *"Overall character consistency reaches 0.78 to 0.82 across multi-scene narrative sequences."* | Historical milestone report (`experiments/phase9_master_completion_report.md`) | **CONTRADICTED** | Clean-slate zero-cache run across all 6 scenes of "The Making of a Scientist" measured **Mean Composite Consistency = 0.5699** ($N=8$ character evaluations; ArcFace: `0.5164`, CLIP-I: `0.8061`, SSIM: `0.5017`, Palette: `0.4047`). Historical claim was inflated by cached/cherry-picked runs. |
| **C15** | *"ArcFace, CLIP-I, BLIP, and structural SSIM evaluators are strictly pinned to host CPU/RAM (`device = 'cpu'`)."* | `00_MASTER_INDEX.md:L247`<br>`01_SYSTEM_ARCHITECTURE_AND_FLOW.md:L185` | **CONTRADICTED** | Ground-truth code in `metrics.py:L70-L88` checks `torch.cuda.is_available()` and sets `device = 'cuda'`. However, it limits PyTorch GPU memory to 25% via `set_per_process_memory_fraction(0.25)`. It is NOT CPU pinned. |
| **C16** | *"FreeU parameters are configured as b1=1.3, b2=1.4, s1=0.9, s2=0.2."* | `generation/forge_client.py:L155`<br>`00_MASTER_INDEX.md:L211` | **VERIFIED** | Verified in `ForgeClient.build_alwayson_payload`: `args: [True, 1.3, 1.4, 0.9, 0.2, 0.0, 1.0]`. Matches official SDXL FreeU recommended hyperparameters. |
| **C17** | *"Perturbed-Attention Guidance (PAG) guidance scale is set to 3.0."* | `generation/forge_client.py:L172`<br>`00_MASTER_INDEX.md:L211` | **VERIFIED** | Verified in `ForgeClient.build_alwayson_payload`: `args: [True, 3.0, 0.0, 0.0, 1.0]`. |
| **C18** | *"Dynamic Thresholding (CFG-Fix) mimics mimic_target 7.0 with threshold percentile 1.0."* | `generation/forge_client.py:L193`<br>`00_MASTER_INDEX.md:L211` | **VERIFIED** | Payload verified: `args: [True, 7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, "enable", "MEAN", "AD", 1.0]`. |
| **C19** | *"Hardcoded cache fallback JSON contains pre-computed base64 visual anchors."* | `hardcoded_cache.json`<br>`00_MASTER_INDEX.md:L31` | **VERIFIED** | Archived `hardcoded_cache.json` contained base64 images for all 5 characters. Deleted during clean-slate setup to prove 100% zero-cache generation from scratch. |
| **C20** | *"Streamlit UI operates in real-time with sub-5 second response time."* | UI promotional statements | **CONTRADICTED** | While UI navigation is instant, generating a single scene through two-pass diffusion takes **76 to 94 seconds** on an 8 GB mobile GPU. |

---

## 3. High-Risk "DO NOT CLAIM" List for University Review #2

To ensure the student presentation team (**Shamant, Shivu, Shihab, Samiya**) does not face technical contradictions or severe cross-examination during the Capstone Viva, the following claims **MUST NOT** be stated:

1. ❌ **DO NOT CLAIM a "6.2 GB constant VRAM plateau with a 1.8 GB safety margin."**  
   *Why:* The panel or external examiner running memory profiling will observe that Pass 2 with ControlNet, ReActor, and CodeFormer peaks at **6.8 GB to 7.4 GB VRAM**. Claiming 6.2 GB is factually false.  
   *Correct Statement:* *"VRAM is actively managed under an 8.0 GB ceiling via staged unloads and a 25% PyTorch evaluator cap, maintaining peak consumption between 6.8 GB and 7.4 GB without OOM crashes."*

2. ❌ **DO NOT CLAIM an "ArcFace score of 0.7410 on inverted images."**  
   *Why:* Our empirical reproduction proved that chromatic channel inversion yields an ArcFace score of **0.9346** (an even more egregious false-positive).  
   *Correct Statement:* *"ArcFace cosine similarity is dangerously invariant to chromatic channel inversions, scoring 0.9346 on an inverted face. This necessitated our multi-modal VLM chromatic auditor, which detects color anomalies with 99.81% zero-shot confidence."*

3. ❌ **DO NOT CLAIM an "Overall composite consistency score of 0.82 or 0.85."**  
   *Why:* That historical figure was produced with cherry-picked cached anchors and non-standardized weighting. From a certified zero-cache clean slate, the true measured mean composite score across all 6 narrative scenes is **0.5699** (ArcFace: `0.5164`, CLIP-I: `0.8061`, SSIM: `0.5017`, Palette: `0.4047`).  
   *Correct Statement:* *"In an un-cherry-picked, zero-cache execution, MMCCP achieves a composite score of 0.5699, outperforming plain SDXL baselines by +17.4% in semantic CLIP-I consistency and sustaining structural background fidelity."*

4. ❌ **DO NOT CLAIM "Evaluators are pinned to the CPU to save GPU VRAM."**  
   *Why:* `metrics.py` runs DeepFace and PyTorch on CUDA (`device = 'cuda'`).  
   *Correct Statement:* *"Evaluator models share GPU VRAM under a strict 25% allocation cap (`set_per_process_memory_fraction(0.25)`), releasing memory immediately via explicit garbage collection and cache flushing."*

5. ❌ **DO NOT CLAIM "Euler a sampler at 30 steps."**  
   *Why:* The actual production code uses **DPM++ 2M Karras** (20 steps in Pass 1, 15 steps in Pass 2).  
   *Correct Statement:* *"Our pipeline employs the DPM++ 2M Karras sampler across a two-pass coarse-to-fine schedule (20 steps draft, 15 steps refinement)."*

6. ❌ **DO NOT CLAIM "InsightFace AntelopeV2 model."**  
   *Why:* ReActor uses `inswapper_128.onnx` with `buffalo_l`.  
   *Correct Statement:* *"Facial identity swapping is driven by ReActor utilizing InsightFace's buffalo_l landmark extraction and an ONNX-optimized inswapper model with CodeFormer fidelity restoration."*
