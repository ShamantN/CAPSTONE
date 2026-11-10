# MMCCP Consistency v2: Master Progress & Execution Log

**Project:** Multi-Modal Character Consistency Pipeline (MMCCP), part of "Novel to Video Generation" (PES University Capstone Phase 3, Project ID: `PW26_CGM_02`)  
**Hardware Platform:** Intel Core i7-12650H, 32 GB DDR4, NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB VRAM, 7.4 GB safe budget)  
**Host OS:** Windows 11 Home, PowerShell  
**Repository:** `https://github.com/ShamantN/CAPSTONE.git`  
**Active Integration Branch:** `consistency-v2`  
**Baseline Commit Tag:** `baseline-v1`  
**Verifier Freeze Tag:** `verifier-frozen`  

---

## 1. Timeline & Milestone Log

### Phase 0: Setup, Repository Pre-Flight & Baseline Execution
- **Pre-flight & Tooling Check:**
  - Verified `git` (2.44.0.windows.1), `git lfs` (3.4.1), and `gh` CLI (2.81.0).
  - Confirmed authentication via `gh auth status` (authenticated as `ShamantN`).
  - Confirmed Git identity configured repo-locally (`ShamantN` with GitHub noreply email).
  - Case C scenario identified: local repo linked to remote `origin`. Committed local baseline onto branch `sync/local-baseline`, tagged `baseline-v1`, pushed, and opened PR #1 to `main` (left open for user review per G6).
  - Created and pushed integration branch `consistency-v2`.
- **Repository Hygiene (G2):**
  - Configured comprehensive `.gitignore` excluding Python caches, venvs, weights, `.safetensors`, `.onnx`, large temporary data, and sensitive environment keys.
  - Configured `.gitattributes` with Git LFS tracking for curated images (`docs_generated/**/*.png`, `results/curated/**/*.png`).
  - Secret scan executed across entire tree: confirmed zero leaked tokens, secrets, or API keys.
  - Installed `.github/pull_request_template.md` and lightweight `.github/workflows/ci.yml`.
- **Hardware & Power Management:**
  - Disabled Windows AC sleep and hibernate timeouts via `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`. Original settings recorded for restoration.
- **Image Viewing Canary:**
  - Verified visual inspection capability by loading and analyzing known anchor images and deliberately inverted canaries via PIL/OpenCV and contact sheet renderers.
- **Baseline v1 Generation & Frozen Evaluation:**
  - Executed complete DEV-FULL baseline generation (18 renders across 6 scenes $\times$ 3 seeds: {11, 23, 37}) using baseline v1 pipeline.
  - Zero crashes, zero OOMs, peak VRAM measured at 6179 MB.
  - Evaluated with frozen verifier on CPU:
    - Biometric Ensemble Mean: **0.6190** (95% CI: [0.5938, 0.6416])
    - Rank-1 Re-ID Accuracy: **95.8%**
    - Spatial & Wardrobe Binding Pass Rate: **25.0%** (75% binding failure in baseline v1!)
    - Mean Defects: **3.22** defects per scene.
    - 18 high-resolution contact sheets generated in `docs_generated/v2/visual_review/contact_sheets/`.

---

### Phase 1: Build & Calibrate Frozen Verification Suite (`feat/verifier`)
- Developed independent biometric evaluation engine in `consistency_v2/verify/` to resolve circularity (D5):
  - 4 distinct CPU judge models: DeepFace `Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`.
  - Built `BindingChecker` utilizing SegFormer (`mattmdjaga/segformer_b2_clothes`) on CPU for person/clothing segmentation and CIELAB $\Delta E$ color difference.
  - Built `AppearanceEvaluator` using DINOv2 (`facebook/dinov2-base`) and CLIP zero-shot attribute probes.
  - Built `SceneAuditor` with RetinaFace face detector, YOLOv8n person counter, and 14-point defect taxonomy checker.
  - Created contact sheet generator `make_contact_sheet.py` and blind re-ID generator `make_blind_sheet.py`.
- Calibrated genuine vs impostor distributions in `calibration.json`.
- Froze verifier code: recorded SHA256 (`cf7430de4ca7c531fffb3eb63f6fa22f30f8898a704814a8d3ec2bfcb314e83a`) in `consistency_v2/verifier_freeze.json`.
- Merged via PR #3 (`fc7c7f8`), tagged `verifier-frozen`, and pushed tag to origin. Merged Windows UTF-8 stdout fix via PR #4 (`fcfaaf5`).

---

### Phase 2: Diagnose & Fix Existing Pipeline Defects (D1–D6)
- **Defect D1 (ControlNet Inactivity):** Proved A03 and A04 renders were byte-identical (`32BED00C2766B2CB7988A174D2B080366CE0CE69BA9E892DF30EC1585A1F97C2`) due to missing input image. Fixed by M2 programmatic OpenPose skeleton generator.
- **Defect D2 (ReActor Payload Mismatch):** Proved 30 vs 31 argument mismatch from `/sdapi/v1/script-info`. Fixed by dynamic script-info builder and direct InsightFace Inswapper (`inswapper_128.onnx`) executing on CPU in Forge's Python.
- **Defect D3 (BREAK Token Bleed):** Proved `BREAK` token only splits CLIP 77-token chunks without spatial isolation. Fixed by installing `sd-forge-couple` extension.
- **Defect D4 (InstantID Incomplete Setup):** Downloaded missing IP-Adapter weights (`ip-adapter_instant_id_sdxl.bin`, 1.69 GB) and linked into Forge's ControlNet models directory to complete the dual-unit setup.
- **Defect D5 (Evaluator Circularity):** Resolved by independent DeepFace 4-model ensemble and spatial-aware Hungarian matching.
- **Defect D6 (Documentation Contradictions):** Documented code vs documentation discrepancies in `docs_generated/v2/claims_corrected.md`.

---

### Phase 3: Implement Consistency Methods (M1 to M10 Only)
All methods implemented, screened on DEV-FAST, and squash-merged to `consistency-v2` via individual PRs:
1. **M1 (Identity Bank):** Multi-angle reference bank with 4-point admission gate and L2 centroid embeddings. PR #5 merged (`31afaf8`).
2. **M2 (Layout Planner):** Non-overlapping horizontal zones with OpenPose COCO-18 skeleton control images. PR #6 merged (`32eea5d`).
3. **M3 (True Regional Conditioning):** Native Forge Couple driver formatting 17-argument API payload. PR #7 merged (`18c5190`).
4. **M4 (Identity Units):** Dual InstantID + IP-Adapter Plus-Face SDXL. PR #9 merged (`96035a7`).
5. **M5 (Reliable Face Transplant):** Direct InsightFace Inswapper + CodeFormer face restoration on CPU. PR #10 merged (`fe592e7`).
6. **M6 (Face Detailer):** Layout-aware high-resolution inpainting on small faces. PR #11 merged (`1bbe16b`).
7. **M7 (Wardrobe Lock):** SegFormer clothes segmentation, CIELAB $\Delta E$ evaluation, and bounded color calibration with strict facial pixel protection. PR #12 merged (`aabe03b`).
8. **M8 (Reference-Only Attention):** ControlNet reference_only unit with region-mask attention isolation. PR #13 merged (`18e76c5`).
9. **M9 (Best-of-N & Healing v2):** Geometric draft candidate screening and systematic diagnostic remedy table. PR #14 merged (`87c51e7`).
10. **M10 (Identity Memory):** High-similarity promotion criteria with centroid drift guard ($\le 0.05$). PR #15 merged (`67dada5`).

*Scope Guard strictly enforced:* M11 (LoRA) remains disabled (`use_character_lora=False`); reserved seeds {606, 707, 808} strictly protected from any use.

---

### Phase 4: Integration, UI v2 & Headless CLI
- Integrated all modules into unified pipeline `consistency_v2/pipeline_v2.py` and canonical `consistency_v2/best_config.json`. PR #16 merged (`228f898`).
- Updated Streamlit UI (`app.py`) with sidebar v1/v2 selector, preset modes (Fast, Default, Quality), M1-M10 toggles, and Tab 4 (`🔍 Frozen Verifier & Contact Sheets`).
- Updated `run_from_scratch.py` supporting `--pipeline v1|v2` and `--mode fast|default|quality`.
- PR #17 merged (`29953c7`). 44/44 unit tests passing cleanly.

---

### Phase 5: Final Held-Out Evaluation (TEST Set)
- Execution of `consistency_v2/run_phase5_evaluation.py` on held-out TEST data splits:
  - 30 renders across 6 Ebright scenes $\times$ seeds {101, 202, 303, 404, 505}.
  - 15 renders across 5 Chrono-Archivists scenes $\times$ seeds {101, 202, 303}.
  - Quality mode renders on seed 101.
  - Leave-one-out ablation study across M1 to M8 on seeds {101, 202, 303}.
  - Complete frozen verifier scoring, contact sheet generation, bootstrap 95% CIs, paired Wilcoxon signed-rank tests against baseline v1, and full `results/MANIFEST.json` artifact registration.
