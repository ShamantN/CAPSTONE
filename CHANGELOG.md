# Changelog: Multi-Modal Character Consistency Pipeline (MMCCP)

All notable changes are recorded strictly from repository tags (`git tag`) and commit history (`git log`) in accordance with [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standards and [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased] - 2026-11-12 (`finalize/v2.0.0`)

### Added & Updated
- Top-level [`README.md`](README.md) rewritten with team/guide/course metadata (`PW26_CGM_02`), accurate `M1–M12` ON/OFF configuration matrix, hardware summary, `results/numbers.json` scorecard, and current `v2.0.0-rc3` release status.
- Added [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) documenting limitations `(a)–(n)` with artifact links.
- Added [`CITATION.cff`](CITATION.cff), [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), [`docs/demo/METHODS_USED.md`](docs/demo/METHODS_USED.md), [`docs/demo/scorecard.csv`](docs/demo/scorecard.csv), [`docs/maintainers/REPO_SETTINGS.md`](docs/maintainers/REPO_SETTINGS.md), and `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`.
- Updated [`docs/run/LICENSES_AND_MODELS.md`](docs/run/LICENSES_AND_MODELS.md) against live HuggingFace/GitHub APIs (`2026-11-12`), correcting WebUI Forge (`AGPL-3.0`), SDXL-Lightning (`openrail++`), and `Juggernaut-XL-v9` (`creativeml-openrail-m`), and marking unverified HF metadata cards explicitly.
- Updated [`docs/run/MAINTAINER_SETUP.md`](docs/run/MAINTAINER_SETUP.md) with Options A and B addressing solo-owner PR self-approval constraints.
- Updated [`tools/render_docs.py`](tools/render_docs.py) and `docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md` to use portable relative repository links instead of workstation-specific local `file://` URLs.

---

## [PR #36] - 2026-11-12 (`0a44e50`)
- Merged collaborator guardrails and governance bundle on `main`: `CONTRIBUTING.md`, `.github/CODEOWNERS`, `.github/pull_request_template.md`, `docs/run/LICENSES_AND_MODELS.md`, and `docs/run/MAINTAINER_SETUP.md`.

## [PR #35] - 2026-11-10 (`36f6646`)
- Merged complete from-scratch execution guides (`docs/run/00_START_HERE.md` through `07_FAQ_AND_COSTS.md`, `TEST_LOG.md`, `REVIEW.md`, `HANDOFF_CHECKLIST.md`), portability layer (`consistency_v2/runtime_paths.py`), preflight checker (`tools/verify_setup.py`), idempotent Forge patch tool (`tools/apply_forge_patches.py` + `patches/forge/*.patch`), and Colab notebook (`notebooks/MMCCP_Colab.ipynb`, marked `UNTESTED`).

## [v2.0.0-rc3] - 2026-11-10 (`6509ac1` / PR #32–#34 on `main` `625b365`)
- Completed honest finalization and pre-registration reconciliation (`results/numbers.json`, `tools/collect_numbers.py`, `tools/render_docs.py`, `tests/test_docs_numbers.py`).
- Retracted `v2.0.0-rc2` post-hoc scorecard shifts (`7904ad5`), separated `defects_per_scene` vs `defects_per_instance_weighted`, logged `0/4` strict pre-registered hypotheses (`P1–P4`), reported whole-system peak VRAM (`7984` MiB / `7.797` GiB), and staged the human blind evaluation pack (`HUMAN_BLIND_TEST.md`).

## [v2.0.0-rc2] - 2026-11-06 (`5a0c7cf`, superseded by `v2.0.0-rc3`)
- Ran sealed one-shot `TEST-2` evaluation (`Ebright` `30` scenes + `Holdout` `12` scenes) and LOO ablation under Frozen Verifier `v1.2`.
- Documentation framing from `7904ad5` was later superseded and retracted in `v2.0.0-rc3` (`docs_generated/v2/PREREG_RECONCILIATION.md`).

## [verifier-v1.2-frozen] - 2026-10-18 (`fab7572` / PR #28)
- Froze Verifier `v1.2` (`SHA: 3fa69071f888643b...`, `15` control tests passing) adding `test2_seal.json` (`test2-sealed` on `2026-10-15`), calibrated thresholds, spatial prominence matching, and `m12` collage-init module (`feat/m12-collage-init`).

---

## [2.0.0-audit] - 2026-10-15 (`ca86e0e`, `a7935c4`)

### Audited & Retracted
- **Retraction of Ungrounded Claims:** Formally audited documentation associated with `2.0.0-rc1`. Moved contradicted and unverified claims to `docs_generated/v2/retracted/` and published `docs_generated/v2/RETRACTIONS.md`.
- **Identified Pipeline v2 Bottlenecks:** Verified and documented Problem Register A1–A13 and B1–B13 in `docs_generated/v2/repair/problem_register.md`.
- **Pre-Repair Baseline:** Confirmed baseline v1 operational and established safety pre-flight with tag `pre-repair-snapshot` (`ca86e0e`) and `verifier-v1.1-frozen` (`a7935c4`).

## [2.0.0-rc1] - 2026-10-14 (`5705f7e`, superseded by `2.0.0-audit` & `v2.0.0-rc3`)


### Added
- **M1 Multi-Reference Identity Bank (`consistency_v2/identity_bank.py`):**
  - Pose and scale clustering for nearest-neighbor reference selection.
  - Multi-model L2-normalized centroid face embeddings $\mathbf{e}^*_c$.
  - 4-point admission gate (single face, face width $\ge 96$px, Laplacian sharpness $\ge 60$, biometric similarity $\ge 0.50$, zero melted face artifacts).
  - Torso CIELAB palette reference extraction.
- **M2 Programmatic Layout Planner (`consistency_v2/layout_planner.py`):**
  - Deterministic non-overlapping spatial bounding boxes from storyboard specs (`character_positions`, `shot_type`, `camera_angle`).
  - Guaranteed minimum face size ($\ge 96$px for medium shots, $\ge 64$px for wide shots).
  - OpenPose COCO-18 skeleton control canvas generator fixing Defect D1.
  - Coarse foreground depth proxy generator.
- **M3 True Regional Conditioning (`consistency_v2/regional.py`):**
  - Native integration with `sd-forge-couple` extension via 17-argument API payload.
  - Spatial cross-attention isolation per character tile fixing Defect D3 (`BREAK` bleed).
  - Blended global background environment prompt support.
- **M4 Dual-Unit Identity Conditioning (`consistency_v2/identity_units.py`):**
  - Coordinated Dual InstantID (IP-Adapter embedding unit `ip-adapter_instant_id_sdxl.bin` + ControlNet keypoints unit `control_instant_id_sdxl.safetensors`) fixing Defect D4.
  - Integrated IP-Adapter Plus-Face SDXL (`ip-adapter-plus-face_sdxl_vit-h.safetensors`).
- **M5 Reliable Face Transplant (`consistency_v2/face_swap.py`):**
  - Direct InsightFace Inswapper (`inswapper_128.onnx`) running in Forge's Python on CPU, consuming 0 MB GPU VRAM and eliminating Defect D2.
  - Dynamic 31-argument ReActor API payload builder conforming to `/sdapi/v1/script-info`.
  - CodeFormer face restoration via Forge `/sdapi/v1/extra-single-image` endpoint.
- **M6 Identity-Conditioned Face Detailer (`consistency_v2/face_detailer.py`):**
  - Post-Pass 2 automatic detection of faces $< 96$px.
  - Soft-padded contextual cropping, upscaled inpainting at $denoise=0.35$, and seamless seam-blended paste-back.
- **M7 Wardrobe & Palette Lock (`consistency_v2/wardrobe.py`):**
  - Semantic person and clothing segmentation using SegFormer (`mattmdjaga/segformer_b2_clothes`) on CPU.
  - CIELAB color distance ($\Delta E$) auditing against StorySpec clothing palettes.
  - Bounded color calibration strictly leaving facial pixels byte-identical.
- **M8 Reference-Only Attention Injection (`consistency_v2/reference_unit.py`):**
  - Cross-image self-attention key/value injection via ControlNet `reference_only` units with layout zone masking (disabled by default, evaluated in ablation).
- **M9 Best-of-N Candidate Selection & Diagnostic Healing (`consistency_v2/candidates.py`, `consistency_v2/healing_v2.py`):**
  - Fast geometric screening of $N=3$ Pass 1 drafts in Quality mode.
  - Deterministic healing table mapping specific defect failure modes to targeted parameter remedies.
- **M10 Cross-Scene Identity Memory (`consistency_v2/identity_memory.py`):**
  - Drift-guarded promotion ($\Delta \le 0.05$, similarity $\ge 0.75$) of accepted face crops to identity bank.
- **Deterministic Frozen Verification Suite (`consistency_v2/verify/`):**
  - 4-model CPU judge ensemble: DeepFace `Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace` to eliminate evaluation circularity (Defect D5).
  - SegFormer semantic wardrobe matcher and spatial zone validator.
  - RetinaFace face detector, YOLOv8n person counter, and 14-point defect taxonomy checker.
  - Frozen under SHA256 in `verifier_freeze.json`, tagged `verifier-frozen`.
- **User Interface & CLI Extensions:**
  - Streamlit UI (`app.py`): Added Consistency v2 sidebar controls, preset mode toggles, and Tab 4 (`🔍 Frozen Verifier & Contact Sheets`).
  - Headless CLI (`run_from_scratch.py`): Added `--pipeline v1|v2` and `--mode fast|default|quality`.

### Fixed
- **D1:** Fixed inactive ControlNet Depth and OpenPose units during Pass 1 by providing synthetically rendered OpenPose skeleton control images.
- **D2:** Fixed ReActor parameter misalignment (30 vs 31 argument mismatch) and implemented direct CPU-based `inswapper_128.onnx` execution.
- **D3:** Replaced non-isolating prompt `BREAK` tokens with true U-Net cross-attention isolation using Forge Couple.
- **D4:** Downloaded missing InstantID IP-Adapter weights (`ip-adapter_instant_id_sdxl.bin`, 1.69 GB) and linked into ControlNet models directory.
- **D5:** Eliminated evaluator circularity by using independent DeepFace judges and position-aware spatial matching.
- **D6:** Reconciled sampler, step count, and denoising strength discrepancies between documentation and codebase.
- **Windows UTF-8 Encoding:** Added UTF-8 stdout/stderr stream wrappers across all runners to prevent CP1252 `UnicodeEncodeError`.

---

## [1.0.0] - 2026-10-14 (`bf27065` / `baseline-v1`)

### Added
- Initial baseline MMCCP pipeline running SDXL Juggernaut-XL_v9 via Stable Diffusion WebUI Forge.
- Two-pass generation architecture (txt2img Pass 1 + img2img Pass 2).
- Tagged `baseline-v1` on branch `sync/local-baseline`.
