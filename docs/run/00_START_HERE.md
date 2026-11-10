# MMCCP v2.0 — Start Here (`docs/run/`)

- **Verification Status:** `NAVIGATION GUIDE — TESTED` (Windows fresh-clone workflow executed in [TEST_LOG.md](TEST_LOG.md); Linux & Google Colab paths marked `UNTESTED`)
- **Target Release / Commit:** Branch `docs/run-guides` (on top of `main` `625b365` / reconciled release candidate `v2.0.0-rc3`)

---

## What This Project Does (In 5 Plain Sentences)

MMCCP (**Multi-Modal Character Consistency Pipeline**) turns a written story JSON into a sequence of cinematic illustrated panels while keeping each character's face and outfit consistent across scenes. Instead of fine-tuning expensive custom LoRAs per character, it works zero-shot on consumer 8 GB GPUs by combining Stable Diffusion XL (SDXL) through WebUI Forge with skeleton pose layouts, regional attention masking, IP-Adapter conditioning, face restoration/swapping, and color-space wardrobe checks. Each run starts either from zero (generating character reference portraits first) or from a canonical anchor bank, and renders every scene through a deterministic two-pass diffusion pipeline. A separate frozen CPU verification suite (`v1.2`) then grades face identity (using four independent biometric models), left/right spatial placement, clothing color accuracy, and visual defect counts. This repository also preserves the reproducible benchmark artifacts and honest scientific scorecard comparing MMCCP v2.0 against Baseline v1.

---

## Choose Your Path (3 Ways to Run MMCCP)

| Option | Target Hardware | Expected Setup Time | First Render Time | Cost | Setup Effort | Status | Next File to Open |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **A. New Windows 10/11 PC** | NVIDIA GPU >= 8 GB VRAM (tested on RTX 3070 Ti 8 GB) + 16–32 GB RAM + ~45 GB disk | ~30–60 min (~15 min `venv`/pip + model downloads) | ~3–5 min (1-scene smoke test: `174.9s` render / `312.7s` w/ anchors); ~18–20 min (6 scenes) | Electricity only ($0 cloud) | Moderate (Python 3.10 + Forge + weights) | **TESTED** (repo/venv/DEV render in `TEST_LOG.md`; empty-PC Forge install *UNTESTED*) | [01_NEW_MACHINE_WINDOWS.md](01_NEW_MACHINE_WINDOWS.md) |
| **B. Linux PC (Ubuntu/Debian CUDA)** | NVIDIA GPU >= 8 GB VRAM + 16–32 GB RAM | ~30–60 min | Same order as Windows | $0 cloud | Moderate | **UNTESTED** (portability layer unit-tested with mocks) | [02_LINUX_NOTES.md](02_LINUX_NOTES.md) |
| **C. Google Colab Notebook** | Free T4 (16 GB VRAM) or Colab Pro (L4/A100) + >= 30 GB Google Drive | ~20–40 min first run (~5 min on reconnect via Drive cache) | ~2–4 min/scene (T4 speeds unmeasured here) | $0 Free Tier (dynamic session limits) or Colab Pro ($9.99/mo) | Low (guided cells in Colab) | **UNTESTED** (validated JSON/syntax locally; see checklist) | [03_GOOGLE_COLAB.md](03_GOOGLE_COLAB.md) & [`MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb) |

> **Just want to score existing images without a GPU?** Skip Forge entirely — you can run the CPU-only verification & statistics pipeline (`tools/check_freeze.py`, `unittest discover tests`, `python -m consistency_v2.verify ...`, `tools/collect_numbers.py`). See [07_FAQ_AND_COSTS.md](07_FAQ_AND_COSTS.md#running-only-the-cpu-verifier-no-forge-gpu).

---

## What You Get When a Run Finishes

1. **Character Anchors (`hardcoded_cache.json` + `experiments/fresh_run_from_scratch/anchors/*.png`):** A 1024x1024 portrait for each story character created from text descriptions before scene rendering starts.
2. **Storyboard Panels (`edu_outputs/v2_default/scene_<N>_seed_<S>.png` and `edu_outputs/images/scene_<N>.png`):** Rendered 1024x1024 (solo) and 1216x832 (multi-character) PNG panels ready to view in Streamlit or any image viewer.
3. **Structured Sidecar Telemetry (`edu_outputs/v2_default/scene_<N>_seed_<S>.json`):** Per-scene runtime, active method flags, face-swap specs, wardrobe CIELAB Delta-E repairs, and memory promotion logs.
4. **Verifier Scorecard (`report_v1_2.json` when scoring a bundle):** Machine-generated JSON/CSV grades for 4-judge identity similarity, Rank-1 identification, left/right spatial pass rate, wardrobe Delta-E pass rate, and defects per scene.

---

## Map of Documentation (`docs/run/`)

| File | Purpose | Audience | Status |
| :--- | :--- | :--- | :---: |
| [00_START_HERE.md](00_START_HERE.md) | Overview, 3-path decision table, outputs map *(this page)* | Everyone | **TESTED** |
| [01_NEW_MACHINE_WINDOWS.md](01_NEW_MACHINE_WINDOWS.md) | Step-by-step Windows setup, Forge launch, smoke test, full demo, Streamlit UI, paper reproduction | Fresh Windows setup | **TESTED** (empty-PC Forge install *UNTESTED*) |
| [02_LINUX_NOTES.md](02_LINUX_NOTES.md) | Shell, Forge `webui.sh`, `/proc/meminfo`, and frozen `torch_bridge` notes for Linux | Linux CUDA users | **UNTESTED** |
| [03_GOOGLE_COLAB.md](03_GOOGLE_COLAB.md) | Drive-backed Colab guide, secret setup, resume workflow, and first-run checklist | Cloud / laptop without GPU | **UNTESTED** |
| [`notebooks/MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb) | Ready-to-open Colab notebook (cells 0–12) | Colab users | **UNTESTED** |
| [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) | Exact filenames, verified download URLs, byte sizes, licenses, destinations, and SHA-256 checksums | Anyone downloading weights | **VERIFIED** (URLs + local SHA-256) |
| [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md) | 29 real "Symptom -> Cause -> Fix" recipes from project logs | Debugging errors | **TESTED / LOG-VERIFIED** |
| [06_HOW_IT_WORKS_FOR_BEGINNERS.md](06_HOW_IT_WORKS_FOR_BEGINNERS.md) | Diagram-led plain-language architecture, ON/OFF module choices, honest results, and glossary | Newcomers & reviewers | **TESTED** (docs/metrics checked) |
| [07_FAQ_AND_COSTS.md](07_FAQ_AND_COSTS.md) | Stage timings, Colab cost tiers, `< 8 GB` GPU reality, CPU-only scoring commands | Planning & Q&A | **TESTED** (local timings) |
| [TEST_LOG.md](TEST_LOG.md) | Complete command-by-command transcript from testing the fresh-clone Windows workflow | Maintainers | **TESTED** |
| [REVIEW.md](REVIEW.md) | Pre-merge self-review checklist, `NEEDS SOURCE` and `UNTESTED` registers | Maintainers | **TESTED** |

