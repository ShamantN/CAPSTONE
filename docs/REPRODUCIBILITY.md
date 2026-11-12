# Reproducibility & Verification Guide (`MMCCP v2.0`)

This guide specifies exact commands, split boundaries, hardware parameters, and non-repository dependencies required to verify published metrics or generate fresh DEV outputs.

---

## 1. Non-Repository Assets (Where They Live)
This repository tracks code, storyboards, preflight tools, verification JSON/CSV results, and markdown docs — **never** large model binaries, identity caches, or raw bulk render PNGs:
- **SDXL / ControlNet / IP-Adapter / InsightFace / YOLO Weights:** Local WebUI Forge installation (default Windows path `C:\Stable-Diffusion\` or `FORGE_ROOT`) and HuggingFace/InsightFace user caches (`~/.cache/huggingface`, `~/.insightface`). Exact URLs, sizes, and SHA-256 checksums are in [`docs/run/04_MODELS_AND_DOWNLOADS.md`](run/04_MODELS_AND_DOWNLOADS.md).
- **Forge Compatibility Patches:** [`patches/forge/`](../patches/forge/README.md), applied idempotently via `my_app_env\Scripts\python.exe tools\apply_forge_patches.py --apply`.
- **Generated Identity Banks & Cache (`gitignored`):** `consistency_v2/banks/` and `hardcoded_cache.json` (restored deterministically from committed ground-truth base64 JSONs in [`experiments/clean_run_ground_truth/anchors/`](../experiments/clean_run_ground_truth/anchors/)).
- **Telemetry CSVs:** `results/telemetry/vram_*.csv` (`9` files, `157,357` samples total; sealed `TEST-2` trace = `vram_test2_sealed.csv`).

---

## 2. Deterministic CPU-Only Reproduction (No GPU / No Forge Required)

Run these Windows PowerShell commands from the repository root using the project Python `3.10` virtual environment (`my_app_env\Scripts\python.exe`, or with `my_app_env` activated):

```powershell
# 1. Verify Frozen Verifier v1.2 integrity (SHA: 3fa69071f888643b...)
my_app_env\Scripts\python.exe tools\check_freeze.py

# 2. Verify TEST-2 & reserved seed isolation across Python code
my_app_env\Scripts\python.exe tools\audit_test2_isolation.py

# 3. Restore canonical 8-character banks + verify 1.0000 cosine similarity
my_app_env\Scripts\python.exe tools\sync_canonical_banks.py

# 4. Run full CPU unit test suite + verifier control suite
my_app_env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
my_app_env\Scripts\python.exe -m unittest -v tests.verifier_controls.test_controls

# 5. Recompute paired statistics, collect results/numbers.json, re-render docs & lint numeric tokens
my_app_env\Scripts\python.exe tools\compute_test2_statistics.py
my_app_env\Scripts\python.exe tools\collect_numbers.py
my_app_env\Scripts\python.exe tools\render_docs.py
my_app_env\Scripts\python.exe -m unittest tests\test_docs_numbers.py
my_app_env\Scripts\python.exe tools\lint_claims.py
```

Expected results:
- `tools/check_freeze.py` exits `0` (`Version 1.2`).
- `tools/audit_test2_isolation.py` exits `0`.
- `unittest discover` runs `85` tests with `0` failures (`83` baseline + blind-scorer & claim-linter suites, or verify live count on your checkout).
- `tests/test_docs_numbers.py` and `tools/lint_claims.py` verify quantitative claims match `results/numbers.json`.

---

## 3. Reproducing DEV Renders (Requires Forge WebUI + >= 8 GB CUDA GPU)

> [!CAUTION]
> **Split & Seed Rules:** Always use `DEV` splits (`DEV-2` seeds `101–505`, `DEV-FAST` seeds `{11, 23, 37}`, or demo seeds `4101–4299`). **Never** set `ALLOW_TEST2=1` or tune on `TEST-2` (`1101–1505`, `storyboards/holdout_v2.json`), and **never** use reserved seeds `{606, 707, 808}`.

```powershell
# Preflight environment, Forge API, weights & patches
my_app_env\Scripts\python.exe tools\verify_setup.py

# Render DEV-2 comparison bundle + run leave-one-out ablation on DEV-2
my_app_env\Scripts\python.exe tools\render_dev2_bundle.py
my_app_env\Scripts\python.exe tools\score_ablation_dev2.py

# Or run an end-to-end 1-scene smoke test from scratch
my_app_env\Scripts\python.exe run_from_scratch.py --pipeline v2 --mode default --scene 1 --seed 11
```

---

## 4. Streamlit UI (`app.py`) Defaults vs Locked `best_config.json`

| Parameter / Surface | `consistency_v2/best_config.json` | `app.py` Interactive UI Default | Notes |
| :--- | :---: | :---: | :--- |
| `enable_dual_instantid` (`M4b`) | `False` | `False` | Fixed in `app.py:696` (`v2_m4` toggle default updated from `True` -> `False`) |
| `enable_ip_adapter_plus` (`M4`) | `True` (`0.55`) | `True` (`0.55`) | Inherited from `ConsistencyV2Config` (not exposed as a toggle) |
| Resolved `ConsistencyV2Config` Hash | `df4a0a2648acdfe1` | `df4a0a2648acdfe1` | Exact parity (`compute_hash()`) in `default` mode |
| Render Inputs & Tab 4 Metrics | `TEST-2` (`1101–1505`, `holdout_v2.json`) | `ebright.json` (`seed=11`); Tab 4 shows legacy static cards (`app.py:1004-1007`) | Interactive `app.py` studio is not an evaluation runner |

---

## 4. Reference Hardware & Runtime Baseline
- **Tested Workstation:** Intel Core i7-12650H (10 cores / 16 threads), 32 GB DDR4 RAM, NVIDIA GeForce RTX 3070 Ti Laptop GPU (`8192` MiB GDDR6 VRAM, 140W TGP), Windows 11 Home, PyTorch `2.3.1+cu121`, Forge started with `--api --xformers --cuda-malloc`.
- **Measured Timings (`results/telemetry_clean_slate_report.json`):** `93.72s/scene` (`562.31s` for 6 scenes in `v2 default`), `114.38s/scene` (`686.26s` in `v2 quality`), vs `94.82s/scene` (`v1` baseline).
- **Measured VRAM (`results/telemetry/vram_test2_sealed.csv`):** Whole-system idle baseline `973` MiB, median `6247.0` MiB, p95 `7265.0` MiB, peak **`7984` MiB** (`7.797` GiB, `52`s > `7400` MiB, `0` OOMs).

