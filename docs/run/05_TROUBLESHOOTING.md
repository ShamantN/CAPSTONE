# MMCCP v2.0 — Troubleshooting Guide (29 Real Failure Modes)

- **Verification Status:** `LOG-VERIFIED & CODE-GROUNDED` (every Symptom, Cause, and Fix below comes directly from repo error logs, `docs_generated/02_issues_and_fixes.md`, `docs_generated/v2/diagnostics.md`, `docs_generated/v2/blockers.md`, `docs_generated/v2/loop_log.md`, and `consistency_v2/` source code)
- **Baseline:** Branch `docs/run-guides` (on top of `main` `625b365` / `v2.0.0-rc3`)

---

## Quick Diagnostic First Step

Before debugging individually, always run:

```powershell
python tools\verify_setup.py --mode new-machine --fix-hints
```

*(Add `--skip-forge` if Forge is not running yet, or `--mode colab` on Linux/Colab.)*

---

## A. Environment, Paths & Python Packages

### 1. Hardcoded Windows Drive Paths Failing on New Machine (`ISSUE-02`)
- **Symptom:** `FileNotFoundError: D:\5TH SEM\HACKATHON\...` or `C:\Stable-Diffusion\...` when cloned to a different drive/folder.
- **Cause:** Earlier scripts hardcoded author workstation paths (`D:\5TH SEM\HACKATHON` / `C:\Stable-Diffusion`).
- **Fix:** Fixed in `consistency_v2/runtime_paths.py` (`commit 35e50ef`). Set environment variables explicitly when non-default:
  ```powershell
  $env:MMCCP_ROOT = (Get-Location).Path
  $env:FORGE_DIR = "C:\Stable-Diffusion"
  $env:FORGE_URL = "http://127.0.0.1:7860"
  ```

### 2. Linux / Colab Crash in `get_host_ram_mb()` (`ctypes.windll`)
- **Symptom:** `AttributeError: module 'ctypes' has no attribute 'windll'` inside `experiments/memory_utils.py`.
- **Cause:** Calling Win32 `GlobalMemoryStatusEx` on POSIX systems.
- **Fix:** Resolved in `commit 35e50ef`: `get_host_ram_mb()` now falls back to `/proc/meminfo` (`MemTotal` / `MemAvailable`) automatically on Linux/Colab.

### 3. Windows PowerShell UTF-8 Crash (`UnicodeEncodeError`)
- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'` (or `Δ`) in terminal output (`blockers.md` §2.3).
- **Cause:** Windows PowerShell defaults stdout/stderr to `cp1252`.
- **Fix:** Set UTF-8 mode in your session:
  ```powershell
  $env:PYTHONUTF8 = "1"
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  ```

### 4. NumPy 2.x Binary Compatibility Crash
- **Symptom:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` when importing `torch` or `insightface`, or `WardrobeLock` NumPy version mismatch.
- **Cause:** `pip` installed NumPy `2.x` while PyTorch `2.3.1` and InsightFace `0.7.3` target NumPy `1.26.x`.
- **Fix:** Pin NumPy `< 2.0`:
  ```powershell
  pip install "numpy>=1.26.0,<2.0.0"
  ```

### 5. `insightface==0.7.3` Build Failure (`Microsoft Visual C++ 14.0 is required`)
- **Symptom:** Installing `insightface` compiles `mesh_core_cython` and crashes if MSVC Build Tools or `g++` are missing.
- **Cause:** PyPI provides only a source tarball for `insightface==0.7.3`, requiring either a compiler or a matching prebuilt wheel.
- **Fix:** On Windows Python 3.10, either rely on Forge's embedded Python (`C:\Stable-Diffusion\system\python\python.exe`, which carries prebuilt `insightface-0.7.3-cp310-cp310-win_amd64.whl` + `onnxruntime-gpu==1.17.1`), or install that matching wheel / Visual Studio Build Tools. On Linux/Colab, install `sudo apt-get install -y build-essential g++ python3-dev` first.

### 6. Metric Function `ImportError` (`calculate_palette_similarity`) (`ISSUE-04`)
- **Symptom:** `ImportError: cannot import name 'calculate_palette_similarity' from 'metrics'`.
- **Cause:** Legacy alias vs `calculate_color_palette_similarity` in `metrics.py`.
- **Fix:** Already aliased in current `metrics.py`; pull latest branch (`main >= 625b365`).

### 7. Wrong Python Environment Launching Streamlit UI
- **Symptom:** `streamlit run app.py` raises `ModuleNotFoundError: No module named 'cv2'` or `'deepface'` or `'google.generativeai'`.
- **Cause:** Global `streamlit` command resolves outside your activated project `venv`.
- **Fix:** Invoke Streamlit through the active `venv` interpreter:
  ```powershell
  python -m streamlit run app.py
  ```

### 8. Streamlit `use_container_width=True` Console Warning Spam (`ISSUE-03`)
- **Symptom:** Logs print `Please replace use_container_width with width`.
- **Cause:** Streamlit `1.51.0` deprecation warning across buttons/columns/images.
- **Fix:** Non-fatal UI warning; harmless in runtime, do not treat as a pipeline error.

---

## B. WebUI Forge, ControlNet & Extensions

### 9. Forge API Connection Refused (`127.0.0.1:7860`)
- **Symptom:** `requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=7860): Max retries exceeded`.
- **Cause:** Forge isn't running, hasn't finished cold boot (`~62.6s`), or `--api` wasn't set in `webui\webui-user.bat`.
- **Fix:** Ensure `set COMMANDLINE_ARGS=--api --xformers` in `webui-user.bat`, run `run.bat`, wait until Forge logs `Running on local URL: http://127.0.0.1:7860`, and re-run `python tools\verify_setup.py`.

### 10. OpenPose ControlNet Not Found (`openpose_xl [f4251cb4]`)
- **Symptom:** Pass 1 ignores skeleton poses or `verify_setup.py` warns `openpose_xl [f4251cb4]` is absent from `/controlnet/model_list`.
- **Cause:** Downloading from HuggingFace (`thibaud/controlnet-openpose-sdxl-1.0`) saves `OpenPoseXL2.safetensors` instead of `openpose_xl.safetensors`, which Forge needs to expose the ID string `openpose_xl [f4251cb4]`.
- **Fix:** Rename `OpenPoseXL2.safetensors` -> `openpose_xl.safetensors` inside `<FORGE_MODELS>\ControlNet\` and restart Forge.

### 11. ControlNet Silently Ignored in Pass 1 (`D1` Defect — Byte-Identical Images)
- **Symptom:** Outputs with/without ControlNet have identical SHA-256 (`diagnostics.md` D1).
- **Cause:** Passing `input_image=None` or omitting the `"image"`/`"input_image"` key causes Forge ControlNet to skip conditioning without raising an HTTP error.
- **Fix:** `MMCCP v2.0` uses `LayoutPlanner` (`M2`) to draw synthetic COCO-18 PNGs and sends both `"image"` and `"input_image"` with `"module": "none"` and `"model": "openpose_xl [f4251cb4]"`. Verify `LayoutPlanner` is ON (`enable_layout_planner: true`).

### 12. Two Characters Collapse Into One Merged Person
- **Symptom:** Multi-character scenes render only a single person or blend Scene 1 `young_ebright` + `mother` attributes (`D3` / `W4` failure).
- **Cause:** Running `v1` (`BREAK`-only syntax without spatial attention partitioning) OR `sd-forge-couple` / `sd-webui-controlnet` missing or Forge didn't restart after adding extensions.
- **Fix:** Run `--pipeline v2`, ensure all 4 extensions (`adetailer`, `sd-forge-couple`, `sd-webui-controlnet`, `sd-webui-reactor`) appear in `webui\extensions\`, and restart Forge cleanly.

### 13. 3rd Character Missing in 3-Character Scenes (e.g., `Holdout Scene 4`)
- **Symptom:** A storyboard scene listing 3 characters only keeps 2 faces/layouts (`PREREG_RECONCILIATION.md`).
- **Cause:** Deliberate design constraint in `v2.0.0-rc3`: M2 layout planner, M3 Forge Couple (2 spatial columns), and M5 Swap enforce a strict **2-character per-scene limit**.
- **Fix:** Known architectural boundary of `v2.0`. Keep `<= 2` active `characters` per scene JSON (multi-char expansion is post-v2 work).

### 14. ReActor / Pass 2 Extension Script Argument Mismatch (`D2` Defect)
- **Symptom:** Old `v1` ReActor swaps drop ArcFace score or silently skip (`diagnostics.md` D2: passed 30 args vs Forge's 31-arg schema).
- **Cause:** Positional `*args` mismatch on Forge Gradio scripts (`0` instead of `"Minimum"`, `"No"`, `"Image(s)"`, missing `"tab_single"`).
- **Fix:** `MMCCP v2.0` bypasses ReActor extension fragilities entirely for primary swaps by using M5 `direct_insightface` (`consistency_v2/face_swap.py`) + Forge Extras CodeFormer API (`0.85`). Keep `swap_backend: "direct_insightface"`.

### 15. Dual InstantID (`M4b`) Missing Weights or Crashing on 8 GB VRAM (`D4`)
- **Symptom:** Enabling `enable_dual_instantid: true` fails or OOMs on 8 GB GPUs.
- **Cause:** InstantID requires two large weights (`control_instant_id_sdxl.safetensors` `2.33 GiB` + `ip-adapter_instant_id_sdxl.bin` `1.58 GiB`) stacked atop OpenPose XL.
- **Fix:** Keep default `best_config.json` setting (`enable_dual_instantid: false`, using `ip-adapter-plus-face_sdxl_vit-h.safetensors` instead). Only download optional InstantID weights on `>= 16 GB` GPUs.

### 16. Checkpoint Hash Confusion (`799b5005` vs `c9e3e68f89`) (`ISSUE-07`)
- **Symptom:** Older docs say `799b5005`, while Forge `/sdapi/v1/sd-models` returns `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors [c9e3e68f89]`.
- **Cause:** 8-char truncated hash vs Forge's 10-char AutoV2 hash of the exact same `7,105,348,188`-byte file.
- **Fix:** No action needed if `sha256 == c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca`.

---

## C. GPU VRAM, Power & Runtime Hangs

### 17. `CUDA out of memory` or Severe Thrashing on 8 GB GPUs
- **Symptom:** Generation crashes in Pass 1 / Pass 2 or spikes > 8192 MiB.
- **Cause:** Web browsers/apps consuming desktop VRAM (~973+ MiB), Pass 2 retaining ControlNet/PAG (`loop-W4-iter1a` hit `8005 MiB` when Pass 2 didn't drop ControlNet), or missing `--xformers`.
- **Fix:**
  1. Close Chrome/Edge/Discord/OBS before rendering.
  2. Verify `webui-user.bat` includes `--api --xformers`.
  3. Keep `v2 default` (`Pass 2` drops ControlNet/PAG; `xl_GPU_MB=1600.0`, `forge_inference_memory=3072.0`).
  4. Use `--mode fast` to disable M6 Face Detailer if you still hit memory pressure.

### 18. `[Low GPU VRAM Warning]` During Evaluation / Rendering (`ISSUE-06`)
- **Symptom:** Forge prints `Your current GPU free memory is ... lower than safe value of 1536.00 MB`.
- **Cause:** Python evaluation process allocated CUDA tensors while Forge held SDXL weights in VRAM.
- **Fix:** `metrics.py` enforces `torch.cuda.set_per_process_memory_fraction(0.25, 0)` + `vram_guard(7400.0)` and `consistency_v2/verify/` runs purely on CPU. Run standalone scoring with Forge closed or CPU-only (`$env:CUDA_VISIBLE_DEVICES = "-1"` during CPU verify).

### 19. Windows Sleeps / Hibernates Mid-Run (`blockers.md` §2.2)
- **Symptom:** Multi-scene or DEV-2 run stalls after 15–30 min of unattended terminal execution.
- **Cause:** Windows 11 AC power plan standby/hibernate timeouts.
- **Fix (PowerShell):**
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```
  *(Restore your preferred minutes after finishing!)*

### 20. First Render / Cold Boot Appears "Frozen" for > 60 Seconds
- **Symptom:** `run_from_scratch.py` hangs at Pass 1 with no immediate image.
- **Cause:** Forge cold start (`~62.6s`) plus loading Juggernaut XL (`6.62 GiB`) + OpenPose XL (`4.66 GiB`) + CLIP-ViT-H (`2.35 GiB`) + first HuggingFace auto-download of `CLIP-ViT-H`, `buffalo_l`, `segformer_b2_clothes` (~104 MiB), and DeepFace weights.
- **Fix:** Watch Forge's console window — download/progress bars appear there first. Give cold boot `3–5 minutes` before aborting.

---

## D. Identity Swap, Wardrobe, Determinism & Scientific Integrity

### 21. Identity-at-Position `0.0` / Near-Zero ArcFace (`0.06`) After Manual Bank Edits
- **Symptom:** Faces look swapped to a different character (`ArcFace 0.06`, `identity_at_position 0/16`).
- **Cause:** Desynchronization between `consistency_v2/banks/` (used by M4/M5) and canonical anchors (`experiments/clean_run_ground_truth/anchors/` or `hardcoded_cache.json`).
- **Fix:** Always run zero-cache runs with `run_from_scratch.py` (which uses a clean scratch bank) or restore canonical 8-character banks via:
  ```powershell
  python tools\sync_canonical_banks.py
  ```

### 22. Wardrobe Delta-E Ignored or Wrong Clothing Color Kept (`wardrobe.py` Bank Override Regression)
- **Symptom:** SegFormer repairs pull a character's shirt toward a stale bank LAB value instead of their storyboard anchor's LAB (`commit 68653e6` post-fix).
- **Cause:** Historical override bug in `get_target_lab()` when `bank_wardrobe_lab` bypassed anchor extraction.
- **Fix:** Fixed before TEST-2 in `main` (`625b365`): `get_target_lab()` derives canonical LAB cleanly from the anchor image/cache (`Delta E <= 38.0`, `blend: 0.70`).

### 23. Stale Anchors or Old Scenes Appear in Streamlit UI (`ISSUE-05` / Quarantine)
- **Symptom:** Streamlit opens and shows old characters instead of starting from scratch.
- **Cause:** `hardcoded_cache.json` and `edu_outputs/` still hold a previous run.
- **Fix:** Follow the zero-cache reset steps in [01_NEW_MACHINE_WINDOWS.md](01_NEW_MACHINE_WINDOWS.md#13-zero-cache-reset-clean-scratch-demo) (back up or delete `hardcoded_cache.json` + clear `edu_outputs/` before launching `run_from_scratch.py`).

### 24. Two Runs With Identical Seed Are Not Byte-Identical (`determinism_report.json`)
- **Symptom:** `check_determinism.py` reports `bitwise_identical: false` (`SSIM 0.997274`, `mean_pixel_diff 0.6665`, `max_pixel_diff 91.0`).
- **Cause:** Expected non-associative `float16` CUDA kernels, `--xformers` flash-attention nondeterminism, and CodeFormer tiling noise.
- **Fix:** Normal behavior on GPU SDXL (`SSIM > 0.997`); never expect strict SHA-256 image equality across diffusion runs.

### 25. Frozen Verifier Integrity Check Fails (`tools/check_freeze.py` exits `1`)
- **Symptom:** `[FAIL]` printed with modified file hash under `consistency_v2/verify/`.
- **Cause:** Accidental edit to `consistency_v2/verify/` or `verifier_freeze.json`.
- **Fix:** Restore immediately and re-verify (`SHA: 3fa69071f888643b...`):
  ```powershell
  git --no-pager checkout -- consistency_v2/verify consistency_v2/verifier_freeze.json
  python tools\check_freeze.py
  ```

### 26. Seed Guardrail Blocks Run (`ValueError` / Split Violation)
- **Symptom:** Runner refuses seeds `606, 707, 808` (`RESERVED` for LoRA) or `1101, 1202, 1303, 1404, 1505` (`TEST-2` sealed evaluation).
- **Cause:** `consistency_v2/splits.py` strictly enforces `TEST-2` / `RESERVED` hygiene (`tools/audit_test2_isolation.py`).
- **Fix:** Use **DEV** seeds (`11, 23, 37` for `DEV-FAST`; `101..505` for `DEV-2`). Never set `ALLOW_TEST2` or `ALLOW_RESERVED`.

### 27. `tools/secret_scan.py` or Push Blocked by File Size / Token Pattern
- **Symptom:** Scanner flags `AIza...`, `hf_...`, or `git push` rejects files > `25–100 MB`.
- **Cause:** Staged `.env`, API key in a scratch script, or accidentally tracked `.safetensors`/`.onnx` weights or `bundle_dev2` PNGs.
- **Fix:** Keep weights inside `$FORGE_MODELS_DIR` (outside git), export keys only as environment variables or Colab Secrets, and run `python tools\secret_scan.py` before every commit.

### 28. `IPAdapterPlus.py` Device Mismatch (`cpu` vs `cuda:0`) on 8 GB GPUs (`UNTESTED` Unpatched Upstream Forge)
- **Symptom:** `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!` inside Forge's built-in `sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py` during M4 Pass 1 conditioning (`experiments/ipadapter_device_fix_report.md`).
- **Cause:** Unpatched upstream WebUI Forge `dfdcbab6...` offloads IP-Adapter linear layers (`to_kvs`) / masks inconsistently with `q.device` on 8 GB GPUs.
- **Fix:** Move `cur_cond`, `cur_uncond`, `k_layer`, `v_layer`, and `mask` onto `q.device` / `org_dtype` in `IPAdapterPlus.py` (exact code documented in `experiments/ipadapter_device_fix_report.md`), OR note the M4 findings in `README_V2.md` §3 (`no_m4_ip_adapter` ablation).

### 29. `7` Unit Tests Fail Right After Fresh `git clone` (`test_identity_bank` / `test_identity_units`)
- **Symptom:** `python -m unittest discover -s tests` reports `65 PASS / 7 FAIL` with `FileNotFoundError: consistency_v2/banks/young_ebright/ref_00_young_ebright_anchor.png` (or wrong global `python` version e.g. `3.14`).
- **Cause:** `consistency_v2/banks/` and `*_ground_truth.png` are `.gitignore`d and not populated until synced from `experiments/clean_run_ground_truth/anchors/*.json`.
- **Fix:** Always create your Windows venv with **`py -3.10 -m venv venv`**, then hydrate canonical banks first:
  ```powershell
  python tools\sync_canonical_banks.py
  python -m unittest discover -s tests -p "test_*.py" -v
  ```

