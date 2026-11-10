# MMCCP v2.0 — New Windows Machine Guide (Step-by-Step)

- **Verification Status:** `TESTED REPO / VENV / DEV RENDER (see TEST_LOG.md)` — *Fresh `git clone`, `Python 3.10.6` `venv` install, preflight, `sync_canonical_banks`, `72/72` unit tests, live Forge API boot, 1-scene zero-cache DEV render, Streamlit health check, and zero-cache cleanup all executed and verified on Windows 11 + RTX 3070 Ti (8 GB); **empty-PC Forge installation & manual weight downloads remain `UNTESTED`** (authored against pre-existing `C:\Stable-Diffusion` + HTTP `HEAD` / SHA-256 manifest)*
- **Target Baseline:** Branch `docs/run-guides` (on top of `main` `625b365` / `v2.0.0-rc3`)

> **Golden Rule:** Keep all generation scripts strictly on **DEV** seeds (`11, 23, 37` or `101..505`). **Never** modify `consistency_v2/verify/`, **never** run `TEST-2` (`1101..1505`) or `RESERVED` LoRA seeds (`606, 707, 808`), and **never** commit model weights or rendered PNG folders to Git.

---

## 1. Prerequisites & System Checks

Before starting, verify your Windows 10/11 (64-bit) PC meets these requirements:

| Requirement | Minimum Tested Target | How to Check |
| :--- | :--- | :--- |
| **NVIDIA GPU & Driver** | >= **8 GB VRAM** (tested on **RTX 3070 Ti Laptop GPU 8192 MiB**, driver `621.01`) | Run `nvidia-smi` in PowerShell |
| **System RAM** | **16 GB minimum / 32 GB recommended** (measured host peak `28.1 GiB`) | Run `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` |
| **Free SSD Space** | **>= 25 GB** minimum install; **>= 50 GB** recommended with caches | Check File Explorer |
| **Python 64-bit** | **Python 3.10.6** (tested) or Python 3.10.x *(with "Add Python to PATH")* | Run `python --version` |
| **Git & 7-Zip** | Git for Windows + 7-Zip *(needed to extract `.7z` Forge archive)* | Run `git --version` |

If `nvidia-smi` fails or shows `< 8192 MiB`, see [07_FAQ_AND_COSTS.md](07_FAQ_AND_COSTS.md) or [03_GOOGLE_COLAB.md](03_GOOGLE_COLAB.md).

---

## 2. Clone the Repository

Open PowerShell in your clean project parent directory (for example `C:\Projects\`, avoid OneDrive/Dropbox-synced folders):

```powershell
mkdir C:\Projects -Force
cd C:\Projects
git clone https://github.com/ShamantN/CAPSTONE.git MMCCP
cd MMCCP
```

Expected output:

```text
Cloning into 'MMCCP'...
...
```

*(Note: While PR #19 merged `v2.0.0` into `main` at `625b365`, check out `docs/run-guides` if cloning before this documentation PR merges into `main`: `git checkout docs/run-guides`.)*

---

## 3. Create Python Virtual Environment & Install Packages

Create an isolated Python **`3.10.6`** virtual environment **inside** or next to your clone so you do not pollute global Python:

> **Multiple Python Versions Warning (`py -3.10`):** On systems that also have Python `3.11+` / `3.14+` installed globally (our test host defaulted `python` to `Python 3.14.0`!), **`python -m venv` creates the wrong Python version**. Always invoke `py -3.10 -m venv` explicitly and verify `3.10.x`:

```powershell
# Ensure UTF-8 console output and allow local activation scripts for this user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
$env:PYTHONUTF8 = "1"

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
# Expected: Python 3.10.6
python -m pip install --upgrade pip setuptools wheel
```

Install CUDA `12.1` PyTorch `2.3.1` + `xformers` first (`~219–288s`, ~2.4 GB download), then MMCCP requirements (`~535s` = ~9 min; total `venv` disk usage = **`~7.72 GiB`**):

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify CUDA and package ABI (`numpy` must stay `< 2.0.0`):

```powershell
python -c "import torch, numpy, cv2, deepface, streamlit; print('CUDA:', torch.cuda.is_available(), 'torch:', torch.__version__, 'numpy:', numpy.__version__)"
```

Expected output:

```text
CUDA: True torch: 2.3.1+cu121 numpy: 1.26.4
```

*(Optional exact snapshot: [`requirements-lock.txt`](../../requirements-lock.txt) records the full `118`-package freeze from the verified Windows workstation `my_app_env`.)*

---

## 4. Install WebUI Forge & Extensions (`UNTESTED` on Empty PC)

> **Honest Note:** On the author machine, `C:\Stable-Diffusion` already existed before this guide was written. The commands below mirror its exact upstream archive, commit hash (`dfdcbab685e57677014f05a3309b48cc87383167`), flags, and 4 extensions verbatim.

### 4.1 Download & Extract Forge One-Click Package
1. Download `webui_forge_cu121_torch231.7z` (`1.74 GiB` verified live on GitHub) from [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md).
2. Extract with **7-Zip** directly to **`C:\Stable-Diffusion\`** so `C:\Stable-Diffusion\run.bat`, `C:\Stable-Diffusion\environment.bat`, and `C:\Stable-Diffusion\webui\webui-user.bat` exist.

### 4.2 Enable Forge REST API & `--xformers`
Open `C:\Stable-Diffusion\webui\webui-user.bat` in Notepad and set `COMMANDLINE_ARGS` so Forge enables its JSON REST API + memory optimizations:

```bat
@echo off
set COMMANDLINE_ARGS=--api --xformers --cuda-malloc
call webui.bat
```

*(Verified author `webui-user.bat` uses `--api --xformers`, and `docs_generated/00_environment_and_run_log.md` adds `--cuda-malloc`. Omit `--cuda-malloc` if your driver throws an allocator error.)*

### 4.3 Install the 4 Required WebUI Extensions

```powershell
cd C:\Stable-Diffusion\webui\extensions
git clone https://github.com/Bing-su/adetailer.git
git clone https://github.com/Haoming02/sd-forge-couple.git
git clone https://github.com/Mikubill/sd-webui-controlnet.git
git clone https://github.com/Gourieff/sd-webui-reactor-sfw.git sd-webui-reactor
```

*(InsightFace Note: Forge's embedded Python `C:\Stable-Diffusion\system\python\python.exe` carries `insightface-0.7.3-cp310-cp310-win_amd64.whl` and `onnxruntime-gpu==1.17.1` used by MMCCP's M5 subprocess face swapper.)*

### 4.4 Pre-Existing Workstation Forge Patches (`UNTESTED` Unpatched Upstream Forge)
Inspection of `C:\Stable-Diffusion\webui` showed three small pre-existing local patches (`Aug–Sep 2026`) in upstream Forge commit `dfdcbab685e57677014f05a3309b48cc87383167`:
1. `extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py`: synchronizes conditioning tensors & masks (`cond`, `to_kvs`, `mask`) to `q.device` / `org_dtype` and guards `tensorToNP` against dict inputs (prevents `RuntimeError: Expected all tensors to be on the same device, cpu and cuda:0!` when 8 GB GPUs offload layers; see [`experiments/ipadapter_device_fix_report.md`](../../experiments/ipadapter_device_fix_report.md)).
2. `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py`: adds `if i in self.current_params:` guards in `process_before_every_sampling` / `postprocess_batch_list`.
3. `modules_forge/utils.py`: adds dict/`Tensor`/`None` type guards to `numpy_to_pytorch()`.

Because running **M4 IP-Adapter Plus-Face SDXL** on unpatched upstream Forge is `UNTESTED` here, keep [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md#28-ipadapterpluspy-device-mismatch-cpu-vs-cuda0-on-8-gb-gpus) handy if upstream Forge raises a CUDA/CPU tensor mismatch.

---

## 5. Download & Verify Model Weights

Download every **Required** model listed in **[04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md#1-base-software--required-generation-models-manual--forge-auto-download)** into `C:\Stable-Diffusion\webui\models\`.

Pay special attention to the **OpenPose rename rule**:
- Save `https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors` as:
  `C:\Stable-Diffusion\webui\models\ControlNet\openpose_xl.safetensors`
- If it stays named `OpenPoseXL2.safetensors`, Forge will **not** expose `openpose_xl [f4251cb4]`, and Pass 1 skeleton conditioning will silently skip.

Verify downloads **before** starting Forge:

```powershell
cd C:\Projects\MMCCP
.\venv\Scripts\Activate.ps1
python tools\verify_setup.py --mode new-machine --skip-forge --verify-sha256 --fix-hints
```

Expected: `FAIL: 0` (Auto-downloaded models like CodeFormer, CLIP-ViT-H, SegFormer, DINOv2, and DeepFace show `WARN` until first run triggers their automatic download, unless already cached).

---

## 6. Configure Environment Variables

Set these environment variables in PowerShell so MMCCP always resolves your clone folder and WebUI Forge directory explicitly (rather than assuming defaults):

```powershell
$env:PYTHONUTF8 = "1"
$env:MMCCP_ROOT = (Get-Location).Path
$env:FORGE_DIR = "C:\Stable-Diffusion"
$env:FORGE_MODELS_DIR = "C:\Stable-Diffusion\webui\models"
$env:FORGE_PYTHON = "C:\Stable-Diffusion\system\python\python.exe"
$env:FORGE_URL = "http://127.0.0.1:7860"
```

*(Optional)* Set `$env:HF_TOKEN = "hf_..."` if HuggingFace rate-limits anonymous weight downloads during first boot. Never write real tokens into tracked `.py` or `.md` files.

---

## 7. Launch WebUI Forge Server (Terminal 1)

Close heavy VRAM consumers (Chrome/Edge tabs, Discord hardware acceleration, games) first so desktop VRAM stays near `~930–980 MiB` on 8 GB GPUs. Then start Forge in **Terminal 1**:

```powershell
cd C:\Stable-Diffusion
.\run.bat
```

Wait ~`48–65 seconds` (measured boot time: **`48.06s`** in Step 3 fresh test; `62.6s` in cold R11 check) until Terminal 1 prints:

```text
Running on local URL:  http://127.0.0.1:7860
```

Verify from **Terminal 2** (with `venv` activated) that Forge responds cleanly and lists OpenPose + IP-Adapter + Juggernaut-XL *(if `/controlnet/model_list` briefly returns `404` right after `/sdapi/v1/sd-models` responds, wait ~10s for Forge's ControlNet extension routes to mount)*:

```powershell
python -c "import requests; r=requests.get('http://127.0.0.1:7860/sdapi/v1/sd-models', timeout=10); print('Models:', len(r.json()), [m['title'] for m in r.json() if 'Juggernaut' in m['title']])"
python -c "import requests; r=requests.get('http://127.0.0.1:7860/controlnet/model_list', timeout=10); print('ControlNets:', [m for m in r.json().get('model_list', []) if 'openpose_xl' in m or 'ip-adapter-plus-face' in m])"
```

---

## 8. Run Full Preflight, Canonical Bank Hydration & CPU Verification Suite (Terminal 2)

```powershell
cd C:\Projects\MMCCP
.\venv\Scripts\Activate.ps1

# 1. Full environment + Forge API preflight
python tools\verify_setup.py --mode new-machine --fix-hints

# 2. Verify Frozen Verifier v1.2 SHA integrity (must exit 0)
python tools\check_freeze.py

# 3. Verify TEST-2 / RESERVED seed isolation (must PASS)
python tools\audit_test2_isolation.py

# 4. Hydrate gitignored canonical character banks from tracked JSON anchors (~211s on first run while DeepFace caches weights)
#    Required on fresh clones so the 7 bank/unit tests in `consistency_v2/banks` pass.
python tools\sync_canonical_banks.py

# 5. Run full CPU unit tests (72 tests, ~21.3s)
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected results:
- `tools\verify_setup.py`: **`FAIL: 0`**
- `tools\check_freeze.py`: **`[PASS] Verifier Freeze Integrity Verified! Version 1.2 (SHA: 3fa69071f888643b...)`**
- `tools\audit_test2_isolation.py`: **`PASSED: 100% clean isolation.`**
- `tools\sync_canonical_banks.py`: **`[SUCCESS] All 8 characters ... achieve EXACT 1.0000 cosine similarity`**
- `unittest`: **`Ran 72 tests ... OK`** (`0` failures).

---

## 9. Quick 1-Scene DEV Smoke Test (~3–5 Minutes)

Run **Scene 1** of the Ebright storyboard on **DEV** seed `11` in `fast` mode (disables M6 Detailer for a quick smoke check while still running Anchor Bootstrap + OpenPose + Forge Couple + IP-Adapter + M5 Face Swap + M7 Wardrobe Lock).

*(Optional telemetry: launch `python tools\vram_sampler.py` in another PowerShell window inside your repo root before starting; press `Ctrl+C` when finished to inspect `results\telemetry\vram_*.csv`.)*

```powershell
# Optional zero-cache cleanup so run_from_scratch starts from 0 cached anchors:
Remove-Item hardcoded_cache.json -ErrorAction SilentlyContinue
Remove-Item edu_outputs -Recurse -Force -ErrorAction SilentlyContinue

python run_from_scratch.py --pipeline v2 --mode fast --story storyboards\ebright.json --scene 1 --seed 11
```

What happens & verified Step 3 measurements ([`TEST_LOG.md`](TEST_LOG.md)):
1. Bootstraps `2` fresh character anchors (`young_ebright` `47s`, `mother` `27s`) into `experiments\fresh_run_from_scratch\anchors\` and writes `hardcoded_cache.json`.
2. Renders `edu_outputs\v2_fast\scene_1_seed_11.png` (`Pass 1: 97.38s`, `Pass 2: 16.08s`, CPU `inswapper_128: ~27s`, CodeFormer: `~10s`, SegFormer WardrobeLock -> pipeline scene runtime **`174.86s` / `2.92 min`**) + telemetry sidecar `scene_1_seed_11.json` + copies `edu_outputs\images\scene_1.png`.
3. Prints evaluation rows (`Composite: 0.5173 (51.7%)` in Step 3 fresh run; total end-to-end command runtime **`312.71s` / `5.21 min`**; **`7887.0 MiB`** peak 1 Hz system VRAM, `0` OOMs).

Open `edu_outputs\v2_fast\scene_1_seed_11.png` visually and confirm you see **two separate figures** (young boy on left in yellow top, mother on right in blue patterned blouse) matching their fresh anchors.

---

## 10. Full 6-Scene `v2 default` Demo (~18–20 Minutes)

Once the 1-scene smoke test succeeds, run all 6 Ebright scenes in default quality mode (`M6 Face Detailer` enabled):

```powershell
python run_from_scratch.py --pipeline v2 --mode default --story storyboards\ebright.json --seed 11
```

Reference benchmark on RTX 3070 Ti (8 GB):
- Scene 1: `271.40s` | Scene 2: `147.14s` | Scene 3: `152.86s`
- Scene 4: `153.54s` | Scene 5: `249.48s` | Scene 6: `148.86s`
- Total across 6 scenes: **`18.72 minutes`** (`1123.28s`, `mean 187.21s`).
- Outputs land in `edu_outputs\v2_default\scene_{1..6}_seed_11.png` and `edu_outputs\images\scene_{1..6}.png`.

---

## 11. Launch the Streamlit Interactive Studio UI

After rendering scenes, inspect anchors, images, diagnostics, and scores in your browser:

```powershell
python -m streamlit run app.py
```

Open **`http://localhost:8501`**. Explore the 4 tabs (reference screenshots checked into repo):
1. **`🎬 6-Scene Storyboard & Consistency Matrix`** ([`01_tab1_storyboard_explorer.png`](../../docs_generated/v2/ui_walkthrough/01_tab1_storyboard_explorer.png)) — Browse rendered storyboard scenes side-by-side with metrics.
2. **`🧬 Visual DNA Character Anchors`** ([`02_tab2_visual_dna_anchors.png`](../../docs_generated/v2/ui_walkthrough/02_tab2_visual_dna_anchors.png)) — Inspect cached studio character portraits (`hardcoded_cache.json`) or regenerate fresh anchors from zero cache.
3. **`📈 Phase 3 Benchmark Metrics & Diagnostics`** ([`03_tab3_diagnostics_benchmark.png`](../../docs_generated/v2/ui_walkthrough/03_tab3_diagnostics_benchmark.png)) — View per-scene breakdown and diagnostics.
4. **`🔍 Frozen Verifier & Contact Sheets`** ([`04_tab4_frozen_verifier.png`](../../docs_generated/v2/ui_walkthrough/04_tab4_frozen_verifier.png)) — Review Verifier `v1.2` grades and comparison sheets.

*(Press `Ctrl+C` in the Streamlit terminal when finished so Chrome/Edge frees GPU memory before launching another render.)*

---

## 12. Reproduce Scorecard Numbers & Determinism Verification (DEV Only)

All headline tables in [`README_V2.md`](../../README_V2.md) are generated programmatically from committed JSON/CSV artifacts via `results/numbers.json`:

```powershell
# 1. Re-collect machine-verified scorecard into results/numbers.json (CPU-only, ~5s)
python tools\collect_numbers.py

# 2. Re-score existing DEV-2 images using Frozen Verifier v1.2 into scratch/ (safe read-only audit)
python -m consistency_v2.verify ^
  --images results\bundle_dev2\renders ^
  --story storyboards\ebright.json ^
  --bank experiments\clean_run_ground_truth\anchors ^
  --out scratch\dev2_verify_check.json ^
  --baseline-report results\rescored_v1.2\baseline_v1_dev2\report.json
```

**Expected Determinism Note (`docs_generated/v2/repair/determinism_report.json`):** Running `python tools\check_determinism.py` renders Scene 1 Seed 11 twice back-to-back. Due to `float16` CUDA / `--xformers` flash-attention kernel non-associativity, identical seeds are **near-identical (`SSIM = 0.997274`, mean pixel diff `0.6665 / 255`), not bitwise identical**.
- **Sealed Split Policy:** `TEST-2` (`1101..1505`) is **burned/sealed** after Phase R10 and `RESERVED` (`606, 707, 808`) is permanently untouched. Never set `ALLOW_TEST2` or `ALLOW_RESERVED`.

---

## 13. Zero-Cache Reset (Run Cleanly From Zero Anchors)

To reset your clone back to **zero cached characters and zero prior scene renders** (without deleting canonical benchmark evidence in `results/` or `experiments/clean_run_ground_truth/`):

```powershell
# Remove only runtime UI/scratch outputs (back them up first if you want to keep renders!)
Remove-Item hardcoded_cache.json -ErrorAction SilentlyContinue
Remove-Item edu_outputs -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item experiments\fresh_run_from_scratch -Recurse -Force -ErrorAction SilentlyContinue

# Restore the canonical 8-character reference banks if you need them later:
python tools\sync_canonical_banks.py
```

---

## 14. Windows Power & 8 GB VRAM Hygiene

1. **Prevent Sleep Mid-Run:**
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```
   *(Re-enable timeouts afterward if running on battery later.)*
2. **One Diffusion Job at a Time:** Never run Streamlit image generation simultaneously with `run_from_scratch.py` or a Verifier GPU job on an 8 GB card.
3. **Free Desktop VRAM:** Close Chromium/Edge/Discord before rendering to keep baseline idle VRAM `< 1000 MiB` (`7984 MiB` observed system peak on TEST-2 vs `8192 MiB` physical limit).
4. **Before Committing Any Changes:** Always run `python tools\secret_scan.py` and verify `git --no-pager status`.

