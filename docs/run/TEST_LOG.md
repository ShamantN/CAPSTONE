# MMCCP v2.0 — Fresh-Machine Windows Test Transcript (`TEST_LOG.md`)

- **Test Status:** `EXECUTED & PASSED` (2026-11-10)
- **Target Tested:** Branch `docs/run-guides` (`858e9bd` + `requirements.txt` & `tools/sync_canonical_banks.py` fixes) on top of `main` `625b365` (`v2.0.0-rc3`)
- **Test Workspace:** `D:\5TH SEM\_fresh_machine_test\CAPSTONE` + fresh `D:\5TH SEM\_fresh_machine_test\venv` (`Python 3.10.6`)
- **Pre-Existing Asset Used Read-Only:** `C:\Stable-Diffusion\` WebUI Forge (`FORGE_DIR`) — **empty-PC Forge installation & manual weight downloads remain `UNTESTED`**
- **Honest Scope Note:** Because `v2.0.0-rc3` (`ee83d5e`) and `main` (`625b365`) pre-date this PR's portability layer (`runtime_paths.py`), updated `requirements.txt`, `sync_canonical_banks.py` JSON hydration, and `verify_setup.py`, the fresh clone was taken from local `docs/run-guides` (which merges back into `main` via this PR).

---

## 1. Original Repo Pre/Post Protection Check

Before & after all Step 3 commands ran in `D:\5TH SEM\_fresh_machine_test\`, original workstation caches & sealed files were hashed cleanly:

| File | Pre-Test SHA-256 | Post-Test SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `D:\5TH SEM\HACKATHON\hardcoded_cache.json` | `A9C901F16D914A8DA1C3450EFB6847586C5368F7A38B47F3BCE628C74AF9510E` | `A9C901F16D914A8DA1C3450EFB6847586C5368F7A38B47F3BCE628C74AF9510E` | **UNCHANGED** |
| `D:\5TH SEM\HACKATHON\results\split_access_log.jsonl` | `7FF6EE7F3693E095B69465ED2A574E28E7BE41545A2BC0EC418D6DF365C21262` | `7FF6EE7F3693E095B69465ED2A574E28E7BE41545A2BC0EC418D6DF365C21262` | **UNCHANGED** |

`C:\Stable-Diffusion\webui` file timestamps (`Aug 27 – Nov 04, 2026`) also stayed untouched (note: we discovered 3 pre-existing workstation patches in `C:\Stable-Diffusion\webui` recorded in `01_NEW_MACHINE_WINDOWS.md` §4.4).

---

## 2. Step-by-Step Executed Commands, Timings & Real Issues Caught

### Step 3.1–3.2 — Host Check & Fresh Clone
```powershell
python --version
# Observed: Python 3.14.0  <-- CATCH #1: default python on PATH was 3.14, not 3.10.6!
py -0p
# Observed: -V:3.14 * C:\Python314\python.exe | -V:3.10 C:\Users\shama\AppData\Local\Programs\Python\Python310\python.exe

git --no-pager clone --branch docs/run-guides "d:\5TH SEM\HACKATHON" "d:\5TH SEM\_fresh_machine_test\CAPSTONE"
# Exit 0 | Elapsed: 2.49s | HEAD: 858e9bd
```
- **Baseline GPU (`nvidia-smi`):** `NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, 931 MiB used, 7087 MiB free`.

### Step 3.3 — Fresh Python `3.10.6` Virtual Environment & Package Install
**First Run (caught real bug in initial `requirements.txt`):**
- Created `venv` (`22.16s`), upgraded `pip/setuptools/wheel` (`25.34s`), installed `torch==2.3.1+cu121 torchvision==0.18.1+cu121` (`287.93s`), installed `xformers==0.0.27` (`16.02s`), ran `pip install -r requirements.txt` (`709.79s`).
- **REAL BUG CAUGHT #2:** Initial `requirements.txt` listed `torch>=2.1.0` & `torchvision>=0.16.0` on PyPI without index pinning, causing `pip` to **uninstall** `torch 2.3.1+cu121` and install PyPI CPU **`torch 2.14.1+cpu`** (`CUDA: False`) + trigger 11-minute `google-api-core`/`grpcio-status` backtracking!
- **Fix Applied:** Removed `torch`/`torchvision` from PyPI `requirements.txt`, bounded `transformers>=4.40.0,<5.0.0`, `ultralytics>=8.0.0,<8.5.0`, `open-clip-torch>=2.20.0,<3.0.0`, `huggingface-hub>=0.23.0,<1.0.0`, `google-generativeai>=0.8.5,<0.9.0`.

**Second Clean Re-Test (`venv` deleted and recreated from scratch after fix):**
```powershell
py -3.10 -m venv "d:\5TH SEM\_fresh_machine_test\venv"            # 21.47s
python -m pip install --upgrade pip setuptools wheel              # 23.18s
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121  # 218.82s
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121                  # 10.52s
pip install -r requirements.txt                                   # 535.49s
```
- **Verification:** `torch: 2.3.1+cu121 CUDA: True numpy: 1.26.4 transformers: 4.57.6` (Exit `0`, zero resolver conflicts).
- **Measured `venv` Disk Size:** `7,909.2 MB` (`~7.72 GiB`); cloned repo + smoke outputs = `72.2 MB`.

### Step 3.4 — Pre-Forge Preflight, Hydration & Unit Tests
1. `python tools\verify_setup.py --mode new-machine --skip-forge --verify-sha256 --fix-hints` -> **`11 PASS | 3 WARN | 0 FAIL`**, Exit **`0`**, **`58.49s`** (hashed ~25 GB weights).
2. `python tools\check_freeze.py` -> **`[PASS] Verifier Freeze Integrity Verified! Version 1.2 (SHA: 3fa69071f888643b...)`**.
3. `python tools\audit_test2_isolation.py` -> **`PASSED: 100% clean isolation.`**.
4. **REAL BUG CAUGHT #3 (`unittest discover` before bank sync):** Ran `72` tests (`25.802s`) -> **`65 PASS / 7 FAIL`** because `.gitignore` excludes `consistency_v2/banks/` and `*_ground_truth.png`, while `test_identity_bank.py` (`4`) and `test_identity_units.py` (`3`) read canonical banks (`young_ebright/ref_00_young_ebright_anchor.png`). And `tools/sync_canonical_banks.py` previously expected `*_ground_truth.png` directly!
   - **Fix Applied:** Added `_hydrate_missing_ground_truth_pngs()` to `tools/sync_canonical_banks.py` to decode missing `*_ground_truth.png` from tracked `ground_truth_anchors.json` & `holdout_anchors.json` (`100%` SHA-256 byte match).
5. `python tools\sync_canonical_banks.py` -> **Exit `0` (`210.97s`)**, hydrated all 8 anchors (`cos sim = 1.0000` across all 4 DeepFace judges!).
6. Re-ran `python -m unittest discover -s tests -p "test_*.py" -v` -> **`Ran 72 tests in 21.346s -> OK`** (Exit **`0`**, total command `35.07s`).
7. `python tools\secret_scan.py` -> **`[Secret Scanner] CLEAN`**.

### Step 3.5–3.8 — Live Forge Boot, Preflight, Zero-Cache 1-Scene Smoke Test & Image Audit
1. Launched `C:\Stable-Diffusion\run.bat` (`--api --xformers`), polled `/sdapi/v1/sd-models` -> **`FORGE_READY=True` in `48.06s`**.
2. First `verify_setup.py --mode new-machine` right at `48s` -> **`12 PASS | 3 WARN | 0 FAIL`** (caught subtle timing quirk: `/controlnet/model_list` returned `404` right when `/sdapi/v1/sd-models` first answered, then returned `200 OK` with `10` models including `openpose_xl [f4251cb4]` once ControlNet router finished mounting!).
3. Cleared `hardcoded_cache.json` + `edu_outputs/` inside `_fresh_machine_test\CAPSTONE` and started `tools\vram_sampler.py`.
4. Executed:
   ```powershell
   python run_from_scratch.py --pipeline v2 --mode fast --story storyboards\ebright.json --scene 1 --seed 11
   ```
   - Bootstrapped 2 fresh anchors (`young_ebright` `47s`, `mother` `27s`) -> wrote `hardcoded_cache.json` inside `_fresh_machine_test\CAPSTONE`.
   - Render `scene_1_seed_11`: Pass 1 **`97.38s`** (`Host RAM used: 21229.3MB`), Pass 2 **`16.08s`** (`Host RAM used: 26060.0MB`), CPU `inswapper_128` **`~27s`**, CodeFormer **`~10s`**, SegFormer `WardrobeLock` init+check -> pipeline scene time **`2.92 min` (`174.86s`)**, total script end-to-end **`312.71s` (`5.21 min`)**, Exit **`0`**.
   - Script metric table output:
     - `Young Richard Ebright`: `ArcFace 0.5115 | CLIP-I 0.7371 | SSIM 0.4847 | Palette 0.3415 | Composite 0.5384`
     - `Ebright's Mother`: `ArcFace 0.4952 | CLIP-I 0.6980 | SSIM 0.5097 | Palette 0.1491 | Composite 0.4962`
     - `Overall Composite`: **`0.5173 (51.7%)`**.
5. **1 Hz VRAM Telemetry (`vram_20261110_180031.csv`, `286` samples):**
   - `MIN_MB=1146.0` | `MEAN_MB=4148.9` | **`PEAK_MB=7887.0`** (`7.702 GiB`, ZERO OOMs).
6. **Literal Visual Inspection (all 3 PNGs opened directly via `read_file`):**
   - `young_ebright_anchor.png`: Crisp white-studio close portrait of a smiling brown-haired blue-eyed boy wearing a yellow unbuttoned over-shirt over a cyan/turquoise shirt and blue jeans.
   - `mother_anchor.png`: Studio light-gray portrait of a smiling adult woman with wavy brown hair and a light-blue patterned puff-sleeve blouse.
   - `scene_1_seed_11.png`: Two clearly separated figures seated at a bright blue/yellow floral tablecloth — left: smiling young boy matching `young_ebright_anchor` face in a yellow t-shirt (minor small-hand/pin distortion at fingers); right: smiling adult woman matching `mother_anchor` face in a blue-and-white patterned puff-sleeve blouse. Clean spatial separation (`left/right`), clear face identity preservation, and clear yellow-vs-blue wardrobe isolation.

### Step 3.9–3.10 — Live Re-Verify, Streamlit UI Health Check & Clean Shutdown
1. Re-ran `python tools\verify_setup.py --mode new-machine --fix-hints` after warm Forge: **`12 PASS | 3 WARN | 0 FAIL`**, **`[PASS] ControlNet Model List: 10 models visible`**.
2. Headless Streamlit launch (`python -m streamlit run app.py --server.headless true --server.port 8599`): **`GET /_stcore/health -> 200 ok`** cleanly with zero import/runtime errors.
3. Stopped Streamlit & WebUI Forge cleanly; `nvidia-smi` returned to **`398 MiB used / 7620 MiB free`**.
4. Tested Section 13 Zero-Cache Reset (`Remove-Item hardcoded_cache.json`, `edu_outputs`, `experiments\fresh_run_from_scratch`): verified cleanly removed without touching `experiments\clean_run_ground_truth` or `results\`.

