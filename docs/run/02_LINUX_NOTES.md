# MMCCP v2.0 — Linux Setup Notes (Ubuntu / Debian CUDA)

- **Verification Status:** `UNTESTED ON LINUX` *(portability paths and `/proc/meminfo` logic unit-tested with mocks in [`tests/test_portability.py`](../../tests/test_portability.py); full live Forge + GPU execution has NOT yet been run on a Linux host)*
- **Target Baseline:** Branch `docs/run-guides` (on top of `main` `625b365` / `v2.0.0-rc3`)

> **Why this file is separate:** Windows setup ([01_NEW_MACHINE_WINDOWS.md](01_NEW_MACHINE_WINDOWS.md)) is the primary tested host. On Linux, almost everything uses the same Python code and Forge REST API once three OS differences (shell syntax, Forge launcher `webui.sh`, and frozen `torch_bridge.py`) are understood.

---

## 1. Key Differences From Windows

| Item | Windows (Tested) | Linux Ubuntu/Debian (`UNTESTED`) |
| :--- | :--- | :--- |
| **Virtual environment activate** | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| **UTF-8 console encoding** | Needs `$env:PYTHONUTF8 = "1"` against CP1252 | UTF-8 by default (`LANG=C.UTF-8` recommended) |
| **RAM telemetry (`memory_utils.py`)** | `ctypes.windll.kernel32.GlobalMemoryStatusEx` | Parses `/proc/meminfo` (`MemTotal` / `MemAvailable`) automatically on POSIX (`35e50ef`) |
| **Forge launcher** | `C:\Stable-Diffusion\run.bat` (`webui-user.bat`) | `~/stable-diffusion-webui-forge/webui.sh` (`webui-user.sh`) |
| **Forge Python (`FORGE_PYTHON`)** | `<FORGE_DIR>\system\python\python.exe` | `<FORGE_DIR>/venv/bin/python` or `<FORGE_DIR>/webui/venv/bin/python` |
| **InsightFace build dependency** | Prebuilt `.whl` for `cp310-win_amd64` | Requires `g++`, `build-essential`, `python3-dev` to compile Cython extensions if no manylinux wheel matches |
| **Windows DLL preloading** | Adds `C:\Stable-Diffusion\system\python\...` | Automatically skipped (`configure_windows_forge_env()` returns immediately when `sys.platform != "win32"`) |
| **Frozen Verifier `torch_bridge.py`** | Calls hardcoded `C:\Stable-Diffusion\system\python\python.exe` | Subprocess path absent on Linux -> YOLO/CLIP/SegFormer/DINOv2 subprocess checks skip while DeepFace (`4` judges) + RetinaFace still run directly in Python |

---

## 2. Step-by-Step Linux Installation (`UNTESTED`)

### 2.1 System Packages & Build Tools
Ubuntu 22.04 / 24.04 with NVIDIA driver (`nvidia-smi` showing CUDA >= 12.1) and Python `3.10.x`:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3.10 python3.10-venv python3.10-dev \
  build-essential g++ libgl1 libglib2.0-0 ffmpeg wget aria2
```

### 2.2 Clone MMCCP & Create App Virtual Environment

```bash
git clone https://github.com/ShamantN/CAPSTONE.git ~/MMCCP
cd ~/MMCCP

python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.3.1 CUDA 12.1 + xformers matching verified Windows build
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121

# App + Verifier + Wardrobe + InsightFace dependencies
pip install -r requirements.txt
pip install insightface==0.7.3 onnxruntime-gpu==1.17.1
```

> **Note:** `requirements.txt` pins `numpy>=1.26.0,<2.0.0`. Do **not** upgrade to NumPy 2.x, because PyTorch `2.3.1` and InsightFace `0.7.3` binary modules were compiled against NumPy 1.x ABI.

### 2.3 Install WebUI Forge & Extensions

```bash
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git ~/stable-diffusion-webui-forge
cd ~/stable-diffusion-webui-forge

# Pin tested Forge upstream commit if you want exact parity (optional):
# git checkout dfdcbab685e57677014f05a3309b48cc87383167

# Required WebUI extensions (same 4 extensions present in author install):
mkdir -p extensions && cd extensions
git clone https://github.com/Bing-su/adetailer.git
git clone https://github.com/Haoming02/sd-forge-couple.git
git clone https://github.com/Mikubill/sd-webui-controlnet.git
git clone https://github.com/Gourieff/sd-webui-reactor-sfw.git sd-webui-reactor
```

Configure `~/stable-diffusion-webui-forge/webui-user.sh`:

```bash
export COMMANDLINE_ARGS="--api --xformers --cuda-malloc"
```

*(Note: `--cuda-malloc` matches `docs_generated/00_environment_and_run_log.md`; omit `--cuda-malloc` if Forge warns on your driver and keep `--api --xformers`.)*

Download weights into `~/stable-diffusion-webui-forge/models/` following the exact destinations and SHA-256 checksums in [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) (especially renaming `OpenPoseXL2.safetensors` -> `models/ControlNet/openpose_xl.safetensors`).

### 2.4 Environment Variables (`runtime_paths.py`)

```bash
cd ~/MMCCP
source venv/bin/activate

export MMCCP_ROOT="$HOME/MMCCP"
export FORGE_DIR="$HOME/stable-diffusion-webui-forge"
export FORGE_MODELS_DIR="$FORGE_DIR/models"
export FORGE_URL="http://127.0.0.1:7860"
# Point FORGE_PYTHON to the venv that has insightface + onnxruntime-gpu:
export FORGE_PYTHON="$MMCCP_ROOT/venv/bin/python"
```

### 2.5 Preflight & Smoke Run

```bash
# Terminal 1: launch WebUI Forge (first boot creates Forge's venv and downloads extensions deps)
cd ~/stable-diffusion-webui-forge && bash webui.sh

# Terminal 2 (once http://127.0.0.1:7860 responds):
cd ~/MMCCP && source venv/bin/activate
python tools/verify_setup.py --mode new-machine --verify-sha256 --fix-hints
python tools/check_freeze.py
python -m unittest discover -s tests -p "test_*.py" -v
python run_from_scratch.py --pipeline v2 --mode fast --story storyboards/ebright.json --scene 1 --seed 11
streamlit run app.py --server.port 8501
```

---

## 3. Known Linux Caveat: Frozen Verifier `torch_bridge.py`

In `consistency_v2/verify/torch_bridge.py` (SHA `3fa69071f888643b...`, frozen by rule and **never modified**), the verifier invokes a hardcoded Windows path:
`C:\Stable-Diffusion\system\python\python.exe`.

On Linux:
- `python tools/verify_setup.py` marks that probe as `[WARN]` instead of `[FAIL]`.
- DeepFace's 4 biometric judges (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`) and RetinaFace face detection still execute normally inside your active Linux Python environment.
- Subprocess-only `torch_bridge` probes (YOLOv8n person count, SegFormer clothes audit, DINOv2 similarity, CLIP props inside `consistency_v2/verify/`) are skipped when that Windows binary does not exist — do **not** report Linux verifier totals as full `v1.2` parity runs without noting this skipped bridge.

## 4. Recommended Upstream Forge Patches (`UNTESTED`)
The tested Windows workstation (`C:\Stable-Diffusion\webui`) carried `3` patches against upstream Forge commit `dfdcbab685e57677014f05a3309b48cc87383167` — shipped in [`patches/forge/`](../../patches/forge/README.md) with [`tools/apply_forge_patches.py`](../../tools/apply_forge_patches.py):
```bash
python tools/apply_forge_patches.py --forge-dir "$FORGE_DIR" --apply
```
*(In clean-copy Windows testing at `D:\5TH SEM\_forge_clean_test`, the unpatched `v2 default` IP-Adapter Plus-Face SDXL smoke path finished without crashing, so these `3` patches are shipped as **recommended** hardening against `FaceID Plus v2` / multi-unit / dict-payload errors documented in `experiments/ipadapter_device_fix_report.md` and `phase8_4_master_validation_report.md`.)*

