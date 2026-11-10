# MMCCP v2.0 — Google Colab Guide & First-Run Checklist

- **Verification Status:** `UNTESTED IN GOOGLE COLAB` *(notebook JSON/AST/schema validated locally on Windows; live Colab T4 execution has **NOT** yet been performed — follow the First-Run Checklist below and report any live runtime diffs)*
- **Target Baseline:** Branch `docs/run-guides` (on top of `main` `625b365` / `v2.0.0-rc3`)
- **Notebook File:** [`notebooks/MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb)

---

## 1. Before You Open the Notebook

### Requirements
1. **Google Account with Google Drive (`>= 30 GB` free space recommended)** to persist WebUI Forge, ~15–22 GB of diffusion/swap weights, HuggingFace caches, and rendered outputs across Colab disconnects.
2. **GPU Runtime:** Colab Free **T4 (16 GB VRAM)** or Colab Pro (**L4 24 GB / A100 40 GB**). Select `Runtime -> Change runtime type -> T4 GPU`.
3. **Optional Secrets (Colab Key icon on left sidebar):**
   - `HF_TOKEN`: HuggingFace read token (prevents anonymous rate-limits when pulling `Juggernaut-XL`, `OpenPose XL`, `SegFormer`, `DINOv2`, `CLIP`).
   - `NGROK_TOKEN`: Only if you prefer ngrok over `cloudflared` (no account needed for `cloudflared`) when viewing the Streamlit UI (`Cell 10`).

### Important Python / PyTorch Reality Check on Colab
Colab's default container image periodically shifts across **Python 3.11 / 3.12** and **CUDA 12.x / PyTorch 2.5+**. Because Forge and MMCCP (`requirements.txt`) share Python in Colab:
- **Do NOT blindly run `pip install torch==2.3.1+cu121`** if Colab comes with a newer CUDA toolkit / Python 3.12 (where older wheels may fail or fight preinstalled `torchvision`). The notebook keeps Colab's working GPU PyTorch (`Cell 3`), checks `torch.cuda.is_available()`, installs MMCCP's packages around it, and **keeps `numpy<2.0`** where supported (or warns cleanly if Colab pins `numpy>=2.0` on newer Python).
- Also remember the frozen verifier caveat from [02_LINUX_NOTES.md](02_LINUX_NOTES.md#3-known-linux-caveat-frozen-verifier-torch_bridgepy): `consistency_v2/verify/torch_bridge.py` (`3fa69071...`) points to Windows `C:\Stable-Diffusion\system\python\python.exe`, so `verify_setup.py --mode colab` reports a non-fatal `[WARN]` on Colab while DeepFace's 4 biometric judges still work.
- **Unpatched Upstream Forge Note (`UNTESTED`):** The tested Windows Forge had local fixes in `sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py`, `sd_forge_controlnet/scripts/controlnet.py`, and `modules_forge/utils.py` (see [`experiments/ipadapter_device_fix_report.md`](../../experiments/ipadapter_device_fix_report.md) & [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md#28-ipadapterpluspy-device-mismatch-cpu-vs-cuda0-on-8-gb-gpus)). On Colab T4 (16 GB VRAM), Forge generally keeps IP-Adapter on GPU without CPU offload, though unpatched upstream behavior remains `UNTESTED` — report any `cpu vs cuda:0` error in Cell 9 if hit.

---

## 2. How Drive Persistence Saves ~25 Minutes Per Reconnect

```
/content/drive/MyDrive/MMCCP/
├── CAPSTONE/                          (git clone of repo; or /content/CAPSTONE synced to Drive)
├── stable-diffusion-webui-forge/      (WebUI Forge + 4 extensions)
├── models/                            (symlinked or FORGE_MODELS_DIR -> Drive persistent weights!)
│   ├── Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors
│   ├── ControlNet/{openpose_xl.safetensors, ip-adapter-plus-face_sdxl_vit-h.safetensors}
│   ├── insightface/{inswapper_128.onnx, models/buffalo_l/...}
│   └── ...
├── hf_cache/                          (HF_HOME -> SegFormer, DINOv2, CLIP)
├── deepface_cache/                    (DEEPFACE_HOME -> Facenet512, ArcFace, SFace, GhostFaceNet)
└── outputs_archive/                   (zipped renders & sidecars)
```

Free Colab runtimes disconnect on inactivity or after dynamic session caps (~12h ceiling). Because weights live on **Google Drive**, after a disconnect you only re-run **Cell 0 (Drive mount + env vars)**, **Cell 3 (fast pip install into fresh VM)**, and **Cell 7 (relaunch Forge)** — **0 GB re-downloaded**.

---

## 3. Colab First-Run Checklist (`Cells 0–12`)

Walk down [`MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb) in exact order on your very first session and verify each expected output:

| Cell | Title | What It Does | Expected Success Signal | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Cell 0** | Mount Drive & Set Secrets/Paths | Mounts `/content/drive`, reads `HF_TOKEN`/`NGROK_TOKEN` from `google.colab.userdata`, sets `MMCCP_ROOT`, `FORGE_DIR`, `FORGE_MODELS_DIR`, `HF_HOME`, `DEEPFACE_HOME`, `PYTHONUTF8=1` | Prints GPU name (`Tesla T4` / `15360+ MiB`), free Drive GB, masked secret presence (`SET`/`UNSET`) | `UNTESTED` |
| **Cell 1** | Clone or Update MMCCP Repo | Clones `ShamantN/CAPSTONE` (checking out `main` or tested branch if needed) into `/content/CAPSTONE` | `git log -1 --oneline` prints commit hash; `tools/check_freeze.py` exits `0` (`[PASS]`) | `UNTESTED` |
| **Cell 2** | Install Linux Build & Media Tools | `apt-get install -y g++ build-essential aria2 libgl1` (needed if `insightface` compiles a wheel) | Exits `0` cleanly | `UNTESTED` |
| **Cell 3** | Install Python Dependencies | Keeps Colab GPU `torch`, installs `requirements.txt` + `insightface` + `onnxruntime-gpu` + Streamlit/DeepFace | Prints `torch` CUDA `True` and `numpy` version cleanly without breaking `deepface` import | `UNTESTED` |
| **Cell 4** | Setup WebUI Forge + 4 Extensions | Clones `lllyasviel/stable-diffusion-webui-forge` to Drive/local + clones `adetailer`, `sd-forge-couple`, `sd-webui-controlnet`, `sd-webui-reactor-sfw`; skips if already cached on Drive | All 4 folders exist under `extensions/` | `UNTESTED` |
| **Cell 5** | Download Required Models (Idempotent) | Uses `aria2c -c -x 8 -s 8` to fetch Juggernaut-XL, OpenPoseXL2 *(saved as `openpose_xl.safetensors`)*, IP-Adapter Plus-Face SDXL, `inswapper_128.onnx`, CodeFormer, `yolov8n.pt`, and verifies sizes + SHA-256 | All `7` required files report `[OK]` with matching byte counts | `UNTESTED` |
| **Cell 6** | Preflight Before Forge | Runs `python tools/verify_setup.py --mode colab --skip-forge --verify-sha256` | `FAIL: 0` *(expected `WARN` on Windows-only `torch_bridge`)* | `UNTESTED` |
| **Cell 7** | Launch WebUI Forge & Wait Until Ready | Starts Forge in background (`python launch.py --api --xformers --listen --port 7860 > forge.log 2>&1 &`), polls `/sdapi/v1/sd-models` and `/controlnet/model_list`; tails last 50 log lines on timeout | Prints `Forge READY!` and confirms `openpose_xl` + `ip-adapter-plus-face` appear | `UNTESTED` |
| **Cell 8** | Post-Forge Full Preflight & Quick Unit Tests | Runs `python tools/verify_setup.py --mode colab` (live Forge check) + `python -m unittest tests/test_portability.py tests/test_splits.py` | Zero failures on Forge API check and split guardrails | `UNTESTED` |
| **Cell 9** | 1-Scene DEV Smoke Test (`--mode fast`) | Runs `python run_from_scratch.py --pipeline v2 --mode fast --story storyboards/ebright.json --scene 1 --seed 11`, verifies output image + sidecar JSON | Prints scene render elapsed seconds and saves `edu_outputs/v2_fast/scene_1_seed_11.png` | `UNTESTED` |
| **Cell 10** | Full 6-Scene Zero-Cache `v2 default` Demo | Runs full `run_from_scratch.py --pipeline v2 --mode default --story storyboards/ebright.json --seed 11` cleanly from empty scratch cache | Generates `5` character anchors + `6` scenes + composite consistency scorecard | `UNTESTED` |
| **Cell 11** | Inspect Gallery & Archive to Google Drive | Displays generated character anchors & 6 scenes inline via `IPython.display`, checks `git status` clean, zips `edu_outputs/` + `hardcoded_cache.json` + logs to Drive | Displays all images and writes `mmccp_run_<timestamp>.zip` on Drive | `UNTESTED` |
| **Cell 12** | Optional: Streamlit Studio via Tunnel (`cloudflared`) | Launches `streamlit run app.py --server.port 8501` and prints temporary public `trycloudflare.com` URL | Prints clickable URL *(optional — stop cell cleanly when finished viewing UI)* | `UNTESTED` |
| **Cell 13** | Clean Shutdown | Stops background Forge/Streamlit processes and syncs final logs to Drive | Clean termination, VRAM freed | `UNTESTED` |
| **Cell 14** | Quick Reconnect / Resume After Disconnect | One-click Drive remount + fast pip restore + Forge relaunch skipping already-downloaded weights | Forge ready in ~2–3 min without re-downloading models | `UNTESTED` |

> **Important Colab Rule:** Never run `TEST-2` seeds (`1101..1505`) or `RESERVED` seeds (`606, 707, 808`). The notebook uses **DEV** seed `11` only.

