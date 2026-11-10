# MMCCP v2.0 — Google Colab Guide & First-Run Checklist

> **UNTESTED on Colab.** Written from the code and validated only statically (JSON, syntax, URL/size/SHA parity). If a cell fails, see ['If something fails'](#if-something-fails) below.

- **Verification Status:** `UNTESTED` *(live Colab T4/L4/A100 execution has **not** been performed)*
- **Target Baseline:** Branch `docs/run-guides` (PR #35, targeting `main` `625b365` / `v2.0.0-rc3`)
- **Notebook File:** [`notebooks/MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb)

---

## 1. Start on Google Colab in 8 Clicks

> **URL & Repo Visibility Check (`PUBLIC`):** Verified via `gh repo view`: `ShamantN/CAPSTONE` is **PUBLIC**. However, this notebook link against `main` will only resolve once **PR #35 is merged into `main`**!
> - **Open-in-Colab Link (after PR #35 merges):** https://colab.research.google.com/github/ShamantN/CAPSTONE/blob/main/notebooks/MMCCP_Colab.ipynb
> - **If using a private fork instead:** Colab cannot open a private GitHub repo without access. Either (a) switch the fork to Public, or (b) upload [`notebooks/MMCCP_Colab.ipynb`](../../notebooks/MMCCP_Colab.ipynb) manually (`File -> Upload notebook`), add a GitHub `GITHUB_TOKEN` in Colab Secrets, and clone via `https://x-access-token:${GITHUB_TOKEN}@github.com/...` in Cell 1.

1. Click the Open-in-Colab link above *(after PR #35 merges to `main`, or upload `MMCCP_Colab.ipynb` manually before merge)*.
2. Click **`Runtime -> Change runtime type -> T4 GPU`** -> **Save**.
3. Run **Cell 0** and approve the Google Drive authorization prompt (`>= 30 GB` free Google Drive recommended).
4. *(Optional)* Click the **Key (Secrets)** icon on Colab's left sidebar, add `HF_TOKEN` (avoids HuggingFace throttling) and toggle **Notebook access** ON *(plus `NGROK_TOKEN` only if preferring ngrok over `cloudflared`)*.
5. Run **Cells 0 through 10** in exact order (`Runtime -> Run cell` step-by-step): watch for `[PASS]` / `[OK]` prints listed in Section 2 below.
6. Find persistent weights & zipped storyboard bundles on Google Drive under **`MyDrive/MMCCP/outputs_archive/`**.
7. If Colab disconnects after inactivity, reconnect and run **ONLY Cell 0 -> Cell 2 -> Cell 3 -> Cell 15 (Drive Reconnect / Resume)** — weights stay cached on Drive!
8. When finished, run **Cell 14 (Clean Shutdown)** and click **`Runtime -> Disconnect and delete runtime`** so free-tier GPU quota stops counting.

---

## 2. Cell-by-Cell First-Run Checklist (`Cells 0–15`)

| Cell | Title | Expected Output When Healthy *(All `UNTESTED` Live)* |
| :---: | :--- | :--- |
| **Cell 0** | Mount Drive, Secrets & Paths | Prints `Tesla T4` GPU, free Drive `GiB`, `HF_TOKEN: SET/UNSET`, sets env vars |
| **Cell 1** | Clone MMCCP Repo & Freeze Check | Prints `HEAD` commit; `tools/check_freeze.py` (`[PASS]`) & `audit_test2_isolation.py` (`PASSED`) |
| **Cell 2** | Install System Tools (`aria2`, `g++`) | Clean exit `0`; `aria2c --version` prints version header |
| **Cell 3** | Install Python Deps Around Colab `torch` | Prints `CUDA available: True`, `numpy` `< 2.0`, and `Imports OK` |
| **Cell 4** | Clone WebUI Forge + `4` Extensions to Drive | All `4` extension dirs exist (`adetailer`, `sd-forge-couple`, `sd-webui-controlnet`, `sd-webui-reactor`) |
| **Cell 5** | Apply Recommended MMCCP Forge Patches | Calls `tools/apply_forge_patches.py --apply` -> `Applied` / `Already applied` at commit `dfdcbab6...` *(skips safely if upstream moved)* |
| **Cell 6** | Download & Verify `7` Core Weights (`aria2c`) | All `7` files report `[OK]` / `[SKIP]` matching exact byte sizes & `openpose_xl.safetensors` filename |
| **Cell 7** | Pre-Forge Preflight (`verify_setup.py --mode colab`) | Reports `FAIL: 0` *(expected `WARN` on Windows-only `torch_bridge`)* |
| **Cell 8** | Start Forge in Background & Poll Readiness | Prints `Forge API is READY!` and lists Juggernaut-XL + OpenPose/IP-Adapter |
| **Cell 9** | Post-Forge Full Preflight & Quick CPU Tests | `FAIL: 0` on Forge API & `test_portability`/`test_verify_setup`/`test_splits` `OK` |
| **Cell 10** | 1-Scene DEV Smoke (`scene 1, seed 11, --mode fast`) | Saves `edu_outputs/v2_fast/scene_1_seed_11.png` + sidecar `.json` |
| **Cell 11** | Full 6-Scene `v2 default` Zero-Cache Demo | Renders `5` fresh anchors + `6` Ebright scenes on DEV `seed 11` |
| **Cell 12** | Display Gallery Inline & Zip to Google Drive | Renders thumbnails inline and saves `MyDrive/MMCCP/outputs_archive/mmccp_colab_run_*.zip` |
| **Cell 13** | *(Optional)* Streamlit UI via `cloudflared` | Prints temporary `https://...trycloudflare.com` UI link *(interrupt when done)* |
| **Cell 14** | Clean Shutdown | Terminates `launch.py` / `streamlit` / `cloudflared`; checks freeze & `git status` |
| **Cell 15** | Drive Reconnect / Resume After Disconnect | Remounts Drive cache & relaunches Forge skipping redownloads |

---

## 3. Known Colab Risks

1. **Ephemeral sessions & idle disconnects:** Colab wipes `/content/` on disconnect and enforces dynamic idle/lifetime limits (~12h max ceiling on free tier). Drive persistence keeps Forge (~1.7 GiB + extensions) and weights, while Python pip deps in `/usr/local` reinstall on reconnect (~3–5 min via Cell 3).
2. **Free-tier T4 GPU not guaranteed:** Under heavy demand Colab may assign no GPU or throttle sessions; never expect dedicated uninterrupted multi-hour queues without Colab Pro / Pay-as-you-go.
3. **Long first-time downloads (`~14.2–25 GiB`):** Initial Drive download pulls `Juggernaut-XL` (`6.62 GiB`) + `openpose_xl` (`4.66 GiB`) + `CLIP-ViT-H-14` (`2.35 GiB`) + `ip-adapter-plus-face` (`808 MiB`) + `inswapper_128` (`529 MiB`) + `CodeFormer` (`359 MiB`) + `buffalo_l` (`275 MiB`) + HF/DeepFace caches (`~22–25 GiB` total minimum).
4. **Different Python/CUDA/PyTorch than tested local `3.10.6 / cu121 / 2.3.1`:** Google Colab images run newer Python (`3.11/3.12`) and `torch 2.5+ / CUDA 12.x`. The notebook intentionally keeps Colab's GPU `torch` (because [`requirements.txt`](../../requirements.txt) excludes `torch`/`torchvision`), pins `numpy<2.0.0`, and installs deps around Colab's runtime — ABI edge cases (such as compiling `insightface==0.7.3`) remain untested live.
5. **Windows-only paths skipped:** `configure_windows_forge_env()` is a no-op on POSIX, [`get_host_ram_mb`](../../experiments/memory_utils.py) reads `/proc/meminfo`, and frozen [`consistency_v2/verify/torch_bridge.py`](../../consistency_v2/verify/torch_bridge.py) skips YOLO/CLIP/SegFormer/DINOv2 subprocess calls because `C:\Stable-Diffusion\system\python\python.exe` does not exist on Colab (DeepFace's 4 judges still run directly).
6. **16 GB T4 vs 8 GB Laptop Memory Settings (`Not Validated on T4`):** [`PipelineV2`](../../consistency_v2/pipeline_v2.py) still pushes `xl_GPU_MB=1600.0` and `forge_inference_memory=3072.0` via `/sdapi/v1/options` tuned for an 8 GB laptop GPU; this runs conservatively on 16 GB T4 in theory, **not** benchmarked live.
7. **Upstream Forge Patch Compatibility:** **Cell 5** runs `tools/apply_forge_patches.py` pinned cleanly to upstream Forge commit `dfdcbab685e57677014f05a3309b48cc87383167` — Cell 4 clones Forge `HEAD`, so **Cell 5** will check out `dfdcbab685e57677014f05a3309b48cc87383167` cleanly before applying the 3 recommended patches.

---

## 4. If Something Fails *(Top 8 Likely Failures — All `[unverified on Colab]`)*

1. **`CUDA GPU runtime required` in Cell 3 `[unverified on Colab]`:** Colab booted CPU runtime. **Fix:** `Runtime -> Change runtime type -> T4 GPU`, then restart from Cell 0.
2. **Drive link `404` / repo clone fails `[unverified on Colab]`:** PR #35 not yet merged into `main`, or running on a private fork. **Fix:** Set `MMCCP_BRANCH="docs/run-guides"` in Cell 1, or upload the `.ipynb` manually and pass a Secret token on private forks.
3. **`insightface==0.7.3` wheel build error on Python 3.12 `[unverified on Colab]`:** C++ header or Cython version issue. **Fix:** Ensure Cell 2 ran (`build-essential g++ python3-dev`), upgrade `cython`/`wheel`, or install with `--no-build-isolation`.
4. **`numpy` upgraded to `2.x` breaking `WardrobeLock` or binary wheels `[unverified on Colab]`:** Colab's base packages pull NumPy 2. **Fix:** Re-run `pip install "numpy>=1.26.0,<2.0.0" --force-reinstall --no-deps` and restart Colab runtime.
5. **HuggingFace `429 Too Many Requests` in Cell 6 `[unverified on Colab]`:** Shared Colab IP rate-limited. **Fix:** Save `HF_TOKEN` in Colab Secrets, re-run Cell 0, and re-run Cell 6 (`aria2c -c` resumes partial files).
6. **Forge startup timeout (`Cell 8`) `[unverified on Colab]`:** First boot installing Forge dependencies takes > 8 minutes or fails on a conflicting pip version inside Forge. **Fix:** Inspect the printed tail of `/content/forge.log`, fix any missing package via `!pip install ...`, and re-run Cell 8.
7. **`RuntimeError: Expected all tensors to be on the same device` (`IPAdapterPlus.py`) `[unverified on Colab]`:** Unpatched Forge triggered offload mismatch. **Fix:** Verify Cell 5 pinned Forge commit `dfdcbab685e57677014f05a3309b48cc87383167` and `tools/apply_forge_patches.py --check` exits `0`.
8. **`cloudflared` tunnel fails to allocate URL (`Cell 13`) `[unverified on Colab]`:** Temporary Cloudflare rate-limit. **Fix:** Add `NGROK_TOKEN` in Colab Secrets and launch `pyngrok` tunnel instead, or skip UI and download the zipped renders from Drive via Cell 12.
