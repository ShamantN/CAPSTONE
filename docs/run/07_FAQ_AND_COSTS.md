# MMCCP v2.0 — FAQ, Timings, Hardware & Costs

- **Verification Status:** `TESTED` (local hardware timings measured across `72` TEST-2/DEV-2 renders, `6`-scene zero-cache demo, and Step 3 fresh-clone smoke run; branch `docs/run-guides` on top of `main` `625b365`; cloud Colab pricing & limits cited from official Google pages — check live Google links before running)

---

## 1. How Long Does Each Stage Take? (Measured Local Timings)

Measured on Windows 11, **NVIDIA RTX 3070 Ti Laptop GPU (8 GB VRAM)**, 32 GB RAM:

| Task / Workload | Measured Duration | Artifact / Source |
| :--- | :--- | :--- |
| **Forge WebUI Cold Start (`--api --xformers`)** | `48.06s` (Step 3 warm Windows test) / `62.6s` (R11 cold start) | `docs/run/TEST_LOG.md`, `v1_regression_verification.md` |
| **1-Scene DEV Smoke Test (`v2 fast`, zero-cache)** | `174.86s` (`2.92 min` scene render) / `312.71s` (`5.21 min` incl. 2 fresh anchors + eval) | `docs/run/TEST_LOG.md` (Step 3 fresh clone, `7887.0 MiB` peak) |
| **1 Scene Render (`v2 default`, sidecar distribution)** | `81s – 303s` (`mean 181.7s` on DEV-2 `n=30`; `mean 130.3s` on Ebright TEST-2) | `results/bundle_dev2/renders/`, `results/test2_ebright/renders/` |
| **Full 6-Scene Zero-Cache `v2 default` Demo (5 anchors + 6 scenes)** | `18.72 min` (`1123.28s` scene generation; Scene 1 `271.4s`, Scenes 2–4 & 6 `~147–154s`, Scene 5 `249.5s`) | Archived live-demo telemetry (`2026-11-10`) |
| **`v2 quality` (3x Best-of-N candidates enabled)** | Longer per multi-candidate scene (`mean 114.4s` in earlier lighter config; multi-fold higher with M4+M6 enabled) | `results/telemetry_clean_slate_report.json` |
| **Canonical Bank Hydration (`sync_canonical_banks.py`, 8 chars)** | `210.97s` (~3.5 min first CPU run while DeepFace caches weights) | `docs/run/TEST_LOG.md` |
| **Frozen Verifier CPU Unit Controls (`15` controls)** | `206.2s` (~3.4 min) on CPU | `unittest -v tests.verifier_controls.test_controls` |
| **Core Unit Test Suite (`tests/test_*.py`, `72` tests)** | `21.35s` (`72/72 OK` after bank hydration) | `docs/run/TEST_LOG.md` |
| **DEV-2 `30`-Render Full Bundle** | `90.86 min` (~1.5 hours) | `results/bundle_dev2/renders/` (`n=30` sidecars) |

*(Your runtime varies with GPU clock limits, power mode, browser VRAM usage, and whether models are already warmed in OS page cache.)*

---

## 2. Cloud Costs & Google Colab Limits

According to Google Colab's official FAQ (`https://colab.research.google.com/faq.html`) and paid tier page (`https://colab.research.google.com/signup`):

| Colab Tier | Price (Official Google Rate, USD) | GPU Access & Key Limits |
| :--- | :--- | :--- |
| **Free Tier** | `$0` | Access to NVIDIA **T4 (16 GB VRAM)** when capacity permits; **limits, idle timeouts, and maximum lifetime (up to ~12 hours max under optimal conditions, often shorter under inactivity/load) vary dynamically** and are **not published as guarantees** by Google. Expect disconnects if left idle. |
| **Pay As You Go** | `$9.99` (100 compute units) or `$49.99` (500 compute units) | No monthly commitment; units expire after **90 days**. Good for finishing multi-hour demo runs if free tier disconnects or rate-limits T4. |
| **Colab Pro** | `$9.99 / month` | `100` compute units/mo, access to faster GPUs (L4 / A100 when available) and high-RAM VMs. Unused units expire after **90 days**. |
| **Colab Pro+** | `$49.99 / month` | `600` compute units/mo (`100` base + `500` extra) and **background execution** up to 24h even if browser tab closes. |

**Recommendation for Free Colab:** Mount Google Drive (`/content/drive/MyDrive/MMCCP/`) so Forge (~1.5 GB + deps) and model weights (~14–22 GB) stay saved on Drive (`requires >= 30 GB free Google Drive`). Reconnecting after an idle timeout only restarts the Forge server (~1–2 min) instead of re-downloading ~22 GB.

---

## 3. What If My GPU Has Less Than 8 GB VRAM (e.g., 4 GB or 6 GB)?

**Honest answer: Not officially supported for this pipeline.**
- Both Pass 1 (`1024x1024` / `1216x832` SDXL UNet + OpenPose XL ControlNet + IP-Adapter Plus-Face SDXL + Forge Couple) and Pass 2 Img2Img + Face Detailer already push an **8 GB (8192 MiB)** RTX 3070 Ti Laptop GPU to **`7011 MiB` process delta / `7984 MiB` whole-system peak** even with `--xformers` and `xl_GPU_MB=1600.0`.
- On 4 GB or 6 GB GPUs, SDXL + ControlNet will likely either hit `CUDA out of memory` or swap constantly over PCIe (making single scenes take 10–15+ minutes).
- If you want to test anyway on 6 GB: switch `--mode fast` (disables M6 Detailer), lower `forge_inference_memory` in Forge settings, add `--lowvram` / `--medvram-sdxl` to `webui-user.bat`, and expect severe slowdown (**UNTESTED**).

---

## 4. Can I Run on AMD / Apple Silicon (MPS) / CPU-Only?

- **Generation (`Forge`):** Requires an NVIDIA CUDA GPU (`nvidia-smi` + `xformers`/CUDA). CPU-only SDXL image generation is impractically slow (hours per image), and Apple MPS / AMD ROCm were **not tested** (`--xformers` is NVIDIA CUDA only).
- **Evaluation, Auditing & Docs (`No GPU Needed`):** Runs cleanly on any CPU laptop (Windows/Linux)!

---

## 5. Running ONLY the CPU Verifier on Existing Images (No Forge GPU Required)

If you already have rendered PNGs in `<IMAGES_DIR>` (named `scene_<N>_seed_<S>.png` matching a storyboard JSON) and ground-truth character anchor PNGs in `<BANK_DIR>`, run:

```powershell
# 1. Always verify the frozen verifier hash first
python tools\check_freeze.py

# 2. Score an existing DEV folder (use Ebright or generalization storyboard, never unlocked TEST-2 unless authorized)
python -m consistency_v2.verify ^
  --images results\bundle_dev2\renders ^
  --story storyboards\ebright.json ^
  --bank experiments\clean_run_ground_truth\anchors ^
  --out scratch\rescore_dev2_check.json ^
  --baseline-report results\rescored_v1.2\baseline_v1_dev2\report.json
```

*(Note: on Linux/Colab without `C:\Stable-Diffusion\system\python\python.exe`, frozen `consistency_v2/verify/torch_bridge.py` skips the YOLO/CLIP/SegFormer/DINOv2 subprocess while DeepFace's 4 biometric judges and RetinaFace still score normally. See [02_LINUX_NOTES.md](02_LINUX_NOTES.md).)*

