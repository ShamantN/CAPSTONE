# Multi-Modal Character Consistency Pipeline (MMCCP) v2

> [!NOTE]
> **ARCHITECTURE REFERENCE ONLY:** Quantitative results, pre-registration reconciliation, and closeout caveats live in root [`README_V2.md`](../../README_V2.md), [`07_measured_results.md`](07_measured_results.md), and [`PREREG_RECONCILIATION.md`](PREREG_RECONCILIATION.md).
> All `v2.0.0-rc1` and `v2.0.0-rc2` overstatements have been retracted and reconciled under Frozen Verifier v1.2.

**PES University Capstone Phase 3 — Project ID:** `PW26_CGM_02`  
**GitHub Repository:** [https://github.com/ShamantN/CAPSTONE.git](https://github.com/ShamantN/CAPSTONE.git)  
**Active Branches:** `main` (reconciled baseline/docs), `consistency-v2` (integration history), `finalize/v2.0.0` (PR #37 closeout)  
**Tags:** `baseline-v1`, `verifier-frozen`, `verifier-v1.1-frozen`, `verifier-v1.2-frozen`, `test2-sealed`, `v2.0.0-rc1` (superseded), `v2.0.0-rc2` (retracted scorecard), `v2.0.0-rc3`, `pre-repair-snapshot`  

---

## 1. Quick Start & One-Command Demo

### Environment Requirements
- **OS:** Windows 11 64-bit (PowerShell)
- **GPU:** NVIDIA RTX 3070 Ti Laptop GPU (8 GB VRAM) or equivalent modern GPU
- **RAM:** 32 GB DDR4
- **WebUI Forge:** Running at `http://127.0.0.1:7860` with `--api --xformers --cuda-malloc`
- **Python:** Python 3.10.6 virtual environment (`my_app_env`)

### Launch WebUI Forge Server
```powershell
cmd.exe /c "cd /d C:\Stable-Diffusion && run.bat"
```
*Verify that the API is responsive at `http://127.0.0.1:7860/sdapi/v1/options` (HTTP 200).*

### Launch Streamlit Visual Studio
```powershell
my_app_env\Scripts\streamlit.exe run app.py --server.port 8501
```
Open your browser to `http://localhost:8501`. Under **⚙️ Pipeline Configuration** -> **Pipeline Architecture:** in the left sidebar, select **`v2 (Consistency v2 SOTA)`** to configure `ConsistencyV2Config`.

### Run Headless Pipeline from CLI
```powershell
# Run single scene in Default mode
my_app_env\Scripts\python.exe run_from_scratch.py --pipeline v2 --mode default --story storyboards/ebright.json

# Run in Fast screening mode (reduced steps, no detailer)
my_app_env\Scripts\python.exe run_from_scratch.py --pipeline v2 --mode fast --story storyboards/ebright.json

# Run in Quality mode (configures higher steps; note best-of-N config flag is not looped inside generate_scene)
my_app_env\Scripts\python.exe run_from_scratch.py --pipeline v2 --mode quality --story storyboards/ebright.json
```

---

## 2. Pipeline Execution Modes (Verified in `consistency_v2/config.py`)

| Parameter | Fast Mode (`--mode fast`) | Default Mode (`--mode default`) | Quality Mode (`--mode quality`) |
| :--- | :--- | :--- | :--- |
| **Pass 1 Steps (`txt2img`)** | `14` steps | `20` steps | `25` steps |
| **Pass 2 Steps (`img2img`)** | `10` steps | `15` steps | `18` steps |
| **M1: Identity Bank** | Active (`pose_and_scale`) | Active (`pose_and_scale`) | Active (`pose_and_scale`) |
| **M2: Layout Planner** | Active (`OpenPose`) | Active (`OpenPose`) | Active (`OpenPose`) |
| **M3: Regional Couple** | Active | Active | Active |
| **M4: IP-Adapter Plus / InstantID** | IP-Adapter Plus (`0.55`), InstantID `OFF` | IP-Adapter Plus (`0.55`), InstantID `OFF` | IP-Adapter Plus (`0.55`), InstantID `OFF` |
| **M5: Reliable Swap** | Active (`inswapper_128` + CodeFormer) | Active (`inswapper_128` + CodeFormer) | Active (`inswapper_128` + CodeFormer) |
| **M6: Face Detailer** | Disabled | Active (`denoise=0.28`, `steps=20`) | Active (`denoise=0.28`, `steps=20`) |
| **M7: Wardrobe Lock** | Active ($\Delta E \le 38.0$, `blend=0.70` in code) | Active ($\Delta E \le 38.0$, `blend=0.70` in code) | Active ($\Delta E \le 38.0$, `blend=0.70` in code) |
| **M9: Best-of-N Flag** | Disabled | Disabled | `enable_best_of_n=True` (`3`, logged only; not executed in `generate_scene()`) |
| **M10: Identity Memory** | Active (`0.75` threshold) | Active (`0.75` threshold) | Active (`0.75` threshold) |
| **Measured Latency (`r9_clean_slate_telemetry`)** | Not separately timed | `93.72` seconds / scene | `114.38` seconds / scene |
| **Measured Peak VRAM (`vram_test2_sealed.csv` / `r9`)** | Not separately profiled | `7984` MiB (`7.797` GiB) whole-system peak (`7011` MiB delta) | `7964` MiB (`7.777` GiB) observed across combined `r9` trace (not isolated to `--mode quality`) |

---

## 3. Rollback to Baseline Pipeline (v1)

The v1 pipeline remains intact and functional without any breaking modifications.

### In the Web UI (`app.py`):
1. In the left sidebar under **⚙️ Pipeline Configuration**, navigate to the **Pipeline Architecture:** radio control.
2. Select **`v1 (Baseline Phase 3)`**.
3. Click **`⚡ Generate 6-Scene Storyboard with Phase 3 Enhancements`** (or **`⚡ Generate 6-Scene Storyboard with Consistency v2 (SOTA)`** when running `v2`).

### In the Headless CLI:
```powershell
my_app_env\Scripts\python.exe run_from_scratch.py --pipeline v1 --story storyboards/ebright.json
```

---

## 4. Git Repository, Branch & Tag Map

- **Remote:** `https://github.com/ShamantN/CAPSTONE.git` (`origin`)
- **Branches:**
  - `main`: Production stable branch (reconciled docs + PR #36). Never committed to directly.
  - `consistency-v2`: Integration branch containing v2 code, verifier, tests, and documentation history.
  - `finalize/v2.0.0`: Closeout PR #37 branch targeting `main`.
  - Feature branches: `feat/verifier` (PR #3), `fix/verifier-encoding` (PR #4), `feat/m1-identity-bank` (PR #5), `feat/m2-layout-planner` (PR #6), `feat/m3-regional-couple` (PR #7), `feat/m4-identity-units` (PR #9), `feat/m5-face-transplant` (PR #10), `feat/m6-face-detailer` (PR #11), `feat/m7-wardrobe-lock` (PR #12), `feat/m8-reference-only` (PR #13), `feat/m9-best-of-n-healing` (PR #14), `feat/m10-identity-memory` (PR #15), `feat/pipeline-v2-integration` (PR #16), `feat/ui-v2` (PR #17).
- **Annotated Tags:**
  - `baseline-v1`: Initial code import and baseline measurement state.
  - `verifier-frozen`, `verifier-v1.1-frozen`, `verifier-v1.2-frozen`: Deterministic verification suite frozen with SHA-256 in `consistency_v2/verifier_freeze.json`.
  - `test2-sealed`: Sealed `TEST-2` snapshot prior to one-shot evaluation.
  - `v2.0.0-rc1`, `v2.0.0-rc2`, `v2.0.0-rc3`: Release candidates (`rc2` post-hoc scorecard shifts formally retracted in `rc3`).

---

## 5. How to Add M11 (Per-Character LoRA Training) Next

M11 was deliberately marked **OUT OF SCOPE** for this run to prioritize inference-time spatial binding, multi-reference identity conditioning, reliable face transplantation, and robust evaluation without the heavy computational and storage overhead of fine-tuning.

A placeholder extension hook is present in config (`v2.1` would need to wire runtime loading):

1. **Config Toggle (Placeholder Only in `v2.0`):**
   - In [`consistency_v2/config.py`](../../consistency_v2/config.py), `use_character_lora: bool = False` exists as a configuration field, but `consistency_v2/pipeline_v2.py` does **not** currently read it at runtime; downstream LoRA prompt/unit wiring must be implemented in `v2.1`.
2. **LoRA Registry Utility:**
   - [`conditioning/lora_registry.py`](../../conditioning/lora_registry.py) defines `LoRARegistry(lora_dir)` with `scan_loras()` and `build_lora_prompt_string()` to enumerate `.safetensors` files under a Forge `models/Lora/` folder (pass `lora_dir` explicitly — passing `None` triggers an assignment bug in `__init__`).
3. **Dedicated Held-Out Seeds for LoRA:**
   - The seeds **{606, 707, 808}** were strictly held out and reserved. They have **NEVER** been used for development, tuning, testing, or ablation in this run. A future LoRA training run can use these clean seeds to benchmark generalization on truly untouched random noise vectors.
4. **Planned Future Training Steps (`v2.1`):**
   - Install `kohya-ss/sd-scripts` or run LoRA training in a separate isolated venv.
   - Curate high-resolution cropped portrait and full-body images per character from the M1 Identity Bank (`consistency_v2/banks/<char_id>/`).
   - Train an SDXL LoRA with trigger tokens, place `.safetensors` into `$FORGE_ROOT/webui/models/Lora/`, pass that path into `LoRARegistry(lora_dir=...)`, and wire `use_character_lora` inside `consistency_v2/pipeline_v2.py`.
