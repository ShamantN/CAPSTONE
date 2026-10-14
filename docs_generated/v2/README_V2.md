# Multi-Modal Character Consistency Pipeline (MMCCP) v2

**PES University Capstone Phase 3 — Project ID:** `PW26_CGM_02`  
**GitHub Repository:** [https://github.com/ShamantN/CAPSTONE.git](https://github.com/ShamantN/CAPSTONE.git)  
**Integration Branch:** `consistency-v2`  
**Tags:** `baseline-v1`, `verifier-frozen`, `v2.0.0-rc1`  

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
& "D:\5TH SEM\HACKATHON\my_app_env\Scripts\streamlit.exe" run app.py --server.port 8501
```
Open your browser to `http://localhost:8501`. Select **Consistency v2 (MMCCP)** in the sidebar to enable methods M1 through M10.

### Run Headless Pipeline from CLI
```powershell
# Run single scene in Default mode
& "D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe" run_from_scratch.py --pipeline v2 --mode default --story storyboards/ebright.json

# Run in Fast screening mode (reduced steps, no detailer)
& "D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe" run_from_scratch.py --pipeline v2 --mode fast --story storyboards/ebright.json

# Run in Quality mode (Best-of-N selection + Face Detailer)
& "D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe" run_from_scratch.py --pipeline v2 --mode quality --story storyboards/ebright.json
```

---

## 2. Pipeline Execution Modes

| Parameter | Fast Mode (`--mode fast`) | Default Mode (`--mode default`) | Quality Mode (`--mode quality`) |
| :--- | :--- | :--- | :--- |
| **Pass 1 Steps (txt2img)** | 14 steps | 20 steps | 25 steps |
| **Pass 2 Steps (img2img)** | 10 steps | 15 steps | 18 steps |
| **M1: Identity Bank** | Active (Centroid) | Active (Nearest Pose/Scale) | Active (Nearest Pose/Scale) |
| **M2: Layout Planner** | Active (OpenPose) | Active (OpenPose + Depth) | Active (OpenPose + Depth) |
| **M3: Regional Couple** | Active | Active | Active |
| **M4: Dual InstantID** | Active ($w=0.55$) | Active ($w=0.65$) | Active ($w=0.70$) |
| **M5: Reliable Swap** | Active (Direct InsightFace) | Active (Direct InsightFace) | Active (Direct InsightFace + CodeFormer) |
| **M6: Face Detailer** | Disabled | Active ($denoise=0.35$) | Active ($denoise=0.38$, steps$=25$) |
| **M7: Wardrobe Lock** | Disabled | Active ($\Delta E \le 38.0$) | Active ($\Delta E \le 35.0$) |
| **M9: Best-of-N Selection** | Disabled ($N=1$) | Disabled ($N=1$) | Active ($N=3$, Screen + Heal) |
| **M10: Identity Memory** | Disabled | Active | Active |
| **Latency Budget** | $\approx 45$ seconds / scene | $\approx 1.8$ minutes / scene | $\approx 4.5$ minutes / scene |
| **Peak VRAM Budget** | $\approx 5.4$ GB | $\approx 6.2$ GB | $\approx 6.4$ GB |

---

## 3. Rollback to Baseline Pipeline (v1)

The v1 pipeline remains 100% intact and functional without any breaking modifications.

### In the Web UI (`app.py`):
1. In the left sidebar, navigate to the **Pipeline Architecture** section.
2. Select the **Consistency v1 (Baseline)** radio button.
3. Click **Execute Storyboard Generation**.

### In the Headless CLI:
```powershell
& "D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe" run_from_scratch.py --pipeline v1 --story storyboards/ebright.json
```

---

## 4. Git Repository, Branch & Tag Map

- **Remote:** `https://github.com/ShamantN/CAPSTONE.git` (`origin`)
- **Branches:**
  - `main`: Production stable branch. Never committed to directly.
  - `consistency-v2`: Long-lived integration branch containing all v2 code, verifier, tests, and documentation.
  - `sync/local-baseline`: Baseline import branch with open PR #1 into `main`.
  - Feature branches: `feat/verifier` (PR #3), `fix/verifier-encoding` (PR #4), `feat/m1-identity-bank` (PR #5), `feat/m2-layout-planner` (PR #6), `feat/m3-regional-couple` (PR #7), `feat/m4-identity-units` (PR #9), `feat/m5-face-transplant` (PR #10), `feat/m6-face-detailer` (PR #11), `feat/m7-wardrobe-lock` (PR #12), `feat/m8-reference-only` (PR #13), `feat/m9-best-of-n-healing` (PR #14), `feat/m10-identity-memory` (PR #15), `feat/pipeline-v2-integration` (PR #16), `feat/ui-v2` (PR #17).
- **Annotated Tags:**
  - `baseline-v1`: Initial code import and baseline measurement state.
  - `verifier-frozen`: Deterministic verification suite frozen with SHA256 in `consistency_v2/verifier_freeze.json`.
  - `v2.0.0-rc1`: Release candidate containing all accepted methods (M1–M10), full test reports, and review presentation materials.

---

## 5. How to Add M11 (Per-Character LoRA Training) Next

M11 was deliberately marked **OUT OF SCOPE** for this run to prioritize inference-time spatial binding, multi-reference identity conditioning, reliable face transplantation, and robust evaluation without the heavy computational and storage overhead of fine-tuning.

A clean extension point has been preserved in the codebase so M11 can be added later without refactoring existing pipelines:

1. **Config Toggle:**
   - In [`consistency_v2/config.py`](file:///D:/5TH%20SEM/HACKATHON/consistency_v2/config.py), the parameter `use_character_lora: bool = False` is already present. Setting `use_character_lora=True` in `best_config.json` will activate downstream LoRA loading.
2. **LoRA Registry:**
   - [`conditioning/lora_registry.py`](file:///D:/5TH%20SEM/HACKATHON/conditioning/lora_registry.py) remains fully functional and provides `resolve_character_lora(char_id)` to map character tokens to trained `.safetensors` files in `C:\Stable-Diffusion\webui\models\Lora\`.
3. **Dedicated Held-Out Seeds for LoRA:**
   - The seeds **{606, 707, 808}** were strictly held out and reserved. They have **NEVER** been used for development, tuning, testing, or ablation in this run. A future LoRA training run can use these clean seeds to benchmark generalization on truly untouched random noise vectors.
4. **Step-by-Step Training Guide:**
   - Install `kohya-ss/sd-scripts` or run LoRA training in a separate isolated venv.
   - For each character, curate 15–25 high-resolution cropped portrait and full-body images from the M1 Identity Bank (`consistency_v2/banks/<char_id>/`).
   - Train an SDXL rank-32 / alpha-16 LoRA with text encoder training using target trigger tokens (e.g., `Ebright_Y7K9`).
   - Copy the resulting `<char_id>.safetensors` into `C:\Stable-Diffusion\webui\models\Lora/`.
   - Update `conditioning/lora_registry.py` to register the file, and set `use_character_lora = True` in `consistency_v2/config.py`.
