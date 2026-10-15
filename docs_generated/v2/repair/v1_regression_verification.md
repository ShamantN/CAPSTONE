# Phase R0: Baseline v1 Regression Render Verification

- **Date:** 2026-10-15
- **Pipeline:** Baseline v1 (`render_phase3_scene` in `app.py`)
- **Checkpoint:** `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`
- **Scene:** Scene 1 "The Spark of Curiosity" (Mother & Young Ebright)
- **Artifact:** `scratch/v1_regression/scene_1_v1_regression.png`
- **Report:** `scratch/v1_regression/v1_regression_report.json`

## Execution Telemetry:
- **Forge Cold Start:** 62.6s (clean startup with `--api --xformers --cuda-malloc`)
- **Generation Time:** 149.4s (Pass 1 + Pass 2 diffusion)
- **Peak VRAM During Render:** 5704 MB (well within 7.4 GB budget)

## Evaluation Results:
| Character | Character Name | ArcFace | CLIP-I | SSIM | Composite | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mother` | Ebright's Mother | **0.9116** | 0.7077 | 0.5021 | 0.6875 | SUCCESS |
| `young_ebright` | Young Richard Ebright | **0.5039** | 0.7714 | 0.4878 | 0.5468 | SUCCESS |

## Verdict:
**PASSED**: Baseline v1 pipeline is fully intact, fully functional, executes cleanly via Forge API, and successfully reproduces historical biometric and composite metric scores.
