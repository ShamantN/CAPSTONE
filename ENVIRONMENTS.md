# Environment Registry & Dependency Manifest

This document records the Python environments, hardware allocation, and runtime isolation policies for the MMCCP (Multi-Modal Character Consistency Pipeline) v2 project.

---

## 1. Primary Runtime Environments

### Environment 1: Project Application Venv (`my_app_env`)
- **Path:** `D:\5TH SEM\HACKATHON\my_app_env`
- **Python Executable:** `D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe`
- **Python Version:** 3.10.6
- **Purpose:** Primary application runtime for Streamlit Web UI (`app.py`), orchestration logic (`pipeline_v2.py`, `GenerationOrchestrator`), verification test suites, CLI runners, and metric evaluators.
- **Key Installed Packages:**
  - `torch` (2.3.1+cu121)
  - `torchvision` (0.18.1+cu121)
  - `transformers`
  - `deepface`
  - `opencv-python`
  - `streamlit`
  - `ultralytics`
  - `scikit-learn`
  - `scipy`
  - `Pillow`
- **Freeze Snapshot:** Tracked in [`docs_generated/v2/my_app_env_freeze.txt`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/my_app_env_freeze.txt)

---

### Environment 2: Stable Diffusion WebUI Forge Embedded Python (`forge_env`)
- **Path:** `C:\Stable-Diffusion\system\python`
- **Python Executable:** `C:\Stable-Diffusion\system\python\python.exe`
- **Python Version:** 3.10.6
- **Purpose:** Host environment for Forge WebUI server, ControlNet extension preprocessors, InsightFace face analysis, ONNX runtime execution, and direct ReActor/Inswapper execution.
- **Key Installed Packages:**
  - `torch` (2.3.1+cu121)
  - `xformers` (0.0.27)
  - `insightface` (0.7.3)
  - `onnxruntime-gpu` (1.17.1)
  - `controlnet_aux` (0.0.10)
  - `facexlib`
  - `codeformer`
- **Freeze Snapshot:** Tracked in [`docs_generated/v2/forge_python_freeze.txt`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/forge_python_freeze.txt)

---

## 2. Stable Diffusion WebUI Forge Extensions

- **Extensions Root:** `C:\Stable-Diffusion\webui\extensions\`
- **Installed Extensions & Commit SHAs:**
  - `sd-webui-controlnet`: Native ControlNet / IP-Adapter integration.
  - `sd-webui-reactor`: ReActor face restoration extension.
  - `sd-forge-couple`: Commit `c7884e81623d7fcbf4c92e3aea14ced8b5b6aa74` (Regional spatial column/box prompt conditioning).
  - `adetailer`: Commit `3a599f5d4607d8f9d8b9fc5a15526197418dae1a` (Automated high-resolution face detection & inpaint detailing).

---

## 3. HuggingFace Cache & Storage Routing

- **System Policy:** Windows Drive C: has restricted capacity (~7.4 GB free). Primary model caches are routed to Drive D: (>195 GB free).
- **HuggingFace Home Directory:** `D:\.cache\huggingface`
- **Transparent NTFS Junction:** `C:\Users\shama\.cache\huggingface <<===>> D:\.cache\huggingface`
- **Environment Variable:** `HF_HOME=D:\.cache\huggingface` configured system-wide.

---

## 4. Hardware and Isolation Policy

- **GPU:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8192 MB VRAM).
- **VRAM Ceiling:** Strict $\le 7.4$ GB peak usage during all generation passes.
- **Policy:**
  1. Only ONE diffusion workload may execute on the GPU at any time.
  2. All metric evaluators (DeepFace, SegFormer, DINOv2, CLIP, YOLOv8) run on CPU by default to preserve VRAM for Forge.
  3. No heavy conflicting packages are installed into Forge's embedded Python.
  4. Any local VLM for visual auditing (e.g., Qwen2-VL or moondream2) will run in an isolated CPU venv.
