# MMCCP Environment and Execution Run Log

- **Project:** Multi-Modal Character Consistency Pipeline (MMCCP)
- **Parent Project:** Novel to Video Generation (NVG)
- **Institution:** PES University (Capstone Project Phase 3, Project ID `PW26_CGM_02`)
- **Execution Date & Time:** 2026-10-14T11:03:00+05:30
- **Team:** Samiya Naaz (PES1UG23CS504), Shamant Nagabhushana (PES1UG23CS536), Shihab K A (PES1UG23CS546), Shivu K C (PES1UG23CS551)
- **Guide:** Prof. Chitra G M

---

## 1. Hardware, Host OS, and Kernel Specifications

| Component | Measured Specification | Verification Source |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (Windows-10-10.0.26300-SP0) 64-bit | `platform.platform()`, `platform.version()` |
| **Host CPU Architecture** | x86_64 / AMD64 Family | `platform.machine()`, `os.environ` |
| **Host System RAM** | 31.71 GB Total Physical (17.91 GB Available at startup) | Windows `GlobalMemoryStatusEx` (`experiments/memory_utils.py`) |
| **Discrete GPU** | NVIDIA GeForce RTX 3070 Ti Laptop GPU | `nvidia-smi`, `torch.cuda.get_device_name(0)` |
| **VRAM Dedicated** | 8192 MiB (8.0 GB GDDR6) | `nvidia-smi`, `torch.cuda.get_device_properties(0)` |
| **NVIDIA Driver Version** | 610.62 | `nvidia-smi` |
| **CUDA UMD Version** | 13.3 (Driver level) | `nvidia-smi` |
| **Primary Project Python** | Python 3.10.6 (`D:\5TH SEM\HACKATHON\my_app_env\Scripts\python.exe`) | `sys.version`, `sys.executable` |
| **Secondary/Forge Python** | Python 3.10.6 embedded (`C:\Stable-Diffusion\system\python\python.exe`) | `C:\Stable-Diffusion\environment.bat` |
| **Deep Learning Engine** | PyTorch 2.3.1+cu121 (dynamically linked via `metrics.py` / `_SD_SITE`) | `torch.__version__`, `torch.version.cuda` |
| **Evaluation VRAM Cap** | 25% Active Memory Allocation Guard | Context-managed `vram_guard` in `metrics.py` |

---

## 2. Stable Diffusion WebUI Forge Backend Environment

- **Forge Root Directory:** `C:\Stable-Diffusion`
- **Forge WebUI Directory:** `C:\Stable-Diffusion\webui`
- **Active Startup Configuration:** `C:\Stable-Diffusion\webui\webui-user.bat`
- **Configured Command Line Flags:** `--api --xformers --cuda-malloc`
- **Default REST Endpoint:** `http://127.0.0.1:7860`

### 2.1 Installed WebUI Extensions (`C:\Stable-Diffusion\webui\extensions`)
1. `sd-webui-controlnet` (ControlNet v1.1.4xx Forge integration)
2. `sd-webui-reactor` (Fast Face-Swap and Restoration using InsightFace Inswapper + CodeFormer)

### 2.2 Models Verified on Disk (`C:\Stable-Diffusion\webui\models`)
| Category | File Path | File Size | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Base SDXL Checkpoint** | `Stable-diffusion\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | 6776.2 MB | Primary photographic SDXL base model |
| **Secondary Checkpoint** | `Stable-diffusion\animagine-xl-3.1.safetensors` | 6616.9 MB | Stylized / anime benchmark model |
| **ControlNet Unit** | `ControlNet\openpose_xl.safetensors` | 4772.3 MB | Pass 1 skeletal pose conditioning |
| **ControlNet Unit** | `ControlNet\depth_xl.safetensors` | 4772.3 MB | Pass 1 spatial geometry conditioning |
| **ControlNet Unit** | `ControlNet\instantid_xl.safetensors` | 2386.2 MB | Facial landmark identity injection |
| **ControlNet Unit** | `ControlNet\ip-adapter_xl.safetensors` | 670.0 MB | Decoupled cross-attention reference portrait identity |
| **ControlNet Unit** | `ControlNet\ip-adapter-faceid-plusv2_sdxl.bin` | 1418.6 MB | FaceID biometric feature injection |
| **ControlNet Unit** | `ControlNet\ip-adapter_sdxl_vit-h.safetensors` | 666.0 MB | Secondary CLIP-ViT-H identity adapter |
| **ControlNet Unit** | `ControlNet\ip-adapter-plus-face_sdxl_vit-h.safetensors` | 808.3 MB | Fine-grained facial patch adapter |
| **ControlNet Preprocessor** | `ControlNetPreprocessor\CLIP-ViT-bigG.safetensors` | 3519.0 MB | Vision token encoder for IP-Adapter SDXL |
| **ControlNet Preprocessor** | `ControlNetPreprocessor\CLIP-ViT-H-14.safetensors` | 2411.2 MB | Auxiliary vision token encoder |
| **ControlNet Preprocessor** | `ControlNetPreprocessor\depth_anything_v2\depth_anything_v2_vitl.safetensors` | 639.6 MB | Depth estimation preprocessor |
| **ControlNet Preprocessor** | `ControlNetPreprocessor\openpose\dw-ll_ucoco_384.onnx` | 128.2 MB | DW-Pose / OpenPose keypoint extractor |
| **LoRA Module** | `Lora\ip-adapter-faceid-plusv2_sdxl_lora.safetensors` | 354.6 MB | Companion LoRA for FaceID Plus v2 |
| **LoRA Module** | `Lora\sdxl_lightning_8step_lora.safetensors` | 375.6 MB | Accelerated step LoRA |
| **Face Restoration** | `Codeformer\codeformer-v0.1.0.pth` | 359.2 MB | ReActor post-swap facial fidelity restoration |
| **InsightFace Model** | `insightface\inswapper_128.onnx` | 528.6 MB | ReActor latent facial replacement kernel |
| **InsightFace Biometrics** | `insightface\models\buffalo_l\w600k_r50.onnx` | 166.3 MB | ArcFace 512-D identity representation |
| **InsightFace Biometrics** | `insightface\models\antelopev2\glintr100.onnx` | 248.6 MB | High-precision ArcFace embedding extractor |

---

## 3. Python Package Environment (`pip freeze`)

```text
absl-py==2.4.0
altair==5.5.0
annotated-types==0.7.0
astunparse==1.6.3
attrs==25.4.0
beautifulsoup4==4.14.3
blinker==1.9.0
cachetools==6.2.2
certifi==2025.11.12
charset-normalizer==3.4.4
click==8.1.8
colorama==0.4.6
decorator==5.2.1
deepface==0.0.99
filelock==3.29.0
fire==0.7.1
Flask==3.1.3
flask-cors==6.0.2
flatbuffers==25.12.19
gast==0.7.0
gdown==6.0.0
gitdb==4.0.12
GitPython==3.1.45
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-api-python-client==2.187.0
google-auth==2.43.0
google-auth-httplib2==0.2.1
google-generativeai==0.8.5
google-pasta==0.2.0
googleapis-common-protos==1.72.0
grpcio==1.76.0
grpcio-status==1.71.2
gTTS==2.5.4
gunicorn==25.3.0
h5py==3.14.0
httplib2==0.31.0
idna==3.11
ImageIO==2.37.2
imageio-ffmpeg==0.6.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
keras==3.12.1
lazy-loader==0.5
libclang==18.1.1
lightdsa==0.0.3
lightecc==0.0.6
lightphe==0.0.24
lz4==4.4.5
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
ml_dtypes==0.5.4
moviepy==2.2.1
mpmath==1.3.0
mtcnn==1.0.0
namex==0.1.0
narwhals==2.11.0
networkx==3.4.2
numpy==1.26.4
opencv-python==4.13.0.92
opt_einsum==3.4.0
optree==0.19.0
packaging==25.0
pandas==2.3.3
pillow==11.3.0
pip==22.2.1
proglog==0.1.12
proto-plus==1.26.1
protobuf==7.34.1
pyarrow==21.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pydantic==2.12.4
pydantic_core==2.41.5
pydeck==0.9.1
Pygments==2.20.0
pyparsing==3.2.5
PyPDF2==3.0.1
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
pytz==2025.2
referencing==0.37.0
requests==2.32.5
retina-face==0.0.17
rich==15.0.0
rpds-py==0.28.0
rsa==4.9.1
scikit-image==0.25.2
scipy==1.15.3
setuptools==63.2.0
six==1.17.0
smmap==5.0.2
soupsieve==2.8.3
streamlit==1.51.0
sympy==1.14.0
tenacity==9.1.2
tensorflow==2.21.0
termcolor==3.3.0
tf_keras==2.21.0
tifffile==2025.5.10
toml==0.10.2
tornado==6.5.2
tqdm==4.67.1
typing_extensions==4.15.0
typing-inspection==0.4.2
tzdata==2025.2
uritemplate==4.2.0
urllib3==2.5.0
watchdog==6.0.0
Werkzeug==3.1.8
wheel==0.46.3
wrapt==2.1.2
```

---

## 4. Codebase Inventory and Lines of Code (LOC)

- **Total Non-Environment Project Files:** 247
- **File Distribution by Extension:**
  - `.png`: 87 files
  - `.json`: 84 files
  - `.py`: 53 files
  - `.md`: 21 files
  - `.ttf`: 1 file
  - `.txt`: 1 file
- **Total Python Lines of Code (LOC):** 9,023 lines

### 4.1 Detailed Breakdown of Python Files
| File Path | LOC | Subsystem / Role |
| :--- | :--- | :--- |
| `app.py` | 902 | Production UI / Streamlit Web Dashboard |
| `metrics.py` | 2,029 | Quad-Modal Evaluation & Biometric Matrix |
| `run_from_scratch.py` | 162 | Headless Production Pipeline Batch Runner |
| `conditioning\controlnet_conditioner.py` | 79 | Multi-Unit ControlNet Payload Builder |
| `conditioning\crossframe_conditioner.py` | 28 | Latent Context & Inter-Frame Continuity |
| `conditioning\lora_registry.py` | 29 | LoRA Weights & Trigger Token Management |
| `conditioning\refinement.py` | 53 | ReActor 30-arg Payload & CodeFormer Builder |
| `conditioning\regional_conditioner.py` | 127 | Spatial Column Partitioning & BREAK Compiling |
| `experiments\benchmark_identity_spatial_methods.py` | 112 | Comparative Adapter & Spatial Harness |
| `experiments\benchmark_native_sota.py` | 151 | Multi-Scene Narrative Benchmark Suite |
| `experiments\controlnet_adapter.py` | 82 | ControlSpec Dataclass & REST Formatter |
| `experiments\evaluate_clean_run.py` | 129 | Metric Extraction & Clean Slate Evaluator |
| `experiments\heal_sota_scene5.py` | 39 | Targeted Scene 5 Auto-Healing Diagnostic |
| `experiments\memory_utils.py` | 105 | Windows Ctypes & PyTorch VRAM Monitor |
| `experiments\phase8_4_execution_suite.py` | 279 | Comprehensive Phase 8.4 Regression Suite |
| `experiments\phase8_4_stress_and_ablation_suite.py` | 303 | LoRA, Refinement, & Spatial Stress Tests |
| `experiments\phase9_anchor_quality_audit.py` | 122 | 4-Point Anchor Certification Auditor |
| `experiments\phase9_audit_collector.py` | 92 | Comprehensive Audit Report Aggregator |
| `experiments\phase9_runner.py` | 420 | Master Phase 9 SOTA Ablation Suite |
| `experiments\phase9_sequence_audit.py` | 160 | Temporal Sequence & Identity Drift Audit |
| `experiments\phase9_stress_suite.py` | 241 | Age Progression & Multi-Char Permutations |
| `experiments\run_ipadapter_fixed_scene2.py` | 71 | Isolated Scene 2 IP-Adapter Diagnostic |
| `experiments\scratch_inspect_audit.py` | 36 | Inspection Tool for Audit JSONs |
| `experiments\sota_generator.py` | 275 | Batch SOTA Two-Pass Renderer |
| `experiments\sota_qa_auditor.py` | 285 | Batch Evaluation of SOTA Outputs |
| `experiments\test_api_smoke.py` | 51 | Forge API Reachability & Model Smoke Test |
| `experiments\test_instantid_runtime.py` | 22 | InstantID ControlNet Preprocessor Test |
| `experiments\test_ipadapter_clean.py` | 173 | IP-Adapter Clean State Validation |
| `generation\forge_client.py` | 67 | Low-Level HTTP REST Interface to Forge |
| `generation\generation_orchestrator.py` | 231 | Master Two-Pass Diffusion Orchestrator |
| `identity\faceid.py` | 43 | Concrete FaceID Plus v2 Implementation |
| `identity\identity_conditioner.py` | 30 | Abstract Identity Conditioner Interface |
| `identity\instantid.py` | 45 | Concrete InstantID Implementation |
| `identity\ipadapter.py` | 61 | Concrete IP-Adapter SDXL Implementation |
| `qa\healing_controller.py` | 116 | Closed-Loop Pose Perturbation & Retries |
| `qa\programmatic_auditor.py` | 214 | Bounding Box & StorySpec QA Rules |
| `qa\test_anchor_manager.py` | 142 | 4-Point Anchor Certification Gate |
| `qa\vlm_visual_auditor.py` | 348 | BLIP Captioning & CLIP Chromatic Anomaly |
| `scratch\gen_new_teen_anchor.py` | 36 | Single Anchor Generation Utility |
| `scratch\generate_consistent_teen_anchor.py` | 36 | Consistent Teen Anchor Generator |
| `scratch\inspect_anchors.py` | 20 | Decode & Save Base64 Anchors |
| `scratch\inspect_prompts.py` | 13 | Scene Prompt Compiler Inspector |
| `scratch\inspect_reactor_args.py` | 10 | ReActor Argument Schema Inspector |
| `scratch\regenerate_visual_dna_anchors.py` | 91 | 5-Character Anchor Regeneration Script |
| `scratch\test_face_detect.py` | 46 | RetinaFace / DeepFace Diagnostic |
| `scratch\test_fixed_faces.py` | 39 | Post-Processed Face Inspection |
| `scratch\test_gpu_throughput_benchmark.py` | 171 | VRAM Plateau & Latency Stress Tester |
| `scratch\update_teen_cache.py` | 24 | Cache Replacement Utility for Teen |
| `story_engine\character_registry.py` | 112 | Visual DNA & Demographic Prompt Registry |
| `story_engine\prompt_builder.py` | 129 | Regional BREAK Syntax Prompt Compiler |
| `story_engine\scene_planner.py` | 113 | Narrative Scene Planner & Denoise Computer |
| `story_engine\schema.py` | 170 | StorySpec, SceneSpec, CharacterSpec Dataclasses|
| `story_engine\story_analyzer.py` | 159 | NLP Parser & Gemini LLM Interface |

---

## 5. Chronological Command Execution Log

1. **`2026-10-14T10:58:28`**: Scanned filesystem tree. Identified `my_app_env` virtual environment and project folders.
2. **`2026-10-14T10:58:40`**: Queried Python executable version: Python 3.10.6 confirmed.
3. **`2026-10-14T10:58:48`**: Computed file counts and LOC across all 53 `.py` files (total 9,023 LOC).
4. **`2026-10-14T10:59:03`**: Executed `nvidia-smi`: NVIDIA RTX 3070 Ti Laptop GPU (8192 MiB), Driver 610.62, CUDA 13.3.
5. **`2026-10-14T10:59:07`**: Checked running processes: No active Forge, WebUI, or Python processes found.
6. **`2026-10-14T10:59:49`**: Verified Forge root path at `C:\Stable-Diffusion`.
7. **`2026-10-14T10:59:58`**: Inspected `run.bat` and `webui-user.bat`. Confirmed launch arguments `--api --xformers --cuda-malloc`.
8. **`2026-10-14T11:00:45`**: Queried system RAM (31.71 GB physical, 17.91 GB available) and scanned model disk inventory (19 models found).
9. **`2026-10-14T11:01:21`**: Verified Forge Python environment: PyTorch 2.3.1+cu121 with CUDA active.
10. **`2026-10-14T11:02:01`**: Verified project dynamic linking: `metrics.py` successfully imported, loading DeepFace and PyTorch 2.3.1+cu121 with 25% VRAM cap.
11. **`2026-10-14T11:02:43`**: Executed AST and regex pattern scan across all files to locate every cache path, output directory, and reporting endpoint.
12. **`2026-10-14T11:03:30`**: Saved complete `pip freeze` to `docs_generated/raw/pip_freeze.txt` (114 packages).
