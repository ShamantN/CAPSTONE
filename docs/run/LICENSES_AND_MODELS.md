# MMCCP v2.0 — Third-Party Licenses & Model Attribution Guide

- **Status:** `VERIFIED AGAINST UPSTREAM MODEL CARDS & REPO REQUIREMENTS` (`docs/collaboration` on `origin/main` `36f6646`)
- **Cross-Reference:** Exact download URLs, byte sizes, local destinations, and SHA-256 checksums are cataloged in [`04_MODELS_AND_DOWNLOADS.md`](04_MODELS_AND_DOWNLOADS.md).

> [!IMPORTANT]
> **No Model Weights Redistributed Here:** This repository contains **zero** binary model weights (`.safetensors`, `.onnx`, `.pth`, `.bin`, `.h5`, `.pt`). End users download all model files directly from their upstream HuggingFace/GitHub authors under each upstream model's own license.
>
> **Repository Source Code License (`OPEN QUESTION FOR MAINTAINER`):** No root `LICENSE` file currently exists in `ShamantN/CAPSTONE`. Before wider public collaboration or third-party forks, **please decide which license you want for MMCCP's own source code** (for example `MIT` or `Apache-2.0`, noting copyleft interactions with GPL-3.0 WebUI Forge & AGPL-3.0 Ultralytics if bundled).

---

## 1. Diffusion, Conditioning & Restoration Weights

| Component / Model | Upstream Source | Upstream License | Commercial Use / Redistribution Notes | Verification Action |
| :--- | :--- | :--- | :--- | :--- |
| **Juggernaut-XL v9 RunDiffusionPhoto v2** *(default SDXL checkpoint)* | `RunDiffusion/Juggernaut-XL-v9` *(HF)* / Civitai | **`CreativeML Open RAIL++-M`** | Research & permissive generation subject to RAIL use-based restrictions; Civitai/RunDiffusion model cards include commercial/service terms | **Verify before redistribution** / check RunDiffusion commercial hosting terms |
| **Animagine XL 3.1** *(optional checkpoint)* | `cagliostrolab/animagine-xl-3.1` | **`Fair AI Public License 1.0-SD`** *(HF card `openrail++`)* | Requires downstream derivatives/modifications to remain shared under compatible terms | **Verify before redistribution** |
| **ControlNet OpenPose SDXL (`OpenPoseXL2`)** | `thibaud/controlnet-openpose-sdxl-1.0` | **`other`** *(HF card custom/unspecified)* | No standard OSI license declared on HF card metadata | **Verify before redistribution** |
| **ControlNet Depth SDXL** *(optional)* | `diffusers/controlnet-depth-sdxl-1.0` | **`OpenRAIL++`** | Standard OpenRAIL++ use-based restrictions apply | Inspect upstream card |
| **IP-Adapter Plus-Face SDXL, IP-Adapter SDXL, `CLIP-ViT-H-14`, `CLIP-ViT-bigG`** | `h94/IP-Adapter` | **`Apache-2.0`** *(code & standard adapters)* | Permissive with attribution & `NOTICE`; *(Exception: **`IP-Adapter FaceID Plus v2`** uses a **Non-commercial** license!)* | **Flag:** Keep optional FaceID Plus v2 non-commercial only |
| **InstantID (`control_instant_id_sdxl`, `ip-adapter.bin`)** *(optional)* | `InstantX/InstantID` | **`Apache-2.0`** *(weights/code; relies on **InsightFace `antelopev2` = Non-commercial**)* | InstantID model weights list Apache-2.0, but **InsightFace `antelopev2` embeddings restrict commercial deployment** | **Verify before redistribution** *(non-commercial end-to-end)* |
| **SDXL-Lightning 8-Step LoRA** *(optional)* | `ByteDance/SDXL-Lightning` | **`Apache-2.0`** | Permissive with attribution | Standard Apache-2.0 |
| **InsightFace `inswapper_128.onnx` & `buffalo_l` (`antelopev2`)** | `deepinsight/insightface` / `Gourieff/ReActor` HF | **Non-commercial research license** | **NON-COMMERCIAL RESEARCH ONLY** (`InsightFace` explicitly prohibits commercial use without separate licensing) | **Verify before redistribution** — commercial use strictly prohibited |
| **CodeFormer (`codeformer-v0.1.0.pth`)** | `sczhou/CodeFormer` | **S-Lab License 1.0** | **NON-COMMERCIAL / research & academic use only** | **Verify before redistribution** — non-commercial only |

---

## 2. Verifier, Segmentation, Detection & Application Stack

| Component / Library | Upstream Source | Upstream License | Key Legal / Redistribution Notes | Verification Action |
| :--- | :--- | :--- | :--- | :--- |
| **Stable Diffusion WebUI Forge** *(runtime server dependency)* | `lllyasviel/stable-diffusion-webui-forge` | **`GPL-3.0`** | Strong copyleft; any bundled/distributed derivative software integrating Forge internals must comply with GPL-3.0 | Separate localhost REST process here (`/sdapi/v1/*`) |
| **Ultralytics YOLOv8n (`yolov8n.pt`)** | `ultralytics/ultralytics` | **`AGPL-3.0`** *(or commercial Ultralytics license)* | **Strong network copyleft (AGPL-3.0):** hosting a public/commercial web service calling `ultralytics` requires releasing server source under AGPL-3.0 unless an enterprise license is purchased | **Verify before commercial hosting / redistribution** |
| **SegFormer Clothes (`mattmdjaga/segformer_b2_clothes`)** | HuggingFace (`mattmdjaga`) | **`other`** *(HF card; MIT base `b2-finetuned` / ATR dataset academic caveats)* | Dataset (ATR) & fine-tuned weights carry non-standardized card metadata | **Verify before redistribution** |
| **DINOv2 Base (`facebook/dinov2-base`)** | Meta / HuggingFace | **`Apache-2.0`** | Permissive with notice & attribution | Standard Apache-2.0 |
| **CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)** | OpenAI / HuggingFace | **`MIT`** *(OpenAI CLIP repo; HF card license field unset)* | Code & weights released under MIT upstream | **Verify before redistribution** *(HF metadata unset)* |
| **BLIP Image Captioning Base** *(optional reviewer)* | `Salesforce/blip-image-captioning-base` | **`BSD-3-Clause`** | Permissive with copyright notice | Standard BSD-3-Clause |
| **DeepFace + Backbones (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`, `RetinaFace`, `MTCNN`)** | `serengil/deepface`, `opencv/opencv_zoo`, `HamadYA/GhostFaceNets` | Code: **`MIT`** (`SFace`: **`Apache-2.0`**) | Framework wrappers are permissive (`MIT`/`Apache-2.0`); upstream face-recognition research datasets/weights (e.g., InsightFace ArcFace `buffalo_l` / WebFace) often carry academic caveats | **Verify before redistribution** for commercial biometric hosting |
| **PyTorch (`2.3.1`), Transformers, OpenCV, Streamlit, NumPy, SciPy** | PyPI / PyTorch | `BSD-3-Clause` / `Apache-2.0` / `MIT` | Standard permissive Python scientific stack | Standard OSS attribution |

---

## 3. Summary Checklist for Contributors

1. **Academic / Research Use (Coursework, Reproducible Benchmarks):** Compatible across all listed components with proper citation and upstream license notices intact.
2. **Public Fork / Redistribution:** **Never** mirror or commit any third-party weight binaries (`.safetensors`, `.onnx`, `.pth`, `.h5`, `.pt`, `.bin`) into Git or GitHub Releases. Link upstream manifests only ([`04_MODELS_AND_DOWNLOADS.md`](04_MODELS_AND_DOWNLOADS.md)).
3. **Commercial Deployment:** **Incompatible out of the box** without replacing or commercially licensing `InsightFace` (`inswapper_128`, `buffalo_l`), `CodeFormer` (`S-Lab 1.0`), `YOLOv8n` (`AGPL-3.0`), `WebUI Forge` (`GPL-3.0`), and `Juggernaut-XL` (`OpenRAIL++-M`).

