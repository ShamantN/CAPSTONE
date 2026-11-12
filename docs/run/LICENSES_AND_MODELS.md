# MMCCP v2.0 — Third-Party Licenses & Model Attribution Guide

- **Status:** `API-VERIFIED` against HuggingFace Hub (`/api/models/*`) and GitHub REST APIs (`/repos/*`) on `2026-11-12` (components with custom/unset metadata explicitly marked `UNVERIFIED` below).
- **Cross-Reference:** Exact download URLs, byte sizes, local destinations, and SHA-256 checksums are cataloged in [`04_MODELS_AND_DOWNLOADS.md`](04_MODELS_AND_DOWNLOADS.md).

> [!IMPORTANT]
> **No Model Weights Redistributed Here:** This repository contains **zero** binary model weights (`.safetensors`, `.onnx`, `.pth`, `.bin`, `.h5`, `.pt`). End users download all model files directly from upstream HuggingFace/GitHub sources under each author's license.
>
> **Repository Source Code License (`NO ROOT LICENSE FILE`):** No `LICENSE` file is included for MMCCP's own code (`LICENSE_FOR_OWN_CODE=""`). Absent an explicit open-source grant, **standard default copyright applies; the university's IP policy has not been checked**.

---

## 1. Diffusion, Conditioning & Restoration Weights (Verified `2026-11-12`)

| Component / Model | Upstream Source | Checked Metadata / License (`2026-11-12`) | Commercial Use / Redistribution Notes | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Juggernaut-XL v9 RunDiffusionPhoto v2** *(default SDXL checkpoint)* | [`RunDiffusion/Juggernaut-XL-v9`](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) | HF API: **`creativeml-openrail-m`** *(CreativeML Open RAIL-M)* | Research & permissive generation subject to RAIL use-based restrictions; check Civitai/RunDiffusion commercial hosting terms | `VERIFIED (HF API HTTP 200)` |
| **Animagine XL 3.1** *(optional checkpoint)* | [`cagliostrolab/animagine-xl-3.1`](https://huggingface.co/cagliostrolab/animagine-xl-3.1) | HF API: **`openrail++`** *(card references Fair AI Public License 1.0-SD)* | Derivative/modified models remain subject to upstream share-alike terms | `VERIFIED (HF API HTTP 200)` |
| **ControlNet OpenPose SDXL (`OpenPoseXL2`)** | [`thibaud/controlnet-openpose-sdxl-1.0`](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0) | HF API: **`other`** *(no standard SPDX/RAIL license tag on card)* | Non-standard card metadata | **`UNVERIFIED`** *(custom/unset card license)* |
| **ControlNet Depth SDXL** *(optional)* | [`diffusers/controlnet-depth-sdxl-1.0`](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) | HF API: **`openrail++`** | Standard OpenRAIL++ use-based restrictions apply | `VERIFIED (HF API HTTP 200)` |
| **IP-Adapter Plus-Face SDXL, `CLIP-ViT-H-14`, `CLIP-ViT-bigG`** | [`h94/IP-Adapter`](https://huggingface.co/h94/IP-Adapter) | HF API: **`apache-2.0`** *(standard adapters)*; note optional `FaceID Plus v2` carries non-commercial terms | Permissive with attribution & `NOTICE` (`IP-Adapter` GitHub org lookup `h94/IP-Adapter` returned HTTP 404; checked via HuggingFace card) | `VERIFIED (HF API HTTP 200)` / GitHub `h94`: `UNVERIFIED` |
| **InstantID (`control_instant_id_sdxl`, `ip-adapter.bin`)** *(optional)* | [`InstantX/InstantID`](https://huggingface.co/InstantX/InstantID) | HF API: **`apache-2.0`** *(adapter weights; depends on InsightFace `antelopev2` = Non-commercial)* | Relies on `antelopev2` embeddings; non-commercial end-to-end | `VERIFIED (HF API HTTP 200)` |
| **SDXL-Lightning 8-Step LoRA** *(optional)* | [`ByteDance/SDXL-Lightning`](https://huggingface.co/ByteDance/SDXL-Lightning) | HF API: **`openrail++`** *(corrected from prior `Apache-2.0` note)* | Subject to OpenRAIL++ use-based restrictions | `VERIFIED (HF API HTTP 200)` |
| **InsightFace `inswapper_128.onnx` & `buffalo_l` (`antelopev2`)** | [`deepinsight/insightface`](https://github.com/deepinsight/insightface) | GitHub API: **`spdx=None`** *(README restricts pretrained weights/models to non-commercial research)* | **NON-COMMERCIAL RESEARCH ONLY** without separate InsightFace license | **`UNVERIFIED` SPDX** *(README restriction)* |
| **CodeFormer (`codeformer-v0.1.0.pth`)** | [`sczhou/CodeFormer`](https://github.com/sczhou/CodeFormer) | GitHub API: **`NOASSERTION`** (`S-Lab License 1.0` in repo) | **NON-COMMERCIAL / academic research only** | `VERIFIED (S-Lab custom license)` |

---

## 2. Verifier, Segmentation, Detection & Application Stack

| Component / Library | Upstream Source | Checked Metadata / License (`2026-11-12`) | Key Legal / Redistribution Notes | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stable Diffusion WebUI Forge** *(local REST runtime)* | [`lllyasviel/stable-diffusion-webui-forge`](https://github.com/lllyasviel/stable-diffusion-webui-forge) | GitHub API: **`AGPL-3.0`** *(GNU Affero GPL v3.0 — corrected from prior `GPL-3.0` note)* | **Strong network copyleft (`AGPL-3.0`)**: hosting/bundling Forge triggers AGPL-3.0 source disclosure obligations | `VERIFIED (GitHub API HTTP 200)` |
| **Ultralytics YOLOv8n (`yolov8n.pt`)** | [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics) | GitHub API: **`AGPL-3.0`** *(or commercial Ultralytics license)* | Hosting public/commercial web services calling `ultralytics` requires AGPL-3.0 compliance or enterprise license | `VERIFIED (GitHub API HTTP 200)` |
| **SegFormer Clothes (`mattmdjaga/segformer_b2_clothes`)** | [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) | HF API: **`other`** *(ATR dataset / fine-tune without SPDX card tag)* | Non-standardized card metadata | **`UNVERIFIED`** *(custom/unset card license)* |
| **DINOv2 Base (`facebook/dinov2-base`)** | [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | HF API: **`apache-2.0`** | Permissive with attribution | `VERIFIED (HF API HTTP 200)` |
| **CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)** | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) / [`openai/CLIP`](https://github.com/openai/CLIP) | HF API: **`None` (`UNVERIFIED` on HF card)**; GitHub `openai/CLIP` API: **`MIT`** | Code/weights under MIT in `openai/CLIP`; HF model card omits license field | HF: **`UNVERIFIED`** / GitHub: `MIT` |
| **BLIP Image Captioning Base** *(optional reviewer)* | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | HF API: **`bsd-3-clause`** | Permissive with copyright notice | `VERIFIED (HF API HTTP 200)` |
| **DeepFace + Backbones (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`, `RetinaFace`, `MTCNN`)** | `serengil/deepface`, `opencv/opencv_zoo`, `HamadYA/GhostFaceNets` | GitHub API: `deepface` **`MIT`**, `opencv_zoo` **`Apache-2.0`**, `GhostFaceNets` **`MIT`** | Wrapper libraries permissive; upstream face datasets/ArcFace weights retain academic research terms | `VERIFIED (GitHub API HTTP 200)` |
| **PyTorch (`2.3.1`), Transformers, OpenCV, Streamlit, NumPy, SciPy** | PyPI / Conda | Standard OSI (`BSD-3-Clause` / `Apache-2.0` / `MIT`) | Standard Python scientific stack attribution | Package metadata |

---

## 3. Summary Checklist for Contributors

1. **Academic / Research Use (Coursework, Reproducible Benchmarks):** Compatible for local academic research when respecting upstream non-commercial model notices (`InsightFace`, `CodeFormer`, dataset terms) and keeping weights used or required by the pipeline external (fetched separately, never redistributed or bundled).
2. **Public Fork / Redistribution:** **Never** mirror or commit any third-party weight binaries (`.safetensors`, `.onnx`, `.pth`, `.h5`, `.pt`, `.bin`) into Git or GitHub Releases. Link upstream sources via [`04_MODELS_AND_DOWNLOADS.md`](04_MODELS_AND_DOWNLOADS.md).
3. **Commercial Deployment:** **Incompatible out of the box** — `InsightFace` (`inswapper_128`, `buffalo_l`) and `CodeFormer` (`S-Lab 1.0`) restrict to non-commercial research, `YOLOv8n` and `WebUI Forge` carry `AGPL-3.0`, `Juggernaut-XL` / `SDXL-Lightning` carry Open RAIL terms, and MMCCP's own repository has no commercial OSI `LICENSE`.
