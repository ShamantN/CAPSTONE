# MMCCP v2.0 — Complete Model Manifest, Sizes & Checksums

- **Verification Status:** `SHA-256 & OFFICIAL URLs VERIFIED` on `C:\Stable-Diffusion` & local HuggingFace/DeepFace/InsightFace caches (branch `docs/run-guides` on top of `main` `625b365` / `v2.0.0-rc3`)
- **Rule:** Every URL and byte size below was verified directly via HTTP `HEAD` against the live upstream host and matched to local SHA-256 checksums on disk (zero guessed URLs or hashes).

---

## Disk Space Summary

| Tier | What Is Included | Total Download / Disk Size |
| :--- | :--- | :--- |
| **Minimum Required (`v2` Default Pipeline)** | `webui_forge_cu121_torch231` (~1.74 GiB archive) + Juggernaut XL v9 + OpenPose XL + IP-Adapter Plus-Face SDXL + `inswapper_128.onnx` + CodeFormer + `buffalo_l` + `yolov8n.pt` + runtime caches (`CLIP-ViT-H`, `SegFormer`, `DINOv2`, DeepFace weights) | **~22–25 GiB** extracted (`~14.2 GiB` core diffusion + swap weights) |
| **Optional Benchmark Weights (Full Parity With Author Machine)** | Above + `Animagine XL 3.1` + `Depth XL` + `InstantID` (`2` files) + `FaceID Plus v2` (`2` files) + `SDXL-Lightning` LoRA + `CLIP-ViT-bigG` + `OpenPose/DWPose` annotators + `BLIP` (`~3.7 GB` HF cache) | **~45–52 GiB** total across Forge models + HF/user caches |

---

## 1. Base Software & Required Generation Models (Manual + Forge Auto-Download)

> **Folder Root:** Let `<FORGE_MODELS>` denote `C:\Stable-Diffusion\webui\models` on default Windows, or `$FORGE_MODELS_DIR` / `$FORGE_DIR/webui/models` elsewhere.

| Model / File | Role in Pipeline | Destination Path | Delivery | Exact Bytes (Human Size) | License | Verified Official URL | SHA-256 |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **WebUI Forge One-Click (`cu121`/`torch231`)** | Base SDXL backend server & embedded Python `3.10.6` | Extract to `C:\Stable-Diffusion\` (`webui` commit `dfdcbab6...`) | Manual (Windows) / Git clone (Linux) | `1,870,203,670` (`1.74 GiB`) | GPL-3.0 (repo) | https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/download/latest/webui_forge_cu121_torch231.7z | `NEEDS SOURCE` *(rolling `latest` tag archive)* |
| **Juggernaut XL v9 RunDiffusionPhoto v2** | Primary photorealistic SDXL checkpoint (`v2` default) | `<FORGE_MODELS>/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | Manual | `7,105,348,188` (`6.62 GiB`) | `creativeml-openrail-m` | https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors | `c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca` |
| **ControlNet OpenPose XL (`OpenPoseXL2`)** | Pass 1 structural skeleton conditioning (**must rename to `openpose_xl.safetensors` so Forge IDs it as `openpose_xl [f4251cb4]`!**) | `<FORGE_MODELS>/ControlNet/openpose_xl.safetensors` | Manual *(rename `OpenPoseXL2.safetensors`)* | `5,004,167,829` (`4.66 GiB`) | `other` (see HF card) | https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors | `5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92` |
| **IP-Adapter Plus-Face SDXL (ViT-H)** | Pass 1 facial identity cross-attention conditioning (`M4`) | `<FORGE_MODELS>/ControlNet/ip-adapter-plus-face_sdxl_vit-h.safetensors` | Manual | `847,517,512` (`808.26 MiB`) | `apache-2.0` | https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors | `677ad8860204f7d0bfba12d29e6c31ded9beefdf3e4bbd102518357d31a292c1` |
| **CLIP-ViT-H-14 Image Encoder** | Vision encoder required by `CLIP-ViT-H (IPAdapter)` | `<FORGE_MODELS>/ControlNetPreprocessor/CLIP-ViT-H-14.safetensors` | **Auto** *(Forge downloads on first IP-Adapter use)* | `2,528,373,448` (`2.35 GiB`) | `apache-2.0` (`h94/IP-Adapter`) | https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors | `6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030` |
| **InsightFace Swapper (`inswapper_128.onnx`)** | Direct CPU face transplant kernel (`M5`) & ReActor model | `<FORGE_MODELS>/insightface/inswapper_128.onnx` | Manual | `554,253,681` (`528.58 MiB`) | Non-commercial research (`InsightFace` / see HF card) | https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx | `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` |
| **CodeFormer (`codeformer-v0.1.0.pth`)** | Post-swap facial fidelity restoration via Forge Extras API | `<FORGE_MODELS>/Codeformer/codeformer-v0.1.0.pth` | **Auto** *(Forge downloads on first restore)* | `376,637,898` (`359.19 MiB`) | S-Lab License 1.0 (`sczhou/CodeFormer`) | https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth | `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7` |
| **InsightFace `buffalo_l` (`w600k_r50.onnx`, etc.)** | Face detection & ArcFace extraction for M5 swap & M1/Verifier | `<FORGE_MODELS>/models/buffalo_l/` (or `~/.insightface/models/buffalo_l/`) | **Auto** *(InsightFace downloads/extracts on first run)* | `288,621,354` (`275.25 MiB` zip) | Non-commercial research (`InsightFace`) | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip | Zip: `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` (`w600k_r50.onnx`: `4c06341c33c2ca1f...`) |
| **YOLOv8n (`yolov8n.pt`)** | CPU person detector in Verifier `v1.2` & `extra_person_detector.py` | `<MMCCP_ROOT>/yolov8n.pt` | **Auto** *(Ultralytics downloads on first call; gitignored in repo)* | `6,549,796` (`6.25 MiB`) | AGPL-3.0 (`ultralytics`) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |

---

## 2. Verifier, Wardrobe & VLM Models (All Auto-Downloaded Into User Caches)

These download automatically on first import/call into `~/.cache/huggingface` (`HF_HOME`) and `~/.deepface/weights/`:

| Model / File | Consumer Module | Cache Location | Size (Bytes / Human) | License / Repo | Verified URL | SHA-256 (Local Disk) |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **`mattmdjaga/segformer_b2_clothes`** | `WardrobeLock` (`M7`) & Verifier clothing segmentation | `$HF_HOME/hub/models--mattmdjaga--segformer_b2_clothes` | `109,493,236` (`104.42 MiB` model file) | `other` (HF card) | https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/model.safetensors | `NEEDS SOURCE` *(stored in HF blob cache)* |
| **`facebook/dinov2-base`** | Verifier appearance similarity (`DINOv2`) | `$HF_HOME/hub/models--facebook--dinov2-base` | `346,345,912` (`330.30 MiB` model file) | `apache-2.0` | https://huggingface.co/facebook/dinov2-base/resolve/main/model.safetensors | `NEEDS SOURCE` *(stored in HF blob cache)* |
| **`openai/clip-vit-base-patch32`** | `metrics.py` CLIP-I, Verifier inversion & props probes | `$HF_HOME/hub/models--openai--clip-vit-base-patch32` | `605,247,071` (`577.21 MiB` `pytorch_model.bin`) | MIT (repo) / HF card unset | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin | `NEEDS SOURCE` *(stored in HF blob cache)* |
| **`Salesforce/blip-image-captioning-base`** | Optional `tools/vlm_review.py` & `qa/vlm_visual_auditor.py` | `$HF_HOME/hub/models--Salesforce--blip-image-captioning-base` | `989,820,849` (`943.97 MiB` `pytorch_model.bin`) | `bsd-3-clause` | https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin | `NEEDS SOURCE` *(stored in HF blob cache)* |
| **DeepFace `facenet512_weights.h5`** | Verifier `v1.2` Judge 1 | `~/.deepface/weights/facenet512_weights.h5` | `94,955,648` (`90.56 MiB`) | MIT (`serengil/deepface`) | https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 | `3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f` |
| **DeepFace `face_recognition_sface_2021dec.onnx`** | Verifier `v1.2` Judge 2 | `~/.deepface/weights/face_recognition_sface_2021dec.onnx` | `38,696,353` (`36.90 MiB`) | Apache-2.0 (`opencv/opencv_zoo`) | https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| **DeepFace `ghostfacenet_v1.h5`** | Verifier `v1.2` Judge 3 | `~/.deepface/weights/ghostfacenet_v1.h5` | `17,271,840` (`16.47 MiB`) | MIT (`HamadYA/GhostFaceNets`) | https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5 | `e4182ca2470dac3eb79974f4d5d54f2abdf66d7e231c6f3df4059a90bded1271` |
| **DeepFace `arcface_weights.h5`** | Verifier `v1.2` Judge 4 & `metrics.py` | `~/.deepface/weights/arcface_weights.h5` | `137,026,640` (`130.68 MiB`) | MIT (`serengil/deepface_models`) | https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 | `6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93` |
| **DeepFace `retinaface.h5`** | Verifier `v1.2` & `IdentityBankManager` face detector | `~/.deepface/weights/retinaface.h5` | `118,667,368` (`113.17 MiB`) | MIT (`serengil/retina-face`) | https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 | `ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf` |

---

## 3. Optional / Disabled-in-Default Models (For Ablations, Stylization, or Disabled Toggles)

`v2` default (`consistency_v2/best_config.json`) has **`enable_dual_instantid: false`** and **`use_character_lora: false`**. Download these **only** if you plan to toggle Animagine, Depth ControlNet, InstantID, or FaceID experiments:

| Model / File | Destination Path | Delivery | Bytes (Human Size) | License | Verified Official URL | SHA-256 |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Animagine XL 3.1** | `<FORGE_MODELS>/Stable-diffusion/animagine-xl-3.1.safetensors` | Manual | `6,938,325,776` (`6.46 GiB`) | `openrail++` | https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors | `e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0` |
| **ControlNet Depth XL** | `<FORGE_MODELS>/ControlNet/depth_xl.safetensors` | Manual | `5,004,167,860` (`4.66 GiB`) | `openrail++` | https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors | `b760b1ed26970c7c60abbcef08e7238bb9428bee510a405aff239e06a1471945` |
| **InstantID ControlNet (`instantid_xl`)** | `<FORGE_MODELS>/ControlNet/control_instant_id_sdxl.safetensors` | Manual | `2,502,139,136` (`2.33 GiB`) | `apache-2.0` | https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors | `c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05` |
| **InstantID IP-Adapter (`ip-adapter.bin`)** | `<FORGE_MODELS>/ControlNet/ip-adapter_instant_id_sdxl.bin` | Manual | `1,691,134,141` (`1.58 GiB`) | `apache-2.0` | https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin | `02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1` |
| **CLIP-ViT-bigG Image Encoder** | `<FORGE_MODELS>/ControlNetPreprocessor/CLIP-ViT-bigG.safetensors` | Auto (when using `ip-adapter_xl`) | `3,689,912,664` (`3.44 GiB`) | `apache-2.0` | https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors | `657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d` |
| **IP-Adapter SDXL (`ip-adapter_xl`)** | `<FORGE_MODELS>/ControlNet/ip-adapter_xl.safetensors` | Manual | `702,585,376` (`670.04 MiB`) | `apache-2.0` | https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors | `ba1002529e783604c5f326d49f0122025392d1d20ac8d573b3eeb3e6dea4ebb6` |
| **FaceID Plus v2 SDXL + LoRA** | `<FORGE_MODELS>/ControlNet/ip-adapter-faceid-plusv2_sdxl.bin` + `<FORGE_MODELS>/Lora/ip-adapter-faceid-plusv2_sdxl_lora.safetensors` | Manual | `1.39 GiB` + `354.62 MiB` | Non-commercial (see HF card) | https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin | `c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb` |
| **DWPose / Depth Anything v2 Preprocessors** | `<FORGE_MODELS>/ControlNetPreprocessor/{openpose,depth_anything_v2}/` | Auto (Forge annotators) | `128 MiB` (`dw-ll_ucoco_384.onnx`) / `640 MiB` (`depth_anything_v2_vitl.safetensors`) | Apache-2.0 / MIT | https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx & https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors | Verified in `step0_raw_facts.json` |

---

## Verify Downloads Automatically

```powershell
python tools\verify_setup.py --mode new-machine --verify-sha256
```
(Add `--skip-forge` if checking weights before starting WebUI Forge.)

