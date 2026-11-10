# 🎬 Multi-Modal Character Consistency Pipeline (MMCCP)

> **MMCCP v2.0 (Reconciled `v2.0.0-rc3` on `main`):**
> - **Technical Scorecard & Honest Limitations:** [`README_V2.md`](README_V2.md)
> - **Run From Scratch Guides (Windows / Linux / Colab):** [`docs/run/00_START_HERE.md`](docs/run/00_START_HERE.md)
> - **Google Colab Notebook:** [`notebooks/MMCCP_Colab.ipynb`](notebooks/MMCCP_Colab.ipynb) *(status: `UNTESTED` live in Colab; see checklist in [`docs/run/03_GOOGLE_COLAB.md`](docs/run/03_GOOGLE_COLAB.md))*
> - **Preflight Verification Tool:** `python tools/verify_setup.py`

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)
![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-XL-5C5CFF?style=for-the-badge&logo=ai)
![ControlNet](https://img.shields.io/badge/ControlNet-SDXL-brightgreen?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8?style=for-the-badge&logo=opencv)

A highly advanced character consistency architecture designed to solve identity bleeding, prompt mixing, and temporal inconsistency in generative AI storyboarding. By orchestrating LLMs (Google Gemini) and Latent Diffusion (Stable Diffusion XL via Forge), MMCCP generates multi-character cinematic storyboards with absolute visual consistency.

---

## 🧠 Architecture Overview: The 6 Pillars

The core innovation of the MMCCP lies in its surgical manipulation of latent space conditioning and text prompt formulation. We solve fundamental flaws in current AI diffusion models through a rigid, six-pillar architectural pipeline.

### 1. Identity Initialization (Solving Concept Bleeding)
* **The Flaw:** Stable Diffusion possesses inherent latent bias for real names (e.g., "Emma" biases towards Emma Watson).
* **The Solution:** We hash character names into unique, mathematically blank alphanumeric tokens (e.g., `YEbright_A1B2C`). This guarantees a zero-shot neural pathway, forcing the AI to rely entirely on explicit physical descriptors rather than latent pre-training biases.

### 2. Visual DNA Caching (Solving Identity Drift)
* **The Flaw:** Text descriptors are not rigid enough to hold facial geometry consistently across multiple frames.
* **The Solution:** The system generates a high-quality, high-step anchor portrait for the character. This image is encoded to Base64 and permanently cached in our JSON database as the character's immutable "Visual DNA" for all subsequent scenes.

### 3. Linguistic Isolation (Solving Prompt Bleeding)
* **The Flaw:** In multi-character scenes, attributes like clothing colors mix and bleed across characters in the cross-attention layers.
* **The Solution:** We dynamically inject spatial coordinates (e.g., "sitting on the left") and utilize the absolute `BREAK` keyword. This forces Stable Diffusion to process characters in mathematically isolated attention layers, preventing attribute contamination.

### 4. Two-Pass Render / Structural Tracing (Solving Action Override)
* **The Flaw:** Feeding a static character reference directly into an Image-to-Image prompt destroys the dynamic action and pose requested by the text.
* **The Solution (Structural Tracing):**
    * **Pass 1:** Text-to-Image acts as a structural sketch, generating the dynamic physics and pose (the draft).
    * **Pass 2:** Img2Img traces this draft using a strict `denoising_strength` of 0.55. This locks the geometric pose while ControlNet IP-Adapter acts as a digital eyedropper, painting the cached clothing/textures over the pre-established bodies.

### 5. Multi-Face Routing (Solving the Clone Glitch)
* **The Flaw:** Global face-swapper extensions blindly map a single reference face to all detected bodies in a scene, creating identical clones.
* **The Solution:** We execute a left-to-right matrix routing strategy. Using PIL, we composite Anchor Portraits side-by-side into a single "Sticker Sheet." By synchronizing the spatial prompt (left/right) with the sticker sheet (left/right), we pass a rigid target payload (e.g., `"0,1"`) to the ReActor extension, guaranteeing Face 0 maps exclusively to Body 0.

### 6. The Max-Match Matrix (Consistency Metrics)
* **The Flaw:** Without quantitative measurement, "consistency" is subjective.
* **The Solution:** The script uses DeepFace to extract faces from the generated scene, converts them into 512-dimensional vectors using the ArcFace neural network, and calculates the Cosine Similarity angle against the Anchor DNA. A one-to-many "Max-Match" array evaluates if the target face exceeds our stringent success threshold (Cosine Similarity > 0.85).

---

## ⚙️ Installation & Setup

### Prerequisites
### Prerequisites (v2.0)
* **Python 3.10.6** (`py -3.10` on Windows) + NVIDIA GPU (`>= 8 GB` VRAM recommended)
* **Stable Diffusion WebUI Forge** running on `--api --xformers` (`http://127.0.0.1:7860`)
* Required Extensions: `adetailer`, `sd-forge-couple`, `sd-webui-controlnet`, `sd-webui-reactor`
* See [`docs/run/04_MODELS_AND_DOWNLOADS.md`](docs/run/04_MODELS_AND_DOWNLOADS.md) for verified URLs, destinations, and SHA-256 checksums.

### Quickstart (See [`docs/run/01_NEW_MACHINE_WINDOWS.md`](docs/run/01_NEW_MACHINE_WINDOWS.md) for Full Walkthrough)
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/ShamantN/CAPSTONE.git
   cd CAPSTONE
   ```
2. **Create a Python 3.10 virtual environment & install CUDA PyTorch + requirements:**
   ```powershell
   py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
   pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. **Hydrate canonical banks, verify setup & run preflight (`--mode new-machine`):**
   ```powershell
   python tools\verify_setup.py --mode new-machine --fix-hints
   python tools\sync_canonical_banks.py
   ```
4. **Render from zero cache or launch the 4-Tab Streamlit Studio UI:**
   ```powershell
   python run_from_scratch.py --pipeline v2 --mode fast --story storyboards\ebright.json --scene 1 --seed 11
   python -m streamlit run app.py
   ```

---

## 📊 Application Flow
1. **Input:** Paste a raw novel chapter or story into the UI.
2. **LLM Extraction:** Gemini extracts structured JSON scenes, characters, and environments.
3. **Pipeline Execution:** The pipeline iterates through scenes, applying the 6 pillars (Tokenization, DNA caching, Two-Pass generation, Routing, and ArcFace validation).
4. **Output:** A fully rendered cinematic storyboard with calculated metric scores attached to each frame.

> Developed as an advanced generative AI capstone project focusing on latent space control, microservices orchestration, and consistency matrix routing.
