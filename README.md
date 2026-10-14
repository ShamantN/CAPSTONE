# 🎬 Multi-Modal Character Consistency Pipeline (MMCCP)

> 🚀 **MMCCP v2.0 Released:** See [README_V2.md](docs_generated/v2/README_V2.md) for full documentation on Methods M1–M10, frozen verifier benchmarks, and one-command execution instructions.

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
* Python 3.9+
* SD WebUI Forge (Running locally on port 7860 with `--api` enabled)
* Required SD Extensions: ControlNet (IP-Adapter XL), ReActor

### Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShamantN/CAPSTONE.git
   cd CAPSTONE
   ```
2. **Install dependencies:**
   ```bash
   pip install streamlit google-generativeai pillow requests opencv-python deepface tf-keras numpy
   ```
3. **Run the Forge API:**
   Ensure your Stable Diffusion Forge instance is running with the `--api` flag enabled on `http://127.0.0.1:7860`.
4. **Launch the Streamlit Interface:**
   ```bash
   streamlit run cap.py
   ```

---

## 📊 Application Flow
1. **Input:** Paste a raw novel chapter or story into the UI.
2. **LLM Extraction:** Gemini extracts structured JSON scenes, characters, and environments.
3. **Pipeline Execution:** The pipeline iterates through scenes, applying the 6 pillars (Tokenization, DNA caching, Two-Pass generation, Routing, and ArcFace validation).
4. **Output:** A fully rendered cinematic storyboard with calculated metric scores attached to each frame.

> Developed as an advanced generative AI capstone project focusing on latent space control, microservices orchestration, and consistency matrix routing.
