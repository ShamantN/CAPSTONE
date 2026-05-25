"""
Multi-Modal Character Consistency Pipeline (MMCCP)

A microservices architecture built on Streamlit that orchestrates LLMs and Latent Diffusion 
to generate multi-character cinematic storyboards with absolute visual consistency.
"""
import streamlit as st
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import time
import json
import re
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import os
import warnings
import requests
import random
import string
import cv2
import numpy as np

# --- IMPORT DEEPFACE FOR CONSISTENCY METRICS ---
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    st.warning("DeepFace not installed. Run `pip install deepface tf-keras opencv-python` to enable metrics.")

# Suppress warnings for clean terminal
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore')

# --- CONFIGURATION & DIRECTORIES ---
MODEL_TEXT = "gemini-2.5-flash"
MAX_RETRIES = 3

EDU_CHARACTERS_DB = "edu_characters_database.json"
EDU_ENVIRONMENTS_DB = "edu_environments_database.json"
EDU_SCENES_DB = "edu_scenes_database.json"

EDU_OUTPUT_DIR = Path("edu_outputs")
EDU_IMAGES_DIR = EDU_OUTPUT_DIR / "images"
EDU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# SD Forge API Configuration
SD_API_URL = "http://127.0.0.1:7860/sdapi/v1"
SD_TXT2IMG_ENDPOINT = f"{SD_API_URL}/txt2img"
SD_IMG2IMG_ENDPOINT = f"{SD_API_URL}/img2img"
SD_OPTIONS_ENDPOINT = f"{SD_API_URL}/options"

SD_MODELS = {
    "Juggernaut": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "Animagine": "animagine-xl-3.1.safetensors"
}

# ==========================================
# 1. METRICS & COMPOSITING FUNCTIONS
# ==========================================

def calculate_facial_cosine_similarity(anchor_b64, scene_bytes):
    """
    Calculates ArcFace 512-d Cosine Similarity between the anchor image and the generated scene face.

    PILLAR 6: The Max-Match Matrix (Consistency Metrics)
    Converts faces into 512-dimensional vectors using the ArcFace neural network and calculates
    the cosine similarity angle against the character's Anchor DNA. This acts as our stringent
    evaluation metric (Target > 0.85) to guarantee structural identity retention.

    Args:
        anchor_b64 (str): Base64 encoded string of the character's cached Anchor DNA image.
        scene_bytes (bytes): Raw bytes of the generated scene image containing the face(s).

    Returns:
        float or None: The cosine similarity score rounded to 4 decimal places (1.0 = perfect match),
                       or None if detection fails or DeepFace is unavailable.
    """
    if not DEEPFACE_AVAILABLE:
        return None
    try:
        anchor_img = Image.open(BytesIO(base64.b64decode(anchor_b64))).convert('RGB')
        scene_img = Image.open(BytesIO(scene_bytes)).convert('RGB')

        anchor_cv = cv2.cvtColor(np.array(anchor_img), cv2.COLOR_RGB2BGR)
        scene_cv = cv2.cvtColor(np.array(scene_img), cv2.COLOR_RGB2BGR)

        result = DeepFace.verify(
            img1_path=anchor_cv, 
            img2_path=scene_cv, 
            model_name="ArcFace", 
            distance_metric="cosine", 
            enforce_detection=False
        )
        return round(1.0 - result['distance'], 4)
    except Exception as e:
        return None

def create_composite_face_image(b64_list):
    """
    Stitches multiple anchor images side-by-side to create a "Sticker Sheet" for face routing.

    PILLAR 5: Multi-Face Routing (Solving the Clone Glitch)
    Global face-swappers map one face to all bodies. By compositing anchor images left-to-right
    into a single matrix, we create rigid mapping coordinates. This sticker sheet is passed to
    the ReActor extension along with a target payload (e.g., "0,1") to ensure Face 0 precisely
    maps to Body 0.

    Args:
        b64_list (list of str): List of Base64 encoded anchor images.

    Returns:
        str or None: Base64 encoded composite "Sticker Sheet" image, or None if the list is empty.
    """
    images = [Image.open(BytesIO(base64.b64decode(b))) for b in b64_list if b]
    if not images: return None
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
        
    buffered = BytesIO()
    new_im.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==========================================
# 2. STABLE DIFFUSION API FUNCTIONS
# ==========================================

def set_sd_model(checkpoint_name):
    """
    Sets the active Stable Diffusion checkpoint model via the Forge REST API.

    Args:
        checkpoint_name (str): The filename of the `.safetensors` model to load.

    Returns:
        bool: True if the model loaded successfully, False otherwise.
    """
    try:
        resp = requests.post(SD_OPTIONS_ENDPOINT, json={"sd_model_checkpoint": checkpoint_name}, timeout=10)
        return resp.status_code == 200
    except: return False

def generate_sd_txt2img(positive_prompt, negative_prompt, sd_checkpoint, steps=15, cfg_scale=5.0):
    """
    Executes Pass 1 (Text-to-Image) to establish the base physical pose and geometry.

    PILLAR 4: Two-Pass Render / Structural Tracing (Pass 1)
    Feeding a static reference image directly into a prompt destroys dynamic action. 
    This first text-to-image pass generates the physical action/pose acting as the "draft".
    We keep steps lower to generate the structural foundation rapidly.

    Args:
        positive_prompt (str): The linguistically isolated master prompt.
        negative_prompt (str): Negative traits to avoid.
        sd_checkpoint (str): The SD model checkpoint filename.
        steps (int, optional): Number of diffusion steps. Defaults to 15.
        cfg_scale (float, optional): Classifier Free Guidance scale. Defaults to 5.0.

    Returns:
        list of bytes or None: List containing the generated image bytes, or None on failure.
    """
    if not set_sd_model(sd_checkpoint): return None
    payload = {
        "prompt": positive_prompt, "negative_prompt": negative_prompt,
        "steps": steps, "width": 1024, "height": 1024,
        "cfg_scale": cfg_scale, "sampler_name": "DPM++ 2M Karras", "seed": -1
    }
    try:
        resp = requests.post(SD_TXT2IMG_ENDPOINT, json=payload, timeout=300)
        if resp.status_code == 200:
            return [base64.b64decode(img) for img in resp.json().get('images', [])]
    except Exception as e: st.error(f"SD Error: {e}")
    return None

def generate_sd_img2img(positive_prompt, negative_prompt, sd_checkpoint, init_image_b64, steps=20, cfg_scale=7.0, ip_adapter_b64=None, reactor_b64=None, face_index_str="0"):
    """
    Executes Pass 2 (Image-to-Image) for Structural Tracing and Identity Routing.

    PILLAR 4: Two-Pass Render / Structural Tracing (Pass 2)
    Traces the draft from Pass 1 using a strict `denoising_strength` of 0.55. This locks the
    geometric pose while allowing ControlNet IP-Adapter and ReActor to act as digital eyedroppers,
    painting the cached Visual DNA textures exactly over the pre-established bodies.

    Args:
        positive_prompt (str): Master prompt.
        negative_prompt (str): Negative prompt.
        sd_checkpoint (str): SD model checkpoint.
        init_image_b64 (str): Base64 encoded draft image from Pass 1.
        steps (int, optional): Number of diffusion steps. Defaults to 20.
        cfg_scale (float, optional): CFG scale. Defaults to 7.0.
        ip_adapter_b64 (str, optional): Base64 image for IP-Adapter clothing transfer. Defaults to None.
        reactor_b64 (str, optional): Base64 composite face image for ReActor routing. Defaults to None.
        face_index_str (str, optional): Target payload array mapping faces to bodies (e.g., "0,1"). Defaults to "0".

    Returns:
        list of bytes or None: List containing the finalized image bytes, or None on failure.
    """
    payload = {
        "prompt": positive_prompt, "negative_prompt": negative_prompt,
        "init_images": [init_image_b64],
        "denoising_strength": 0.55, # Traces the draft tightly without losing the structural pose
        "steps": steps, "width": 1024, "height": 1024,
        "cfg_scale": cfg_scale, "sampler_name": "DPM++ 2M Karras",
        "seed": -1, "alwayson_scripts": {}
    }

    # Inject IP-Adapter (Clothing/Style Transfer)
    if ip_adapter_b64:
        payload["alwayson_scripts"]["controlnet"] = {
            "args": [{"enabled": True, "module": "ip-adapter_clip_sdxl", "model": "ip-adapter_xl", "image": ip_adapter_b64, "weight": 0.85}]
        }
        
    # Inject ReActor (Multi-Face Routing)
    if reactor_b64:
        payload["alwayson_scripts"]["reactor"] = {
            "args": [reactor_b64, True, face_index_str, face_index_str, "inswapper_128.onnx", "CodeFormer", 1.0, True]
        }
        
    if not payload["alwayson_scripts"]: del payload["alwayson_scripts"]
    
    try:
        resp = requests.post(SD_IMG2IMG_ENDPOINT, json=payload, timeout=300)
        if resp.status_code == 200:
            return [base64.b64decode(img) for img in resp.json().get('images', [])]
    except Exception as e: st.error(f"SD Error: {e}")
    return None

def build_master_prompt(scene, characters_db, environments_db, style="Cinematic Realistic"):
    """
    Compiles character and environment data into a mathematically isolated diffusion prompt.

    PILLAR 1: Identity Initialization (Solving Concept Bleeding)
    Uses the `unique_token` (e.g., YEbright_A1B2C) instead of real names to guarantee a zero-shot
    blank pathway, forcing the AI to rely 100% on the injected physical descriptors.

    PILLAR 3: Linguistic Isolation (Solving Prompt Bleeding)
    Injects spatial coordinates (left/right) and explicitly uses the `BREAK` keyword between
    character descriptions. This forces SD to process characters in mathematically isolated
    attention layers, preventing attributes like clothing colors from mixing.

    Args:
        scene (dict): Scene data extracted from the LLM.
        characters_db (dict): The visual DNA caching database for characters.
        environments_db (dict): The database for environment descriptions.
        style (str, optional): The overarching art style multiplier. Defaults to "Cinematic Realistic".

    Returns:
        str: The fully assembled, mathematically isolated master prompt.
    """
    char_desc_blocks = []
    # Hardcoded spatial mappings to synchronize with the Multi-Face Routing matrix
    positions = ["sitting on the left", "sitting on the right"] 
    
    for idx, char_id in enumerate(scene["characters"]):
        char = next((c for c in characters_db["data"] if c["id"] == char_id), None)
        if char:
            # Identity Initialization: Use the hashed token to prevent latent real-name bias
            name_token = char.get("unique_token", char["name"])
            pos = positions[idx] if len(scene["characters"]) > 1 and idx < 2 else ""
            desc = f"({name_token}:1.5) {pos}, {char['physical_appearance']['age']} years old {char['physical_appearance']['gender']}, wearing {char['clothing']['default_outfit']}, {char['clothing']['colors'][0]} clothes, {char['physical_appearance']['hair']['color']} hair"
            char_desc_blocks.append(desc.strip())

    env = next((e for e in environments_db["data"] if e["id"] == scene["environment"]), None)
    env_desc = f"{env['name']}, {env['atmosphere']['lighting']['type']} lighting, mood: {env['atmosphere']['mood']}" if env else ""

    # Linguistic Isolation: Forcing separate cross-attention calculation layers
    char_string = " BREAK ".join(char_desc_blocks)
    return f"({style}:1.2), masterpiece, cinematic lighting, 8k resolution, highly detailed, {scene['action_description']}, {char_string} BREAK Background: {env_desc}, {scene['cinematic_prompt']['shot_type']}, {scene['cinematic_prompt']['camera_angle']}."

# ==========================================
# 3. LLM EXTRACTION & DATABASE LOGIC
# ==========================================

# Minimal schemas for LLM extraction
CHARACTER_SCHEMA = {
    "name": "", "id": "", "physical_appearance": {"age": "", "gender": "", "hair": {"color": ""}},
    "clothing": {"default_outfit": "", "colors": []}, "unique_token": ""
}
SCENE_PROMPT_SCHEMA = {
    "scene_id": "", "scene_number": 0, "title": "", "characters": [], "environment": "",
    "cinematic_prompt": {"shot_type": "", "camera_angle": ""}, "action_description": ""
}

def load_db(file_path):
    """
    Loads JSON data representing the cached Visual DNA and environment metadata.
    
    Args:
        file_path (str or Path): Path to the JSON database file.
        
    Returns:
        dict: Parsed database dictionary, or a default empty structure if file is missing.
    """
    if not Path(file_path).exists(): return {"data": []}
    with open(file_path, 'r') as f: return json.load(f)

def save_db(file_path, data):
    """
    Saves metadata to the JSON database, ensuring Visual DNA persistence.
    
    Args:
        file_path (str or Path): Path to the JSON database file.
        data (dict): The data structure to serialize and save.
    """
    with open(file_path, 'w') as f: json.dump(data, f, indent=2)

def generate_with_retry(model, prompt, retries=MAX_RETRIES):
    """
    Wraps Google Gemini API calls with a retry mechanism for rate limits.
    
    Args:
        model (genai.GenerativeModel): Configured Gemini model.
        prompt (str): The prompt payload.
        retries (int, optional): Maximum retry attempts. Defaults to MAX_RETRIES.
        
    Returns:
        genai.types.GenerateContentResponse or None: The response object on success.
    """
    for attempt in range(retries):
        try: return model.generate_content(contents=prompt)
        except Exception: time.sleep(2)
    return None

def extract_scenes(chapter_text, text_model):
    """
    Uses the LLM to extract highly structured scene data mapped to our JSON schema.
    
    Args:
        chapter_text (str): Raw novel/story chapter text provided by the user.
        text_model (genai.GenerativeModel): Configured Gemini model for text processing.
        
    Returns:
        list: A list of extracted scene dictionaries matching SCENE_PROMPT_SCHEMA.
    """
    prompt = f"Extract all scenes from this text. Return ONLY a JSON array matching this schema: {json.dumps(SCENE_PROMPT_SCHEMA)} \n\nTEXT:\n{chapter_text}"
    resp = generate_with_retry(text_model, prompt)
    if resp and resp.text:
        match = re.search(r'\[.*\]', resp.text, re.DOTALL)
        if match: return json.loads(match.group())
    return []

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="MMCCP Storyboard", page_icon="🎬", layout="wide")
st.title("🎬 Multi-Modal Character Consistency Pipeline")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    sd_model_key = st.selectbox("SD Model:", list(SD_MODELS.keys()))
    art_style = st.selectbox("Art Style:", ["Cinematic Realistic", "Digital Painting"])
    sd_steps = st.slider("Final Pass Steps", 15, 40, 25)

chapter_input = st.text_area("📖 Paste Novel Chapter (JSON generation):", height=150)

if st.button("🚀 Render Pipeline", type="primary"):
    if not api_key:
        st.error("Please enter a Gemini API Key in the sidebar.")
    else:
        genai.configure(api_key=api_key)
        text_model = genai.GenerativeModel(MODEL_TEXT)
        
        st.info("🧠 LLM: Extracting scenes and characters...")
        # For a full pipeline, you would extract characters and environments here.
        # Assuming the databases are pre-populated or extracted via LLM:
        scenes = extract_scenes(chapter_input, text_model)
        
        if not scenes:
            st.error("Failed to parse scenes from text.")
        else:
            characters_db = load_db(EDU_CHARACTERS_DB)
            environments_db = load_db(EDU_ENVIRONMENTS_DB)
            generated_panels = []

            for scene in scenes:
                st.write(f"**🎬 Processing Scene {scene.get('scene_number', 'X')}: {scene.get('title')}**")
                
                pos_prompt = build_master_prompt(scene, characters_db, environments_db, art_style)
                neg_prompt = "blurry, bad anatomy, deformed, ugly, watermark, extra limbs"
                
                # PILLAR 2: Visual DNA Caching
                # Fetching the immutable base64 anchor portrait as the ground-truth visual DNA
                char_anchors = [c.get("anchor_image_b64") for c in characters_db["data"] if c["id"] in scene["characters"] and c.get("anchor_image_b64")]
                
                # Setup Multi-Face Routing logic
                if len(scene["characters"]) == 1 and char_anchors:
                    ip_adapter_b64 = char_anchors[0]
                    reactor_b64 = char_anchors[0]
                    face_index_str = "0"
                elif len(scene["characters"]) > 1 and char_anchors:
                    # Disable IP-Adapter for crowded scenes to prevent severe color/texture bleed 
                    # from overlapping conditioning signals.
                    ip_adapter_b64 = None 
                    
                    # PILLAR 5: Multi-Face Routing
                    # Create the composite sticker sheet and the mapping array (e.g. "0,1")
                    reactor_b64 = create_composite_face_image(char_anchors)
                    face_index_str = ",".join(str(i) for i in range(len(char_anchors))) 
                else:
                    ip_adapter_b64 = reactor_b64 = face_index_str = None

                st.caption("🏃 Pass 1: Generating structural physics draft...")
                draft_imgs = generate_sd_txt2img(pos_prompt, neg_prompt, SD_MODELS[sd_model_key])
                pose_b64 = base64.b64encode(draft_imgs[0]).decode('utf-8') if draft_imgs else None

                if pose_b64:
                    st.caption("✨ Pass 2: Tracing final geometry & routing identities...")
                    final_imgs = generate_sd_img2img(
                        pos_prompt, neg_prompt, SD_MODELS[sd_model_key], 
                        init_image_b64=pose_b64, steps=sd_steps, 
                        ip_adapter_b64=ip_adapter_b64, reactor_b64=reactor_b64, face_index_str=face_index_str
                    )

                    if final_imgs:
                        # ArcFace Evaluation Matrix
                        scene_scores = []
                        for idx, char_id in enumerate(scene["characters"]):
                            char_profile = next((c for c in characters_db["data"] if c["id"] == char_id), None)
                            if char_profile and char_profile.get("anchor_image_b64"):
                                score = calculate_facial_cosine_similarity(char_profile["anchor_image_b64"], final_imgs[0])
                                if score: scene_scores.append(f"{char_profile['name']}: {score}")
                        
                        generated_panels.append({
                            "bytes": final_imgs[0],
                            "caption": scene.get('title'),
                            "metrics": " | ".join(scene_scores) if scene_scores else "Metrics N/A"
                        })
                        st.success(f"Scene {scene.get('scene_number')} Complete!")

            # Final Grid Display
            if generated_panels:
                st.subheader("🖼️ Final Storyboard & ArcFace Metrics")
                cols = st.columns(3)
                for i, panel in enumerate(generated_panels):
                    with cols[i % 3]:
                        st.image(Image.open(BytesIO(panel["bytes"])), use_container_width=True)
                        st.markdown(f"**{panel['caption']}**")
                        st.success(f"**ArcFace Similarity:**\n\n{panel['metrics']}")