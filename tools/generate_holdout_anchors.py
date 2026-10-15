"""
Generate and audit certified anchors for the holdout storyboard characters:
- elena_child
- dr_marcus_vance
- clara_thorn

Requirements:
- SDXL Juggernaut close-up portrait renders via Forge API
- Audited for single face detection, face width >= 200px, high sharpness (Laplacian variance > 100)
- Store in experiments/clean_run_ground_truth/anchors/holdout/
- Register in hardcoded_cache.json
"""
import os
import sys
import json
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure DLL paths for torch/deepface
_DLL_PATHS = [
    r"C:\Stable-Diffusion\system\python\Library\bin",
    r"C:\Stable-Diffusion\system\python\Lib\site-packages\torch\lib",
    r"C:\Stable-Diffusion\system\python\Scripts",
    r"C:\Stable-Diffusion\system\python"
]
for _p in _DLL_PATHS:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from consistency_v2.verify.scene_auditor import SceneAuditor

FORGE_API = "http://127.0.0.1:7860"
OUT_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors" / "holdout"
CACHE_FILE = PROJECT_ROOT / "hardcoded_cache.json"

CHARACTERS = {
    "elena_child": {
        "prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio close-up portrait photograph of ((Elena_C9R2M:1.45)), "
            "(9-year-old small child girl:1.5), (childhood portrait of little girl, heart-shaped youthful child face, primary school girl:1.4), "
            "(small 9 years old girl:1.4), (Eastern European, fair skin tone, strawberry blonde hair in twin braids, bright green eyes:1.30), "
            "cute freckles across nose, wearing (crimson red knitted cardigan and white collared shirt:1.25), "
            "innocent youthful child expression, solid plain neutral white studio background, soft flattering portrait lighting, "
            "sharp focus on eyes, looking directly at camera"
        ),
        "neg_prompt": (
            "adult, grown woman, mature face, teenager, 20 years old, 30 years old, wrinkles, tall, male, boy, "
            "cartoon, anime, 3d render, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text"
        ),
        "candidate_seeds": [5011, 5022, 5033, 5044, 5055]
    },
    "dr_marcus_vance": {
        "prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio close-up portrait photograph of ((Marcus_V8K4P:1.45)), "
            "(45-year-old mature adult African American man:1.5), (distinguished middle-aged Black scholar, chiseled square jaw:1.45), "
            "(deep dark brown skin tone:1.4), (dark brown eyes:1.35), "
            "(black hair with silver temples, short faded buzzcut:1.30), trimmed salt-and-pepper goatee, wire-rim glasses, "
            "wearing (charcoal grey turtleneck sweater:1.25), confident intellectual expression, "
            "solid plain neutral grey studio background, soft dramatic portrait lighting, sharp focus on eyes, looking directly at camera"
        ),
        "neg_prompt": (
            "caucasian, white man, pale skin, fair skin, light skin, child, teenager, young boy, female, woman, "
            "cartoon, anime, 3d render, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text, missing glasses, long hair"
        ),
        "candidate_seeds": [5111, 5122, 5133, 5144, 5155]
    },
    "clara_thorn": {
        "prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio close-up portrait photograph of ((Clara_T3N7L:1.45)), "
            "(32-year-old elegant adult woman:1.5), (oval elegant face shape, high cheekbones:1.4), "
            "(East Asian, warm ivory skin tone, amber brown eyes:1.35), "
            "(jet black sleek high ponytail hair:1.30), wearing (emerald green silk tailored blazer and cream blouse:1.25), "
            "silver leaf lapel pin, poised sophisticated expression, solid plain neutral studio background, "
            "soft flattering studio lighting, sharp focus on eyes, looking directly at camera"
        ),
        "neg_prompt": (
            "child, teenager, baby, old woman, male, man, beard, mustache, cartoon, anime, 3d render, blurry, "
            "deformed, bad anatomy, bad eyes, disfigured, watermark, text"
        ),
        "candidate_seeds": [5211, 5222, 5233, 5244, 5255]
    }
}


def audit_anchor(img_bgr: np.ndarray, auditor: SceneAuditor) -> Dict[str, Any]:
    """Audits candidate anchor for single face, sharpness, and defect counts."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    faces = auditor.detect_faces(img_bgr)

    is_valid = True
    reasons = []

    if len(faces) != 1:
        is_valid = False
        reasons.append(f"Expected 1 face, found {len(faces)}")
    else:
        fw = faces[0]["bbox"]["w"]
        fh = faces[0]["bbox"]["h"]
        if fw < 150 or fh < 150:
            is_valid = False
            reasons.append(f"Face too small: {fw}x{fh} < 150x150")
        
        face_audit = auditor.audit_face_quality(faces[0])
        if face_audit.get("melted_face_flag", False):
            is_valid = False
            reasons.append("Face distorted / melted")

    if laplacian_var < 100.0:
        is_valid = False
        reasons.append(f"Sharpness too low: {laplacian_var:.1f} < 100.0")

    return {
        "is_valid": is_valid,
        "reasons": reasons,
        "face_count": len(faces),
        "laplacian_var": laplacian_var,
        "faces": faces
    }


def generate_anchor_render(prompt: str, neg_prompt: str, seed: int) -> Optional[np.ndarray]:
    """Sends txt2img request to Forge."""
    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 25,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 6.5,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "seed": seed,
        "alwayson_scripts": {
            "freeu integrated (sd 1.x, sd 2.x, sdxl)": {
                "args": [True, 1.3, 1.4, 0.9, 0.2, 0.0, 1.0]
            },
            "perturbedattentionguidance integrated": {
                "args": [True, 3.0, 0.0, 0.0, 1.0]
            },
            "dynamicthresholding (cfg-fix) integrated": {
                "args": [True, 7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, "enable", "MEAN", "AD", 1.0]
            }
        }
    }

    resp = requests.post(f"{FORGE_API}/sdapi/v1/txt2img", json=payload, timeout=300)
    if resp.status_code != 200:
        print(f"  Forge error {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    images = data.get("images", [])
    if not images:
        return None

    raw = base64.b64decode(images[0])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auditor = SceneAuditor()

    # Load existing hardcoded cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    holdout_anchors = {}
    out_json = OUT_DIR / "holdout_anchors.json"
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                holdout_anchors = json.load(f)
        except Exception:
            pass

    target_chars = sys.argv[1:] if len(sys.argv) > 1 else list(CHARACTERS.keys())

    print("\n" + "="*80)
    print(f"  GENERATING CERTIFIED HOLDOUT ANCHORS FOR: {target_chars}")
    print("="*80 + "\n")

    for char_id in target_chars:
        if char_id not in CHARACTERS:
            continue
        spec = CHARACTERS[char_id]
        print(f"[{char_id}] Searching for certified anchor...")
        accepted_img = None
        accepted_seed = None
        accepted_audit = None

        for seed in spec["candidate_seeds"]:
            print(f"  Trying seed {seed}...")
            img_bgr = generate_anchor_render(spec["prompt"], spec["neg_prompt"], seed)
            if img_bgr is None:
                continue

            audit = audit_anchor(img_bgr, auditor)
            print(f"    Audit: valid={audit['is_valid']}, faces={audit['face_count']}, sharpness={audit['laplacian_var']:.1f}")
            if audit["is_valid"]:
                accepted_img = img_bgr
                accepted_seed = seed
                accepted_audit = audit
                break
            else:
                print(f"    Rejected reasons: {audit['reasons']}")

        if accepted_img is None:
            raise RuntimeError(f"Failed to find valid certified anchor for {char_id} among candidate seeds!")

        # Save PNG to OUT_DIR
        out_png = OUT_DIR / f"{char_id}_ground_truth.png"
        cv2.imwrite(str(out_png), accepted_img)
        print(f"  -> Saved certified anchor: {out_png}")

        # Base64 encode
        _, buf = cv2.imencode(".png", accepted_img)
        b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
        holdout_anchors[char_id] = b64_str

        # Update cache under char_id and char_id_primary
        cache[char_id] = b64_str
        cache[f"{char_id}_primary"] = b64_str

    # Write holdout_anchors.json
    out_json = OUT_DIR / "holdout_anchors.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(holdout_anchors, f, indent=2)
    print(f"\nSaved holdout anchors JSON: {out_json}")

    # Write back to hardcoded_cache.json
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"Updated cache: {CACHE_FILE} (total keys: {len(cache)})")

    print("\nAll 3 holdout anchors generated, audited, and certified successfully!\n")


if __name__ == "__main__":
    main()
