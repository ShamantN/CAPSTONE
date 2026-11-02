"""
MMCCP Showcase Anchor Generator & Auditor
Generates from-scratch demographic anchors for Ebright storyboard characters
and verifies each with the 4-point certification gate.
"""
import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import cv2
import numpy as np

# Ensure DLL paths for torch/deepface on Windows
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.character_registry import CharacterRegistry
from metrics import detect_all_scene_faces, DeepFeatureExtractor

FORGE_API = "http://127.0.0.1:7860"


def generate_txt2img(prompt: str, neg_prompt: str, seed: int) -> Optional[np.ndarray]:
    """Generates an image via Forge txt2img API."""
    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 25,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 6.0,
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

    try:
        resp = requests.post(f"{FORGE_API}/sdapi/v1/txt2img", json=payload, timeout=300)
        if resp.status_code != 200:
            print(f"  [ERROR] Forge status {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        images = data.get("images", [])
        if not images:
            return None
        raw = base64.b64decode(images[0])
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [EXCEPTION] txt2img call failed: {e}")
        return None


def audit_anchor_image(img_cv: np.ndarray, char_id: str) -> Dict[str, Any]:
    """
    4-Point Anchor Certification Gate:
    1. Resolution: H >= 512, W >= 512
    2. Face count: Exactly 1 face
    3. Face area: Between 4.0% and 90.0% of canvas
    4. Embeddability: DeepFeatureExtractor norm > 0.5
    """
    h, w = img_cv.shape[:2]
    res_pass = (h >= 512 and w >= 512)

    faces = detect_all_scene_faces(img_cv, expected_faces=1)
    face_count = len(faces)
    face_pass = (face_count == 1)

    face_area_pct = 0.0
    facial_bbox = None
    if face_count >= 1:
        facial_bbox = faces[0].get("facial_area", {})
        fw = facial_bbox.get("w", 0)
        fh = facial_bbox.get("h", 0)
        face_area_pct = round(float(fw * fh) / float(h * w) * 100, 2)

    size_pass = (4.0 <= face_area_pct <= 90.0)

    emb = DeepFeatureExtractor.get_feature_vector(img_cv)
    emb_pass = (emb is not None and np.linalg.norm(emb) > 0.5)

    all_passed = bool(res_pass and face_pass and size_pass and emb_pass)

    return {
        "char_id": char_id,
        "passed": all_passed,
        "resolution": f"{w}x{h}",
        "resolution_pass": res_pass,
        "detected_faces_count": face_count,
        "face_count_pass": face_pass,
        "face_area_percentage": face_area_pct,
        "face_size_pass": size_pass,
        "facial_bbox": facial_bbox,
        "biometric_embedding_valid": bool(emb_pass),
        "embedding_norm": float(np.linalg.norm(emb)) if emb is not None else 0.0
    }


def generate_anchors_for_set(
    out_dir: Path,
    seed_map: Dict[str, int],
    max_retries: int = 6
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = CharacterRegistry()

    summary = {}
    print(f"\n==========================================")
    print(f"Generating Anchors to: {out_dir}")
    print(f"==========================================")

    for char_id, base_seed in seed_map.items():
        c_spec = registry.get_character(char_id)
        if not c_spec:
            print(f"[ERROR] Character '{char_id}' not found in registry!")
            continue

        primary_state = c_spec.age_states.get("primary")
        pos_prompt, neg_prompt = registry.build_anchor_prompt(c_spec, primary_state)

        print(f"\n---> Processing character: {char_id} (base seed {base_seed})")
        print(f"Prompt: {pos_prompt[:120]}...")

        selected_img = None
        selected_audit = None
        selected_seed = None

        for attempt in range(max_retries):
            current_seed = base_seed + (attempt * 10)
            print(f"  Attempt {attempt+1}/{max_retries} with seed {current_seed}...")
            t0 = time.time()
            img_bgr = generate_txt2img(pos_prompt, neg_prompt, current_seed)
            dt = time.time() - t0

            if img_bgr is None:
                print(f"  [FAIL] txt2img returned None (time: {dt:.1f}s)")
                continue

            audit = audit_anchor_image(img_bgr, char_id)
            print(f"  Audit: pass={audit['passed']} faces={audit['detected_faces_count']} area={audit['face_area_percentage']}% emb_norm={audit['embedding_norm']:.3f} (time: {dt:.1f}s)")

            if audit["passed"]:
                selected_img = img_bgr
                selected_audit = audit
                selected_seed = current_seed
                break
            else:
                print(f"  [RETRY] Gate failed. Reasons: res={audit['resolution_pass']}, face_count={audit['face_count_pass']}, size={audit['face_size_pass']}, emb={audit['biometric_embedding_valid']}")

        if selected_img is not None:
            out_path = out_dir / f"{char_id}.png"
            cv2.imwrite(str(out_path), selected_img)
            selected_audit["seed"] = selected_seed
            selected_audit["file_path"] = str(out_path)
            with open(out_dir / f"{char_id}_audit.json", "w", encoding="utf-8") as f:
                json.dump(selected_audit, f, indent=2)
            print(f"  [SUCCESS] Certified anchor saved to {out_path} (seed {selected_seed})")
            summary[char_id] = selected_audit
        else:
            print(f"  [FATAL] Failed to generate certified anchor for {char_id} after {max_retries} attempts.")
            summary[char_id] = {"passed": False, "attempts": max_retries}

    # Save summary report
    with open(out_dir / "anchors_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and certify character anchors.")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for anchors")
    parser.add_argument("--set-name", type=str, default="A", choices=["A", "B"], help="Set name (A or B)")
    args = parser.parse_args()

    if args.set_name == "A":
        seed_map = {
            "young_ebright": 4101,
            "teen_ebright": 4102,
            "college_ebright": 4103,
            "mother": 4104,
            "james_wong": 4105
        }
    else:
        seed_map = {
            "young_ebright": 5101,
            "teen_ebright": 5102,
            "college_ebright": 5103,
            "mother": 5104,
            "james_wong": 5105
        }

    generate_anchors_for_set(Path(args.out_dir), seed_map)
