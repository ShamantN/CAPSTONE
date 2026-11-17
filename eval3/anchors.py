"""
MMCCP TEST-3 Evaluation Protocol: Anchor Synthesis, Certification & Selection (eval3/anchors.py).

Implements deterministic anchor candidate generation (up to 8 candidates per character),
the strict 4-point admission gate, deterministic selection rule (highest confidence x face width),
fallback uncertified anchor tagging, intra-replicate separation checking, and isolated bank export.
"""
import sys
import json
import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import CharacterSpec, AgeState, StorySpec
from consistency_v2.runtime_paths import get_forge_url
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from eval3.splits import get_anchor_seed

DEFAULT_SEPARATION_THRESHOLD = 0.42  # Refined on DEV-3 impostor 99th percentile

CERTIFIED_ADMISSION_RULES: Dict[str, Any] = {
    "min_confidence": 0.80,
    "min_face_width_px": 80,
    "min_sharpness": 60.0,
    "single_face_required": True,
    "no_defects_required": True,
}


def load_all_evaluation_characters() -> Dict[str, CharacterSpec]:
    """Loads all characters across Story A (ebright), Story B (holdout_v2), and Story C (eval3_story_c)."""
    chars: Dict[str, CharacterSpec] = {}
    sb_dir = PROJECT_ROOT / "storyboards"
    for fname in ["ebright.json", "holdout_v2.json", "eval3_story_c.json"]:
        fpath = sb_dir / fname
        if fpath.exists():
            spec = StorySpec.load_json(str(fpath))
            for cid, cdata in spec.characters.items():
                if isinstance(cdata, CharacterSpec):
                    chars[cid] = cdata
                elif isinstance(cdata, dict):
                    chars[cid] = CharacterSpec.from_dict(cdata)
    return chars


def build_anchor_prompt(char: CharacterSpec, age_state: Optional[AgeState] = None) -> Tuple[str, str]:
    """Builds demographic anchor prompt and negative prompt."""
    age = age_state.age_years if age_state else "25"
    stage = age_state.stage_name if age_state else "adult"
    hair_style = age_state.hairstyle if age_state and age_state.hairstyle else char.hair_style
    clothing = age_state.clothing_description if age_state and age_state.clothing_description else char.recurring_clothing

    demo = f"((strictly {char.ethnicity}:1.45)), ({char.skin_tone} skin tone:1.35), ({char.face_shape} face shape:1.25), ({char.eye_color} eyes:1.2)"
    hair = f"({char.hair_color} hair, {hair_style}:1.3)"
    outfit = f"wearing ({clothing}:1.3)" if clothing else "wearing casual clothes"

    pos_prompt = (
        f"(Cinematic Realistic:1.2), masterpiece, 8k resolution, raw photo, highly detailed, "
        f"a close-up portrait of {char.name}, {age} years old {stage} {char.gender}, "
        f"{demo}, {hair}, {outfit}, neutral white studio background, studio portrait lighting, sharp focus, clean skin texture."
    )

    neg_prompt = (
        f"blurry, deformed, bad anatomy, bad hands, missing fingers, extra fingers, ugly, watermark, text, signature, "
        f"opposite gender, wrong ethnicity, cartoon, illustration, low quality"
    )

    try:
        age_int = int(age)
        if age_int < 12:
            neg_prompt += ", teenager, adult, mature face, beard, mustache, stubble, deep voice, wrinkles, clean-shaven man"
        elif age_int < 18:
            neg_prompt += ", child, toddler, old man, mature adult, full beard, mustache, wrinkles"
        else:
            neg_prompt += ", child, baby, boy, kid, toddler, adolescent"
    except Exception:
        pass

    return pos_prompt, neg_prompt


def generate_candidate_txt2img(
    prompt: str,
    neg_prompt: str,
    seed: int,
    forge_url: Optional[str] = None,
    timeout: int = 180
) -> Optional[np.ndarray]:
    """Calls Forge txt2img API to render a single anchor candidate."""
    base_url = (forge_url or get_forge_url()).rstrip("/")
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
        resp = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        images = data.get("images", [])
        if not images:
            return None
        raw = base64.b64decode(images[0])
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def audit_candidate_4point_gate(
    img_bgr: np.ndarray,
    auditor: SceneAuditor
) -> Dict[str, Any]:
    """
    Evaluates 4-point admission gate:
    1. Single face detected
    2. Size pass: face width >= 96 px
    3. Sharpness pass: Laplacian variance >= 60.0
    4. Zero defects (no melted face)
    Returns gate result dictionary with computed selection score = (confidence * width_px).
    """
    faces = auditor.detect_faces(img_bgr)
    res = {
        "passed": False,
        "face_count": len(faces),
        "single_face": len(faces) == 1,
        "face_width_px": 0,
        "face_confidence": 0.0,
        "sharpness": 0.0,
        "size_pass": False,
        "sharpness_pass": False,
        "no_defects": False,
        "selection_score": 0.0,
        "crop_bgr": None,
    }

    if len(faces) != 1:
        return res

    face = faces[0]
    bbox = face.get("bbox", {})
    conf = float(face.get("confidence", 0.0))
    w = int(bbox.get("w", 0))

    quality = auditor.audit_face_quality(face)
    sharpness = float(quality.get("sharpness_laplacian", 0.0))
    melted = bool(quality.get("melted_face", False))

    res["face_width_px"] = w
    res["face_confidence"] = conf
    res["sharpness"] = round(sharpness, 2)
    res["size_pass"] = bool(w >= 96)
    res["sharpness_pass"] = bool(sharpness >= 60.0)
    res["no_defects"] = not melted
    res["selection_score"] = round(conf * float(w), 3)
    res["crop_bgr"] = face.get("crop_bgr")

    res["passed"] = bool(
        res["single_face"] and
        res["size_pass"] and
        res["sharpness_pass"] and
        res["no_defects"]
    )
    return res


def generate_character_candidates(
    char_id: str,
    char_spec: CharacterSpec,
    replicate_idx: int,
    char_idx: int,
    is_test3: bool,
    out_dir: Path,
    auditor: SceneAuditor,
    forge_url: Optional[str] = None,
    max_candidates: int = 8
) -> List[Dict[str, Any]]:
    """Generates up to max_candidates for a single character and audits each against 4-point gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_prompt, neg_prompt = build_anchor_prompt(char_spec)

    candidates: List[Dict[str, Any]] = []
    for k in range(max_candidates):
        seed = get_anchor_seed(replicate_idx, char_idx, k, is_test3=is_test3)
        img_bgr = generate_candidate_txt2img(pos_prompt, neg_prompt, seed, forge_url=forge_url)
        if img_bgr is None:
            continue

        gate = audit_candidate_4point_gate(img_bgr, auditor)
        cand_record = {
            "char_id": char_id,
            "replicate_idx": replicate_idx,
            "char_idx": char_idx,
            "candidate_idx": k,
            "seed": seed,
            "passed_gate": gate["passed"],
            "selection_score": gate["selection_score"],
            "face_width_px": gate["face_width_px"],
            "face_confidence": gate["face_confidence"],
            "sharpness": gate["sharpness"],
            "image_path": str(out_dir / f"{char_id}_cand_{k}_seed_{seed}.png"),
        }

        # Save candidate image
        cv2.imwrite(cand_record["image_path"], img_bgr)
        # Save sidecar JSON
        with open(out_dir / f"{char_id}_cand_{k}_seed_{seed}.json", "w", encoding="utf-8") as f:
            json.dump({k2: v2 for k2, v2 in cand_record.items() if k2 != "crop_bgr"}, f, indent=2)

        cand_record["img_bgr"] = img_bgr
        cand_record["crop_bgr"] = gate["crop_bgr"]
        candidates.append(cand_record)

    return candidates


def select_best_anchor(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic Selection Rule:
    Among certified candidates, select highest (detector confidence x face width),
    breaking ties to the lowest seed.
    If none certify, take highest score among uncertified and set uncertified_anchor=True.
    """
    assert len(candidates) > 0, "No candidates available for anchor selection!"

    certified = [c for c in candidates if c["passed_gate"]]
    if certified:
        # Sort by selection_score descending, then seed ascending
        sorted_cands = sorted(certified, key=lambda c: (-c["selection_score"], c["seed"]))
        best = sorted_cands[0].copy()
        best["uncertified_anchor"] = False
        return best
    else:
        # Fallback to best uncertified
        sorted_cands = sorted(candidates, key=lambda c: (-c["selection_score"], c["seed"]))
        best = sorted_cands[0].copy()
        best["uncertified_anchor"] = True
        return best


select_best_anchor_candidate = select_best_anchor
