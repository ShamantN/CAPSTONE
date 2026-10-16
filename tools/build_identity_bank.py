"""
MMCCP Phase R6: Multi-Angle Identity Bank Builder
Generates and gates 5 to 8 multi-pose/expression views per character into local identity bank.
Uses certified anchors, Forge txt2img pose generation, DirectFaceSwapper CPU execution,
CodeFormer facial restoration, and strict 4-point admission gating.
Computes multi-model centroid embeddings and saves metadata.json.
"""
import os
import sys
import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import cv2
import numpy as np
from PIL import Image

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DLL paths for Torch on Windows
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

from consistency_v2.face_swap import DirectFaceSwapper
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble, normalize_to_unit_vector
from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.splits import assert_split_allowed

FORGE_API = "http://127.0.0.1:7860"
BANKS_ROOT = PROJECT_ROOT / "consistency_v2" / "banks"
ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"

# Define character specifications and variation prompts
CHARACTER_SPECS = {
    "young_ebright": {
        "anchor_file": ANCHORS_DIR / "young_ebright_ground_truth.png",
        "char_type": "child",
        "base_prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio portrait photograph of ((YEbright_K4M9Z:1.4)), "
            "(8-year-old Caucasian young boy:1.4), fair skin, short light brown side-parted hair, hazel eyes, "
            "wearing yellow long-sleeve t-shirt with navy blue collar, solid neutral studio background, soft lighting"
        ),
        "neg_prompt": "adult, woman, mature, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text",
        "views": [
            {
                "name": "three_quarter_left",
                "prompt_suffix": "head turned 30 degrees to the left looking to the side",
                "seed": 8011
            },
            {
                "name": "three_quarter_right",
                "prompt_suffix": "head turned 30 degrees to the right looking to the side",
                "seed": 8012
            },
            {
                "name": "slight_upward",
                "prompt_suffix": "head tilted slightly upward, inquisitive upward gaze",
                "seed": 8013
            },
            {
                "name": "slight_downward",
                "prompt_suffix": "head tilted slightly downward, thoughtful downward look",
                "seed": 8014
            },
            {
                "name": "warm_smile",
                "prompt_suffix": "warm cheerful toothy smile, joyful boyish expression",
                "seed": 8015
            },
            {
                "name": "close_up",
                "prompt_suffix": "tight close-up portrait of face, detailed facial features",
                "seed": 8016
            }
        ]
    },
    "mother": {
        "anchor_file": ANCHORS_DIR / "mother_ground_truth.png",
        "char_type": "adult",
        "base_prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio portrait photograph of ((Mother_M2B4Q:1.4)), "
            "(40-year-old mature Caucasian mother:1.4), fair skin, shoulder-length wavy dark brown hair with grey streaks, "
            "hazel eyes, round mature face, wearing pale blue floral blouse, solid neutral studio background, soft studio lighting"
        ),
        "neg_prompt": "child, teenager, young girl, male, man, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text",
        "views": [
            {
                "name": "three_quarter_left",
                "prompt_suffix": "head turned 30 degrees to the left looking to the side",
                "seed": 8021
            },
            {
                "name": "three_quarter_right",
                "prompt_suffix": "head turned 30 degrees to the right looking to the side",
                "seed": 8022
            },
            {
                "name": "slight_upward",
                "prompt_suffix": "head tilted slightly upward, thoughtful maternal look",
                "seed": 8023
            },
            {
                "name": "slight_downward",
                "prompt_suffix": "head tilted slightly downward, gentle serene expression",
                "seed": 8024
            },
            {
                "name": "warm_smile",
                "prompt_suffix": "warm loving maternal smile, happy expression",
                "seed": 8025
            },
            {
                "name": "close_up",
                "prompt_suffix": "tight close-up portrait of face, mature refined features",
                "seed": 8026
            }
        ]
    },
    "teen_ebright": {
        "anchor_file": ANCHORS_DIR / "teen_ebright_ground_truth.png",
        "char_type": "adult",
        "base_prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio portrait photograph of ((TEbright_W3R8K:1.4)), "
            "(15-year-old Caucasian teenage boy:1.4), fair skin, textured side-part brown hair, hazel eyes, "
            "adolescent facial bone structure, wearing dark green plaid button-up shirt, solid neutral studio background, soft dramatic lighting"
        ),
        "neg_prompt": "child, toddler, woman, female, beard, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text",
        "views": [
            {
                "name": "three_quarter_left",
                "prompt_suffix": "head turned 30 degrees to the left looking to the side",
                "seed": 8031
            },
            {
                "name": "three_quarter_right",
                "prompt_suffix": "head turned 30 degrees to the right looking to the side",
                "seed": 8032
            },
            {
                "name": "slight_upward",
                "prompt_suffix": "head tilted slightly upward, focused intellectual gaze",
                "seed": 8033
            },
            {
                "name": "slight_downward",
                "prompt_suffix": "head tilted slightly downward, inspecting something below",
                "seed": 8034
            },
            {
                "name": "warm_smile",
                "prompt_suffix": "slight confident adolescent smile",
                "seed": 8035
            },
            {
                "name": "close_up",
                "prompt_suffix": "tight close-up portrait of face, adolescent features",
                "seed": 8036
            }
        ]
    },
    "college_ebright": {
        "anchor_file": ANCHORS_DIR / "college_ebright_ground_truth.png",
        "char_type": "adult",
        "base_prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio portrait photograph of ((CEbright_X9F2A:1.4)), "
            "(21-year-old Caucasian young adult man:1.4), fair skin, neatly combed back brown hair, hazel eyes, "
            "mature defined jawline, clean-shaven, wearing mustard yellow polo shirt, solid neutral grey studio background, studio lighting"
        ),
        "neg_prompt": "child, teenager, woman, beard, mustache, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text",
        "views": [
            {
                "name": "three_quarter_left",
                "prompt_suffix": "head turned 30 degrees to the left looking to the side",
                "seed": 8041
            },
            {
                "name": "three_quarter_right",
                "prompt_suffix": "head turned 30 degrees to the right looking to the side",
                "seed": 8042
            },
            {
                "name": "slight_upward",
                "prompt_suffix": "head tilted slightly upward, visionary scholar expression",
                "seed": 8043
            },
            {
                "name": "slight_downward",
                "prompt_suffix": "head tilted slightly downward, serious scholarly look",
                "seed": 8044
            },
            {
                "name": "warm_smile",
                "prompt_suffix": "friendly confident smile, young scholar",
                "seed": 8045
            },
            {
                "name": "close_up",
                "prompt_suffix": "tight close-up portrait of face, chiseled jawline",
                "seed": 8046
            }
        ]
    },
    "james_wong": {
        "anchor_file": ANCHORS_DIR / "james_wong_ground_truth.png",
        "char_type": "adult",
        "base_prompt": (
            "(Cinematic Realistic:1.2), masterpiece, 8k resolution, highly detailed, "
            "professional studio portrait photograph of ((Wong_J7P1L:1.4)), "
            "(21-year-old East Asian adult man:1.4), light tan skin, short parted black hair, dark brown eyes, "
            "square sharp face shape, sharp jawline, clean-shaven, wearing crimson red Harvard sweater, solid neutral grey studio background, soft lighting"
        ),
        "neg_prompt": "child, teenager, caucasian, white, woman, beard, blurry, deformed, bad anatomy, bad eyes, disfigured, watermark, text",
        "views": [
            {
                "name": "three_quarter_left",
                "prompt_suffix": "head turned 30 degrees to the left looking to the side",
                "seed": 8051
            },
            {
                "name": "three_quarter_right",
                "prompt_suffix": "head turned 30 degrees to the right looking to the side",
                "seed": 8052
            },
            {
                "name": "slight_upward",
                "prompt_suffix": "head tilted slightly upward, confident analytical gaze",
                "seed": 8053
            },
            {
                "name": "slight_downward",
                "prompt_suffix": "head tilted slightly downward, deep concentration",
                "seed": 8054
            },
            {
                "name": "warm_smile",
                "prompt_suffix": "charismatic collegial smile",
                "seed": 8055
            },
            {
                "name": "close_up",
                "prompt_suffix": "tight close-up portrait of face, sharp jawline and focused eyes",
                "seed": 8056
            }
        ]
    }
}


def generate_pose_image(prompt: str, neg_prompt: str, seed: int) -> Optional[np.ndarray]:
    """Generates an image via Forge txt2img."""
    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 16,
        "width": 768,
        "height": 768,
        "cfg_scale": 6.5,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "seed": seed
    }
    try:
        resp = requests.post(f"{FORGE_API}/sdapi/v1/txt2img", json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("images"):
                b64 = data["images"][0]
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [ERROR] txt2img failed: {e}")
    return None


def extract_torso_lab(image_bgr: np.ndarray, face_bbox: Dict[str, int]) -> List[float]:
    """Extracts median CIELAB color vector from upper torso area below face."""
    h_img, w_img = image_bgr.shape[:2]
    fx = face_bbox.get("x", 0)
    fy = face_bbox.get("y", 0)
    fw = face_bbox.get("w", 0)
    fh = face_bbox.get("h", 0)

    y1 = min(h_img, fy + int(0.8 * fh))
    y2 = min(h_img, fy + int(3.0 * fh))
    x1 = max(0, fx - int(0.5 * fw))
    x2 = min(w_img, fx + int(1.5 * fw))

    if (y2 - y1) >= 20 and (x2 - x1) >= 20:
        torso_bgr = image_bgr[y1:y2, x1:x2]
        torso_lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = torso_lab[:, :, 0] * (100.0 / 255.0)
        a_chan = torso_lab[:, :, 1] - 128.0
        b_chan = torso_lab[:, :, 2] - 128.0
        return [
            float(round(np.median(l_chan), 2)),
            float(round(np.median(a_chan), 2)),
            float(round(np.median(b_chan), 2))
        ]
    return [50.0, 0.0, 0.0]


def build_character_bank(cid: str, spec: Dict[str, Any], auditor: SceneAuditor, ensemble: BiometricEnsemble):
    """Builds a complete multi-view identity bank for a character."""
    print(f"\n{'='*70}\n[BUILDING BANK] Character: {cid} ({spec['char_type']})\n{'='*70}")

    char_dir = BANKS_ROOT / cid
    char_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = PROJECT_ROOT / "scratch" / "bank_builder"
    temp_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = spec["anchor_file"]
    if not anchor_path.exists():
        raise FileNotFoundError(f"Certified anchor missing for {cid}: {anchor_path}")

    anchor_bgr = cv2.imread(str(anchor_path))
    anchor_faces = auditor.detect_faces(anchor_bgr)
    if not anchor_faces:
        raise ValueError(f"No face detected in certified anchor for {cid}!")

    # Extract anchor embeddings
    anchor_embs = ensemble.extract_face_embeddings(anchor_bgr)
    print(f"Anchor embeddings extracted across {len(anchor_embs)} judge models.")

    # Pose estimation for anchor
    anchor_face = anchor_faces[0]
    anchor_pose = IdentityBankManager().estimate_head_pose(
        anchor_face.get("landmarks", []),
        anchor_face.get("bbox", {"w": 100, "h": 100})
    )
    anchor_lab = extract_torso_lab(anchor_bgr, anchor_face.get("bbox", {}))

    # Copy anchor as ref_00
    ref_00_path = char_dir / f"ref_00_{cid}_anchor.png"
    cv2.imwrite(str(ref_00_path), anchor_bgr)

    references = [{
        "index": 0,
        "filename": ref_00_path.name,
        "view_name": "anchor_frontal",
        "path": str(ref_00_path),
        "pose": anchor_pose,
        "face_width_px": anchor_face["bbox"]["w"],
        "sharpness": round(auditor.audit_face_quality(anchor_face)["sharpness_laplacian"], 1),
        "biometric_score": 1.0,
        "wardrobe_lab": anchor_lab
    }]

    all_embs: Dict[str, List[np.ndarray]] = {m: [anchor_embs[m]] for m in anchor_embs}
    all_wardrobe_labs: List[List[float]] = [anchor_lab]

    # Generate each view
    for v_idx, v_spec in enumerate(spec["views"], start=1):
        v_name = v_spec["name"]
        seed = v_spec["seed"]
        print(f"\n--- Generating View {v_idx}/6: {v_name} (Seed: {seed}) ---")

        full_prompt = f"{spec['base_prompt']}, {v_spec['prompt_suffix']}"
        raw_pose_bgr = generate_pose_image(full_prompt, spec["neg_prompt"], seed)
        if raw_pose_bgr is None:
            print(f"  [WARN] Failed to generate raw pose image for {v_name}")
            continue

        temp_in = str(temp_dir / f"{cid}_{v_name}_raw.png")
        temp_out = str(temp_dir / f"{cid}_{v_name}_swapped.png")
        cv2.imwrite(temp_in, raw_pose_bgr)

        # Execute DirectFaceSwapper
        swap_specs = [{
            "character_id": cid,
            "target_zone": "center",
            "source_path": str(anchor_path)
        }]
        ok = DirectFaceSwapper.execute_face_swap(
            target_image_path=temp_in,
            swap_specs=swap_specs,
            out_image_path=temp_out,
            codeformer_fidelity=0.60,
            forge_api_url=FORGE_API
        )

        if not (ok and os.path.exists(temp_out)):
            print(f"  [WARN] Face swap failed for {v_name}")
            continue

        swapped_bgr = cv2.imread(temp_out)
        faces = auditor.detect_faces(swapped_bgr)
        if len(faces) != 1:
            print(f"  [GATE FAIL] Expected 1 face, detected {len(faces)} for {v_name}")
            continue

        face = faces[0]
        q = auditor.audit_face_quality(face)
        fw = q["width_px"]
        sharpness = q["sharpness_laplacian"]

        if fw < 96:
            print(f"  [GATE FAIL] Face width {fw}px < 96px for {v_name}")
            continue

        # Extract embeddings
        embs = ensemble.extract_face_embeddings(swapped_bgr)
        sims = []
        for m in BiometricEnsemble.JUDGE_MODELS:
            if m in anchor_embs and m in embs:
                sims.append(float(np.dot(anchor_embs[m], embs[m])))
        mean_sim = float(np.mean(sims)) if sims else 0.0

        calib_thresh = 0.13 if spec["char_type"] == "child" else 0.12
        if mean_sim < calib_thresh:
            print(f"  [GATE FAIL] Biometric similarity {mean_sim:.4f} < {calib_thresh} for {v_name}")
            continue

        # Passed 4-point gate!
        pose = IdentityBankManager().estimate_head_pose(face.get("landmarks", []), face.get("bbox", {}))
        w_lab = extract_torso_lab(swapped_bgr, face.get("bbox", {}))

        ref_fn = f"ref_{len(references):02d}_{cid}_{v_name}.png"
        ref_path = char_dir / ref_fn
        cv2.imwrite(str(ref_path), swapped_bgr)

        ref_entry = {
            "index": len(references),
            "filename": ref_fn,
            "view_name": v_name,
            "path": str(ref_path),
            "pose": pose,
            "face_width_px": fw,
            "sharpness": round(sharpness, 1),
            "biometric_score": round(mean_sim, 4),
            "wardrobe_lab": w_lab
        }
        references.append(ref_entry)
        all_wardrobe_labs.append(w_lab)
        for m in embs:
            if m in all_embs:
                all_embs[m].append(embs[m])

        print(f"  [GATE PASS] {v_name}: fw={fw}px, sharp={sharpness:.1f}, sim={mean_sim:.4f}, yaw={pose.get('yaw', 0.0)}deg")

    # Compute centroid embeddings across all judge models
    centroid_embeddings = {}
    for m in BiometricEnsemble.JUDGE_MODELS:
        if m in all_embs and all_embs[m]:
            mean_vec = np.mean(all_embs[m], axis=0)
            centroid_embeddings[m] = normalize_to_unit_vector(mean_vec).tolist()

    avg_wardrobe_lab = np.mean(all_wardrobe_labs, axis=0).tolist()

    bank_meta = {
        "character_id": cid,
        "character_type": spec["char_type"],
        "num_references": len(references),
        "references": references,
        "centroid_embeddings": centroid_embeddings,
        "average_wardrobe_lab": [round(x, 2) for x in avg_wardrobe_lab],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "R6"
    }

    meta_path = char_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(bank_meta, f, indent=2)

    print(f"\n[BANK COMPLETE] {cid}: {len(references)} references registered in {meta_path.name}")
    return bank_meta


def main():
    print("=" * 80)
    print("MMCCP Phase R6: Multi-Angle Identity Bank Builder")
    print("=" * 80)

    # Pre-flight check: ensure Forge is running
    try:
        r = requests.get(f"{FORGE_API}/sdapi/v1/progress", timeout=5)
        if r.status_code != 200:
            print("[FATAL] Forge is not responding on 127.0.0.1:7860")
            sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Could not connect to Forge: {e}")
        sys.exit(1)

    auditor = SceneAuditor()
    ensemble = BiometricEnsemble()

    summary = {}
    t0 = time.time()

    for cid, spec in CHARACTER_SPECS.items():
        meta = build_character_bank(cid, spec, auditor, ensemble)
        summary[cid] = {
            "num_references": meta["num_references"],
            "views": [r["view_name"] for r in meta["references"]],
            "mean_sharpness": round(float(np.mean([r["sharpness"] for r in meta["references"]])), 1),
            "mean_face_width": round(float(np.mean([r["face_width_px"] for r in meta["references"]])), 1)
        }

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"IDENTITY BANK BUILD SUMMARY (Total Time: {total_time:.1f}s)")
    print("=" * 80)
    for cid, s in summary.items():
        print(f"  * {cid}: {s['num_references']} references | Mean Width: {s['mean_face_width']}px | Mean Sharpness: {s['mean_sharpness']}")

    summary_file = BANKS_ROOT / "bank_build_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_characters": len(summary),
            "summary": summary
        }, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()
