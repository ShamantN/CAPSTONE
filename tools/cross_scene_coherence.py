"""
MMCCP Phase R7: Cross-Scene Identity Coherence Evaluator (Tools Only)
Strictly independent tool (freeze untouched).

Objective:
Measures same-character face similarity across scenes and seeds divided by
impostor similarity (inter-character confusion) across the 4 independent judges:
Facenet512, SFace, GhostFaceNet, ArcFace, and the ensemble average.

Evaluates:
1. Baseline v1 on DEV-FAST (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37)
2. Bundle iter3 on DEV-FAST (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37)
"""
import os
import sys
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Force UTF-8
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
        try: os.add_dll_directory(_p)
        except: pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.cli import assign_faces_to_characters

DEV_FAST_SCENES = [1, 2, 4, 5]
DEV_FAST_SEEDS = [11, 23, 37]

STORYBOARD_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
    story_spec = json.load(f)
SCENE_MAP = {s["scene_number"]: s for s in story_spec["scenes"]}

def extract_dataset_character_faces(
    images_dir: Path,
    auditor: SceneAuditor,
    verifier: BiometricEnsemble
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts detected face crops and embeddings grouped by character ID."""
    char_faces = {}
    
    for s_num in DEV_FAST_SCENES:
        scene = SCENE_MAP[s_num]
        expected_chars = scene.get("characters_present", scene.get("characters", []))
        char_positions = scene.get("character_positions", {})
        
        for seed in DEV_FAST_SEEDS:
            img_path = images_dir / f"scene_{s_num}_seed_{seed}.png"
            if not img_path.exists():
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h_img, w_img = img_bgr.shape[:2]
            
            detected = auditor.detect_faces(img_bgr)
            assigned = assign_faces_to_characters(
                detected_faces=detected,
                expected_chars=expected_chars,
                character_positions=char_positions,
                image_width=w_img
            )
            
            for cid in expected_chars:
                tup = assigned.get(cid)
                if tup is not None:
                    idx, face_item = tup
                    crop = face_item.get("crop_bgr")
                    if crop is not None and crop.size > 0:
                        embs = verifier.extract_face_embeddings(crop)
                        if cid not in char_faces:
                            char_faces[cid] = []
                        char_faces[cid].append({
                            "scene": s_num,
                            "seed": seed,
                            "bbox": face_item["bbox"],
                            "embeddings": embs
                        })
    return char_faces

def compute_coherence_matrix(char_faces: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    characters = sorted(list(char_faces.keys()))
    models = BiometricEnsemble.JUDGE_MODELS
    
    genuine_sims = {m: [] for m in models}
    impostor_sims = {m: [] for m in models}
    
    # 1. Genuine pairs: same character, different scenes/seeds
    for cid in characters:
        instances = char_faces[cid]
        n = len(instances)
        for i in range(n):
            for j in range(i + 1, n):
                embs_i = instances[i]["embeddings"]
                embs_j = instances[j]["embeddings"]
                for m in models:
                    if m in embs_i and m in embs_j:
                        v_i = embs_i[m]
                        v_j = embs_j[m]
                        sim = float(np.dot(v_i, v_j))
                        genuine_sims[m].append(sim)
                        
    # 2. Impostor pairs: different characters
    for i_c, cid1 in enumerate(characters):
        for j_c in range(i_c + 1, len(characters)):
            cid2 = characters[j_c]
            for inst1 in char_faces[cid1]:
                for inst2 in char_faces[cid2]:
                    embs1 = inst1["embeddings"]
                    embs2 = inst2["embeddings"]
                    for m in models:
                        if m in embs1 and m in embs2:
                            v1 = embs1[m]
                            v2 = embs2[m]
                            sim = float(np.dot(v1, v2))
                            impostor_sims[m].append(sim)
                            
    # Summary per model
    model_results = {}
    for m in models:
        gen_mean = float(np.mean(genuine_sims[m])) if genuine_sims[m] else 0.0
        imp_mean = float(np.mean(impostor_sims[m])) if impostor_sims[m] else 0.0
        ratio = gen_mean / imp_mean if imp_mean > 0 else 0.0
        gap = gen_mean - imp_mean
        model_results[m] = {
            "genuine_mean": round(gen_mean, 4),
            "impostor_mean": round(imp_mean, 4),
            "coherence_ratio": round(ratio, 4),
            "separation_gap": round(gap, 4),
            "num_genuine_pairs": len(genuine_sims[m]),
            "num_impostor_pairs": len(impostor_sims[m])
        }
        
    # Ensemble average
    all_gen = [model_results[m]["genuine_mean"] for m in models]
    all_imp = [model_results[m]["impostor_mean"] for m in models]
    ens_gen = float(np.mean(all_gen))
    ens_imp = float(np.mean(all_imp))
    ens_ratio = ens_gen / ens_imp if ens_imp > 0 else 0.0
    
    return {
        "ensemble": {
            "genuine_mean": round(ens_gen, 4),
            "impostor_mean": round(ens_imp, 4),
            "coherence_ratio": round(ens_ratio, 4),
            "separation_gap": round(ens_gen - ens_imp, 4)
        },
        "per_model": model_results,
        "character_instance_counts": {cid: len(char_faces[cid]) for cid in characters}
    }

def main():
    print("=" * 80)
    print("MMCCP Phase R7: Cross-Scene Identity Coherence Evaluation")
    print("=" * 80)
    
    auditor = SceneAuditor()
    verifier = BiometricEnsemble()
    
    v1_dir = PROJECT_ROOT / "results" / "baseline_v1" / "dev_full"
    iter3_dir = PROJECT_ROOT / "results" / "iterations" / "loop-W3W4-iter3-closeup-scale-fidelity" / "renders"
    
    print("\n[1/2] Processing Baseline v1 DEV-FAST...")
    v1_faces = extract_dataset_character_faces(v1_dir, auditor, verifier)
    v1_coherence = compute_coherence_matrix(v1_faces)
    
    print("\n[2/2] Processing Bundle (iter3) DEV-FAST...")
    iter3_faces = extract_dataset_character_faces(iter3_dir, auditor, verifier)
    iter3_coherence = compute_coherence_matrix(iter3_faces)
    
    print("\n" + "=" * 80)
    print("CROSS-SCENE COHERENCE COMPARISON:")
    print("=" * 80)
    print(f"Baseline v1 Ensemble Coherence Ratio:  {v1_coherence['ensemble']['coherence_ratio']:.4f} (Genuine={v1_coherence['ensemble']['genuine_mean']:.4f}, Impostor={v1_coherence['ensemble']['impostor_mean']:.4f}, Gap={v1_coherence['ensemble']['separation_gap']:+.4f})")
    print(f"Bundle iter3 Ensemble Coherence Ratio: {iter3_coherence['ensemble']['coherence_ratio']:.4f} (Genuine={iter3_coherence['ensemble']['genuine_mean']:.4f}, Impostor={iter3_coherence['ensemble']['impostor_mean']:.4f}, Gap={iter3_coherence['ensemble']['separation_gap']:+.4f})")
    
    out_json = PROJECT_ROOT / "results" / "cross_scene_coherence.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_v1": v1_coherence,
            "bundle_iter3": iter3_coherence
        }, f, indent=2)
    print(f"\nSaved cross-scene coherence to: {out_json}")

if __name__ == "__main__":
    main()
