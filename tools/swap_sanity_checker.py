"""
MMCCP Phase R7: Swap Sanity Checker (Tools Only)
Strictly independent tool (freeze untouched).

Objective:
Compares the native InsightFace similarity of final post-swap faces to the certified
character anchor against the four independent DeepFace judges:
- Facenet512
- SFace
- GhostFaceNet
- ArcFace

Tests the hypothesis:
Does facial identity only match the swap model's native feature space (inswapper/InsightFace),
or is the identity transferred in a way that independent visual judges recognize?
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

import insightface
from insightface.app import FaceAnalysis
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.cli import assign_faces_to_characters

ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
DEV_FAST_SCENES = [1, 2, 4, 5]
DEV_FAST_SEEDS = [11, 23, 37]

STORYBOARD_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
    story_spec = json.load(f)
SCENE_MAP = {s["scene_number"]: s for s in story_spec["scenes"]}

def load_anchor_images() -> Dict[str, np.ndarray]:
    anchors = {}
    for p in ANCHORS_DIR.glob("*_ground_truth.png"):
        cid = p.stem.replace("_ground_truth", "")
        img = cv2.imread(str(p))
        if img is not None:
            anchors[cid] = img
    return anchors

def run_swap_sanity_check():
    print("=" * 80)
    print("MMCCP Phase R7: Swap Sanity Evaluation")
    print("=" * 80)

    # 1. Initialize InsightFace app (buffalo_l on CPU)
    print("Initializing InsightFace FaceAnalysis (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # 2. Initialize DeepFace independent ensemble
    print("Initializing DeepFace Biometric Ensemble (4 judges)...")
    verifier = BiometricEnsemble()
    auditor = SceneAuditor()

    # 3. Load anchor images and compute baseline embeddings
    anchor_imgs = load_anchor_images()
    print(f"Loaded {len(anchor_imgs)} certified character anchor images: {list(anchor_imgs.keys())}")

    anchor_embs_insight = {}
    anchor_embs_deepface = {}

    for cid, img in anchor_imgs.items():
        # InsightFace
        faces_if = app.get(img)
        if faces_if:
            emb = faces_if[0].embedding
            emb_norm = emb / np.linalg.norm(emb)
            anchor_embs_insight[cid] = emb_norm

        # DeepFace
        df_embs = verifier.extract_face_embeddings(img)
        anchor_embs_deepface[cid] = df_embs

    # 4. Evaluate Bundle iter3 renders
    iter3_dir = PROJECT_ROOT / "results" / "iterations" / "loop-W3W4-iter3-closeup-scale-fidelity" / "renders"
    print(f"Auditing face crops in: {iter3_dir}")

    records = []

    for s_num in DEV_FAST_SCENES:
        scene = SCENE_MAP[s_num]
        expected_chars = scene.get("characters_present", scene.get("characters", []))
        char_positions = scene.get("character_positions", {})

        for seed in DEV_FAST_SEEDS:
            img_path = iter3_dir / f"scene_{s_num}_seed_{seed}.png"
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

            # A. Detect faces with InsightFace on full image
            faces_img_if = app.get(img_bgr)

            for cid in expected_chars:
                tup = assigned.get(cid)
                if tup is None:
                    continue
                idx, face_item = tup
                crop = face_item.get("crop_bgr")
                if crop is None or crop.size == 0:
                    continue

                # Match closest InsightFace face detection
                rf_cx = face_item["bbox"]["x"] + face_item["bbox"]["w"] / 2.0
                rf_cy = face_item["bbox"]["y"] + face_item["bbox"]["h"] / 2.0
                sim_insight = None
                best_dist = float("inf")
                matched_if_face = None
                for if_f in faces_img_if:
                    if_cx = (if_f.bbox[0] + if_f.bbox[2]) / 2.0
                    if_cy = (if_f.bbox[1] + if_f.bbox[3]) / 2.0
                    dist = float(np.hypot(rf_cx - if_cx, rf_cy - if_cy))
                    if dist < best_dist and dist < max(face_item["bbox"]["w"], face_item["bbox"]["h"]) * 1.5:
                        best_dist = dist
                        matched_if_face = if_f

                if matched_if_face is not None and cid in anchor_embs_insight:
                    c_emb = matched_if_face.embedding
                    c_emb_norm = c_emb / np.linalg.norm(c_emb)
                    sim_insight = float(np.dot(anchor_embs_insight[cid], c_emb_norm))

                # B. DeepFace similarities to anchor
                df_crop_embs = verifier.extract_face_embeddings(crop)
                sims_deepface = {}
                for m in BiometricEnsemble.JUDGE_MODELS:
                    if m in df_crop_embs and m in anchor_embs_deepface.get(cid, {}):
                        v1 = anchor_embs_deepface[cid][m]
                        v2 = df_crop_embs[m]
                        sims_deepface[m] = float(np.dot(v1, v2))

                records.append({
                    "scene": s_num,
                    "seed": seed,
                    "character_id": cid,
                    "face_width_px": face_item["bbox"]["w"],
                    "insightface_similarity": round(sim_insight, 4) if sim_insight is not None else None,
                    "facenet512_similarity": round(sims_deepface.get("Facenet512", 0.0), 4),
                    "sface_similarity": round(sims_deepface.get("SFace", 0.0), 4),
                    "ghostfacenet_similarity": round(sims_deepface.get("GhostFaceNet", 0.0), 4),
                    "arcface_similarity": round(sims_deepface.get("ArcFace", 0.0), 4),
                    "deepface_ensemble_mean": round(float(np.mean(list(sims_deepface.values()))), 4) if sims_deepface else 0.0
                })

    print(f"\nEvaluated {len(records)} face crops.")

    # 5. Aggregate comparisons
    if_sims = [r["insightface_similarity"] for r in records if r["insightface_similarity"] is not None]
    fn_sims = [r["facenet512_similarity"] for r in records]
    sf_sims = [r["sface_similarity"] for r in records]
    gf_sims = [r["ghostfacenet_similarity"] for r in records]
    af_sims = [r["arcface_similarity"] for r in records]
    df_means = [r["deepface_ensemble_mean"] for r in records]

    print("\n" + "=" * 80)
    print("SWAP SANITY COMPARISON RESULTS:")
    print("=" * 80)
    print(f"Native InsightFace (buffalo_l) Mean Sim: {np.mean(if_sims):.4f} (std={np.std(if_sims):.4f})")
    print(f"DeepFace ArcFace Mean Sim:               {np.mean(af_sims):.4f} (std={np.std(af_sims):.4f})")
    print(f"DeepFace Facenet512 Mean Sim:            {np.mean(fn_sims):.4f} (std={np.std(fn_sims):.4f})")
    print(f"DeepFace SFace Mean Sim:                 {np.mean(sf_sims):.4f} (std={np.std(sf_sims):.4f})")
    print(f"DeepFace GhostFaceNet Mean Sim:          {np.mean(gf_sims):.4f} (std={np.std(gf_sims):.4f})")
    print(f"DeepFace 4-Judge Ensemble Mean:          {np.mean(df_means):.4f} (std={np.std(df_means):.4f})")

    # Correlations between InsightFace and independent judges
    if len(if_sims) == len(records):
        corr_fn = float(np.corrcoef(if_sims, fn_sims)[0, 1])
        corr_sf = float(np.corrcoef(if_sims, sf_sims)[0, 1])
        corr_gf = float(np.corrcoef(if_sims, gf_sims)[0, 1])
        corr_af = float(np.corrcoef(if_sims, af_sims)[0, 1])
        corr_ens = float(np.corrcoef(if_sims, df_means)[0, 1])
        print("\nCorrelation with native InsightFace similarity:")
        print(f"  ArcFace:      r = {corr_af:+.4f}")
        print(f"  Facenet512:   r = {corr_fn:+.4f}")
        print(f"  SFace:        r = {corr_sf:+.4f}")
        print(f"  GhostFaceNet: r = {corr_gf:+.4f}")
        print(f"  4-Judge Mean: r = {corr_ens:+.4f}")

    output_data = {
        "num_faces_evaluated": len(records),
        "means": {
            "insightface_buffalo_l": round(float(np.mean(if_sims)), 4) if if_sims else None,
            "deepface_arcface": round(float(np.mean(af_sims)), 4),
            "deepface_facenet512": round(float(np.mean(fn_sims)), 4),
            "deepface_sface": round(float(np.mean(sf_sims)), 4),
            "deepface_ghostfacenet": round(float(np.mean(gf_sims)), 4),
            "deepface_ensemble": round(float(np.mean(df_means)), 4)
        },
        "per_face_records": records
    }

    out_json = PROJECT_ROOT / "results" / "swap_sanity_check.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved swap sanity check report to: {out_json}")

if __name__ == "__main__":
    run_swap_sanity_check()
