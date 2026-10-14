import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from metrics import (
    HARDCODED_CHARACTERS,
    HARDCODED_SCENES,
    HARDCODED_ENVIRONMENTS,
    evaluate_character_consistency,
    evaluate_multicharacter_scene,
    decode_image_to_cv2,
    encode_image_to_base64,
    ConsistencyReport
)

def run_clean_run_evaluation():
    print("=" * 90)
    print("  EXECUTING COMPREHENSIVE BIOMETRIC & MULTIMODAL CONSISTENCY QA AUDIT")
    print("=" * 90)

    ground_truth_dir = ROOT_DIR / "experiments" / "clean_run_ground_truth" / "anchors"
    gt_json_path = ground_truth_dir / "ground_truth_anchors.json"
    
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_anchors_b64 = json.load(f)

    images_dir = ROOT_DIR / "edu_outputs" / "images"
    
    evaluation_results = []
    
    for scene in HARDCODED_SCENES:
        s_num = scene["scene_number"]
        s_id = scene["scene_id"]
        s_title = scene["title"]
        char_ids = scene["characters"]
        
        img_path = images_dir / f"scene_{s_num}.png"
        if not img_path.exists():
            print(f"[ERROR] Missing scene image: {img_path}")
            continue
            
        scene_cv = cv2.imread(str(img_path))
        if scene_cv is None:
            print(f"[ERROR] Could not read scene image: {img_path}")
            continue

        char_profiles = [c for c in HARDCODED_CHARACTERS if c["id"] in char_ids]
        char_anchors = [gt_anchors_b64.get(cid, "") for cid in char_ids]
        
        print(f"\nEvaluating Scene {s_num}: '{s_title}' (Characters: {char_ids})...")
        
        if len(char_ids) == 1:
            rep = evaluate_character_consistency(
                anchor_input=char_anchors[0],
                scene_input=scene_cv,
                char_id=char_ids[0],
                char_name=char_profiles[0]["name"]
            )
            reports = [rep]
        else:
            reports, sim_mat = evaluate_multicharacter_scene(
                anchors_map=gt_anchors_b64,
                scene_input=scene_cv,
                characters_info=char_profiles
            )

        for rep in reports:
            rep_dict = {
                "scene_number": s_num,
                "scene_id": s_id,
                "scene_title": s_title,
                "character_id": rep.character_id,
                "character_name": rep.character_name,
                "status": rep.status,
                "detected_faces_count": rep.detected_faces_count,
                "matched_face_index": rep.matched_face_index,
                "arcface_score": round(float(rep.arcface_score), 4),
                "clip_score": round(float(rep.clip_score), 4),
                "structural_score": round(float(rep.structural_score), 4),
                "palette_score": round(float(rep.palette_score), 4),
                "composite_score": round(float(rep.composite_score), 4),
                "diagnostics": rep.diagnostics
            }
            evaluation_results.append(rep_dict)
            print(f"  -> [{rep.character_name}] ArcFace: {rep.arcface_score:.4f} | CLIP-I: {rep.clip_score:.4f} | Struct: {rep.structural_score:.4f} | Palette: {rep.palette_score:.4f} | Composite: {rep.composite_score:.4f}")

    # Summary calculations
    arcface_scores = [r["arcface_score"] for r in evaluation_results if r["arcface_score"] > 0]
    clip_scores = [r["clip_score"] for r in evaluation_results]
    struct_scores = [r["structural_score"] for r in evaluation_results]
    palette_scores = [r["palette_score"] for r in evaluation_results]
    composite_scores = [r["composite_score"] for r in evaluation_results]

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluations": len(evaluation_results),
        "mean_arcface": round(float(np.mean(arcface_scores)), 4) if arcface_scores else 0.0,
        "mean_clip_i": round(float(np.mean(clip_scores)), 4),
        "mean_structural_ssim": round(float(np.mean(struct_scores)), 4),
        "mean_palette": round(float(np.mean(palette_scores)), 4),
        "mean_composite": round(float(np.mean(composite_scores)), 4),
        "evaluations": evaluation_results
    }

    report_json_path = ROOT_DIR / "experiments" / "clean_user_generation_final_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print(f"  OVERALL CLEAN RUN METRIC SUMMARY (N={len(evaluation_results)})")
    print("=" * 90)
    print(f"  Mean ArcFace Cosine Similarity:  {summary['mean_arcface']:.4f}")
    print(f"  Mean CLIP-I Image Similarity:    {summary['mean_clip_i']:.4f}")
    print(f"  Mean Structural SSIM Score:      {summary['mean_structural_ssim']:.4f}")
    print(f"  Mean Color Palette Fidelity:     {summary['mean_palette']:.4f}")
    print(f"  Mean Composite Consistency:      {summary['mean_composite']:.4f} ({summary['mean_composite']*100:.1f}%)")
    print(f"  Saved JSON report to:            {report_json_path}")
    print("=" * 90)

if __name__ == "__main__":
    run_clean_run_evaluation()
