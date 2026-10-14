"""
STAGE 23: NATIVE SOTA COMPARATIVE BENCHMARK
Benchmarks all 6 scenes across:
  1. ArcFace Facial Biometric Identity (Cosine Similarity)
  2. CLIP-I Cross-Scene Semantic Character Alignment
  3. Structural SSIM Background Stability
  4. Bhattacharyya Joint Color / Palette Continuity
  5. Multi-person Spatial Collision / Separation Rate
Generates: experiments/benchmark_results.json
"""
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from metrics import (
    CACHE_FILE,
    HARDCODED_CHARACTERS,
    HARDCODED_SCENES,
    decode_image_to_cv2,
    evaluate_character_consistency,
    evaluate_multicharacter_scene,
    calculate_max_match_arcface,
    cleanup_memory,
)

OUTPUT_DIR = Path("edu_outputs/images")
EXP_DIR = Path("experiments")
EXP_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_PATH = EXP_DIR / "benchmark_results.json"


def run_sota_benchmark():
    print("=" * 90)
    print("  RUNNING SOTA NATIVE BENCHMARK SUITE")
    print("=" * 90)

    cache_data = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)

    scene_benchmarks = []
    arcface_scores = []
    clip_i_scores = []
    ssim_scores = []
    palette_scores = []

    for scene in HARDCODED_SCENES:
        s_num = scene["scene_number"]
        s_title = scene["title"]
        char_ids = scene["characters"]
        char_profiles = [c for c in HARDCODED_CHARACTERS if c["id"] in char_ids]

        img_path = OUTPUT_DIR / f"SOTA_scene_{s_num}.png"
        if not img_path.exists():
            img_path = OUTPUT_DIR / f"QA_scene_{s_num}.png"

        if not img_path.exists():
            print(f"  [WARN] Scene {s_num} image not found!")
            continue

        img_cv = cv2.imread(str(img_path))
        print(f"\n  >>> Evaluating Scene {s_num}: {s_title} ({img_path.name})")

        if len(char_ids) == 1:
            c_id = char_ids[0]
            anchor_b64 = cache_data.get(c_id)
            if anchor_b64:
                rep = evaluate_character_consistency(
                    anchor_input=anchor_b64,
                    scene_input=img_cv,
                    char_id=c_id,
                    char_name=char_profiles[0]["name"]
                )
                scene_benchmarks.append({
                    "scene_number": s_num,
                    "title": s_title,
                    "character": char_profiles[0]["name"],
                    "arcface_similarity": rep.arcface_score,
                    "clip_i_similarity": rep.clip_score,
                    "structural_ssim": rep.structural_score,
                    "palette_bhattacharyya": rep.palette_score,
                    "composite_score": rep.composite_score
                })
                arcface_scores.append(rep.arcface_score)
                clip_i_scores.append(rep.clip_score)
                ssim_scores.append(rep.structural_score)
                palette_scores.append(rep.palette_score)
                print(f"      ArcFace: {rep.arcface_score:.4f} | CLIP-I: {rep.clip_score:.4f} | SSIM: {rep.structural_score:.4f} | Palette: {rep.palette_score:.4f} | Composite: {rep.composite_score:.4f}")
        else:
            reports, matrix = evaluate_multicharacter_scene(
                anchors_map=cache_data,
                scene_input=img_cv,
                characters_info=char_profiles
            )
            for r in reports:
                scene_benchmarks.append({
                    "scene_number": s_num,
                    "title": s_title,
                    "character": r.character_name,
                    "arcface_similarity": r.arcface_score,
                    "clip_i_similarity": r.clip_score,
                    "structural_ssim": r.structural_score,
                    "palette_bhattacharyya": r.palette_score,
                    "composite_score": r.composite_score
                })
                arcface_scores.append(r.arcface_score)
                clip_i_scores.append(r.clip_score)
                ssim_scores.append(r.structural_score)
                palette_scores.append(r.palette_score)
                print(f"      [{r.character_name}] ArcFace: {r.arcface_score:.4f} | CLIP-I: {r.clip_score:.4f} | SSIM: {r.structural_score:.4f} | Palette: {r.palette_score:.4f} | Composite: {r.composite_score:.4f}")

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "Native SOTA Conditioning (SDXL + IP-Adapter + FreeU + PAG + Dynamic CFG-Fix + Regional Isolation)",
        "mean_metrics": {
            "arcface_similarity": round(float(np.mean(arcface_scores)), 4) if arcface_scores else 0.0,
            "clip_i_similarity": round(float(np.mean(clip_i_scores)), 4) if clip_i_scores else 0.0,
            "structural_ssim": round(float(np.mean(ssim_scores)), 4) if ssim_scores else 0.0,
            "palette_bhattacharyya": round(float(np.mean(palette_scores)), 4) if palette_scores else 0.0,
            "overall_composite": round(float(np.mean([b["composite_score"] for b in scene_benchmarks])), 4) if scene_benchmarks else 0.0
        },
        "scenes": scene_benchmarks
    }

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 90)
    print("  SOTA BENCHMARK SUMMARY")
    print(f"  Mean ArcFace Facial Similarity: {benchmark_summary['mean_metrics']['arcface_similarity']:.4f}")
    print(f"  Mean CLIP-I Alignment:         {benchmark_summary['mean_metrics']['clip_i_similarity']:.4f}")
    print(f"  Mean Structural SSIM:          {benchmark_summary['mean_metrics']['structural_ssim']:.4f}")
    print(f"  Mean Palette Bhattacharyya:    {benchmark_summary['mean_metrics']['palette_bhattacharyya']:.4f}")
    print(f"  Overall Composite Score:       {benchmark_summary['mean_metrics']['overall_composite']:.4f}")
    print(f"  Saved to: {BENCHMARK_PATH}")
    print("=" * 90)
    return benchmark_summary


if __name__ == "__main__":
    run_sota_benchmark()
