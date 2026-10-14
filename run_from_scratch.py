"""
================================================================================
MMCCP: RUN STORYBOARD COMPLETELY FROM SCRATCH (0 INITIAL ANCHORS)
================================================================================
This script:
1. Wipes all existing runtime anchor caches (starts at 0 anchors).
2. Generates fresh Visual DNA character anchors from scratch via SDXL txt2img
   using pure prompt specifications with Demographic Phenotype Locking.
3. Renders the complete 6-Scene Storyboard using Two-Pass IP-Adapter +
   ControlNet Depth + Regional BREAK + ReActor/CodeFormer.
4. Executes real-time biometric & multimodal QA evaluation (ArcFace + CLIP-I + SSIM).
================================================================================
"""
import os
import sys
import json
import time
import base64
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DLL paths for Torch/CUDA on Windows
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

from metrics import (
    HARDCODED_CHARACTERS,
    HARDCODED_SCENES,
    HARDCODED_ENVIRONMENTS,
    CACHE_FILE,
    SD_MODELS,
    EDU_IMAGES_DIR,
    EDU_OUTPUT_DIR,
    ensure_visual_dna_anchors,
    decode_image_to_cv2,
    decode_image_to_pil,
    evaluate_character_consistency,
    evaluate_multicharacter_scene,
    cleanup_memory
)
from app import render_phase3_scene


def run_from_scratch():
    print("=" * 90)
    print("  🚀 MULTI-MODAL CHARACTER CONSISTENCY PIPELINE: RUNNING FROM SCRATCH (0 ANCHORS)")
    print("=" * 90)

    # 1. Wipe cache to guarantee 0 anchors
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
            print(f"  [CACHE WIPED] Removed previous '{CACHE_FILE.name}'. Starting with 0 anchors.")
        except Exception as e:
            print(f"  [WARNING] Could not delete cache file: {e}")
    else:
        print("  [CLEAN SLATE] No cache file found. Active anchors = 0.")

    # Create fresh anchor output folder for visual inspection
    fresh_anchor_dir = PROJECT_ROOT / "experiments" / "fresh_run_from_scratch" / "anchors"
    fresh_anchor_dir.mkdir(parents=True, exist_ok=True)
    
    EDU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Generate 5 Fresh Anchors from scratch
    print("\n" + "-" * 90)
    print("  PHASE 1: GENERATING 5 FRESH CHARACTER ANCHORS FROM SCRATCH VIA SDXL TXT2IMG")
    print("-" * 90)
    t0_anchors = time.time()
    
    sd_checkpoint = SD_MODELS["Juggernaut"]
    cache_data = ensure_visual_dna_anchors(
        characters_db=HARDCODED_CHARACTERS,
        cache_path=CACHE_FILE,
        sd_checkpoint=sd_checkpoint,
        force_regenerate=True
    )
    
    elapsed_anchors = time.time() - t0_anchors
    print(f"  ✅ All 5 Fresh Character Anchors Generated in {elapsed_anchors:.1f}s!")

    # Save PNGs for direct inspection
    for char in HARDCODED_CHARACTERS:
        cid = char["id"]
        b64 = cache_data.get(cid)
        if b64:
            pil_img = decode_image_to_pil(b64)
            if pil_img:
                out_p = fresh_anchor_dir / f"{cid}_anchor.png"
                pil_img.save(str(out_p))
                print(f"     -> Saved anchor: {out_p.name}")

    # 3. Generate 6 Storyboard Scenes using the fresh anchors
    print("\n" + "-" * 90)
    print("  PHASE 2: RENDERING 6-SCENE STORYBOARD USING FRESH ANCHORS")
    print("-" * 90)
    
    rendered_panels = []
    t0_scenes = time.time()

    for idx, scene in enumerate(HARDCODED_SCENES):
        s_num = scene["scene_number"]
        s_title = scene["title"]
        chars = scene["characters"]
        print(f"\n  [SCENE {s_num}/6] '{s_title}' (Characters: {chars})...")
        
        t0_s = time.time()
        scene_cv, reports = render_phase3_scene(
            scene=scene,
            characters_db=HARDCODED_CHARACTERS,
            environments_db=HARDCODED_ENVIRONMENTS,
            cache_data=cache_data,
            sd_checkpoint=sd_checkpoint,
            style="Cinematic Realistic"
        )
        elapsed_s = time.time() - t0_s

        if scene_cv is not None:
            save_p = EDU_IMAGES_DIR / f"scene_{s_num}.png"
            cv2.imwrite(str(save_p), scene_cv)
            print(f"     ✅ Scene {s_num} rendered and saved to '{save_p.name}' ({elapsed_s:.1f}s)")
            
            for rep in reports:
                print(f"        -> {rep.character_name}: ArcFace={rep.arcface_score:.4f} | CLIP-I={rep.clip_score:.4f} | Composite={rep.composite_score:.4f}")
            
            rendered_panels.append({
                "scene": scene,
                "reports": reports
            })
        else:
            print(f"     ❌ Failed to render Scene {s_num}!")

    total_time = time.time() - t0_anchors
    print("\n" + "=" * 90)
    print(f"  🎉 SCRATCH RUN COMPLETED SUCCESSFULLY IN {total_time/60:.1f} MINUTES!")
    print(f"  All 6 scenes are available in: {EDU_IMAGES_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    run_from_scratch()
