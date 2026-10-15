"""
================================================================================
MMCCP: RUN STORYBOARD COMPLETELY FROM SCRATCH (0 INITIAL ANCHORS)
================================================================================
Supports:
--pipeline v1: Phase 3 baseline pipeline
--pipeline v2: Consistency v2 SOTA pipeline (M1-M10)
--mode fast|default|quality: execution mode presets
================================================================================
"""
import os
import sys
import io
import json
import time
import base64
import argparse

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
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


def run_v1_scratch():
    from app import render_phase3_scene
    print("=" * 90)
    print("  🚀 RUNNING BASELINE v1 FROM SCRATCH (0 INITIAL ANCHORS)")
    print("=" * 90)

    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
            print(f"  [CACHE WIPED] Removed previous '{CACHE_FILE.name}'. Starting with 0 anchors.")
        except Exception as e:
            print(f"  [WARNING] Could not delete cache file: {e}")

    fresh_anchor_dir = PROJECT_ROOT / "experiments" / "fresh_run_from_scratch" / "anchors"
    fresh_anchor_dir.mkdir(parents=True, exist_ok=True)
    EDU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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

    for char in HARDCODED_CHARACTERS:
        cid = char["id"]
        b64 = cache_data.get(cid)
        if b64:
            pil_img = decode_image_to_pil(b64)
            if pil_img:
                out_p = fresh_anchor_dir / f"{cid}_anchor.png"
                pil_img.save(str(out_p))

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
            print(f"     ✅ Scene {s_num} rendered in {elapsed_s:.1f}s")


def run_v2_scratch(mode: str = "default"):
    from consistency_v2.config import ConsistencyV2Config
    from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
    from story_engine.schema import StorySpec

    print("=" * 90)
    print(f"  🌟 RUNNING MMCCP CONSISTENCY v2 (SOTA) - MODE: {mode.upper()}")
    print("=" * 90)

    cfg = ConsistencyV2Config(mode=mode)
    pipeline = ConsistencyPipelineV2(config=cfg)

    # Load canonical storyboard
    story_file = PROJECT_ROOT / "storyboards" / "ebright.json"
    story = StorySpec.load_json(str(story_file))

    output_dir = PROJECT_ROOT / "edu_outputs" / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = pipeline.generate_story(
        story=story,
        output_dir=output_dir,
        seed=11,
        prefix="scene_"
    )
    total_time = time.time() - t0

    print("\n" + "=" * 90)
    print(f"  🎉 MMCCP v2 RENDERING COMPLETE IN {total_time/60:.2f} MINUTES!")
    print(f"  Outputs saved in: {output_dir}")
    print("=" * 90)

    # Sync to EDU_IMAGES_DIR for Streamlit UI and evaluate consistency against fresh visual DNA
    EDU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

    print("\n" + "=" * 90)
    print("  📊 EVALUATING CONSISTENCY ON FRESHLY GENERATED VISUAL DNA")
    print("=" * 90)
    print(f"{'Scene':<10} | {'Character':<20} | {'ArcFace':<10} | {'CLIP-I':<10} | {'SSIM':<10} | {'Palette':<10} | {'Composite':<10}")
    print("-" * 90)

    all_scores = []
    for s_idx, (img_cv, meta) in enumerate(results, start=1):
        if img_cv is not None:
            save_p = EDU_IMAGES_DIR / f"scene_{s_idx}.png"
            cv2.imwrite(str(save_p), img_cv)

            scene_hardcoded = next((s for s in HARDCODED_SCENES if s["scene_number"] == s_idx), None)
            if scene_hardcoded and cache_data:
                chars = scene_hardcoded.get("characters", [])
                char_profiles = [c for c in HARDCODED_CHARACTERS if c["id"] in chars]
                reports = []
                if len(chars) > 1:
                    reports, _ = evaluate_multicharacter_scene(
                        anchors_map=cache_data,
                        scene_input=img_cv,
                        characters_info=char_profiles
                    )
                elif len(chars) == 1 and char_profiles:
                    cid = chars[0]
                    a_b64 = cache_data.get(cid)
                    if a_b64:
                        rep = evaluate_character_consistency(
                            anchor_input=a_b64,
                            scene_input=img_cv,
                            char_id=cid,
                            char_name=char_profiles[0]["name"]
                        )
                        reports = [rep]

                for rep in reports:
                    all_scores.append(rep.composite_score)
                    print(f"{s_idx:<10} | {rep.character_name[:20]:<20} | {rep.arcface_score:<10.4f} | {rep.clip_score:<10.4f} | {rep.structural_score:<10.4f} | {rep.palette_score:<10.4f} | {rep.composite_score:<10.4f}")

    if all_scores:
        avg_comp = sum(all_scores) / len(all_scores)
        print("=" * 90)
        print(f"  🏆 OVERALL COMPOSITE CONSISTENCY SCORE: {avg_comp:.4f} ({avg_comp*100:.1f}%)")
        print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MMCCP Storyboard Headless Runner")
    parser.add_argument(
        "--pipeline",
        choices=["v1", "v2"],
        default="v2",
        help="Select pipeline architecture: 'v1' (baseline Phase 3) or 'v2' (Consistency v2 SOTA)"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "default", "quality"],
        default="default",
        help="Execution mode preset for v2 pipeline ('fast', 'default', 'quality')"
    )

    args = parser.parse_args()

    if args.pipeline == "v2":
        run_v2_scratch(mode=args.mode)
    else:
        run_v1_scratch()


if __name__ == "__main__":
    main()
