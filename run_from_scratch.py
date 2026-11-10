"""
================================================================================
MMCCP: RUN STORYBOARD COMPLETELY FROM SCRATCH (0 INITIAL ANCHORS)
================================================================================
Supports:
--pipeline v1: Phase 3 baseline pipeline
--pipeline v2: Consistency v2 SOTA pipeline (M1-M10)
--mode fast|default|quality: execution mode presets
--story: path to storyboard JSON specification
================================================================================
"""
import os
import sys
import io
import json
import time
import base64
import argparse
from typing import Optional

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

from consistency_v2.runtime_paths import get_project_root, configure_windows_forge_env

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
configure_windows_forge_env()

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


def run_v1_scratch(story_path: str = "storyboards/ebright.json"):
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


def run_v2_scratch(
    mode: str = "default",
    story_path: str = "storyboards/ebright.json",
    scene_num: Optional[int] = None,
    seed: int = 11,
):
    from consistency_v2.config import ConsistencyV2Config
    from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
    from consistency_v2.splits import assert_seed_allowed
    from story_engine.schema import StorySpec

    assert_seed_allowed(seed)

    print("=" * 90)
    print(f"  🌟 RUNNING MMCCP CONSISTENCY v2 (SOTA) - MODE: {mode.upper()} | SEED: {seed}")
    print("=" * 90)

    cfg = ConsistencyV2Config(mode=mode)
    cfg.validate_seed(seed)

    # Isolate cold-start scratch banks BEFORE pipeline init so canonical consistency_v2/banks is never touched
    scratch_bank_root = PROJECT_ROOT / "experiments" / "fresh_run_from_scratch" / "banks"
    scratch_bank_root.mkdir(parents=True, exist_ok=True)
    pipeline = ConsistencyPipelineV2(config=cfg, bank_root=scratch_bank_root)

    # Load storyboard
    story_file = Path(story_path) if Path(story_path).is_absolute() else (PROJECT_ROOT / story_path)
    story = StorySpec.load_json(str(story_file))

    target_scenes = [s for s in story.scenes if scene_num is None or s.scene_number == scene_num]
    if not target_scenes:
        raise ValueError(f"Scene {scene_num} not found in {story_file}")

    # Determine characters actually needed for target_scenes (or full story)
    needed_cids = []
    for sc in target_scenes:
        for cid in (sc.characters_present or []):
            if cid not in needed_cids:
                needed_cids.append(cid)
    chars_to_bootstrap = needed_cids if scene_num is not None else (
        list(story.characters.keys()) if isinstance(story.characters, dict) else [c.id for c in story.characters]
    )
    missing_banks = [cid for cid in chars_to_bootstrap if cid not in pipeline.bank_manager._banks]
    if missing_banks:
        print(f"  [BOOTSTRAP] Identity banks empty for {missing_banks}. Generating certified fresh anchors...")
        fresh_anchor_dir = PROJECT_ROOT / "experiments" / "fresh_run_from_scratch" / "anchors"
        fresh_anchor_dir.mkdir(parents=True, exist_ok=True)
        sd_checkpoint = SD_MODELS["Juggernaut"]
        target_chars = [c for c in HARDCODED_CHARACTERS if c["id"] in missing_banks] or HARDCODED_CHARACTERS
        cache_data = ensure_visual_dna_anchors(
            characters_db=target_chars,
            cache_path=CACHE_FILE,
            sd_checkpoint=sd_checkpoint,
            force_regenerate=(scene_num is None)
        )
        for char in target_chars:
            cid = char["id"]
            b64 = cache_data.get(cid)
            if b64:
                pil_img = decode_image_to_pil(b64)
                if pil_img:
                    out_p = fresh_anchor_dir / f"{cid}_anchor.png"
                    pil_img.save(str(out_p))
                    pipeline.bank_manager.initialize_character_bank(cid, [out_p])
        print("  ✅ Identity banks initialized from fresh certified anchors!")

    output_dir = PROJECT_ROOT / "edu_outputs" / f"v2_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    for sc in target_scenes:
        img_cv, meta = pipeline.generate_scene(story=story, scene=sc, seed=seed)
        results.append((sc.scene_number, img_cv, meta))
        if img_cv is not None:
            img_path = output_dir / f"scene_{sc.scene_number}_seed_{seed}.png"
            sidecar_path = output_dir / f"scene_{sc.scene_number}_seed_{seed}.json"
            cv2.imwrite(str(img_path), img_cv)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
    total_time = time.time() - t0

    print("\n" + "=" * 90)
    print(f"  🎉 MMCCP v2 ({mode}) RENDERING COMPLETE IN {total_time/60:.2f} MINUTES!")
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
    for s_idx, img_cv, meta in results:
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
    parser.add_argument(
        "--story",
        default="storyboards/ebright.json",
        help="Path to storyboard JSON specification"
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="Optional single scene number to render (for fast DEV smoke tests)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed (DEV splits only; reserved/TEST-2 seeds blocked)"
    )

    args = parser.parse_args()

    if args.pipeline == "v2":
        run_v2_scratch(mode=args.mode, story_path=args.story, scene_num=args.scene, seed=args.seed)
    else:
        run_v1_scratch(story_path=args.story)


if __name__ == "__main__":
    main()
