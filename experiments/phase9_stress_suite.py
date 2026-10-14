"""
PHASE 13 - 21: ADVANCED CHARACTER CONSISTENCY & STRESS TEST SUITE
Executes Sequence Testing, Anchor-to-Scene Matrix, Age Progression, Multi-Character Permutation,
Viewpoint, Lighting, Expression, and Clothing stress tests with Fresh Anchor lifecycle enforcement.
"""
import sys
import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from generation.generation_orchestrator import GenerationOrchestrator
from qa.vlm_visual_auditor import VLMVisualAuditor
from metrics import (
    decode_image_to_cv2,
    calculate_max_match_arcface,
    calculate_clip_i_similarity,
    calculate_structural_similarity,
    cleanup_memory
)

logger = logging.getLogger("MMCCP.Phase9StressSuite")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_STRESS_DIR = Path(r"D:\5TH SEM\HACKATHON\experiments\phase9_validation\stress_tests")


def run_age_progression_test(orchestrator: GenerationOrchestrator, story: StorySpec) -> Dict[str, Any]:
    """
    PHASE 15: Age Progression Testing (8 -> 15 -> 21).
    Evaluates: Young Ebright (Scene 2, age 8), Teen Ebright (Scene 3, age 15), College Ebright (Scene 6, age 21).
    """
    logger.info(">>> Running Phase 15 Age Progression Suite (8 -> 15 -> 21)...")
    out_dir = BASE_STRESS_DIR / "age_progression"
    out_dir.mkdir(parents=True, exist_ok=True)

    age_scenes = [
        ("age_08_child", story.scenes[1]),    # Scene 2: Young Ebright (8)
        ("age_15_teen", story.scenes[2]),     # Scene 3: Teen Ebright (15)
        ("age_21_adult", story.scenes[5])     # Scene 6: College Ebright (21)
    ]

    auditor = VLMVisualAuditor()
    results = []

    for label, scene in age_scenes:
        char_id = scene.characters_present[0]
        anchor_b64 = orchestrator.registry.get_anchor_b64(char_id)
        logger.info(f"    - Generating {label} (Scene {scene.scene_number}, Char: {char_id})...")
        t0 = time.time()
        img_cv, _ = orchestrator.generate_scene(story=story, scene=scene, steps_pass1=20, steps_pass2=10, seed=42)
        dur = round(time.time() - t0, 2)

        if img_cv is not None:
            img_p = out_dir / f"{label}.png"
            cv2.imwrite(str(img_p), img_cv)
            audit = auditor.audit_scene(anchor_b64, img_cv, story, scene)
            audit["label"] = label
            audit["runtime_s"] = dur
            audit["image_file"] = str(img_p.name)
            results.append(audit)
            logger.info(f"      [RESULT] {label} -> ArcFace: {audit['arcface_score']} | Confidence: {audit['identity_confidence']} | Caption: '{audit['scene_visual_caption']}'")

    summary_file = out_dir / "age_progression_report.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return {"age_progression": results}


def run_multi_character_spatial_permutation(orchestrator: GenerationOrchestrator, story: StorySpec) -> Dict[str, Any]:
    """
    PHASE 16: Multi-Character Spatial Permutation Testing.
    Run A: Young Ebright = LEFT, Mother = RIGHT
    Run B: Mother = LEFT, Young Ebright = RIGHT
    Verifies that identities follow the characters, NOT the spatial slot.
    """
    logger.info(">>> Running Phase 16 Multi-Character Permutation Suite...")
    out_dir = BASE_STRESS_DIR / "multi_char_permutation"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_scene = story.scenes[0]  # Scene 1: Dining Room Classroom
    auditor = VLMVisualAuditor()
    results = {}

    permutations = [
        ("pos_A_left_B_right", {"young_ebright": "left", "mother": "right"}),
        ("pos_B_left_A_right", {"mother": "left", "young_ebright": "right"})
    ]

    for tag, pos_map in permutations:
        test_scene = SceneSpec.from_dict(base_scene.to_dict())
        test_scene.character_positions = pos_map
        logger.info(f"    - Testing Permutation '{tag}': {pos_map}...")

        t0 = time.time()
        img_cv, _ = orchestrator.generate_scene(story=story, scene=test_scene, steps_pass1=20, steps_pass2=10, seed=42)
        dur = round(time.time() - t0, 2)

        if img_cv is not None:
            img_p = out_dir / f"{tag}.png"
            cv2.imwrite(str(img_p), img_cv)

            anchor_ebright = orchestrator.registry.get_anchor_b64("young_ebright")
            anchor_mother = orchestrator.registry.get_anchor_b64("mother")

            audit_ebright = auditor.audit_scene(anchor_ebright, img_cv, story, test_scene)
            audit_mother = auditor.audit_scene(anchor_mother, img_cv, story, test_scene)

            rec = {
                "permutation": tag,
                "positions": pos_map,
                "runtime_s": dur,
                "image_file": str(img_p.name),
                "ebright_arcface": audit_ebright["arcface_score"],
                "mother_arcface": audit_mother["arcface_score"],
                "detected_faces": audit_ebright["detected_faces_count"],
                "vlm_caption": audit_ebright["scene_visual_caption"],
                "separation_status": audit_ebright["multi_character_separation"]
            }
            results[tag] = rec
            logger.info(f"      [RESULT] {tag} -> Ebright ArcFace: {rec['ebright_arcface']} | Mother ArcFace: {rec['mother_arcface']} | Faces: {rec['detected_faces']}")

    summary_file = out_dir / "spatial_permutation_report.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_environmental_stress_tests(orchestrator: GenerationOrchestrator, story: StorySpec) -> Dict[str, Any]:
    """
    PHASES 17, 18, 19, 20: Viewpoint, Lighting, Expression, and Clothing stress tests.
    """
    logger.info(">>> Running Phases 17-20 Environmental & Visual Stress Tests...")
    out_dir = BASE_STRESS_DIR / "environmental_stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_scene = story.scenes[1]  # Scene 2
    anchor_b64 = orchestrator.registry.get_anchor_b64("young_ebright")
    auditor = VLMVisualAuditor()

    stress_variations = [
        ("viewpoint_three_quarter", {"shot_type": "Three-quarter view close-up portrait", "camera_angle": "slight 45 degree angle"}),
        ("viewpoint_high_angle", {"camera_angle": "high angle looking down"}),
        ("lighting_dramatic_sunset", {"lighting": "dramatic golden hour sunset with warm rim lighting and deep shadows"}),
        ("lighting_low_key", {"lighting": "moody low-key chiaroscuro lighting with candle illumination"}),
        ("expression_laughing_happy", {"action_description": "Young Richard Ebright is laughing happily with wide joyful smile holding a butterfly jar"}),
        ("expression_intense_focus", {"action_description": "Young Richard Ebright has an intensely serious, focused, determined analytical expression"}),
        ("clothing_winter_parka", {"character_outfits": {"young_ebright": "heavy navy blue hooded winter parka coat"}})
    ]

    results = []
    for tag, overrides in stress_variations:
        test_scene = SceneSpec.from_dict(base_scene.to_dict())
        for k, v in overrides.items():
            setattr(test_scene, k, v)

        logger.info(f"    - Testing Stress Condition '{tag}'...")
        t0 = time.time()
        img_cv, _ = orchestrator.generate_scene(story=story, scene=test_scene, steps_pass1=20, steps_pass2=10, seed=42)
        dur = round(time.time() - t0, 2)

        if img_cv is not None:
            img_p = out_dir / f"{tag}.png"
            cv2.imwrite(str(img_p), img_cv)

            audit = auditor.audit_scene(anchor_b64, img_cv, story, test_scene)
            rec = {
                "condition": tag,
                "overrides": overrides,
                "runtime_s": dur,
                "image_file": str(img_p.name),
                "arcface": audit["arcface_score"],
                "clip_i": audit["clip_i_score"],
                "confidence": audit["identity_confidence"],
                "same_character": audit["same_character"],
                "caption": audit["scene_visual_caption"]
            }
            results.append(rec)
            logger.info(f"      [RESULT] {tag} -> ArcFace: {rec['arcface']} | Confidence: {rec['confidence']} | Caption: '{rec['caption']}'")

    summary_file = out_dir / "environmental_stress_report.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return {"environmental_stress": results}


def run_full_stress_suite():
    """Executes the complete stress testing matrix."""
    BASE_STRESS_DIR.mkdir(parents=True, exist_ok=True)
    story = StorySpec.load_json("storyboards/ebright.json")
    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")

    t_start = time.time()
    age_res = run_age_progression_test(orchestrator, story)
    perm_res = run_multi_character_spatial_permutation(orchestrator, story)
    env_res = run_environmental_stress_tests(orchestrator, story)

    master_stress = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_s": round(time.time() - t_start, 2),
        "age_progression": age_res,
        "spatial_permutation": perm_res,
        "environmental_stress": env_res
    }
    master_file = BASE_STRESS_DIR / "phase9_stress_suite_summary.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_stress, f, indent=2)
    logger.info(f"\nAll Phase 9 Stress Tests Completed in {master_stress['total_elapsed_s']}s!")


if __name__ == "__main__":
    run_full_stress_suite()
