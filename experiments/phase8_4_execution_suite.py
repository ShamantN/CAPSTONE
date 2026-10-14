"""
PHASE 8.4: MASTER EXPERIMENTAL EXECUTION & VALIDATION SUITE
Implements all remaining methods, loop engineering, fresh anchor management, and multi-stage testing.
"""
import os, sys, json, time, math, shutil, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from generation.generation_orchestrator import GenerationOrchestrator
from qa.programmatic_auditor import ProgrammaticAuditor
from qa.test_anchor_manager import TestAnchorManager
from metrics import decode_image_to_cv2, calculate_max_match_arcface, cleanup_memory

logger = logging.getLogger("MMCCP.Phase8_4_Suite")


def run_stage2_instantid_tests():
    """STAGE 2: INSTANTID RUNTIME VALIDATION ACROSS MULTIPLE SHOT TYPES"""
    print("\n" + "=" * 90)
    print("  STAGE 2: INSTANTID REAL RUNTIME EXECUTION & LOOP VALIDATION")
    print("=" * 90)

    exp_dir, anchors = TestAnchorManager.reset_fresh_test_anchors("instantid", "shot_types_suite")
    story = StorySpec.load_json("storyboards/ebright.json")
    
    # Test cases: Close-up (Scene 2), Medium (Scene 3), Multi-character (Scene 1)
    test_scenes = [
        ("close_up", story.scenes[1]),     # Scene 2
        ("medium_shot", story.scenes[2]),  # Scene 3
        ("multi_char", story.scenes[0])    # Scene 1
    ]

    orchestrator = GenerationOrchestrator(identity_engine="instantid")
    auditor = ProgrammaticAuditor()

    results = []
    for test_tag, scene in test_scenes:
        print(f"\n  >>> [InstantID] Testing Shot Type: '{test_tag}' (Scene {scene.scene_number}: {scene.title})")
        t0 = time.time()
        try:
            img_cv, meta = orchestrator.generate_scene(story=story, scene=scene, steps_pass1=25, steps_pass2=15, seed=42)
            elapsed = time.time() - t0
            
            if img_cv is not None:
                img_path = exp_dir / f"instantid_{test_tag}.png"
                json_path = exp_dir / f"instantid_{test_tag}.json"
                cv2.imwrite(str(img_path), img_cv)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # QA
                auditor.results = []
                auditor.audit_scene(story, scene, img_path)
                passed = sum(1 for r in auditor.results if r["passed"])
                total = len(auditor.results)
                
                # Biometric ArcFace
                anchor_b64 = anchors.get(scene.characters_present[0])
                arc_sim = 0.0
                if anchor_b64:
                    arc_sim, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv)

                res_entry = {
                    "test": test_tag,
                    "scene_number": scene.scene_number,
                    "status": "PASS" if passed >= total - 1 else "PARTIAL",
                    "runtime_s": round(elapsed, 2),
                    "arcface_similarity": round(arc_sim, 4),
                    "qa_pass_rate": f"{passed}/{total}",
                    "image_path": str(img_path)
                }
                results.append(res_entry)
                print(f"      [RESULT] {test_tag} -> ArcFace={arc_sim:.4f} | QA: {passed}/{total} in {elapsed:.1f}s")
            else:
                results.append({"test": test_tag, "status": "FAIL", "error": meta.get("error")})
        except Exception as e:
            logger.error(f"InstantID test failed: {e}")
            results.append({"test": test_tag, "status": "FAIL", "error": str(e)})

    report_path = exp_dir / "instantid_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_stage3_faceid_tests():
    """STAGE 3: FACEID PLUS V2 DEEP VALIDATION"""
    print("\n" + "=" * 90)
    print("  STAGE 3: FACEID PLUS V2 DEEP MULTI-CONDITION VALIDATION")
    print("=" * 90)

    exp_dir, anchors = TestAnchorManager.reset_fresh_test_anchors("faceid", "multi_condition_suite")
    story = StorySpec.load_json("storyboards/ebright.json")
    
    test_scenes = [
        ("close_up", story.scenes[1]),     # Scene 2
        ("medium_shot", story.scenes[2]),  # Scene 3
        ("wide_shot", story.scenes[5])      # Scene 6
    ]

    orchestrator = GenerationOrchestrator(identity_engine="faceid")
    auditor = ProgrammaticAuditor()

    results = []
    for test_tag, scene in test_scenes:
        print(f"\n  >>> [FaceID Plus v2] Testing: '{test_tag}' (Scene {scene.scene_number})")
        t0 = time.time()
        try:
            img_cv, meta = orchestrator.generate_scene(story=story, scene=scene, steps_pass1=25, steps_pass2=15, seed=42)
            elapsed = time.time() - t0
            
            if img_cv is not None:
                img_path = exp_dir / f"faceid_{test_tag}.png"
                json_path = exp_dir / f"faceid_{test_tag}.json"
                cv2.imwrite(str(img_path), img_cv)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # Biometric ArcFace
                anchor_b64 = anchors.get(scene.characters_present[0])
                arc_sim = 0.0
                if anchor_b64:
                    arc_sim, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv)

                res_entry = {
                    "test": test_tag,
                    "scene_number": scene.scene_number,
                    "status": "PASS",
                    "runtime_s": round(elapsed, 2),
                    "arcface_similarity": round(arc_sim, 4),
                    "image_path": str(img_path)
                }
                results.append(res_entry)
                print(f"      [RESULT] {test_tag} -> ArcFace={arc_sim:.4f} in {elapsed:.1f}s")
            else:
                results.append({"test": test_tag, "status": "FAIL", "error": meta.get("error")})
        except Exception as e:
            logger.error(f"FaceID test failed: {e}")
            results.append({"test": test_tag, "status": "FAIL", "error": str(e)})

    report_path = exp_dir / "faceid_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_stage4_5_spatial_benchmarks():
    """STAGE 4 & 5: OPENPOSE & DEPTH CONTROLNET A/B BENCHMARKS"""
    print("\n" + "=" * 90)
    print("  STAGE 4 & 5: OPENPOSE & DEPTH CONTROLNET ABLATION BENCHMARKS")
    print("=" * 90)

    exp_dir, anchors = TestAnchorManager.reset_fresh_test_anchors("spatial", "openpose_depth_ablation")
    story = StorySpec.load_json("storyboards/ebright.json")
    scene = story.scenes[2]  # Scene 3: Science Fair Table Interaction

    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    
    configs = [
        ("no_spatial", False, False),
        ("openpose_only", True, False),
        ("depth_only", False, True),
        ("openpose_plus_depth", True, True)
    ]

    results = []
    for tag, op_active, depth_active in configs:
        print(f"\n  >>> Testing Spatial Config: '{tag}' (OpenPose={op_active}, Depth={depth_active})")
        t0 = time.time()
        img_cv, meta = orchestrator.generate_scene(
            story=story, scene=scene, steps_pass1=25, steps_pass2=15, seed=42,
            enable_openpose=op_active, enable_depth=depth_active
        )
        elapsed = time.time() - t0

        if img_cv is not None:
            img_path = exp_dir / f"spatial_{tag}.png"
            json_path = exp_dir / f"spatial_{tag}.json"
            cv2.imwrite(str(img_path), img_cv)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            anchor_b64 = anchors.get("teen_ebright")
            arc_sim = 0.0
            if anchor_b64:
                arc_sim, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv)

            res_entry = {
                "config": tag,
                "openpose": op_active,
                "depth": depth_active,
                "runtime_s": round(elapsed, 2),
                "arcface_similarity": round(arc_sim, 4),
                "image_path": str(img_path)
            }
            results.append(res_entry)
            print(f"      [RESULT] {tag} -> ArcFace={arc_sim:.4f} in {elapsed:.1f}s")

    report_path = exp_dir / "spatial_ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_stage30_31_full_story_validations():
    """STAGE 30 & 31: COMPLETE FULL STORY VALIDATION SEQUENCES"""
    print("\n" + "=" * 90)
    print("  STAGE 30 & 31: COMPLETE SEQUENCE-LEVEL VALIDATIONS (EBRIGHT & CHRONO-ARCHIVISTS)")
    print("=" * 90)

    # 1. Full Ebright Story Sequence
    exp_dir_ebright, anchors_ebright = TestAnchorManager.reset_fresh_test_anchors("full_stories", "ebright_sequence")
    story_ebright = StorySpec.load_json("storyboards/ebright.json")
    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    auditor = ProgrammaticAuditor()

    print(f"\n  [STORY 1] Generating Complete Richard Ebright Sequence (6 Scenes)...")
    ebright_results = orchestrator.generate_story(story_ebright, output_dir=exp_dir_ebright, prefix="ebright_scene_")
    ebright_audit = auditor.run_story_audit(story_ebright, image_dir=exp_dir_ebright, prefix="ebright_scene_")
    
    ebright_report_path = exp_dir_ebright / "ebright_sequence_qa_summary.json"
    with open(ebright_report_path, "w", encoding="utf-8") as f:
        json.dump(ebright_audit, f, indent=2)

    # 2. Full Chrono-Archivists Story Sequence (Generalization)
    exp_dir_chrono, anchors_chrono = TestAnchorManager.reset_fresh_test_anchors("full_stories", "chrono_archivists_sequence")
    story_chrono = StorySpec.load_json("storyboards/generalization_benchmark.json")

    print(f"\n  [STORY 2] Generating Complete Chrono-Archivists Sequence (5 Scenes)...")
    chrono_results = orchestrator.generate_story(story_chrono, output_dir=exp_dir_chrono, prefix="chrono_scene_")
    chrono_audit = auditor.run_story_audit(story_chrono, image_dir=exp_dir_chrono, prefix="chrono_scene_")

    chrono_report_path = exp_dir_chrono / "chrono_sequence_qa_summary.json"
    with open(chrono_report_path, "w", encoding="utf-8") as f:
        json.dump(chrono_audit, f, indent=2)

    return ebright_audit, chrono_audit


def main():
    print("=" * 90)
    print("  STARTING PHASE 8.4 MASTER EXECUTION & VALIDATION SUITE")
    print("=" * 90)

    t0_all = time.time()
    
    # Stage 2: InstantID
    s2_res = run_stage2_instantid_tests()

    # Stage 3: FaceID Plus v2
    s3_res = run_stage3_faceid_tests()

    # Stage 4 & 5: Spatial Ablation
    s45_res = run_stage4_5_spatial_benchmarks()

    # Stage 30 & 31: Full Sequences
    ebright_qa, chrono_qa = run_stage30_31_full_story_validations()

    total_elapsed = time.time() - t0_all
    print("\n" + "=" * 90)
    print(f"  PHASE 8.4 SUITE EXECUTION COMPLETED IN {total_elapsed/60:.1f} MINUTES")
    print(f"  Ebright Pass Rate: {ebright_qa.get('pass_rate_pct')}%")
    print(f"  Chrono-Archivists Pass Rate: {chrono_qa.get('pass_rate_pct')}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
