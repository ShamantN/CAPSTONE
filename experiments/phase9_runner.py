"""
PHASE 9: SOTA METHODS EXPERIMENTATION, COMPARISON, & ABLATION RUNNER
Executes controlled A/B tests across Identity Engines, ControlNets, LoRAs, Refinements, and Sequences.
"""
import sys, os, json, time, shutil, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from generation.generation_orchestrator import GenerationOrchestrator
from qa.programmatic_auditor import ProgrammaticAuditor
from qa.test_anchor_manager import TestAnchorManager
from metrics import (
    decode_image_to_cv2,
    calculate_max_match_arcface,
    calculate_clip_i_similarity,
    calculate_structural_similarity,
    cleanup_memory
)

logger = logging.getLogger("MMCCP.Phase9Runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_EXP_DIR = Path(r"D:\5TH SEM\HACKATHON\experiments\phase9_validation")


def evaluate_generated_image(
    img_cv: np.ndarray,
    anchor_b64: str,
    story: StorySpec,
    scene: SceneSpec
) -> Dict[str, Any]:
    """Computes Quad-Modal metrics: ArcFace, CLIP-I, Structural SSIM, and Programmatic QA."""
    anchor_cv = decode_image_to_cv2(anchor_b64)
    arc_sim, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv, expected_faces=len(scene.characters_present))
    clip_sim = calculate_clip_i_similarity(anchor_cv, img_cv)
    ssim_val = calculate_structural_similarity(anchor_cv, img_cv)
    composite = round(0.45 * arc_sim + 0.25 * clip_sim + 0.15 * ssim_val + 0.15 * 0.60, 4)

    auditor = ProgrammaticAuditor()
    tmp_path = BASE_EXP_DIR / "temp_eval.png"
    cv2.imwrite(str(tmp_path), img_cv)
    auditor.audit_scene(story, scene, tmp_path)
    if tmp_path.exists():
        tmp_path.unlink()

    qa_passed = sum(1 for r in auditor.results if r["passed"])
    qa_total = len(auditor.results)
    cleanup_memory(force=True)

    return {
        "arcface": round(float(arc_sim), 4),
        "clip_i": round(float(clip_sim), 4),
        "structural_ssim": round(float(ssim_val), 4),
        "composite": composite,
        "qa_passed": qa_passed,
        "qa_total": qa_total,
        "qa_rate_pct": round(qa_passed / max(1, qa_total) * 100, 1)
    }


def run_identity_engine_comparisons():
    """
    PHASES 2, 3, 4: Controlled comparison of Generic IP-Adapter vs InstantID vs FaceID Plus V2.
    Tests across:
      1. Close-up (Scene 2: The Butterfly Basement)
      2. Medium Shot (Scene 3: The County Science Fair)
      3. Multi-Character (Scene 1: The Dining Room Classroom)
    """
    print("\n" + "=" * 95)
    print("  PHASE 2-4: CONTROLLED IDENTITY ENGINE A/B BENCHMARK (IP-Adapter vs InstantID vs FaceID)")
    print("=" * 95)

    suite_dir = BASE_EXP_DIR / "identity_ab_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    story = StorySpec.load_json("storyboards/ebright.json")
    test_scenes = [
        ("close_up", story.scenes[1]),     # Scene 2
        ("medium_shot", story.scenes[2]),  # Scene 3
        ("multi_char", story.scenes[0])    # Scene 1
    ]

    engines = ["ip_adapter", "instantid", "faceid"]
    results = {}

    for eng in engines:
        print(f"\n>>> Running Identity Engine: {eng.upper()}")
        orchestrator = GenerationOrchestrator(identity_engine=eng)
        eng_results = []

        for tag, scene in test_scenes:
            img_p = suite_dir / f"{eng}_{tag}.png"
            json_p = suite_dir / f"{eng}_{tag}.json"
            if img_p.exists() and json_p.exists():
                print(f"    - Shot '{tag}' for {eng} already generated. Loading existing record...")
                with open(json_p, "r", encoding="utf-8") as f:
                    eng_results.append(json.load(f))
                continue

            print(f"    - Testing Shot '{tag}' (Scene {scene.scene_number})...")
            t0 = time.time()
            img_cv, meta = orchestrator.generate_scene(
                story=story,
                scene=scene,
                steps_pass1=20,
                steps_pass2=12,
                seed=42
            )
            dur = round(time.time() - t0, 2)

            if img_cv is not None:
                img_p = suite_dir / f"{eng}_{tag}.png"
                json_p = suite_dir / f"{eng}_{tag}.json"
                cv2.imwrite(str(img_p), img_cv)

                anchor_b64 = orchestrator.registry.get_anchor_b64(scene.characters_present[0])
                metrics_res = evaluate_generated_image(img_cv, anchor_b64, story, scene)

                record = {
                    "engine": eng,
                    "shot_type": tag,
                    "scene_number": scene.scene_number,
                    "runtime_s": dur,
                    "image_file": str(img_p.name),
                    "metrics": metrics_res
                }
                with open(json_p, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

                eng_results.append(record)
                print(f"      [RESULT] ArcFace: {metrics_res['arcface']} | CLIP-I: {metrics_res['clip_i']} | SSIM: {metrics_res['structural_ssim']} | QA: {metrics_res['qa_passed']}/{metrics_res['qa_total']} ({dur}s)")
            else:
                eng_results.append({"engine": eng, "shot_type": tag, "status": "FAILED", "meta": meta})

        results[eng] = eng_results

    summary_file = suite_dir / "identity_ab_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Identity A/B comparison summary saved to {summary_file}")
    return results


def run_controlnet_spatial_ablations():
    """
    PHASES 5 & 6: ControlNet OpenPose and Depth controlled ablations.
    A. Baseline Identity Only
    B. Identity + OpenPose
    C. Identity + Depth
    D. Identity + OpenPose + Depth
    """
    print("\n" + "=" * 95)
    print("  PHASE 5 & 6: CONTROLNET SPATIAL ABLATION BENCHMARK (Identity vs +OpenPose vs +Depth vs Both)")
    print("=" * 95)

    suite_dir = BASE_EXP_DIR / "spatial_ablation_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    story = StorySpec.load_json("storyboards/ebright.json")
    scene = story.scenes[3]  # Scene 4: The Starling Experiment (dynamic pose & outdoors)
    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    anchor_b64 = orchestrator.registry.get_anchor_b64(scene.characters_present[0])

    configs = [
        ("identity_only", False, False),
        ("identity_plus_openpose", True, False),
        ("identity_plus_depth", False, True),
        ("identity_openpose_depth", True, True)
    ]

    results = []
    for tag, use_op, use_depth in configs:
        img_p = suite_dir / f"{tag}.png"
        json_p = suite_dir / f"{tag}.json"
        if img_p.exists() and json_p.exists():
            print(f"    - Spatial configuration '{tag}' already generated. Loading existing record...")
            with open(json_p, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        print(f"\n>>> Testing Configuration: '{tag}' (OpenPose={use_op}, Depth={use_depth})...")
        t0 = time.time()
        img_cv, meta = orchestrator.generate_scene(
            story=story,
            scene=scene,
            steps_pass1=20,
            steps_pass2=12,
            seed=42,
            enable_openpose=use_op,
            enable_depth=use_depth
        )
        dur = round(time.time() - t0, 2)

        if img_cv is not None:
            img_p = suite_dir / f"{tag}.png"
            json_p = suite_dir / f"{tag}.json"
            cv2.imwrite(str(img_p), img_cv)

            metrics_res = evaluate_generated_image(img_cv, anchor_b64, story, scene)
            rec = {
                "config": tag,
                "openpose": use_op,
                "depth": use_depth,
                "runtime_s": dur,
                "image_file": str(img_p.name),
                "metrics": metrics_res
            }
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            results.append(rec)
            print(f"      [RESULT] ArcFace: {metrics_res['arcface']} | CLIP-I: {metrics_res['clip_i']} | SSIM: {metrics_res['structural_ssim']} | QA: {metrics_res['qa_passed']}/{metrics_res['qa_total']} ({dur}s)")
        else:
            results.append({"config": tag, "status": "FAILED", "meta": meta})

    summary_file = suite_dir / "spatial_ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Spatial ablation summary saved to {summary_file}")
    return results


def run_lora_sweep():
    """
    PHASE 7: Dynamic LoRA Weight Sweeps (0.0, 0.25, 0.50, 0.75, 1.00).
    """
    print("\n" + "=" * 95)
    print("  PHASE 7: DYNAMIC LORA WEIGHT SWEEP (0.0 -> 1.0)")
    print("=" * 95)

    suite_dir = BASE_EXP_DIR / "lora_sweep_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    story = StorySpec.load_json("storyboards/ebright.json")
    scene = story.scenes[1]  # Scene 2
    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    anchor_b64 = orchestrator.registry.get_anchor_b64(scene.characters_present[0])

    weights = [0.0, 0.25, 0.50, 0.75, 1.00]
    results = []

    for w in weights:
        tag = f"lora_weight_{int(w*100):03d}"
        img_p = suite_dir / f"{tag}.png"
        json_p = suite_dir / f"{tag}.json"
        if img_p.exists() and json_p.exists():
            print(f"    - LoRA weight {w} already generated. Loading existing record...")
            with open(json_p, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        print(f"\n>>> Testing LoRA Weight: {w}...")
        lora_token = f" <lora:sdxl_lightning_8step_lora:{w}>" if w > 0.0 else ""
        
        # Inject LoRA token into prompt override
        plan = orchestrator.registry
        t0 = time.time()
        
        # Build payload directly with prompt mod
        from story_engine.scene_planner import ScenePlanner
        p = ScenePlanner.plan_scene(story, scene, orchestrator.registry)
        p.positive_prompt += lora_token

        # Generate via standard orchestrator client
        cnet_specs = orchestrator.identity_conditioner.build_control_specs(p)
        from conditioning.controlnet_conditioner import ControlNetConditioner
        cnet_payload = ControlNetConditioner.build_payload(cnet_specs)

        alwayson_scripts = {
            "freeu integrated (sd 1.x, sd 2.x, sdxl)": {"args": [True, 1.3, 1.4, 0.9, 0.2, 0.0, 1.0]},
            "perturbedattentionguidance integrated": {"args": [True, 3.0, 0.0, 0.0, 1.0]},
            "dynamicthresholding (cfg-fix) integrated": {"args": [True, 7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, "enable", "MEAN", "AD", 1.0]}
        }
        if cnet_payload.get("controlnet"):
            alwayson_scripts["controlnet"] = cnet_payload["controlnet"]

        txt2img_payload = {
            "prompt": p.positive_prompt,
            "negative_prompt": p.negative_prompt,
            "steps": 20,
            "width": p.width,
            "height": p.height,
            "cfg_scale": 5.5,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "seed": 42,
            "alwayson_scripts": alwayson_scripts
        }
        imgs, _ = orchestrator.client.txt2img(txt2img_payload)
        dur = round(time.time() - t0, 2)

        if imgs:
            img_cv = decode_image_to_cv2(imgs[0])
            img_p = suite_dir / f"{tag}.png"
            json_p = suite_dir / f"{tag}.json"
            cv2.imwrite(str(img_p), img_cv)

            metrics_res = evaluate_generated_image(img_cv, anchor_b64, story, scene)
            rec = {
                "lora_weight": w,
                "runtime_s": dur,
                "image_file": str(img_p.name),
                "metrics": metrics_res
            }
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            results.append(rec)
            print(f"      [RESULT] Weight {w} -> ArcFace: {metrics_res['arcface']} | CLIP-I: {metrics_res['clip_i']} | SSIM: {metrics_res['structural_ssim']} ({dur}s)")
        else:
            results.append({"lora_weight": w, "status": "FAILED"})

    summary_file = suite_dir / "lora_sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] LoRA sweep summary saved to {summary_file}")
    return results


def run_seed_robustness():
    """
    PHASE 22: Random-Seed Robustness across 5 seeds per major identity config.
    Reports Mean, Median, Min, Max, StdDev, and Failure Rate.
    """
    print("\n" + "=" * 95)
    print("  PHASE 22: RANDOM-SEED ROBUSTNESS TESTING (5 SEEDS)")
    print("=" * 95)

    suite_dir = BASE_EXP_DIR / "seed_robustness_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    story = StorySpec.load_json("storyboards/ebright.json")
    scene = story.scenes[1]  # Scene 2
    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    anchor_b64 = orchestrator.registry.get_anchor_b64(scene.characters_present[0])

    seeds = [42, 107, 219, 384, 501]
    scores = []
    results = []

    for s in seeds:
        img_p = suite_dir / f"seed_{s}.png"
        json_p = suite_dir / f"seed_{s}.json"
        if img_p.exists() and json_p.exists():
            print(f"    - Seed {s} already generated. Loading existing record...")
            with open(json_p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            scores.append(rec["metrics"]["arcface"])
            results.append(rec)
            continue

        print(f"    - Testing Seed {s}...")
        t0 = time.time()
        img_cv, meta = orchestrator.generate_scene(story=story, scene=scene, steps_pass1=20, steps_pass2=12, seed=s)
        dur = round(time.time() - t0, 2)
        if img_cv is not None:
            img_p = suite_dir / f"seed_{s}.png"
            cv2.imwrite(str(img_p), img_cv)
            metrics_res = evaluate_generated_image(img_cv, anchor_b64, story, scene)
            scores.append(metrics_res["arcface"])
            rec = {"seed": s, "runtime_s": dur, "image_file": str(img_p.name), "metrics": metrics_res}
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            results.append(rec)
            print(f"      [RESULT] Seed {s} -> ArcFace: {metrics_res['arcface']} | Composite: {metrics_res['composite']}")
        else:
            results.append({"seed": s, "status": "FAILED"})

    stats = {
        "seeds_tested": len(seeds),
        "mean_arcface": round(float(np.mean(scores)), 4) if scores else 0.0,
        "median_arcface": round(float(np.median(scores)), 4) if scores else 0.0,
        "min_arcface": round(float(np.min(scores)), 4) if scores else 0.0,
        "max_arcface": round(float(np.max(scores)), 4) if scores else 0.0,
        "std_arcface": round(float(np.std(scores)), 4) if scores else 0.0,
        "failure_rate_pct": 0.0 if len(scores) == len(seeds) else 100.0 * (len(seeds) - len(scores)) / len(seeds),
        "individual_runs": results
    }

    summary_file = suite_dir / "seed_robustness_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Saved] Seed robustness stats: Mean={stats['mean_arcface']}, Std={stats['std_arcface']}, Range=[{stats['min_arcface']}, {stats['max_arcface']}]")
    return stats


def run_full_phase9_suite():
    """Executes the full Phase 9 validation suite."""
    BASE_EXP_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    id_ab = run_identity_engine_comparisons()
    sp_ab = run_controlnet_spatial_ablations()
    lora_sw = run_lora_sweep()
    seed_rob = run_seed_robustness()

    master_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_s": round(time.time() - t_start, 2),
        "identity_ab": id_ab,
        "spatial_ablation": sp_ab,
        "lora_sweep": lora_sw,
        "seed_robustness": seed_rob
    }
    out_json = BASE_EXP_DIR / "phase9_master_benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\n{'='*95}")
    print(f"  ALL PHASE 9 BENCHMARK SUITES COMPLETED IN {master_summary['total_elapsed_s']}s!")
    print(f"  Summary written to: {out_json}")
    print(f"{'='*95}\n")


if __name__ == "__main__":
    run_full_phase9_suite()
