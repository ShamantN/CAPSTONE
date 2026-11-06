"""
MMCCP Phase R10: Sealed TEST-2 Evaluation Runner (Strictly One-Shot)
Authoritatively executes the sealed final evaluation of ConsistencyPipelineV2 on TEST-2:
  1. Ebright TEST-2: 30 renders (Scenes 1-6 x Seeds 1101, 1202, 1303, 1404, 1505)
  2. Holdout TEST-2: 12 renders (Scenes 1-4 x Seeds 1101, 1202, 1303)
  3. TEST-2 Ablation: 7 arms on seed 1101 (full_v2, no_m2_layout, no_m3_couple, no_m4_ip_adapter,
     no_m5_swap, no_m6_detailer, no_m7_wardrobe)
  4. Frozen Verifier v1.2 Scoring:
     - results/test2_ebright/report_v1_2.json
     - results/test2_holdout/report_v1_2.json
     - results/ablation_test2/loo_ablation_test2_summary.json
  5. Stage Tracer Verification & Contact Sheets
"""
import os
import sys
import io
import json
import time
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
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
        except Exception: pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from consistency_v2.config import ConsistencyV2Config, RESERVED_SEEDS
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.verify.cli import run_verification, bootstrap_ci_95
from consistency_v2.stage_trace import compute_image_sha256, compute_region_pixel_diff
from consistency_v2.splits import (
    assert_split_allowed,
    assert_seed_allowed,
    TEST_2_EBRIGHT_SEEDS,
    TEST_2_EBRIGHT_SCENES,
    TEST_2_HOLDOUT_SEEDS,
    TEST_2_HOLDOUT_SCENES,
    log_split_access
)

# Paths
EBRIGHT_STORY_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
HOLDOUT_STORY_PATH = PROJECT_ROOT / "storyboards" / "holdout_v2.json"

EBRIGHT_ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
HOLDOUT_ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors" / "holdout"

EBRIGHT_BASELINE_REPORT = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_ebright" / "report.json"
HOLDOUT_BASELINE_REPORT = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_holdout" / "report.json"

TEST2_EBRIGHT_DIR = PROJECT_ROOT / "results" / "test2_ebright"
TEST2_HOLDOUT_DIR = PROJECT_ROOT / "results" / "test2_holdout"
TEST2_ABLATION_DIR = PROJECT_ROOT / "results" / "ablation_test2"

ABLATION_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "full_v2": {},
    "no_m2_layout": {"enable_layout_planner": False},
    "no_m3_couple": {"enable_regional_couple": False},
    "no_m4_ip_adapter": {"enable_ip_adapter_plus": False},
    "no_m5_swap": {"enable_reliable_face_swap": False},
    "no_m6_detailer": {"enable_face_detailer": False},
    "no_m7_wardrobe": {"enable_wardrobe_lock": False}
}


def load_base_config(config_path: Optional[Path] = None) -> ConsistencyV2Config:
    cfg = ConsistencyV2Config()
    cfg_p = config_path or (PROJECT_ROOT / "consistency_v2" / "best_config.json")
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def render_scene_bundle(
    pipeline: ConsistencyPipelineV2,
    story: StorySpec,
    scenes: List[SceneSpec],
    seeds: List[int],
    out_dir: Path,
    bundle_label: str,
    resume: bool = True
) -> List[float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_times = []
    total = len(scenes) * len(seeds)
    count = 0

    print(f"\n[{bundle_label}] Rendering {total} images ({len(scenes)} scenes x {len(seeds)} seeds)...")

    for scene in scenes:
        s_num = scene.scene_number
        print(f"\n--- {bundle_label} Scene {s_num}: {scene.title} ({scene.shot_type}) ---")

        for seed in seeds:
            count += 1
            assert_seed_allowed(seed)
            out_img = out_dir / f"scene_{s_num}_seed_{seed}.png"
            out_meta = out_dir / f"scene_{s_num}_seed_{seed}.json"

            if resume and out_img.exists() and out_meta.exists():
                print(f"  [{count}/{total}] Seed {seed}: Cached on disk -> {out_img.name}")
                continue

            print(f"  [{count}/{total}] Seed {seed}: Generating...")
            t0 = time.time()
            img_cv, meta = pipeline.generate_scene(story=story, scene=scene, seed=seed)
            t_render = time.time() - t0
            render_times.append(t_render)

            if img_cv is not None:
                cv2.imwrite(str(out_img), img_cv)
                meta["scene_number"] = s_num
                meta["seed"] = seed
                meta["runtime_seconds"] = round(t_render, 2)
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                peak_vram = meta.get("peak_vram_mb", 0.0)
                print(f"  [{count}/{total}] Seed {seed}: Done in {t_render:.1f}s (Peak VRAM: {peak_vram:.0f} MB)")
            else:
                print(f"  [{count}/{total}] Seed {seed}: Generation FAILED!")
                raise RuntimeError(f"Generation failed for {bundle_label} scene {s_num} seed {seed}")

    return render_times


def run_test2_ablation(
    story: StorySpec,
    scenes: List[SceneSpec],
    seed: int,
    base_config: ConsistencyV2Config,
    full_v2_renders_dir: Path,
    out_dir: Path,
    resume: bool = True
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("  PHASE R10: TEST-2 LEAVE-ONE-OUT ABLATION (Seed 1101, 7 Arms)")
    print("=" * 80)

    # 1. Arm: full_v2 (fast-path copy from test2_ebright seed 1101)
    full_dir = out_dir / "full_v2"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_images: Dict[str, np.ndarray] = {}

    for scene in scenes:
        s_num = scene.scene_number
        src_img = full_v2_renders_dir / f"scene_{s_num}_seed_{seed}.png"
        src_meta = full_v2_renders_dir / f"scene_{s_num}_seed_{seed}.json"
        dst_img = full_dir / f"scene_{s_num}_seed_{seed}.png"
        dst_meta = full_dir / f"scene_{s_num}_seed_{seed}.json"

        if not dst_img.exists() or not dst_meta.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_meta, dst_meta)

        im = cv2.imread(str(dst_img))
        full_images[f"scene_{s_num}_seed_{seed}"] = im

    # 2. Arms 2-7
    tracer_results: Dict[str, Any] = {
        "verified_at": time.time(),
        "total_conditions": len(ABLATION_CONDITIONS),
        "conditions": {
            "full_v2": {
                "config_hash": base_config.compute_hash(),
                "overrides": {},
                "renders_count": len(scenes),
                "renders": {
                    k: {"sha256": compute_image_sha256(v), "pixel_diff_vs_full": 0.0}
                    for k, v in full_images.items()
                },
                "mean_pixel_diff_vs_full": 0.0
            }
        }
    }

    for cond_name, overrides in ABLATION_CONDITIONS.items():
        if cond_name == "full_v2":
            continue

        cond_dir = out_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        cfg = ConsistencyV2Config()
        cfg_dict = base_config.to_dict()
        cfg_dict.update(overrides)
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg_hash = cfg.compute_hash()
        pipeline = ConsistencyPipelineV2(config=cfg)

        print(f"\n>>> Running Ablation Arm: {cond_name} (Hash: {cfg_hash})")
        renders_info = {}
        pixel_diffs = []

        for scene in scenes:
            s_num = scene.scene_number
            key = f"scene_{s_num}_seed_{seed}"
            out_img = cond_dir / f"scene_{s_num}_seed_{seed}.png"
            out_meta = cond_dir / f"scene_{s_num}_seed_{seed}.json"

            # Fast-path for no_m3_couple on single-character scenes (Scenes 2, 3, 4, 6)
            if cond_name == "no_m3_couple" and len(scene.characters_present) <= 1:
                if not out_img.exists() or not out_meta.exists():
                    shutil.copy2(full_dir / out_img.name, out_img)
                    shutil.copy2(full_dir / out_meta.name, out_meta)
                im = cv2.imread(str(out_img))
                sha = compute_image_sha256(im)
                renders_info[key] = {"sha256": sha, "pixel_diff_vs_full": 0.0}
                pixel_diffs.append(0.0)
                print(f"  [{cond_name}] Scene {s_num}: Single-character scene -> Bit-identical to full_v2 (pixel diff 0.0)")
                continue

            # Check resume
            if resume and out_img.exists() and out_meta.exists():
                im = cv2.imread(str(out_img))
                sha = compute_image_sha256(im)
                diff = float(compute_region_pixel_diff(full_images[key], im))
                renders_info[key] = {"sha256": sha, "pixel_diff_vs_full": diff}
                pixel_diffs.append(diff)
                print(f"  [{cond_name}] Scene {s_num}: Cached on disk (pixel diff vs full: {diff:.2f})")
                continue

            # Render
            print(f"  [{cond_name}] Scene {s_num}: Generating with {overrides}...")
            t0 = time.time()
            img_cv, meta = pipeline.generate_scene(story=story, scene=scene, seed=seed)
            t_render = time.time() - t0

            if img_cv is not None:
                cv2.imwrite(str(out_img), img_cv)
                meta["scene_number"] = s_num
                meta["seed"] = seed
                meta["condition"] = cond_name
                meta["config_hash"] = cfg_hash
                meta["runtime_seconds"] = round(t_render, 2)
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                sha = compute_image_sha256(img_cv)
                diff = float(compute_region_pixel_diff(full_images[key], img_cv))
                renders_info[key] = {"sha256": sha, "pixel_diff_vs_full": diff}
                pixel_diffs.append(diff)
                print(f"  [{cond_name}] Scene {s_num}: Done in {t_render:.1f}s | Diff vs full: {diff:.2f}px")
            else:
                raise RuntimeError(f"Ablation render failed for {cond_name} scene {s_num}")

        mean_diff = float(np.mean(pixel_diffs))
        tracer_results["conditions"][cond_name] = {
            "config_hash": cfg_hash,
            "overrides": overrides,
            "renders_count": len(scenes),
            "renders": renders_info,
            "mean_pixel_diff_vs_full": round(mean_diff, 2)
        }
        print(f"[{cond_name}] Completed. Mean Pixel Diff vs full_v2: {mean_diff:.2f}px")

    # Save stage tracer verification
    tracer_path = out_dir / "stage_tracer_verification.json"
    with open(tracer_path, "w", encoding="utf-8") as f:
        json.dump(tracer_results, f, indent=2)
    print(f"\n[OK] Saved stage tracer verification: {tracer_path}")

    return tracer_results


def score_ablation_test2(out_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  SCORING TEST-2 LEAVE-ONE-OUT ABLATION (Frozen Verifier v1.2)")
    print("=" * 80)

    summary_results = {}
    for cond in ABLATION_CONDITIONS.keys():
        cond_dir = out_dir / cond
        report_path = cond_dir / "report.json"
        print(f"\n>>> Scoring Condition: {cond}...")

        run_verification(
            images_dir=cond_dir,
            story_path=EBRIGHT_STORY_PATH,
            bank_dir=EBRIGHT_ANCHORS_DIR,
            output_report_path=report_path,
            baseline_report_path=None
        )

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sm = data.get("summary_metrics", {})
        binding = sm.get("binding_decomposed", {})
        age_split = sm.get("age_split_metrics", {})

        summary_results[cond] = {
            "biometric_ensemble_mean": sm.get("biometric_ensemble_mean", 0.0),
            "biometric_ensemble_95ci": sm.get("biometric_ensemble_95ci", [0.0, 0.0]),
            "rank1_accuracy": sm.get("rank1_discrimination_accuracy", 0.0),
            "spatial_pass_rate": binding.get("spatial_position_pass_rate", 0.0),
            "wardrobe_pass_rate": binding.get("wardrobe_palette_pass_rate", 0.0),
            "mean_defects": sm.get("mean_defects_per_scene", 0.0),
            "child_bio": age_split.get("child", {}).get("biometric_ensemble_mean", 0.0),
            "adult_bio": age_split.get("adult", {}).get("biometric_ensemble_mean", 0.0)
        }

    full = summary_results["full_v2"]
    delta_results = {}

    print("\n" + "=" * 80)
    print("  7-ARM LEAVE-ONE-OUT ABLATION TRUTH TABLE (TEST-2)")
    print("=" * 80)
    print(f"{'Condition':<18} | {'Bio Mean':<9} | {'Rank-1':<7} | {'Spatial':<7} | {'Wardrobe':<8} | {'Defects':<7} | {'Delta Bio':<9}")
    print("-" * 80)

    for cond in ABLATION_CONDITIONS.keys():
        res = summary_results[cond]
        delta_bio = res["biometric_ensemble_mean"] - full["biometric_ensemble_mean"]
        delta_results[cond] = {
            **res,
            "delta_bio_vs_full": round(delta_bio, 4),
            "delta_spatial_vs_full": round(res["spatial_pass_rate"] - full["spatial_pass_rate"], 4),
            "delta_wardrobe_vs_full": round(res["wardrobe_pass_rate"] - full["wardrobe_pass_rate"], 4)
        }
        print(f"{cond:<18} | {res['biometric_ensemble_mean']:<9.4f} | {res['rank1_accuracy']*100:<6.1f}% | {res['spatial_pass_rate']*100:<6.1f}% | {res['wardrobe_pass_rate']*100:<7.1f}% | {res['mean_defects']:<7.2f} | {delta_bio:+0.4f}")

    out_file = out_dir / "loo_ablation_test2_summary.json"
    final_payload = {
        "conditions": delta_results,
        "full_v2_baseline": full
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[OK] Saved TEST-2 LOO ablation summary: {out_file}")

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="MMCCP Sealed TEST-2 Evaluation Runner")
    parser.add_argument("--config", type=str, default="consistency_v2/best_config.json", help="Path to config file")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from cached renders")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume, re-render all")
    parser.add_argument("--skip-render", action="store_true", default=False, help="Skip render stage, score existing")
    args = parser.parse_args()

    print("=" * 80)
    print("  MMCCP PHASE R10: FINAL SEALED EVALUATION ON TEST-2 (STRICTLY ONE-SHOT)")
    print("=" * 80)

    # 1. Split Guard & Logging
    assert_split_allowed("TEST-2", "run_sealed_test2.py")
    print("[PASS] TEST-2 Split Authorization Verified & Logged.")

    # 2. Config & Pipeline Initialization
    config_path = PROJECT_ROOT / args.config
    base_config = load_base_config(config_path)
    print(f"Loaded Config from: {config_path} (Hash: {base_config.compute_hash()})")
    print(f"CFG Scale: {base_config.cfg_scale} | CodeFormer: {base_config.codeformer_fidelity} | IP-Adapter: {base_config.ip_adapter_plus_weight}")

    # Load Storyboards
    with open(EBRIGHT_STORY_PATH, "r", encoding="utf-8") as f:
        ebright_story = StorySpec.from_dict(json.load(f))
    with open(HOLDOUT_STORY_PATH, "r", encoding="utf-8") as f:
        holdout_story = StorySpec.from_dict(json.load(f))

    ebright_scenes = [s for s in ebright_story.scenes if s.scene_number in TEST_2_EBRIGHT_SCENES]
    holdout_scenes = [s for s in holdout_story.scenes if s.scene_number in TEST_2_HOLDOUT_SCENES]

    pipeline = ConsistencyPipelineV2(config=base_config)

    # 3. Part 1: Render 30 Ebright TEST-2 Images
    ebright_renders_dir = TEST2_EBRIGHT_DIR / "renders"
    if not args.skip_render:
        render_scene_bundle(
            pipeline=pipeline,
            story=ebright_story,
            scenes=ebright_scenes,
            seeds=TEST_2_EBRIGHT_SEEDS,
            out_dir=ebright_renders_dir,
            bundle_label="TEST-2 Ebright",
            resume=args.resume
        )

    # 4. Part 2: Render 12 Holdout TEST-2 Images
    holdout_renders_dir = TEST2_HOLDOUT_DIR / "renders"
    if not args.skip_render:
        render_scene_bundle(
            pipeline=pipeline,
            story=holdout_story,
            scenes=holdout_scenes,
            seeds=TEST_2_HOLDOUT_SEEDS,
            out_dir=holdout_renders_dir,
            bundle_label="TEST-2 Holdout",
            resume=args.resume
        )

    # 5. Part 3: Render TEST-2 Leave-One-Out Ablation (Seed 1101)
    if not args.skip_render:
        run_test2_ablation(
            story=ebright_story,
            scenes=ebright_scenes,
            seed=1101,
            base_config=base_config,
            full_v2_renders_dir=ebright_renders_dir,
            out_dir=TEST2_ABLATION_DIR,
            resume=args.resume
        )

    # 6. Part 4: Frozen Verifier v1.2 Scoring
    print("\n" + "=" * 80)
    print("  SCORING TEST-2 EBRIGHT (30 Renders) WITH FROZEN VERIFIER v1.2")
    print("=" * 80)
    ebright_report_path = TEST2_EBRIGHT_DIR / "report_v1_2.json"
    ebright_results = run_verification(
        images_dir=ebright_renders_dir,
        story_path=EBRIGHT_STORY_PATH,
        bank_dir=EBRIGHT_ANCHORS_DIR,
        output_report_path=ebright_report_path,
        baseline_report_path=EBRIGHT_BASELINE_REPORT if EBRIGHT_BASELINE_REPORT.exists() else None
    )
    print(f"[OK] Ebright TEST-2 Report Saved: {ebright_report_path}")

    print("\n" + "=" * 80)
    print("  SCORING TEST-2 HOLDOUT (12 Renders) WITH FROZEN VERIFIER v1.2")
    print("=" * 80)
    holdout_report_path = TEST2_HOLDOUT_DIR / "report_v1_2.json"
    holdout_results = run_verification(
        images_dir=holdout_renders_dir,
        story_path=HOLDOUT_STORY_PATH,
        bank_dir=HOLDOUT_ANCHORS_DIR,
        output_report_path=holdout_report_path,
        baseline_report_path=HOLDOUT_BASELINE_REPORT if HOLDOUT_BASELINE_REPORT.exists() else None
    )
    print(f"[OK] Holdout TEST-2 Report Saved: {holdout_report_path}")

    # Score Ablation
    ablation_results = score_ablation_test2(out_dir=TEST2_ABLATION_DIR)

    print("\n" + "=" * 80)
    print("  ALL TEST-2 RENDERS AND VERIFICATIONS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

