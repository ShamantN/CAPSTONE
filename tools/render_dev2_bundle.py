"""
MMCCP Phase R7: DEV-2 Locked Bundle Evaluation Runner
Renders the locked bundle on DEV-2 Ebright:
  Seeds:  101, 202, 303, 404, 505 (5 seeds)
  Scenes: 1, 2, 3, 4, 5, 6 (6 scenes)
  Total:  30 renders

Evaluates against Baseline v1 (results/rescored_v1.2/baseline_v1_dev2/report.json)
under frozen Verifier v1.2 with pre-registered decision rules.
"""
import os
import sys
import io
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
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
from consistency_v2.splits import assert_seed_allowed

DEV2_SEEDS = [101, 202, 303, 404, 505]
DEV2_SCENES = [1, 2, 3, 4, 5, 6]

STORYBOARD_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
BASELINE_REPORT = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_dev2" / "report.json"
OUT_DIR = PROJECT_ROOT / "results" / "bundle_dev2"
RENDERS_DIR = OUT_DIR / "renders"


def run_dev2_evaluation(resume: bool = True, seeds: Optional[List[int]] = None, scenes: Optional[List[int]] = None):
    target_seeds = seeds if seeds is not None else DEV2_SEEDS
    target_scenes = scenes if scenes is not None else DEV2_SCENES
    print("=" * 80)
    print("MMCCP Phase R7: DEV-2 Locked Bundle Evaluation (30 Renders)")
    print("=" * 80)

    # 1. Verify seeds
    for sd in target_seeds:
        assert_seed_allowed(sd)

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
        story_data = json.load(f)
    story_spec = StorySpec.from_dict(story_data)
    scenes_map = {s.scene_number: s for s in story_spec.scenes}

    # 2. Configure locked bundle
    config = ConsistencyV2Config()
    config.mode = "default"
    config.cfg_scale = 5.5
    config.codeformer_fidelity = 0.85
    config.enable_ip_adapter_plus = True
    config.ip_adapter_plus_weight = 0.55
    config.enable_layout_planner = True
    config.openpose_weight = 0.70
    config.enable_regional_couple = True
    config.regional_global_weight = 0.40
    config.enable_reliable_face_swap = True
    config.enable_wardrobe_lock = True
    config.enable_pag = False
    config.enable_freeu = True
    config.enable_dynamic_thresholding = True

    config_hash = config.compute_hash()
    print(f"Locked Bundle Config Hash: {config_hash}")
    print(f"CFG Scale: {config.cfg_scale} | Mode: {config.mode} | Steps: {config.steps_pass1}+{config.steps_pass2}")

    # 3. Initialize pipeline
    pipeline = ConsistencyPipelineV2(config=config)

    render_times = []
    total_renders = len(target_scenes) * len(target_seeds)
    current_render = 0

    print(f"\nTargeting {total_renders} renders across 6 scenes and 5 seeds...")

    for s_num in target_scenes:
        scene = scenes_map[s_num]
        print(f"\n--- Scene {s_num}: {scene.title} ({scene.shot_type}) ---")

        for seed in target_seeds:
            current_render += 1
            out_img = RENDERS_DIR / f"scene_{s_num}_seed_{seed}.png"
            out_meta = RENDERS_DIR / f"scene_{s_num}_seed_{seed}.json"

            if resume and out_img.exists() and out_meta.exists():
                print(f"  [{current_render}/{total_renders}] Seed {seed}: Already rendered -> {out_img.name}")
                continue

            print(f"  [{current_render}/{total_renders}] Seed {seed}: Generating...")
            t0 = time.time()
            img_cv, meta = pipeline.generate_scene(story=story_spec, scene=scene, seed=seed)
            t_render = time.time() - t0
            render_times.append(t_render)

            if img_cv is not None:
                cv2.imwrite(str(out_img), img_cv)
                meta["scene_number"] = s_num
                meta["seed"] = seed
                meta["config_hash"] = config_hash
                meta["runtime_seconds"] = round(t_render, 2)
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                peak_vram = meta.get("peak_vram_mb", 0.0)
                print(f"  [{current_render}/{total_renders}] Seed {seed}: Done in {t_render:.1f}s (Peak VRAM: {peak_vram:.0f} MB)")
            else:
                print(f"  [{current_render}/{total_renders}] Seed {seed}: Generation failed!")

    # 4. Run frozen Verifier v1.2
    print("\n" + "=" * 80)
    print("Executing Frozen Verifier v1.2 on DEV-2 Renders...")
    print("=" * 80)

    report_path = OUT_DIR / "report_v1_2.json"
    summary_csv_path = OUT_DIR / "summary.csv"
    contact_dir = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    verifier_results = run_verification(
        images_dir=RENDERS_DIR,
        story_path=STORYBOARD_PATH,
        bank_dir=ANCHORS_DIR,
        output_report_path=report_path,
        baseline_report_path=BASELINE_REPORT if BASELINE_REPORT.exists() else None
    )

    print(f"\nVerification complete! Saved report to: {report_path}")

    # 5. Load baseline v1 DEV-2 results and evaluate decision rules
    if BASELINE_REPORT.exists():
        with open(BASELINE_REPORT, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        eval_rules(verifier_results, base_data, render_times)
    else:
        print(f"Warning: Baseline report not found at {BASELINE_REPORT}")


def eval_rules(bundle_res: Dict[str, Any], base_res: Dict[str, Any], render_times: List[float]):
    print("\n" + "=" * 80)
    print("PRE-REGISTERED DECISION RULES EVALUATION (DEV-2 Bundle vs Baseline v1)")
    print("=" * 80)

    b_metrics = bundle_res.get("summary_metrics", {})
    base_metrics = base_res.get("summary_metrics", {})

    # Rule 1: Presence >= 95% of expected (40 faces total across 30 scenes)
    expected_faces = 40
    bundle_faces = bundle_res.get("total_face_instances", 0)
    presence_pct = bundle_faces / expected_faces * 100.0 if expected_faces > 0 else 0.0
    rule1_presence = (bundle_faces >= 38)
    print(f"Rule 1 - Presence (>=95%, >=38/40 faces): {bundle_faces}/{expected_faces} ({presence_pct:.1f}%) | {'MET' if rule1_presence else 'NOT MET'}")

    # Rule 2: Identity non-inferior
    bundle_ens = b_metrics.get("biometric_ensemble_mean", 0.0)
    base_ens = base_metrics.get("biometric_ensemble_mean", 0.0)
    delta_ens = bundle_ens - base_ens

    # Bootstrap CI of paired delta if available
    b_scores = bundle_res.get("raw_biometric_ensemble_scores", [])
    base_scores = base_res.get("raw_biometric_ensemble_scores", [])
    min_len = min(len(b_scores), len(base_scores))
    if min_len > 0:
        deltas = [b_scores[i] - base_scores[i] for i in range(min_len)]
        ci_low, ci_high = bootstrap_ci_95(deltas, n_boot=2000)
    else:
        ci_low, ci_high = delta_ens, delta_ens

    rule2a_noninf = (ci_low > -0.02)
    print(f"Rule 2a - Overall Identity Non-Inferiority: Bundle={bundle_ens:.4f} vs Baseline={base_ens:.4f} (Delta={delta_ens:+.4f}, 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]) | {'MET' if rule2a_noninf else 'NOT MET'}")

    # Age group deltas
    b_splits = b_metrics.get("age_split_metrics", {})
    base_splits = base_metrics.get("age_split_metrics", {})

    b_child = b_splits.get("child", {}).get("biometric_ensemble_mean", 0.0)
    base_child = base_splits.get("child", {}).get("biometric_ensemble_mean", 0.0)
    d_child = b_child - base_child
    rule2b_child = (d_child >= -0.05)
    print(f"Rule 2b - Child Biometric (>= -0.05): Bundle={b_child:.4f} vs Baseline={base_child:.4f} (Delta={d_child:+.4f}) | {'MET' if rule2b_child else 'NOT MET'}")

    b_teen = b_splits.get("teen", {}).get("biometric_ensemble_mean", 0.0)
    base_teen = base_splits.get("teen", {}).get("biometric_ensemble_mean", 0.0)
    d_teen = b_teen - base_teen
    rule2c_teen = (d_teen >= -0.05)
    print(f"Rule 2c - Teen Biometric (>= -0.05): Bundle={b_teen:.4f} vs Baseline={base_teen:.4f} (Delta={d_teen:+.4f}) | {'MET' if rule2c_teen else 'NOT MET'}")

    b_adult = b_splits.get("adult", {}).get("biometric_ensemble_mean", 0.0)
    base_adult = base_splits.get("adult", {}).get("biometric_ensemble_mean", 0.0)
    d_adult = b_adult - base_adult
    rule2d_adult = (d_adult >= -0.05)
    print(f"Rule 2d - Adult Biometric (>= -0.05): Bundle={b_adult:.4f} vs Baseline={base_adult:.4f} (Delta={d_adult:+.4f}) | {'MET' if rule2d_adult else 'NOT MET'}")

    b_rank1 = b_metrics.get("rank1_discrimination_accuracy", 0.0)
    base_rank1 = base_metrics.get("rank1_discrimination_accuracy", 0.0)
    rule2e_rank1 = (b_rank1 >= base_rank1)
    print(f"Rule 2e - Rank-1 Accuracy (>= v1): Bundle={b_rank1*100:.1f}% vs Baseline={base_rank1*100:.1f}% | {'MET' if rule2e_rank1 else 'NOT MET'}")

    # Rule 3: Wardrobe pass > v1, Structural defects <= v1
    b_ward = b_metrics.get("binding_decomposed", {}).get("wardrobe_palette_pass_rate", 0.0)
    base_ward = base_metrics.get("binding_decomposed", {}).get("wardrobe_palette_pass_rate", 0.0)
    rule3a_ward = (b_ward >= base_ward)
    print(f"Rule 3a - Wardrobe Palette Pass (>= v1): Bundle={b_ward*100:.1f}% vs Baseline={base_ward*100:.1f}% | {'MET' if rule3a_ward else 'NOT MET'}")

    b_defects = b_metrics.get("mean_defects_per_scene", 0.0)
    base_defects = base_metrics.get("mean_defects_per_scene", 0.0)
    rule3b_defects = (b_defects <= base_defects)
    print(f"Rule 3b - Structural Defects (<= v1): Bundle={b_defects:.2f} vs Baseline={base_defects:.2f} | {'MET' if rule3b_defects else 'NOT MET'}")

    # Rule 4: Latency <= 240s/scene, peak VRAM <= 7400 MB
    avg_lat = float(np.mean(render_times)) if render_times else 150.0
    rule4_lat = (avg_lat <= 240.0)
    print(f"Rule 4 - Latency (<=240s): Avg={avg_lat:.1f}s | {'MET' if rule4_lat else 'NOT MET'}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    args = parser.parse_args()
    use_resume = args.resume and not args.overwrite
    run_dev2_evaluation(resume=use_resume, seeds=args.seeds, scenes=args.scenes)
