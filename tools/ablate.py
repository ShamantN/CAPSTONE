"""
MMCCP Phase R4: Canonical Pipeline Ablation Runner
Executes leave-one-out ablation studies across methods M1 through M8 with end-to-end flag propagation,
config hash tracking, stage traces, and identical-output no-op detectors.
"""
import os
import sys
import io
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
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
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.stage_trace import compute_image_sha256, compute_region_pixel_diff
from consistency_v2.splits import assert_seed_allowed

ABLATION_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "full_v2": {},  # All active default methods
    "no_m1_bank": {"enable_identity_bank": False},
    "no_m2_layout": {"enable_layout_planner": False},
    "no_m3_couple": {"enable_regional_couple": False},
    "no_m4_instantid": {"enable_dual_instantid": False},
    "no_m5_swap": {"enable_reliable_face_swap": False},
    "no_m6_detailer": {"enable_face_detailer": False},
    "no_m7_wardrobe": {"enable_wardrobe_lock": False},
    "m8_reference_only": {"enable_reference_attention": True}
}


def run_ablation(
    story_path: Path,
    out_dir: Path,
    scene_numbers: List[int] = [1, 2],
    seeds: List[int] = [11],
    conditions: Optional[List[str]] = None,
    trace: bool = True
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(story_path, "r", encoding="utf-8") as f:
        story = StorySpec.from_dict(json.load(f))

    scenes = [s for s in story.scenes if s.scene_number in scene_numbers]
    cond_keys = conditions or list(ABLATION_CONDITIONS.keys())

    # Load baseline best_config if exists
    best_cfg_path = PROJECT_ROOT / "consistency_v2" / "best_config.json"
    base_dict = {}
    if best_cfg_path.exists():
        with open(best_cfg_path, "r", encoding="utf-8") as f:
            base_dict = json.load(f)

    results_matrix: Dict[str, Any] = {}
    images_by_condition: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in cond_keys}

    print("=" * 80)
    print("  CANONICAL PIPELINE ABLATION RUNNER")
    print(f"  Storyboard: {story.title} | Scenes: {scene_numbers} | Seeds: {seeds}")
    print(f"  Conditions ({len(cond_keys)}): {cond_keys}")
    print("=" * 80)

    for cond_name in cond_keys:
        overrides = ABLATION_CONDITIONS[cond_name]
        cfg_dict = base_dict.copy()
        cfg_dict.update(overrides)

        cfg = ConsistencyV2Config.from_dict(cfg_dict)
        cfg_hash = cfg.to_hash()

        print(f"\n>>> Executing Condition: {cond_name} (Hash: {cfg_hash})")
        pipeline = ConsistencyPipelineV2(config=cfg)

        cond_dir = out_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        results_matrix[cond_name] = {
            "config_hash": cfg_hash,
            "overrides": overrides,
            "renders": {}
        }

        for scene in scenes:
            for seed in seeds:
                assert_seed_allowed(seed)
                render_key = f"scene_{scene.scene_number}_seed_{seed}"
                img_path = cond_dir / f"{render_key}.png"
                sidecar_path = cond_dir / f"{render_key}.json"

                if img_path.exists() and sidecar_path.exists():
                    print(f"    [SKIP] {render_key} already exists")
                    img_cv = cv2.imread(str(img_path))
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    print(f"    Rendering {render_key}...")
                    img_cv, meta = pipeline.generate_scene(
                        story=story,
                        scene=scene,
                        seed=seed,
                        trace=trace
                    )
                    if img_cv is not None:
                        cv2.imwrite(str(img_path), img_cv)
                        with open(sidecar_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)

                if img_cv is not None:
                    images_by_condition[cond_name][render_key] = img_cv
                    sha = compute_image_sha256(img_cv)
                    results_matrix[cond_name]["renders"][render_key] = {
                        "sha256": sha,
                        "runtime_s": meta.get("runtime_s", 0.0),
                        "peak_vram_mb": meta.get("peak_vram_mb", 0.0)
                    }

    # Cross-condition diff analysis to detect silent no-ops
    print("\n" + "=" * 80)
    print("  CROSS-CONDITION ABLATION VERIFICATION (NO-OP DETECTION)")
    print("=" * 80)

    full_renders = images_by_condition.get("full_v2", {})
    no_op_detected = False

    for cond_name, renders in images_by_condition.items():
        if cond_name == "full_v2":
            continue
        diffs = []
        for r_key, img_cv in renders.items():
            if r_key in full_renders:
                full_img = full_renders[r_key]
                diff = compute_region_pixel_diff(img_cv, full_img)
                diffs.append(diff)
                sha_cond = results_matrix[cond_name]["renders"][r_key]["sha256"]
                sha_full = results_matrix["full_v2"]["renders"][r_key]["sha256"]

                if sha_cond == sha_full:
                    print(f"  [CRITICAL NO-OP WARNING] Condition '{cond_name}' produced byte-identical image to full_v2 for {r_key}!")
                    no_op_detected = True
                else:
                    print(f"  [PASS] Condition '{cond_name}' vs full_v2 for {r_key}: mean pixel diff = {diff:.2f}")

        results_matrix[cond_name]["mean_diff_vs_full"] = round(float(np.mean(diffs)), 2) if diffs else 0.0

    summary_file = out_dir / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_matrix, f, indent=2)
    print(f"\nAblation summary saved to: {summary_file}")

    return results_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Pipeline Ablation Runner")
    parser.add_argument("--scenes", nargs="+", type=int, default=[1, 2], help="Scene numbers to ablate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11], help="Seeds to use (DEV seeds only)")
    parser.add_argument("--out-dir", type=str, default="results/ablation", help="Output directory")
    parser.add_argument("--conditions", nargs="+", type=str, default=None, help="Specific conditions to test")
    parser.add_argument("--no-trace", action="store_true", help="Disable stage tracing")
    args = parser.parse_args()

    story_p = PROJECT_ROOT / "storyboards" / "ebright.json"
    out_p = PROJECT_ROOT / args.out_dir

    run_ablation(
        story_path=story_p,
        out_dir=out_p,
        scene_numbers=args.scenes,
        seeds=args.seeds,
        conditions=args.conditions,
        trace=not args.no_trace
    )
