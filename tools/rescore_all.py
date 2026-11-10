"""
MMCCP Phase R2 & R7: Canonical Historical Rescoring Engine
Rescores historical baseline and v2-rc1 renders using frozen Verifier v1.2:
1. Baseline v1 DEV-FULL (results/baseline_v1/dev_full)
2. Baseline v1 DEV-2 (results/baseline_v1/dev2)
3. v2-rc1 DEV-2 / old TEST (results/v2_default/test)
4. v2-rc1 Generalization (results/v2_default/generalization)
5. v2-rc1 Ablations (results/ablation)
Optionally with --include-test2 (requires ALLOW_TEST2=1):
6. Baseline v1 TEST-2 Ebright (results/baseline_v1/test2/ebright)
7. Baseline v1 TEST-2 Holdout (results/baseline_v1/test2/holdout)

Outputs structured canonical reports and JSON summaries to results/rescored_v1.2/
"""
import sys
import os
import io
import json
import logging
import argparse
from pathlib import Path

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
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from consistency_v2.verify.cli import run_verification
from consistency_v2.splits import assert_split_allowed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MMCCP.RescoreAll")


def rescore_all(version: str = "1.2", include_test2: bool = False):
    print("=" * 80)
    print(f"  CANONICAL HISTORICAL RESCORING WITH VERIFIER v{version}")
    print("=" * 80)

    out_base = PROJECT_ROOT / "results" / f"rescored_v{version}"
    out_base.mkdir(parents=True, exist_ok=True)

    anchor_dir = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
    holdout_anchor_dir = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors" / "holdout"
    ebright_story = PROJECT_ROOT / "storyboards" / "ebright.json"
    chrono_story = PROJECT_ROOT / "storyboards" / "generalization_benchmark.json"
    holdout_story = PROJECT_ROOT / "storyboards" / "holdout_v2.json"

    runs = [
        {
            "name": "baseline_v1_dev_full",
            "images": PROJECT_ROOT / "results" / "baseline_v1" / "dev_full",
            "story": ebright_story,
            "bank": anchor_dir,
            "out": out_base / "baseline_v1_dev_full" / "report.json",
            "baseline": None,
            "is_test2": False
        },
        {
            "name": "baseline_v1_dev2",
            "images": PROJECT_ROOT / "results" / "baseline_v1" / "dev2",
            "story": ebright_story,
            "bank": anchor_dir,
            "out": out_base / "baseline_v1_dev2" / "report.json",
            "baseline": None,
            "is_test2": False
        },
        {
            "name": "v2_rc1_dev2",
            "images": PROJECT_ROOT / "results" / "v2_default" / "test",
            "story": ebright_story,
            "bank": anchor_dir,
            "out": out_base / "v2_rc1_dev2" / "report.json",
            "baseline": out_base / "baseline_v1_dev_full" / "report.json",
            "is_test2": False
        },
        {
            "name": "v2_rc1_generalization",
            "images": PROJECT_ROOT / "results" / "v2_default" / "generalization",
            "story": chrono_story,
            "bank": anchor_dir,
            "out": out_base / "v2_rc1_generalization" / "report.json",
            "baseline": None,
            "is_test2": False
        },
        {
            "name": "v2_rc1_ablation",
            "images": PROJECT_ROOT / "results" / "ablation",
            "story": ebright_story,
            "bank": anchor_dir,
            "out": out_base / "v2_rc1_ablation" / "report.json",
            "baseline": None,
            "is_test2": False
        }
    ]

    if include_test2:
        runs.extend([
            {
                "name": "baseline_v1_test2_ebright",
                "images": PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "ebright",
                "story": ebright_story,
                "bank": anchor_dir,
                "out": out_base / "baseline_v1_test2_ebright" / "report.json",
                "baseline": None,
                "is_test2": True
            },
            {
                "name": "baseline_v1_test2_holdout",
                "images": PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "holdout",
                "story": holdout_story,
                "bank": holdout_anchor_dir,
                "out": out_base / "baseline_v1_test2_holdout" / "report.json",
                "baseline": None,
                "is_test2": True
            }
        ])

    summary_records = {}

    for run_cfg in runs:
        r_name = run_cfg["name"]
        imgs_dir = run_cfg["images"]
        if not imgs_dir.exists():
            print(f"[SKIP] Directory not found for {r_name}: {imgs_dir}")
            continue

        png_count = len(list(imgs_dir.glob("*.png")))
        if png_count == 0:
            print(f"[SKIP] No PNG images in {imgs_dir}")
            continue

        if run_cfg["is_test2"]:
            assert_split_allowed("TEST-2", caller_info=f"rescore_all.py for {r_name} under Verifier v{version}")

        print(f"\n>>> Rescoring {r_name} ({png_count} images)...")
        rep = run_verification(
            images_dir=imgs_dir,
            story_path=run_cfg["story"],
            bank_dir=run_cfg["bank"],
            output_report_path=run_cfg["out"],
            baseline_report_path=run_cfg["baseline"] if (run_cfg["baseline"] and run_cfg["baseline"].exists()) else None
        )

        sm = rep.get("summary_metrics", {})
        age_splits = sm.get("age_split_metrics", {})
        summary_records[r_name] = {
            "images_count": rep.get("images_evaluated_count", 0),
            "faces_count": rep.get("total_face_instances", 0),
            "biometric_ensemble_mean": sm.get("biometric_ensemble_mean", 0.0),
            "biometric_ensemble_95ci": sm.get("biometric_ensemble_95ci", [0.0, 0.0]),
            "rank1_discrimination_accuracy": sm.get("rank1_discrimination_accuracy", 0.0),
            "combined_binding_pass_rate": sm.get("binding_decomposed", {}).get("combined_binding_pass_rate", 0.0),
            "spatial_position_pass_rate": sm.get("binding_decomposed", {}).get("spatial_position_pass_rate", 0.0),
            "wardrobe_palette_pass_rate": sm.get("binding_decomposed", {}).get("wardrobe_palette_pass_rate", 0.0),
            "identity_at_position_pass_rate": sm.get("binding_decomposed", {}).get("identity_at_position_pass_rate", 0.0),
            "mean_defects_per_scene": sm.get("mean_defects_per_scene", 0.0),
            "adult_biometric_mean": age_splits.get("adult", {}).get("biometric_ensemble_mean", 0.0),
            "teen_biometric_mean": age_splits.get("teen", {}).get("biometric_ensemble_mean", 0.0),
            "child_biometric_mean": age_splits.get("child", {}).get("biometric_ensemble_mean", 0.0),
            "seg_failed_count": sm.get("seg_failed_count", 0)
        }

    # Save comparative summary
    summary_path = out_base / "rescored_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 100)
    print(f"  CANONICAL RESCORING COMPLETE - TRUTH TABLE (Verifier v{version})")
    print("=" * 100)
    for name, s in summary_records.items():
        print(f"  {name:28s} | Bio: {s['biometric_ensemble_mean']:.4f} (95% CI: [{s['biometric_ensemble_95ci'][0]:.4f}, {s['biometric_ensemble_95ci'][1]:.4f}]) | Rank1: {s['rank1_discrimination_accuracy']*100:.1f}% | Binding: {s['combined_binding_pass_rate']*100:.1f}% | IdPos: {s['identity_at_position_pass_rate']*100:.1f}% | Defects: {s['mean_defects_per_scene']:.2f} | Adult: {s['adult_biometric_mean']:.4f} | Teen: {s['teen_biometric_mean']:.4f} | Child: {s['child_biometric_mean']:.4f}")
    print("=" * 100)
    print(f"✅ Canonical rescored summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="1.2", help="Verifier version")
    parser.add_argument("--include-test2", action="store_true", help="Include baseline TEST-2 runs (requires ALLOW_TEST2=1)")
    args = parser.parse_args()

    rescore_all(version=args.version, include_test2=args.include_test2)
