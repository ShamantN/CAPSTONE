"""
Score newly rendered Baseline v1 runs using frozen Verifier v1.1:
1. baseline_v1_dev2 (30 images)
2. baseline_v1_test2_ebright (30 images)
3. baseline_v1_test2_holdout (12 images)
"""
import os
import sys
import io
import json
import logging
from pathlib import Path

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

from consistency_v2.verify.cli import run_verification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MMCCP.ScoreBaselineRuns")


def score_all():
    print("=" * 80)
    print("  SCORING BASELINE v1 DEV-2 & TEST-2 RUNS WITH VERIFIER v1.1")
    print("=" * 80)

    out_base = PROJECT_ROOT / "results" / "rescored_v1.1"
    out_base.mkdir(parents=True, exist_ok=True)

    ebright_story = PROJECT_ROOT / "storyboards" / "ebright.json"
    holdout_story = PROJECT_ROOT / "storyboards" / "holdout_v2.json"

    ebright_bank = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
    holdout_bank = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors" / "holdout"

    runs = [
        {
            "name": "baseline_v1_dev2",
            "images": PROJECT_ROOT / "results" / "baseline_v1" / "dev2",
            "story": ebright_story,
            "bank": ebright_bank,
            "out": out_base / "baseline_v1_dev2" / "report.json",
        },
        {
            "name": "baseline_v1_test2_ebright",
            "images": PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "ebright",
            "story": ebright_story,
            "bank": ebright_bank,
            "out": out_base / "baseline_v1_test2_ebright" / "report.json",
        },
        {
            "name": "baseline_v1_test2_holdout",
            "images": PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "holdout",
            "story": holdout_story,
            "bank": holdout_bank,
            "out": out_base / "baseline_v1_test2_holdout" / "report.json",
        }
    ]

    summary_records = {}

    for run_spec in runs:
        name = run_spec["name"]
        print(f"\n>>> Scoring run: {name}")
        print(f"    Images: {run_spec['images']}")
        print(f"    Story: {run_spec['story']}")
        print(f"    Output: {run_spec['out']}")

        if not run_spec["images"].exists():
            print(f"    [WARN] Images dir not found: {run_spec['images']}")
            continue

        rep = run_verification(
            images_dir=run_spec["images"],
            story_path=run_spec["story"],
            bank_dir=run_spec["bank"],
            output_report_path=run_spec["out"],
            baseline_report_path=None
        )

        if run_spec["out"].exists():
            sm = rep.get("summary_metrics", {})
            summary_records[name] = {
                "images_count": rep.get("images_evaluated_count", 0),
                "faces_count": rep.get("total_face_instances", 0),
                "biometric_ensemble_mean": sm.get("biometric_ensemble_mean", 0.0),
                "biometric_ensemble_95ci": sm.get("biometric_ensemble_95ci", [0.0, 0.0]),
                "rank1_discrimination_accuracy": sm.get("rank1_discrimination_accuracy", 0.0),
                "combined_binding_pass_rate": sm.get("binding_decomposed", {}).get("combined_binding_pass_rate", 0.0),
                "spatial_position_pass_rate": sm.get("binding_decomposed", {}).get("spatial_position_pass_rate", 0.0),
                "wardrobe_palette_pass_rate": sm.get("binding_decomposed", {}).get("wardrobe_palette_pass_rate", 0.0),
                "mean_defects_per_scene": sm.get("mean_defects_per_scene", 0.0),
                "adult_biometric_mean": sm.get("age_split_metrics", {}).get("adult", {}).get("biometric_ensemble_mean", 0.0),
                "child_biometric_mean": sm.get("age_split_metrics", {}).get("child", {}).get("biometric_ensemble_mean", 0.0),
                "seg_failed_count": sm.get("seg_failed_count", 0)
            }
            print(f"    -> [DONE] {name}: Bio={sm.get('biometric_ensemble_mean')} "
                  f"(Rank1={sm.get('rank1_discrimination_accuracy')}), "
                  f"Binding={sm.get('binding_decomposed', {}).get('combined_binding_pass_rate')}, "
                  f"Defects={sm.get('mean_defects_per_scene')}")

    out_summary_file = out_base / "baseline_new_scores_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved summary to: {out_summary_file}")


if __name__ == "__main__":
    score_all()
