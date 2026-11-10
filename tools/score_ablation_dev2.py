"""
Score all 7 Leave-One-Out Ablation arms on DEV-2 using Frozen Verifier v1.2.
Computes Deltas relative to Full Bundle (full_v2).
"""
import os
import sys
import json
from pathlib import Path

# Force UTF-8 and CPU for verifier
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.cli import run_verification

CONDITIONS = [
    "full_v2",
    "no_m2_layout",
    "no_m3_couple",
    "no_m4_ip_adapter",
    "no_m5_swap",
    "no_m6_detailer",
    "no_m7_wardrobe"
]

STORY_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
BANK_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
ABLATION_DIR = PROJECT_ROOT / "results" / "ablation_dev2"


def main():
    print("=" * 80)
    print("  SCORING 7-ARM DEV-2 ABLATION WITH FROZEN VERIFIER v1.2")
    print("=" * 80)

    summary_results = {}

    for cond in CONDITIONS:
        cond_dir = ABLATION_DIR / cond
        report_path = cond_dir / "report.json"
        print(f"\n>>> Scoring Condition: {cond}...")
        
        # Run verification CLI directly
        run_verification(
            images_dir=cond_dir,
            story_path=STORY_PATH,
            bank_dir=BANK_DIR,
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

    # Compute deltas vs full_v2
    full = summary_results["full_v2"]
    delta_results = {}

    print("\n" + "=" * 80)
    print("  7-ARM LEAVE-ONE-OUT ABLATION TRUTH TABLE (DEV-2)")
    print("=" * 80)
    print(f"{'Condition':<18} | {'Bio Mean':<9} | {'Rank-1':<7} | {'Spatial':<7} | {'Wardrobe':<8} | {'Defects':<7} | {'Delta Bio':<9}")
    print("-" * 80)

    for cond in CONDITIONS:
        res = summary_results[cond]
        delta_bio = res["biometric_ensemble_mean"] - full["biometric_ensemble_mean"]
        delta_results[cond] = {
            **res,
            "delta_bio_vs_full": round(delta_bio, 4),
            "delta_spatial_vs_full": round(res["spatial_pass_rate"] - full["spatial_pass_rate"], 4),
            "delta_wardrobe_vs_full": round(res["wardrobe_pass_rate"] - full["wardrobe_pass_rate"], 4)
        }
        print(f"{cond:<18} | {res['biometric_ensemble_mean']:<9.4f} | {res['rank1_accuracy']*100:<6.1f}% | {res['spatial_pass_rate']*100:<6.1f}% | {res['wardrobe_pass_rate']*100:<7.1f}% | {res['mean_defects']:<7.2f} | {delta_bio:+0.4f}")

    out_file = ABLATION_DIR / "loo_ablation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "conditions": delta_results,
            "full_v2_baseline": full
        }, f, indent=2)

    print(f"\nAblation truth table saved to: {out_file}")


if __name__ == "__main__":
    main()

