"""
MMCCP TEST-3 Evaluation Protocol: DEV-3 Pilot Scoring & Config Selection (tools/score_dev3_pilot.py).

Scores Candidates C0, C1, C2 using Eval3Evaluator:
- Measures PM1 AUC (out-of-loop 4-judge ensemble across scenes)
- Measures PM2 Rank-1 (leave-one-scene-out intention-to-treat denominator)
- Measures PM3 Scene Correctness (all present, no extra, left/right order correct)
- Measures T_render (seconds per scene render)
- Selects winning config (PM1 -> PM2 -> C0) and saves results/eval3/selected_config.json
- Calculates R and budget projections for Stage B and Stage D
"""
import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec
from eval3.runner import EVAL_OUTPUT_BASE, CONFIG_PRESETS
from eval3.metrics.evaluator import Eval3Evaluator
from eval3.metrics.calibration import DEFAULT_CALIBRATION


def score_candidate(cand_id: str, evaluator: Eval3Evaluator, story: StorySpec) -> Dict[str, Any]:
    cand_dir = EVAL_OUTPUT_BASE / f"dev3_pilot_{cand_id}"
    manifest_path = cand_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing for {cand_id} at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load character anchors for replicate 0 and 1
    shared_anchors_dir = EVAL_OUTPUT_BASE / "dev3_shared" / "replicates"

    candidate_replicate_audits = []
    all_render_timings = []

    for r_idx in [0, 1]:
        rep_render_dir = cand_dir / f"rep_{r_idx:02d}" / "scenes"
        rep_anchors_dir = shared_anchors_dir / f"rep_{r_idx:02d}" / "anchors"

        # Load anchor crops
        anchor_crops = {}
        for c_file in rep_anchors_dir.glob("*_selected_anchor.json"):
            with open(c_file, "r", encoding="utf-8") as af:
                ameta = json.load(af)
                cid = ameta["char_id"]
                img_p = ameta["image_path"]
                if Path(img_p).exists():
                    img = cv2.imread(img_p)
                    anchor_crops[img_p] = img

        # Audit each scene render in replicate (with caching)
        cache_path = rep_render_dir / "audits_cache.json"
        scene_audits = []
        if cache_path.exists():
            print(f"  [CACHE] Loading cached scene audits from {cache_path.name}")
            with open(cache_path, "r", encoding="utf-8") as cf:
                scene_audits = json.load(cf)
            for a in scene_audits:
                elapsed = a.get("elapsed_seconds", 0.0)
                if elapsed > 0:
                    all_render_timings.append(elapsed)
        else:
            for sc_file in sorted(rep_render_dir.glob("rep_*.json")):
                with open(sc_file, "r", encoding="utf-8") as sf:
                    sc_data = json.load(sf)

                img_p = rep_render_dir / f"{sc_data['render_id']}.png"
                if not img_p.exists():
                    continue

                elapsed = sc_data.get("elapsed_seconds", 0.0)
                if elapsed > 0:
                    all_render_timings.append(elapsed)

                sc_num = sc_data.get("scene_number", 1)
                exp_chars = sc_data.get("expected_characters", [])
                char_pos = sc_data.get("character_positions", {})

                # Run single scene audit
                audit = evaluator.audit_single_scene_image(
                    image_path=img_p,
                    expected_characters=exp_chars,
                    character_positions=char_pos,
                    anchor_crops=anchor_crops
                )
                audit["scene_number"] = sc_num
                audit["render_id"] = sc_data["render_id"]
                audit["seed"] = sc_data.get("seed", 0)
                audit["elapsed_seconds"] = elapsed
                scene_audits.append(audit)

            # Save cache
            def _clean_json(obj):
                if isinstance(obj, dict):
                    return {k: _clean_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_clean_json(v) for v in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                return obj

            with open(cache_path, "w", encoding="utf-8") as cf:
                json.dump(_clean_json(scene_audits), cf)

        # Expected characters across Story A scenes 1 to 5
        story_chars = ["young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong"]
        rep_eval = evaluator.evaluate_replicate(
            replicate_idx=r_idx,
            scene_audits=scene_audits,
            expected_chars=story_chars
        )
        rep_eval["scenes"] = scene_audits
        candidate_replicate_audits.append(rep_eval)

    # Compute aggregate metrics across the 2 replicates
    pm1_scores = [r["pm1_cross_scene_auc"] for r in candidate_replicate_audits]
    pm2_scores = [r["pm2_cross_scene_rank1"] for r in candidate_replicate_audits]
    pm3_scores = [r["pm3_scene_correctness_rate"] for r in candidate_replicate_audits]

    # Model judges average
    judge_keys = ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]
    judge_avgs = {}
    for jk in judge_keys:
        j_vals = [r["judge_aucs"].get(jk, 0.5) for r in candidate_replicate_audits]
        judge_avgs[jk] = round(float(np.mean(j_vals)), 4)

    mean_pm1 = round(float(np.mean(pm1_scores)), 4)
    mean_pm2 = round(float(np.mean(pm2_scores)), 4)
    mean_pm3 = round(float(np.mean(pm3_scores)), 4)
    avg_timing = round(float(np.mean(all_render_timings)), 2) if all_render_timings else 18.0

    return {
        "candidate_id": cand_id,
        "config_hash": CONFIG_PRESETS[cand_id].to_hash(),
        "config_compute_hash": CONFIG_PRESETS[cand_id].compute_hash(),
        "pm1_cross_scene_auc": mean_pm1,
        "pm2_cross_scene_rank1": mean_pm2,
        "pm3_scene_correctness_rate": mean_pm3,
        "judge_aucs": judge_avgs,
        "mean_seconds_per_render": avg_timing,
        "replicate_results": [
            {
                "replicate_idx": r["replicate_idx"],
                "pm1_auc": r["pm1_cross_scene_auc"],
                "pm2_rank1": r["pm2_cross_scene_rank1"],
                "pm3_rate": r["pm3_scene_correctness_rate"],
                "n_genuine_pairs": r["n_genuine_pairs"],
                "n_impostor_pairs": r["n_impostor_pairs"],
            }
            for r in candidate_replicate_audits
        ],
        "all_audits": candidate_replicate_audits
    }


def main():
    print("=" * 70)
    print("SCORING DEV-3 PILOT CANDIDATES (C0, C1, C2) VIA EVAL3EVALUATOR")
    print("=" * 70)

    evaluator = Eval3Evaluator()
    story_a_path = PROJECT_ROOT / "storyboards" / "ebright.json"
    story = StorySpec.load_json(str(story_a_path))

    candidates = ["C0", "C1", "C2"]
    scored_candidates = {}

    for c_id in candidates:
        print(f"\nScoring Candidate {c_id}...")
        res = score_candidate(c_id, evaluator, story)
        scored_candidates[c_id] = res
        print(f"  [{c_id}] PM1 AUC: {res['pm1_cross_scene_auc']:.4f} | PM2 Rank-1: {res['pm2_cross_scene_rank1']:.4f} | PM3 Scene Correctness: {res['pm3_scene_correctness_rate']:.4f} | Mean Render Time: {res['mean_seconds_per_render']}s")

    # Winner Selection Rule:
    # 1. Highest DEV-3 PM1 AUC.
    # 2. Ties broken by DEV-3 PM2 Rank-1.
    # 3. Ties broken to C0 (status quo).
    def _sort_key(c_id: str):
        d = scored_candidates[c_id]
        is_c0 = 1 if c_id == "C0" else 0
        return (d["pm1_cross_scene_auc"], d["pm2_cross_scene_rank1"], is_c0)

    winner_id = max(candidates, key=_sort_key)
    winner_data = scored_candidates[winner_id]
    winner_cfg = CONFIG_PRESETS[winner_id]

    print("\n" + "=" * 70)
    print(f"SELECTED WINNER CONFIGURATION: {winner_id}")
    print(f"Config Hash: {winner_cfg.to_hash()}")
    print(f"PM1 AUC: {winner_data['pm1_cross_scene_auc']:.4f} | PM2 Rank-1: {winner_data['pm2_cross_scene_rank1']:.4f} | PM3: {winner_data['pm3_scene_correctness_rate']:.4f}")
    print("=" * 70)

    # Save selected configuration
    selected_cfg_path = PROJECT_ROOT / "results" / "eval3" / "selected_config.json"
    selected_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selected_cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_candidate": winner_id,
            "config_hash": winner_cfg.to_hash(),
            "config_compute_hash": winner_cfg.compute_hash(),
            "selection_reason": "Highest DEV-3 PM1 AUC across out-of-loop 4-judge ensemble (ties broken by PM2 Rank-1 then C0)",
            "config": winner_cfg.to_dict(),
            "metrics": {
                "pm1_auc": winner_data["pm1_cross_scene_auc"],
                "pm2_rank1": winner_data["pm2_cross_scene_rank1"],
                "pm3_scene_correctness": winner_data["pm3_scene_correctness_rate"],
                "mean_seconds_per_render": winner_data["mean_seconds_per_render"]
            }
        }, f, indent=2)

    # Timing & Replicate Budget Calculation
    # Cap: BUDGET_HOURS = 6.0 hours = 21,600 seconds
    budget_hours = 6.0
    budget_seconds = budget_hours * 3600.0
    t_render = winner_data["mean_seconds_per_render"]

    # Protocol structure:
    # 12 scenes total across Story A (5), Story B (3), Story C (4)
    # v2_final renders 3 seeds per scene = 36 renders per replicate
    # v2_final_bon3 is selected from the same 3 seeds (0 extra generation!)
    # Anchor generation: 11 characters * 4 candidates = 44 anchor renders per replicate (at ~10s per anchor txt2img = 440s)
    # Plus evaluation scoring overhead: ~60s per replicate
    total_scenes_per_rep = 12
    seeds_per_scene = 3
    scene_renders_per_rep = total_scenes_per_rep * seeds_per_scene  # 36
    scene_time_per_rep = scene_renders_per_rep * t_render
    anchor_time_per_rep = 11 * 4 * 10.0  # 440s
    eval_overhead_per_rep = 80.0
    total_time_per_rep = scene_time_per_rep + anchor_time_per_rep + eval_overhead_per_rep
    fixed_overhead = 600.0  # 10 min setup, calibration, final bootstrap analysis

    raw_r = (budget_seconds - fixed_overhead) / total_time_per_rep
    r_chosen = int(max(4, min(8, math.floor(raw_r))))

    # Budget projections
    stage_b_seconds = (r_chosen * total_time_per_rep) + fixed_overhead
    stage_b_hours = round(stage_b_seconds / 3600.0, 2)

    # Stage D (Baselines: naive, v1) projections:
    # Naive: 12 scenes * 2 seeds * 12s = 288s per replicate
    # v1: 12 scenes * 2 seeds * 18s = 432s per replicate
    # Total Stage D per replicate = ~720s
    stage_d_seconds = (r_chosen * 720.0) + 300.0
    stage_d_hours = round(stage_d_seconds / 3600.0, 2)

    budget_plan = {
        "budget_hours_cap": budget_hours,
        "mean_seconds_per_scene_render_T_render": t_render,
        "total_scenes_per_replicate": total_scenes_per_rep,
        "seeds_per_scene": seeds_per_scene,
        "renders_per_replicate": scene_renders_per_rep,
        "seconds_per_replicate": round(total_time_per_rep, 1),
        "chosen_replicates_R": r_chosen,
        "stage_b_estimated_hours": stage_b_hours,
        "stage_d_estimated_hours": stage_d_hours,
        "stage_b_estimated_renders": r_chosen * (scene_renders_per_rep + 44),
        "stage_b_fits_budget": stage_b_hours <= budget_hours
    }

    report = {
        "candidates": {c: {k: v for k, v in scored_candidates[c].items() if k != "all_audits"} for c in candidates},
        "winner": winner_id,
        "winner_config_hash": winner_cfg.to_hash(),
        "budget_plan": budget_plan
    }

    report_path = PROJECT_ROOT / "results" / "eval3" / "dev3_pilot_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBudget calculation complete:")
    print(f"  T_render: {t_render}s | Seconds per replicate: {total_time_per_rep:.1f}s")
    print(f"  Chosen R: {r_chosen} replicates (Target in [5, 6]: {r_chosen in [5, 6]})")
    print(f"  Stage B Estimated Duration: {stage_b_hours} hours (Fits {budget_hours}h cap: {stage_b_hours <= budget_hours})")
    print(f"  Stage D Estimated Duration: {stage_d_hours} hours")
    print(f"Saved full pilot report to {report_path}")


if __name__ == "__main__":
    main()
