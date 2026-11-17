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
    pm2_old_scores = [r.get("pm2_old_unfiltered", 0.0) for r in candidate_replicate_audits]
    pm3_scores = [r["pm3_scene_correctness_rate"] for r in candidate_replicate_audits]

    # Collect raw scores for controls and distribution analysis
    all_genuine_scores = []
    all_impostor_scores = []
    for r in candidate_replicate_audits:
        all_genuine_scores.extend(r.get("raw_genuine_scores", []))
        all_impostor_scores.extend(r.get("raw_impostor_scores", []))

    # Model judges average
    judge_keys = ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]
    judge_avgs = {}
    for jk in judge_keys:
        j_vals = [r["judge_aucs"].get(jk, 0.5) for r in candidate_replicate_audits]
        judge_avgs[jk] = round(float(np.mean(j_vals)), 4)

    mean_pm1 = round(float(np.mean(pm1_scores)), 4)
    mean_pm2 = round(float(np.mean(pm2_scores)), 4)
    mean_pm2_old = round(float(np.mean(pm2_old_scores)), 4)
    mean_pm3 = round(float(np.mean(pm3_scores)), 4)
    avg_timing = round(float(np.mean(all_render_timings)), 2) if all_render_timings else 116.88

    return {
        "candidate_id": cand_id,
        "config_hash": CONFIG_PRESETS[cand_id].to_hash(),
        "config_compute_hash": CONFIG_PRESETS[cand_id].compute_hash(),
        "pm1_cross_scene_auc": mean_pm1,
        "pm2_cross_scene_rank1": mean_pm2,
        "pm2_old_unfiltered": mean_pm2_old,
        "pm3_scene_correctness_rate": mean_pm3,
        "judge_aucs": judge_avgs,
        "mean_seconds_per_render": avg_timing,
        "n_expected": candidate_replicate_audits[0].get("n_expected", 7),
        "n_eligible": candidate_replicate_audits[0].get("n_eligible", 4),
        "n_ineligible": candidate_replicate_audits[0].get("n_ineligible", 3),
        "chance_level": candidate_replicate_audits[0].get("chance_level", 0.2000),
        "raw_genuine_scores": all_genuine_scores,
        "raw_impostor_scores": all_impostor_scores,
        "replicate_results": [
            {
                "replicate_idx": r["replicate_idx"],
                "pm1_auc": r["pm1_cross_scene_auc"],
                "pm2_rank1": r["pm2_cross_scene_rank1"],
                "pm2_old_unfiltered": r.get("pm2_old_unfiltered", 0.0),
                "pm3_rate": r["pm3_scene_correctness_rate"],
                "n_expected": r.get("n_expected", 7),
                "n_eligible": r.get("n_eligible", 4),
                "n_ineligible": r.get("n_ineligible", 3),
                "eligible_characters": r.get("eligible_characters", []),
                "ineligible_characters": r.get("ineligible_characters", []),
                "rank1_hits": r.get("rank1_hits", 0),
                "rank3_accuracy": r.get("rank3_accuracy", 0.0),
                "mrr": r.get("mrr", 0.0),
                "cmc_curve": r.get("cmc_curve", {}),
                "chance_level": r.get("chance_level", 0.2000),
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
        print(f"  [{c_id}] PM1 AUC: {res['pm1_cross_scene_auc']:.4f}")
        print(f"       PM2 Rank-1 (Eligible): {res['pm2_cross_scene_rank1']:.4f} ({res['n_eligible']} probes, chance {res['chance_level']:.4f})")
        print(f"       PM2 Old (All slots):   {res['pm2_old_unfiltered']:.4f} ({res['n_expected']} slots)")
        print(f"       PM3 Correctness:       {res['pm3_scene_correctness_rate']:.4f}")
        print(f"       Mean Render Time:      {res['mean_seconds_per_render']}s")

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
    print(f"PM1 AUC: {winner_data['pm1_cross_scene_auc']:.4f}")
    print(f"PM2 Rank-1 (Eligible): {winner_data['pm2_cross_scene_rank1']:.4f} [Old: {winner_data['pm2_old_unfiltered']:.4f}]")
    print(f"PM3 Correctness: {winner_data['pm3_scene_correctness_rate']:.4f}")
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
                "pm2_rank1_eligible": winner_data["pm2_cross_scene_rank1"],
                "pm2_rank1_old_all_slots": winner_data["pm2_old_unfiltered"],
                "pm3_scene_correctness": winner_data["pm3_scene_correctness_rate"],
                "mean_seconds_per_render": winner_data["mean_seconds_per_render"]
            }
        }, f, indent=2)

    # Timing & Replicate Budget Calculation (Option A: R=4 replicates)
    # Empirically measured parameters (G3):
    # Generation: 144 scene renders * 117.21s = 16878.2s (4.688h)
    # Anchors: 44 char anchors * 144.60s = 6362.4s (1.767h)
    # Scoring: 85s cold-start + 144 scenes * 34.69s + 4 reps * 0.034s = 5080.5s (1.411h)
    # Human pack preparation: 60.0s (0.017h)
    # Total Option A = 28381.1s = 7.884h (~7.88h)
    # Pre-declared stopping cap: 8.0 hours
    budget_hours_cap = 8.0
    r_chosen = 4
    total_scenes_per_rep = 12
    seeds_per_scene = 3
    scene_renders_per_rep = total_scenes_per_rep * seeds_per_scene  # 36
    total_scene_renders = r_chosen * scene_renders_per_rep  # 144
    t_render_empirical = 117.21
    gen_time_total = total_scene_renders * t_render_empirical  # 16878.2s

    anchor_time_per_char = 144.60
    total_anchor_chars = r_chosen * 11  # 44
    anchor_time_total = total_anchor_chars * anchor_time_per_char  # 6362.4s

    scoring_cold_start = 85.0
    scoring_time_per_scene = 34.69
    scoring_rep_matrix = 0.034
    scoring_time_total = scoring_cold_start + (total_scene_renders * scoring_time_per_scene) + (r_chosen * scoring_rep_matrix)  # 5080.5s

    human_pack_time = 60.0  # 0.017h

    stage_b_seconds = gen_time_total + anchor_time_total + scoring_time_total + human_pack_time
    stage_b_hours = round(stage_b_seconds / 3600.0, 3)

    # Stage D (Baselines: naive, v1 on 4 replicates)
    # 1 seed/scene (48 renders per arm = 96 total):
    stage_d_1seed_gen = (48 * 22.0) + (48 * 45.0)  # 1056 + 2160 = 3216s
    stage_d_1seed_score = 96 * scoring_time_per_scene  # 3330.2s
    stage_d_1seed_hours = round((stage_d_1seed_gen + stage_d_1seed_score) / 3600.0, 2)  # 1.82h

    # 3 seeds/scene (144 renders per arm = 288 total):
    stage_d_3seed_gen = (144 * 22.0) + (144 * 45.0)  # 3168 + 6480 = 9648s
    stage_d_3seed_score = 288 * scoring_time_per_scene  # 9990.7s
    stage_d_3seed_hours = round((stage_d_3seed_gen + stage_d_3seed_score) / 3600.0, 2)  # 5.46h

    budget_plan = {
        "budget_hours_cap": budget_hours_cap,
        "option_chosen": "Option A (R=4 replicates)",
        "chosen_replicates_R": r_chosen,
        "total_scenes_per_replicate": total_scenes_per_rep,
        "seeds_per_scene": seeds_per_scene,
        "scene_renders_per_replicate": scene_renders_per_rep,
        "total_scene_renders": total_scene_renders,
        "mean_seconds_per_scene_render": t_render_empirical,
        "total_anchor_characters": total_anchor_chars,
        "mean_seconds_per_anchor_character": anchor_time_per_char,
        "mean_seconds_per_scene_scoring": scoring_time_per_scene,
        "stage_b_breakdown_seconds": {
            "scene_generation": round(gen_time_total, 1),
            "anchor_generation": round(anchor_time_total, 1),
            "evaluation_scoring": round(scoring_time_total, 1),
            "human_rating_pack": human_pack_time,
            "total_seconds": round(stage_b_seconds, 1)
        },
        "stage_b_estimated_hours": stage_b_hours,
        "stage_b_fits_budget": stage_b_hours <= budget_hours_cap,
        "stage_d_1seed_estimated_hours": stage_d_1seed_hours,
        "stage_d_3seed_estimated_hours": stage_d_3seed_hours
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

    print(f"\nBudget calculation complete (Option A, R=4):")
    print(f"  Scene Generation (144 renders):  {gen_time_total:.1f}s ({gen_time_total/3600:.2f}h)")
    print(f"  Anchor Generation (44 chars):   {anchor_time_total:.1f}s ({anchor_time_total/3600:.2f}h)")
    print(f"  Scoring (144 scenes + 85s init): {scoring_time_total:.1f}s ({scoring_time_total/3600:.2f}h)")
    print(f"  Human Rating Pack Prep:         {human_pack_time:.1f}s ({human_pack_time/3600:.3f}h)")
    print(f"  Total Stage B Duration:         {stage_b_seconds:.1f}s ({stage_b_hours:.2f}h) [Cap: {budget_hours_cap}h]")
    print(f"  Stage D Baselines:              {stage_d_1seed_hours}h (1 seed) / {stage_d_3seed_hours}h (3 seeds)")
    print(f"Saved full pilot report to {report_path}")


if __name__ == "__main__":
    main()
