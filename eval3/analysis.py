"""
MMCCP TEST-3 Evaluation Protocol: Statistical Analysis & Aggregation (eval3/analysis.py).

Provides:
- Hierarchical cluster bootstrap (scenes clustered within replicates) with 10,000 resamples and 95% CIs.
- Export of per-face CSV (test3_per_face.csv).
- Export of per-scene CSV (test3_per_scene.csv).
- Export of summary JSON (test3_summary.json).
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def hierarchical_cluster_bootstrap(
    replicate_audits: List[Dict[str, Any]],
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Hierarchical bootstrap:
    Resamples replicates with replacement, then scenes within each replicate with replacement.
    Computes 95% Confidence Intervals (percentile method).
    """
    rng = np.random.RandomState(seed)
    n_replicates = len(replicate_audits)
    if n_replicates == 0:
        return {}

    pm1_boot = []
    pm2_boot = []
    pm3_boot = []

    for _ in range(n_resamples):
        # Sample replicates with replacement
        rep_indices = rng.choice(n_replicates, size=n_replicates, replace=True)
        rep_pm1s = []
        rep_pm2s = []
        rep_pm3s = []

        for r_idx in rep_indices:
            rep_data = replicate_audits[r_idx]
            # Use replicate level metrics directly or resample scenes
            scenes = rep_data.get("scenes", [])
            if scenes:
                # Resample scenes
                s_indices = rng.choice(len(scenes), size=len(scenes), replace=True)
                s_sampled = [scenes[si] for si in s_indices]
                correct = sum(1 for s in s_sampled if s.get("pm3_correct", False))
                rep_pm3s.append(float(correct) / float(len(s_sampled)))
            else:
                rep_pm3s.append(rep_data.get("pm3_scene_correctness_rate", 0.0))

            rep_pm1s.append(rep_data.get("pm1_cross_scene_auc", 0.50))
            rep_pm2s.append(rep_data.get("pm2_cross_scene_rank1", 0.0))

        pm1_boot.append(float(np.mean(rep_pm1s)))
        pm2_boot.append(float(np.mean(rep_pm2s)))
        pm3_boot.append(float(np.mean(rep_pm3s)))

    def _ci(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a)), 4),
            "ci_95_lower": round(float(np.percentile(a, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(a, 97.5)), 4),
        }

    return {
        "pm1_cross_scene_auc": _ci(pm1_boot),
        "pm2_cross_scene_rank1": _ci(pm2_boot),
        "pm3_scene_correctness_rate": _ci(pm3_boot),
    }


def export_evaluation_results(
    run_id: str,
    output_dir: Path,
    replicate_audits: List[Dict[str, Any]],
    scene_records: List[Dict[str, Any]],
    face_records: List[Dict[str, Any]],
    extra_summary_meta: Optional[Dict[str, Any]] = None
) -> Tuple[Path, Path, Path]:
    """
    Exports:
    1. results/eval3/test3_per_face.csv
    2. results/eval3/test3_per_scene.csv
    3. results/eval3/test3_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_face_path = output_dir / "test3_per_face.csv"
    per_scene_path = output_dir / "test3_per_scene.csv"
    summary_path = output_dir / "test3_summary.json"

    # 1. Per Face CSV
    if face_records:
        df_faces = pd.DataFrame(face_records)
    else:
        df_faces = pd.DataFrame(columns=[
            "run_id", "replicate_idx", "story_id", "scene_idx", "face_idx",
            "character_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "confidence", "sharpness", "buffalo_l_anchor_sim", "v1_2_ensemble_score",
            "yolo_person_match", "extra_person_flag", "rank1_match"
        ])
    df_faces.to_csv(per_face_path, index=False)

    # 2. Per Scene CSV
    if scene_records:
        df_scenes = pd.DataFrame(scene_records)
    else:
        df_scenes = pd.DataFrame(columns=[
            "run_id", "replicate_idx", "story_id", "scene_idx", "seed",
            "all_expected_present", "no_extra_person", "order_correct",
            "pm3_correct", "yolo_count", "faces_detected", "extra_person_flag",
            "render_time_s"
        ])
    df_scenes.to_csv(per_scene_path, index=False)

    # 3. Bootstrap & Summary JSON
    boot_stats = hierarchical_cluster_bootstrap(replicate_audits, n_resamples=10000)

    # Calculate aggregate secondary metrics
    pm1_scores = [r.get("pm1_cross_scene_auc", 0.5) for r in replicate_audits]
    pm2_scores = [r.get("pm2_cross_scene_rank1", 0.0) for r in replicate_audits]
    pm3_scores = [r.get("pm3_scene_correctness_rate", 0.0) for r in replicate_audits]
    recalls = [r.get("face_recall", 0.0) for r in replicate_audits]
    mrrs = [r.get("mrr", 0.0) for r in replicate_audits]
    extra_person_rates = [r.get("extra_person_rate", 0.0) for r in replicate_audits]
    mean_sharpnesses = [r.get("mean_face_sharpness", 0.0) for r in replicate_audits]

    # Model judge breakdown across replicates
    judge_keys = ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]
    judge_summary = {}
    for jk in judge_keys:
        j_aucs = [r.get("judge_aucs", {}).get(jk, 0.5) for r in replicate_audits]
        judge_summary[jk] = round(float(np.mean(j_aucs)), 4) if j_aucs else 0.5

    summary_data = {
        "run_id": run_id,
        "protocol_version": "TEST-3 v1.0",
        "description": "Pre-registered absolute evaluation of MMCCP pipeline (TEST-3 Phase 1)",
        "bootstrap_iterations": 10000,
        "n_replicates": len(replicate_audits),
        "total_scenes_evaluated": len(scene_records),
        "total_faces_evaluated": len(face_records),
        "primary_metrics": {
            "pm1_cross_scene_auc": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm1_scores)), 4) if pm1_scores else 0.5,
                "bootstrap": boot_stats.get("pm1_cross_scene_auc", {}),
                "chance_floor": 0.50,
            },
            "pm2_cross_scene_rank1": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm2_scores)), 4) if pm2_scores else 0.0,
                "bootstrap": boot_stats.get("pm2_cross_scene_rank1", {}),
                "chance_floor": replicate_audits[0].get("chance_level", 0.20) if replicate_audits else 0.20,
            },
            "pm3_scene_correctness_rate": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm3_scores)), 4) if pm3_scores else 0.0,
                "bootstrap": boot_stats.get("pm3_scene_correctness_rate", {}),
                "chance_floor": 0.0,
            }
        },
        "secondary_metrics": {
            "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
            "face_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "extra_person_rate": round(float(np.mean(extra_person_rates)), 4) if extra_person_rates else 0.0,
            "mean_face_sharpness": round(float(np.mean(mean_sharpnesses)), 2) if mean_sharpnesses else 0.0,
            "judge_aucs": judge_summary,
        },
        "per_replicate": replicate_audits,
    }

    if extra_summary_meta:
        summary_data.update(extra_summary_meta)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return per_face_path, per_scene_path, summary_path
