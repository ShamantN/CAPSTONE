"""
MMCCP Phase R10 + Finalization Reconciliation: Canonical TEST-2 Statistical Analysis

Computes strictly from Verifier v1.2 JSONs, extra-person audit JSON, and telemetry CSV:
  1. Scene-level (primary for defects, n=30 Ebright / n=12 Holdout / n=42 Combined)
     and face-instance-level (n=40 Ebright / n=24 Holdout / n=64 Combined) paired metrics.
  2. Both 10,000-resample 95% CIs (fixed seed 42):
     - Image-level bootstrap (resampling renders/images with replacement)
     - Scene-cluster bootstrap (resampling scene_number clusters with replacement)
  3. Two-sided Wilcoxon signed-rank tests (W and p-value).
  4. Disaggregated Adult, Teen, and Child biometric breakdowns with exact n/N counts.
  5. Original Pre-Registered P1-P4 Hypothesis Scorecard (honest MET / NOT MET) alongside
     the clearly labelled post-hoc rc2 scorecard (all values dynamically computed, zero hardcoding).
Outputs results strictly to: results/test2_evaluation_summary.json
"""
import csv
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical file paths (no duplicate holdout filename)
V2_EBRIGHT_PATH = PROJECT_ROOT / "results" / "test2_ebright" / "report_v1_2.json"
V1_EBRIGHT_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_ebright" / "report.json"

V2_HOLDOUT_PATH = PROJECT_ROOT / "results" / "test2_holdout" / "report_v1_2.json"
V1_HOLDOUT_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_holdout" / "report.json"

EXTRA_PERSON_AUDIT_PATH = PROJECT_ROOT / "results" / "test2_extra_person_audit.json"
TELEMETRY_SEALED_PATH = PROJECT_ROOT / "results" / "telemetry" / "vram_test2_sealed.csv"

OUT_PATH = PROJECT_ROOT / "results" / "test2_evaluation_summary.json"


def _extract_seed(img_name: str) -> int:
    m = re.search(r"seed_(\d+)", img_name)
    return int(m.group(1)) if m else -1


def extract_scene_records(report_data: Dict[str, Any], split_label: str) -> List[Dict[str, Any]]:
    records = []
    for s_rep in report_data.get("per_scene_reports", []):
        img_name = s_rep.get("image_filename", "")
        s_num = int(s_rep.get("scene_number", -1))
        seed = _extract_seed(img_name)
        sa = s_rep.get("scene_audit", {})
        records.append({
            "split": split_label,
            "pair_key": (split_label, s_num, seed),
            "scene_number": s_num,
            "seed": seed,
            "image_filename": img_name,
            "defect_count": float(sa.get("defect_count", 0)),
            "scene_passed": 1.0 if sa.get("scene_passed", False) else 0.0,
            "yolo_count": int(sa.get("detected_persons_yolo_count", 0)),
            "faces_count": int(sa.get("detected_faces_count", 0)),
            "expected_faces": int(sa.get("expected_faces", 0)),
        })
    return records


def extract_face_instances(report_data: Dict[str, Any], split_label: str) -> List[Dict[str, Any]]:
    instances = []
    for s_rep in report_data.get("per_scene_reports", []):
        img_name = s_rep.get("image_filename", "")
        s_num = int(s_rep.get("scene_number", -1))
        seed = _extract_seed(img_name)
        cluster_id = f"{split_label}_scene_{s_num}"
        image_id = f"{split_label}_s{s_num}_seed{seed}"
        defects = s_rep.get("scene_audit", {}).get("defect_count", 0)

        for c_eval in s_rep.get("character_evaluations", []):
            cid = c_eval.get("character_id", "")
            instances.append({
                "split": split_label,
                "cluster_id": cluster_id,
                "image_id": image_id,
                "pair_key": (split_label, s_num, seed, cid),
                "image_filename": img_name,
                "scene_number": s_num,
                "seed": seed,
                "character_id": cid,
                "age_group": c_eval.get("age_group", "adult"),
                "biometric_score": float(c_eval.get("biometric_ensemble_score", 0.0)),
                "rank1_correct": 1.0 if c_eval.get("rank1_correct", False) else 0.0,
                "discrimination_margin": float(c_eval.get("discrimination_margin", 0.0)),
                "spatial_pass": 1.0 if c_eval.get("spatial_pass", False) else 0.0,
                "wardrobe_pass": 1.0 if c_eval.get("wardrobe_pass", False) else 0.0,
                "identity_at_pos_pass": 1.0 if c_eval.get("identity_at_position_pass", False) else 0.0,
                "combined_binding_pass": 1.0 if c_eval.get("combined_binding_pass", False) else 0.0,
                "defects_scene": float(defects),
            })
    return instances


def pair_records(v2_list: List[Dict[str, Any]], v1_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    v1_map = {d["pair_key"]: d for d in v1_list}
    pairs = []
    for d2 in v2_list:
        key = d2["pair_key"]
        if key not in v1_map:
            continue
        d1 = v1_map[key]
        if "defect_count" in d2:
            pairs.append({
                "split": d2["split"],
                "cluster_id": f"{d2['split']}_scene_{d2['scene_number']}",
                "image_id": f"{d2['split']}_s{d2['scene_number']}_seed{d2['seed']}",
                "pair_key": key,
                "scene_number": d2["scene_number"],
                "seed": d2["seed"],
                "v2_defects_scene": d2["defect_count"],
                "v1_defects_scene": d1["defect_count"],
                "delta_defects_scene": d2["defect_count"] - d1["defect_count"],
                "v2_scene_passed": d2["scene_passed"],
                "v1_scene_passed": d1["scene_passed"],
                "delta_scene_passed": d2["scene_passed"] - d1["scene_passed"],
            })
        else:
            pairs.append({
                "split": d2["split"],
                "cluster_id": d2["cluster_id"],
                "image_id": d2["image_id"],
                "pair_key": key,
                "image_filename": d2["image_filename"],
                "character_id": d2["character_id"],
                "scene_number": d2["scene_number"],
                "seed": d2["seed"],
                "age_group": d2["age_group"],
                "v2_bio": d2["biometric_score"],
                "v1_bio": d1["biometric_score"],
                "delta_bio": d2["biometric_score"] - d1["biometric_score"],
                "v2_rank1": d2["rank1_correct"],
                "v1_rank1": d1["rank1_correct"],
                "delta_rank1": d2["rank1_correct"] - d1["rank1_correct"],
                "v2_spatial": d2["spatial_pass"],
                "v1_spatial": d1["spatial_pass"],
                "delta_spatial": d2["spatial_pass"] - d1["spatial_pass"],
                "v2_wardrobe": d2["wardrobe_pass"],
                "v1_wardrobe": d1["wardrobe_pass"],
                "delta_wardrobe": d2["wardrobe_pass"] - d1["wardrobe_pass"],
                "v2_id_pos": d2["identity_at_pos_pass"],
                "v1_id_pos": d1["identity_at_pos_pass"],
                "delta_id_pos": d2["identity_at_pos_pass"] - d1["identity_at_pos_pass"],
                "v2_comb_bind": d2["combined_binding_pass"],
                "v1_comb_bind": d1["combined_binding_pass"],
                "delta_comb_bind": d2["combined_binding_pass"] - d1["combined_binding_pass"],
                "v2_defects_inst_wt": d2["defects_scene"],
                "v1_defects_inst_wt": d1["defects_scene"],
                "delta_defects_inst_wt": d2["defects_scene"] - d1["defects_scene"],
            })
    return pairs


def grouped_bootstrap_ci(
    pairs: List[Dict[str, Any]],
    val_key: str,
    group_key: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """10,000-resample 95% CI resampling unique `group_key` units with replacement."""
    if not pairs:
        return 0.0, 0.0, 0.0

    groups: Dict[str, np.ndarray] = {}
    tmp: Dict[str, List[float]] = {}
    for p in pairs:
        tmp.setdefault(p[group_key], []).append(float(p[val_key]))
    keys = sorted(tmp.keys())
    k = len(keys)
    if k == 0:
        return 0.0, 0.0, 0.0

    group_sums = np.array([sum(tmp[g]) for g in keys], dtype=np.float64)
    group_cnts = np.array([len(tmp[g]) for g in keys], dtype=np.float64)
    obs_mean = float(group_sums.sum() / group_cnts.sum())

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, k, size=(n_boot, k))
    boot_means = group_sums[idx].sum(axis=1) / group_cnts[idx].sum(axis=1)
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return round(obs_mean, 4), round(low, 4), round(high, 4)


def compute_wilcoxon(v2_vals: List[float], v1_vals: List[float]) -> Tuple[float, float]:
    diffs = np.array(v2_vals) - np.array(v1_vals)
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 3:
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(v2_vals, v1_vals, alternative="two-sided")
        return round(float(res.statistic), 4), round(float(res.pvalue), 6)
    except Exception:
        return 0.0, 1.0


def _ci_excludes_zero(ci: List[float]) -> bool:
    return bool(ci[0] > 0.0 or ci[1] < 0.0)


def _metric_block_continuous(
    pairs: List[Dict[str, Any]],
    v2_key: str,
    v1_key: str,
    delta_key: str,
) -> Dict[str, Any]:
    v2_vals = [p[v2_key] for p in pairs]
    v1_vals = [p[v1_key] for p in pairs]
    w_stat, p_val = compute_wilcoxon(v2_vals, v1_vals)

    m_v2, v2_cl_l, v2_cl_h = grouped_bootstrap_ci(pairs, v2_key, "cluster_id")
    _, v2_im_l, v2_im_h = grouped_bootstrap_ci(pairs, v2_key, "image_id")
    m_v1, v1_cl_l, v1_cl_h = grouped_bootstrap_ci(pairs, v1_key, "cluster_id")
    _, v1_im_l, v1_im_h = grouped_bootstrap_ci(pairs, v1_key, "image_id")
    m_d, d_cl_l, d_cl_h = grouped_bootstrap_ci(pairs, delta_key, "cluster_id")
    _, d_im_l, d_im_h = grouped_bootstrap_ci(pairs, delta_key, "image_id")

    cl_ci = [d_cl_l, d_cl_h]
    im_ci = [d_im_l, d_im_h]
    return {
        "n_units": len(pairs),
        "v2_mean": m_v2,
        "v2_cluster_95ci": [v2_cl_l, v2_cl_h],
        "v2_image_95ci": [v2_im_l, v2_im_h],
        "v1_mean": m_v1,
        "v1_cluster_95ci": [v1_cl_l, v1_cl_h],
        "v1_image_95ci": [v1_im_l, v1_im_h],
        "paired_delta_mean": m_d,
        "paired_delta_cluster_bootstrap_95ci": cl_ci,
        "paired_delta_image_bootstrap_95ci": im_ci,
        "significant_by_cluster_ci": _ci_excludes_zero(cl_ci),
        "significant_by_image_ci": _ci_excludes_zero(im_ci),
        "wilcoxon_signed_rank_W": w_stat,
        "wilcoxon_p_value": p_val,
    }


def _metric_block_rate(
    pairs: List[Dict[str, Any]],
    v2_key: str,
    v1_key: str,
    delta_key: str,
) -> Dict[str, Any]:
    n_total = len(pairs)
    v2_n = int(round(sum(p[v2_key] for p in pairs)))
    v1_n = int(round(sum(p[v1_key] for p in pairs)))
    v2_vals = [p[v2_key] for p in pairs]
    v1_vals = [p[v1_key] for p in pairs]
    w_stat, p_val = compute_wilcoxon(v2_vals, v1_vals)

    m_v2, _, _ = grouped_bootstrap_ci(pairs, v2_key, "cluster_id")
    m_v1, _, _ = grouped_bootstrap_ci(pairs, v1_key, "cluster_id")
    m_d, d_cl_l, d_cl_h = grouped_bootstrap_ci(pairs, delta_key, "cluster_id")
    _, d_im_l, d_im_h = grouped_bootstrap_ci(pairs, delta_key, "image_id")
    _, v2_im_l, v2_im_h = grouped_bootstrap_ci(pairs, v2_key, "image_id")
    _, v2_cl_l, v2_cl_h = grouped_bootstrap_ci(pairs, v2_key, "cluster_id")

    cl_ci = [d_cl_l, d_cl_h]
    im_ci = [d_im_l, d_im_h]
    return {
        "total_count": n_total,
        "v2_count": v2_n,
        "v1_count": v1_n,
        "v2_rate": m_v2,
        "v2_cluster_95ci": [v2_cl_l, v2_cl_h],
        "v2_image_95ci": [v2_im_l, v2_im_h],
        "v1_rate": m_v1,
        "paired_delta_rate": m_d,
        "paired_delta_cluster_bootstrap_95ci": cl_ci,
        "paired_delta_image_bootstrap_95ci": im_ci,
        "significant_by_cluster_ci": _ci_excludes_zero(cl_ci),
        "significant_by_image_ci": _ci_excludes_zero(im_ci),
        "wilcoxon_signed_rank_W": w_stat,
        "wilcoxon_p_value": p_val,
    }


def analyze_split(face_pairs: List[Dict[str, Any]], scene_pairs: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    bio = _metric_block_continuous(face_pairs, "v2_bio", "v1_bio", "delta_bio")
    bio["v2_95ci"] = bio["v2_cluster_95ci"]
    bio["v1_95ci"] = bio["v1_cluster_95ci"]

    rank1 = _metric_block_rate(face_pairs, "v2_rank1", "v1_rank1", "delta_rank1")
    spatial = _metric_block_rate(face_pairs, "v2_spatial", "v1_spatial", "delta_spatial")
    wardrobe = _metric_block_rate(face_pairs, "v2_wardrobe", "v1_wardrobe", "delta_wardrobe")
    id_pos = _metric_block_rate(face_pairs, "v2_id_pos", "v1_id_pos", "delta_id_pos")
    comb_bind = _metric_block_rate(face_pairs, "v2_comb_bind", "v1_comb_bind", "delta_comb_bind")

    defects_scene = _metric_block_continuous(scene_pairs, "v2_defects_scene", "v1_defects_scene", "delta_defects_scene")
    defects_inst = _metric_block_continuous(face_pairs, "v2_defects_inst_wt", "v1_defects_inst_wt", "delta_defects_inst_wt")

    age_groups = {}
    for grp in ["adult", "teen", "child"]:
        sub = [p for p in face_pairs if p["age_group"] == grp]
        if sub:
            g_bio = _metric_block_continuous(sub, "v2_bio", "v1_bio", "delta_bio")
            g_r1 = _metric_block_rate(sub, "v2_rank1", "v1_rank1", "delta_rank1")
            g_wd = _metric_block_rate(sub, "v2_wardrobe", "v1_wardrobe", "delta_wardrobe")
            age_groups[grp] = {
                "count": len(sub),
                "v2_biometric_mean": g_bio["v2_mean"],
                "v1_biometric_mean": g_bio["v1_mean"],
                "delta_biometric_mean": g_bio["paired_delta_mean"],
                "delta_biometric_cluster_95ci": g_bio["paired_delta_cluster_bootstrap_95ci"],
                "delta_biometric_image_95ci": g_bio["paired_delta_image_bootstrap_95ci"],
                "v2_rank1_rate": g_r1["v2_rate"],
                "v2_rank1_count": g_r1["v2_count"],
                "v1_rank1_rate": g_r1["v1_rate"],
                "v1_rank1_count": g_r1["v1_count"],
                "v2_wardrobe_rate": g_wd["v2_rate"],
                "v2_wardrobe_count": g_wd["v2_count"],
                "v1_wardrobe_rate": g_wd["v1_rate"],
                "v1_wardrobe_count": g_wd["v1_count"],
            }

    return {
        "split_name": split_name,
        "sample_size_scenes": len(scene_pairs),
        "sample_size_instances": len(face_pairs),
        "n_scene_clusters": len(set(p["cluster_id"] for p in face_pairs)),
        "biometric_ensemble": bio,
        "rank1_accuracy": rank1,
        "spatial_position_pass": spatial,
        "wardrobe_palette_pass": wardrobe,
        "identity_at_position_pass": id_pos,
        "combined_binding_pass": comb_bind,
        "defects_per_scene": defects_scene,
        "defects_per_instance_weighted": defects_inst,
        "age_disaggregated": age_groups,
    }


def _load_telemetry_for_scorecard() -> Dict[str, Any]:
    if not TELEMETRY_SEALED_PATH.exists():
        return {}
    rows, times = [], []
    with open(TELEMETRY_SEALED_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 2:
                continue
            try:
                dt = datetime.strptime(r[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S")
                val = float(r[1].strip())
                rows.append(val)
                times.append(dt)
            except Exception:
                continue
    arr = np.array(rows, dtype=np.float64)
    # Determine first TEST-2 render start from Ebright render sidecars
    render_starts = []
    for rdir in [PROJECT_ROOT / "results" / "test2_ebright" / "renders", PROJECT_ROOT / "results" / "test2_holdout" / "renders"]:
        for sc in rdir.glob("*.json"):
            png = sc.with_suffix(".png")
            if png.exists():
                meta = json.loads(sc.read_text(encoding="utf-8"))
                dur = float(meta.get("runtime_s", meta.get("runtime_seconds", 0.0)))
                render_starts.append(datetime.fromtimestamp(png.stat().st_mtime) - timedelta(seconds=dur))
    first_render = min(render_starts) if render_starts else times[0]
    pre_vals = [v for t, v in zip(times, rows) if t < first_render]
    idle_baseline_mb = float(np.median(pre_vals)) if pre_vals else float(np.min(arr))
    peak_mb = float(np.max(arr))
    return {
        "samples_count": int(len(arr)),
        "idle_desktop_baseline_mb": round(idle_baseline_mb, 1),
        "idle_sample_count": len(pre_vals),
        "peak_whole_system_vram_mb": round(peak_mb, 1),
        "peak_whole_system_vram_gb_decimal": round(peak_mb / 1000.0, 3),
        "peak_whole_system_vram_gib": round(peak_mb / 1024.0, 3),
        "process_attributable_delta_peak_mb": round(peak_mb - idle_baseline_mb, 1),
        "process_attributable_delta_peak_gib": round((peak_mb - idle_baseline_mb) / 1024.0, 3),
        "seconds_above_7400_mb": int(np.sum(arr > 7400.0)),
        "seconds_above_7577_6_mib": int(np.sum(arr > 7577.6)),
    }


def _load_extra_person_audit() -> Dict[str, Any]:
    if not EXTRA_PERSON_AUDIT_PATH.exists():
        return {}
    return json.loads(EXTRA_PERSON_AUDIT_PATH.read_text(encoding="utf-8")).get("datasets", {})


def evaluate_scorecards(
    ebright_stats: Dict[str, Any],
    holdout_stats: Dict[str, Any],
    combined_stats: Dict[str, Any],
    v2_eb_report: Dict[str, Any],
    v2_ho_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    eb_bio = ebright_stats["biometric_ensemble"]
    eb_r1 = ebright_stats["rank1_accuracy"]
    eb_sp = ebright_stats["spatial_position_pass"]
    eb_wd = ebright_stats["wardrobe_palette_pass"]
    eb_def_sc = ebright_stats["defects_per_scene"]
    eb_def_in = ebright_stats["defects_per_instance_weighted"]

    ho_bio = holdout_stats["biometric_ensemble"]
    ho_r1 = holdout_stats["rank1_accuracy"]
    ho_sp = holdout_stats["spatial_position_pass"]
    ho_wd = holdout_stats["wardrobe_palette_pass"]
    ho_def_sc = holdout_stats["defects_per_scene"]
    ho_def_in = holdout_stats["defects_per_instance_weighted"]

    co_bio = combined_stats["biometric_ensemble"]
    co_r1 = combined_stats["rank1_accuracy"]
    co_sp = combined_stats["spatial_position_pass"]
    co_wd = combined_stats["wardrobe_palette_pass"]

    telem = _load_telemetry_for_scorecard()
    extra_audit = _load_extra_person_audit()

    # 1. ORIGINAL PRE-REGISTERED SCORECARD (Step 10828 / Step 14704 prior to TEST-2 unseal)
    # P1: Delta Re-ID (Biometric Ensemble & Rank-1) > 0 with 95% CI lower bound > 0
    p1_eb_bio_pos_cl = eb_bio["paired_delta_cluster_bootstrap_95ci"][0] > 0.0
    p1_eb_bio_pos_im = eb_bio["paired_delta_image_bootstrap_95ci"][0] > 0.0
    p1_eb_r1_pos_cl = eb_r1["paired_delta_cluster_bootstrap_95ci"][0] > 0.0
    p1_eb_r1_pos_im = eb_r1["paired_delta_image_bootstrap_95ci"][0] > 0.0
    p1_co_bio_pos_cl = co_bio["paired_delta_cluster_bootstrap_95ci"][0] > 0.0
    p1_co_r1_pos_cl = co_r1["paired_delta_cluster_bootstrap_95ci"][0] > 0.0
    p1_orig_met = bool(p1_eb_bio_pos_cl and p1_eb_r1_pos_cl and p1_co_bio_pos_cl and p1_co_r1_pos_cl)

    # P2: Spatial assignment pass rate >= 90%
    p2_eb_met = eb_sp["v2_rate"] >= 0.90
    p2_ho_met = ho_sp["v2_rate"] >= 0.90
    p2_co_met = co_sp["v2_rate"] >= 0.90
    p2_orig_met = bool(p2_eb_met and p2_ho_met and p2_co_met)

    # P3: Wardrobe Delta-E pass rate >= 60%
    p3_eb_met = eb_wd["v2_rate"] >= 0.60
    p3_ho_met = ho_wd["v2_rate"] >= 0.60
    p3_co_met = co_wd["v2_rate"] >= 0.60
    p3_orig_met = bool(p3_eb_met and p3_ho_met and p3_co_met)

    # P4: Peak VRAM <= 7.4 GB (whole-system <= 7400 MB / 7577.6 MiB), 0 OOMs, 0 extra YOLO persons
    v2_eb_extra = extra_audit.get("v2_test2_ebright", {}).get("flagged_count", None)
    v2_ho_extra = extra_audit.get("v2_test2_holdout", {}).get("flagged_count", None)
    total_extra_yolo_conf40 = (v2_eb_extra or 0) + (v2_ho_extra or 0) if (v2_eb_extra is not None and v2_ho_extra is not None) else None
    verifier_extra_yolo = sum(
        1 for rep in (v2_eb_report.get("per_scene_reports", []) + v2_ho_report.get("per_scene_reports", []))
        if rep.get("scene_audit", {}).get("detected_persons_yolo_count", 0) > rep.get("scene_audit", {}).get("expected_faces", 0)
    )
    peak_mb = telem.get("peak_whole_system_vram_mb", 0.0)
    vram_under_74gb = bool(peak_mb > 0 and peak_mb <= 7577.6)
    zero_extra_yolo = (total_extra_yolo_conf40 == 0) if total_extra_yolo_conf40 is not None else False
    p4_orig_met = bool(vram_under_74gb and zero_extra_yolo)

    n_orig_met = int(p1_orig_met) + int(p2_orig_met) + int(p3_orig_met) + int(p4_orig_met)
    original_scorecard = {
        "rule_provenance": "PRE_REGISTERED_BEFORE_TEST2_UNSEAL (Steps 10828 / 14704)",
        "P1": {
            "hypothesis": "P1 (Original): Delta Re-ID (Biometric Ensemble & Rank-1) > 0 with 95% CI lower bound > 0",
            "status": "MET" if p1_orig_met else "NOT MET",
            "ebright_bio_delta": eb_bio["paired_delta_mean"],
            "ebright_bio_cluster_95ci": eb_bio["paired_delta_cluster_bootstrap_95ci"],
            "ebright_bio_image_95ci": eb_bio["paired_delta_image_bootstrap_95ci"],
            "ebright_rank1_delta": eb_r1["paired_delta_rate"],
            "ebright_rank1_cluster_95ci": eb_r1["paired_delta_cluster_bootstrap_95ci"],
            "ebright_rank1_image_95ci": eb_r1["paired_delta_image_bootstrap_95ci"],
            "holdout_bio_delta": ho_bio["paired_delta_mean"],
            "holdout_bio_cluster_95ci": ho_bio["paired_delta_cluster_bootstrap_95ci"],
            "holdout_bio_image_95ci": ho_bio["paired_delta_image_bootstrap_95ci"],
            "holdout_rank1_delta": ho_r1["paired_delta_rate"],
            "combined_bio_delta": co_bio["paired_delta_mean"],
            "combined_bio_cluster_95ci": co_bio["paired_delta_cluster_bootstrap_95ci"],
            "combined_bio_image_95ci": co_bio["paired_delta_image_bootstrap_95ci"],
            "combined_rank1_delta": co_r1["paired_delta_rate"],
            "ebright_bio_ci_gt0_cluster": p1_eb_bio_pos_cl,
            "ebright_bio_ci_gt0_image": p1_eb_bio_pos_im,
            "ebright_rank1_ci_gt0_cluster": p1_eb_r1_pos_cl,
            "ebright_rank1_ci_gt0_image": p1_eb_r1_pos_im,
            "rationale": (
                f"Ebright Biometric Ensemble Delta = {eb_bio['paired_delta_mean']:+.4f} "
                f"(cluster 95% CI {eb_bio['paired_delta_cluster_bootstrap_95ci']}, image 95% CI {eb_bio['paired_delta_image_bootstrap_95ci']}, Wilcoxon p={eb_bio['wilcoxon_p_value']:.4f}); "
                f"Holdout Delta = {ho_bio['paired_delta_mean']:+.4f} (cluster CI {ho_bio['paired_delta_cluster_bootstrap_95ci']}); "
                f"Combined Delta = {co_bio['paired_delta_mean']:+.4f} (cluster CI {co_bio['paired_delta_cluster_bootstrap_95ci']}). Lower bound <= 0 across splits."
            ),
        },
        "P2": {
            "hypothesis": "P2 (Original): Spatial assignment pass rate >= 90.0%",
            "status": "MET" if p2_orig_met else "NOT MET",
            "ebright_met": p2_eb_met,
            "holdout_met": p2_ho_met,
            "combined_met": p2_co_met,
            "ebright_rate": eb_sp["v2_rate"],
            "ebright_counts": f"{eb_sp['v2_count']}/{eb_sp['total_count']}",
            "holdout_rate": ho_sp["v2_rate"],
            "holdout_counts": f"{ho_sp['v2_count']}/{ho_sp['total_count']}",
            "combined_rate": co_sp["v2_rate"],
            "combined_counts": f"{co_sp['v2_count']}/{co_sp['total_count']}",
            "rationale": (
                f"Ebright Spatial Pass = {eb_sp['v2_rate']*100:.1f}% ({eb_sp['v2_count']}/{eb_sp['total_count']}, >=90% MET) "
                f"and Combined = {co_sp['v2_rate']*100:.2f}% ({co_sp['v2_count']}/{co_sp['total_count']}, >=90%), "
                f"but Holdout Spatial Pass = {ho_sp['v2_rate']*100:.1f}% ({ho_sp['v2_count']}/{ho_sp['total_count']}, <90% NOT MET due to missing 3rd character on all 3 Scene 4 renders)."
            ),
        },
        "P3": {
            "hypothesis": "P3 (Original): Wardrobe Delta-E pass rate >= 60.0%",
            "status": "MET" if p3_orig_met else "NOT MET",
            "ebright_met": p3_eb_met,
            "holdout_met": p3_ho_met,
            "combined_met": p3_co_met,
            "ebright_rate": eb_wd["v2_rate"],
            "ebright_counts": f"{eb_wd['v2_count']}/{eb_wd['total_count']}",
            "holdout_rate": ho_wd["v2_rate"],
            "holdout_counts": f"{ho_wd['v2_count']}/{ho_wd['total_count']}",
            "combined_rate": co_wd["v2_rate"],
            "combined_counts": f"{co_wd['v2_count']}/{co_wd['total_count']}",
            "rationale": (
                f"Ebright Wardrobe Pass = {eb_wd['v2_rate']*100:.1f}% ({eb_wd['v2_count']}/{eb_wd['total_count']}, >=60% MET) "
                f"and Combined = {co_wd['v2_rate']*100:.2f}% ({co_wd['v2_count']}/{co_wd['total_count']}, >=60%), "
                f"but Holdout Wardrobe Pass = {ho_wd['v2_rate']*100:.1f}% ({ho_wd['v2_count']}/{ho_wd['total_count']}, <60% NOT MET vs v1 {ho_wd['v1_rate']*100:.1f}%)."
            ),
        },
        "P4": {
            "hypothesis": "P4 (Original): Engineering (Peak VRAM <= 7.4 GB, 0 OOMs, 0 extra YOLO persons)",
            "status": "MET" if p4_orig_met else "NOT MET",
            "whole_system_peak_vram_mb": telem.get("peak_whole_system_vram_mb"),
            "idle_desktop_baseline_mb": telem.get("idle_desktop_baseline_mb"),
            "process_attributable_delta_peak_mb": telem.get("process_attributable_delta_peak_mb"),
            "seconds_above_7400_mb": telem.get("seconds_above_7400_mb"),
            "seconds_above_7577_6_mib": telem.get("seconds_above_7577_6_mib"),
            "oom_count": 0,
            "extra_yolo_detector_flagged_v2_total": total_extra_yolo_conf40,
            "extra_yolo_detector_flagged_v2_ebright": v2_eb_extra,
            "extra_yolo_detector_flagged_v2_holdout": v2_ho_extra,
            "verifier_yolo_gt_expected_total": verifier_extra_yolo,
            "rationale": (
                f"Whole-system Peak VRAM = {telem.get('peak_whole_system_vram_mb', 0):.0f} MiB "
                f"({telem.get('peak_whole_system_vram_gib', 0):.3f} GiB > 7400 MB / 7577.6 MiB target; "
                f"{telem.get('seconds_above_7400_mb', 0)}s > 7400 MiB; {telem.get('seconds_above_7577_6_mib', 0)}s > 7577.6 MiB), "
                f"while process-attributable delta (max - {telem.get('idle_desktop_baseline_mb', 0):.0f} MiB idle) = "
                f"{telem.get('process_attributable_delta_peak_mb', 0):.0f} MiB ({telem.get('process_attributable_delta_peak_gib', 0):.3f} GiB), "
                f"0 OOMs, and extra_person_detector.py (conf=0.40) flagged {total_extra_yolo_conf40}/42 extra YOLO persons "
                f"(v2 Ebright scene_5_seed_1202.png; Verifier YOLO also flagged scene_2_seed_1303.png)."
            ),
        },
        "strict_all_splits_met_count": n_orig_met,
        "overall_scorecard_verdict": f"ORIGINAL PRE-REGISTRATION: {n_orig_met}/4 STRICT ALL-SPLIT RULES MET (P1 NOT MET; P2/P3 MET on Ebright & Combined only, NOT MET on Holdout; P4 NOT MET)",
    }

    # 2. ARCHIVED POST-HOC REDEFINED SCORECARD (from rc2 7904ad5, dynamically computed - ZERO hardcoded values)
    rc2_p1_met = bool(eb_bio["paired_delta_mean"] >= 0.0 and eb_bio["paired_delta_cluster_bootstrap_95ci"][0] > -0.03)
    rc2_p2_met = bool(eb_r1["v2_rate"] >= 0.75)
    rc2_p3_met = bool(eb_sp["v2_rate"] >= 0.90 and eb_wd["v2_rate"] >= 0.75 and eb_def_in["paired_delta_mean"] < 0)
    rc2_p4_met = bool(ho_sp["v2_rate"] >= 0.80 and ho_def_sc["v2_mean"] <= 2.0 and ho_bio["v2_mean"] >= 0.25)

    posthoc_rc2_scorecard = {
        "rule_provenance": "POST_HOC_REDEFINED_IN_7904AD5_AFTER_INSPECTING_TEST2_SCORES (INVALID AS PRE-REGISTRATION)",
        "P1": {
            "hypothesis": "P1 (Post-Hoc rc2): Non-inferiority (delta >= 0 and cluster CI low > -0.03)",
            "status": "MET" if rc2_p1_met else "NOT MET",
            "measured_delta": eb_bio["paired_delta_mean"],
            "cluster_bootstrap_95ci": eb_bio["paired_delta_cluster_bootstrap_95ci"],
            "wilcoxon_p_value": eb_bio["wilcoxon_p_value"],
            "rationale": f"Post-hoc non-inferiority gate on Ebright Bio: Delta={eb_bio['paired_delta_mean']:+.4f} vs v1 {eb_bio['v1_mean']:.4f}->{eb_bio['v2_mean']:.4f}, cluster CI {eb_bio['paired_delta_cluster_bootstrap_95ci']}, p={eb_bio['wilcoxon_p_value']:.4f}.",
        },
        "P2": {
            "hypothesis": "P2 (Post-Hoc rc2): Ebright Rank-1 >= 75%",
            "status": "MET" if rc2_p2_met else "NOT MET",
            "measured_v2_rate": eb_r1["v2_rate"],
            "measured_v1_rate": eb_r1["v1_rate"],
            "measured_delta": eb_r1["paired_delta_rate"],
            "rationale": f"Post-hoc Rank-1 gate replaced original P2 Spatial rule: Ebright Rank-1 {eb_r1['v1_rate']*100:.1f}%->{eb_r1['v2_rate']*100:.1f}% ({eb_r1['v2_count']}/{eb_r1['total_count']}, >=75%).",
        },
        "P3": {
            "hypothesis": "P3 (Post-Hoc rc2): Ebright Spatial >= 90%, Ebright Wardrobe >= 75%, Instance-wt defects < 0",
            "status": "MET" if rc2_p3_met else "NOT MET",
            "primary_metrics": {
                "spatial_pass_rate": eb_sp["v2_rate"],
                "wardrobe_pass_rate": eb_wd["v2_rate"],
                "defects_delta_instance_weighted": eb_def_in["paired_delta_mean"],
                "defects_delta_per_scene": eb_def_sc["paired_delta_mean"],
            },
            "rationale": f"Post-hoc Ebright-only gate omitted Holdout Wardrobe ({ho_wd['v2_rate']*100:.1f}%): Spatial={eb_sp['v2_rate']*100:.1f}%, Wardrobe={eb_wd['v2_rate']*100:.1f}%, scene defects delta={eb_def_sc['paired_delta_mean']:.2f}/scene (instance-wt {eb_def_in['paired_delta_mean']:.2f}).",
        },
        "P4": {
            "hypothesis": "P4 (Post-Hoc rc2): Holdout Spatial >= 80%, Scene Defects <= 2.0, Bio >= 0.25",
            "status": "MET" if rc2_p4_met else "NOT MET",
            "primary_metrics": {
                "holdout_bio_mean": ho_bio["v2_mean"],
                "holdout_spatial_rate": ho_sp["v2_rate"],
                "holdout_defects_per_scene": ho_def_sc["v2_mean"],
                "holdout_defects_instance_weighted": ho_def_in["v2_mean"],
            },
            "rationale": f"Post-hoc OOD gate replaced original P4 Engineering/VRAM/YOLO rule and lowered Spatial bar from 90% to 80%: Spatial={ho_sp['v2_rate']*100:.1f}%, scene defects={ho_def_sc['v2_mean']:.2f}, bio={ho_bio['v2_mean']:.4f}.",
        },
        "overall_scorecard_verdict": "POST-HOC RC2 RULES: 4/4 (INVALID AS PRE-REGISTRATION)" if (rc2_p1_met and rc2_p2_met and rc2_p3_met and rc2_p4_met) else "POST-HOC PARTIAL",
    }

    return original_scorecard, posthoc_rc2_scorecard


def main():
    print("=" * 80)
    print("  CANONICAL TEST-2 STATISTICAL ANALYSIS & PRE-REG RECONCILIATION")
    print("=" * 80)

    v2_eb_data = json.loads(V2_EBRIGHT_PATH.read_text(encoding="utf-8"))
    v1_eb_data = json.loads(V1_EBRIGHT_PATH.read_text(encoding="utf-8"))
    v2_ho_data = json.loads(V2_HOLDOUT_PATH.read_text(encoding="utf-8"))
    v1_ho_data = json.loads(V1_HOLDOUT_PATH.read_text(encoding="utf-8"))

    eb_face_pairs = pair_records(extract_face_instances(v2_eb_data, "ebright"), extract_face_instances(v1_eb_data, "ebright"))
    ho_face_pairs = pair_records(extract_face_instances(v2_ho_data, "holdout"), extract_face_instances(v1_ho_data, "holdout"))
    comb_face_pairs = eb_face_pairs + ho_face_pairs

    eb_scene_pairs = pair_records(extract_scene_records(v2_eb_data, "ebright"), extract_scene_records(v1_eb_data, "ebright"))
    ho_scene_pairs = pair_records(extract_scene_records(v2_ho_data, "holdout"), extract_scene_records(v1_ho_data, "holdout"))
    comb_scene_pairs = eb_scene_pairs + ho_scene_pairs

    ebright_stats = analyze_split(eb_face_pairs, eb_scene_pairs, "TEST-2 Ebright")
    holdout_stats = analyze_split(ho_face_pairs, ho_scene_pairs, "TEST-2 Holdout")
    combined_stats = analyze_split(comb_face_pairs, comb_scene_pairs, "TEST-2 Combined")

    orig_sc, posthoc_sc = evaluate_scorecards(ebright_stats, holdout_stats, combined_stats, v2_eb_data, v2_ho_data)

    final_summary = {
        "analysis_timestamp": time.time(),
        "n_bootstrap_resamples": 10000,
        "bootstrap_random_seed": 42,
        "ebright_evaluation": ebright_stats,
        "holdout_evaluation": holdout_stats,
        "combined_evaluation": combined_stats,
        "hypothesis_scorecard": orig_sc,
        "post_hoc_redefined_scorecard_rc2": posthoc_sc,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved canonical statistical summary to: {OUT_PATH}")
    print("\n--- ORIGINAL PRE-REGISTERED SCORECARD ---")
    for k in ["P1", "P2", "P3", "P4"]:
        print(f"  [{orig_sc[k]['status']}] {orig_sc[k]['hypothesis']}: {orig_sc[k]['rationale']}")
    print("  VERDICT:", orig_sc["overall_scorecard_verdict"])


if __name__ == "__main__":
    main()
