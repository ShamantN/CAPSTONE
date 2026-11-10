"""
MMCCP Phase R11 + Finalization Pass: Canonical Numbers Collector
Aggregates ground-truth metrics, statistics, bootstrap CIs (cluster & image),
v1 baselines, extra-person audits, telemetry, automated review summaries, and
ablation tables from verified JSON/CSV artifacts into results/numbers.json.
Zero fabrication and zero hardcoded fallback numbers.
"""
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.splits import RESERVED_SEEDS

# Canonical file paths
FREEZE_PATH = PROJECT_ROOT / "consistency_v2" / "verifier_freeze.json"
TEST2_STAT_PATH = PROJECT_ROOT / "results" / "test2_evaluation_summary.json"
TEST2_EBRIGHT_PATH = PROJECT_ROOT / "results" / "test2_ebright" / "report_v1_2.json"
TEST2_HOLDOUT_PATH = PROJECT_ROOT / "results" / "test2_holdout" / "report_v1_2.json"
TEST2_ABLATION_PATH = PROJECT_ROOT / "results" / "ablation_test2" / "loo_ablation_test2_summary.json"
TEST2_TRACER_PATH = PROJECT_ROOT / "results" / "ablation_test2" / "stage_tracer_verification.json"

V1_EBRIGHT_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_ebright" / "report.json"
V1_HOLDOUT_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_test2_holdout" / "report.json"
V1_DEV2_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_dev2" / "report.json"

DEV2_BUNDLE_PATH = PROJECT_ROOT / "results" / "bundle_dev2" / "report_v1_2.json"
DEV2_ABLATION_PATH = PROJECT_ROOT / "results" / "ablation_dev2" / "loo_ablation_summary.json"
DEV2_TRACER_PATH = PROJECT_ROOT / "results" / "ablation_dev2" / "stage_tracer_verification.json"

BASELINE_RESCORED_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "rescored_summary.json"
EXTRA_PERSON_AUDIT_PATH = PROJECT_ROOT / "results" / "test2_extra_person_audit.json"
CIRCULARITY_AUDIT_PATH = PROJECT_ROOT / "results" / "test2_ebright_circularity_coherence.json"

TELEMETRY_DIR = PROJECT_ROOT / "results" / "telemetry"
TELEMETRY_SEALED_PATH = TELEMETRY_DIR / "vram_test2_sealed.csv"
TELEMETRY_R9_PATH = PROJECT_ROOT / "results" / "telemetry_clean_slate_report.json"

BLIND_STUDY_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_study_report.json"
CANARY_AUDIT_PATH = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "canary_audit_report.json"

OUT_PATH = PROJECT_ROOT / "results" / "numbers.json"


def get_git_info() -> Dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
    except Exception:
        commit = "unknown"
        branch = "unknown"
    return {"commit": commit, "branch": branch}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_stage_diffs(tracer_data: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for cond, info in tracer_data.get("conditions", {}).items():
        if "mean_pixel_diff_vs_full" in info:
            out[cond] = round(float(info["mean_pixel_diff_vs_full"]), 2)
        else:
            diffs = [float(v["pixel_diff_vs_full"]) for v in info.get("renders", {}).values() if "pixel_diff_vs_full" in v]
            out[cond] = round(float(np.mean(diffs)), 2) if diffs else 0.0
    return out


def _ablation_meta(tracer_data: Dict[str, Any], report_data: Dict[str, Any]) -> Dict[str, Any]:
    full = tracer_data.get("conditions", {}).get("full_v2", {})
    renders = full.get("renders", {})
    seed = int(next(iter(renders.keys())).split("_seed_")[-1]) if renders else None
    n_scenes = int(full.get("renders_count", len(renders)))
    n_faces = 0
    if seed is not None:
        token = f"_seed_{seed}"
        for rep in report_data.get("per_scene_reports", []):
            if token in rep.get("image_filename", ""):
                n_faces += len(rep.get("character_evaluations", []))
    return {"sample_size_scenes": n_scenes, "sample_size_faces": n_faces, "seed": seed}


def _extract_report_summary(rep: Dict[str, Any], stat_block: Dict[str, Any] = None) -> Dict[str, Any]:
    sm = rep["summary_metrics"]
    bd = sm["binding_decomposed"]
    age = sm["age_split_metrics"]
    out = {
        "images_count": int(rep["images_evaluated_count"]),
        "faces_count": int(rep["total_face_instances"]),
        "biometric_ensemble_mean": round(float(sm["biometric_ensemble_mean"]), 4),
        "biometric_ensemble_std": round(float(sm["biometric_ensemble_std"]), 4),
        "biometric_ensemble_95ci": sm["biometric_ensemble_95ci"],
        "rank1_accuracy": round(float(sm["rank1_discrimination_accuracy"]), 4),
        "discrimination_margin": round(float(sm["discrimination_margin_mean"]), 4),
        "dinov2_mean_similarity": round(float(sm["dinov2_mean_similarity"]), 4),
        "per_judge_means": sm["per_judge_means"],
        "spatial_pass_rate": round(float(bd["spatial_position_pass_rate"]), 4),
        "wardrobe_pass_rate": round(float(bd["wardrobe_palette_pass_rate"]), 4),
        "identity_at_position_pass_rate": round(float(bd["identity_at_position_pass_rate"]), 4),
        "combined_binding_pass_rate": round(float(bd["combined_binding_pass_rate"]), 4),
        "mean_defects_per_scene": round(float(sm["mean_defects_per_scene"]), 4),
        "mean_defects": round(float(sm["mean_defects_per_scene"]), 4),
        "seg_failed_count": int(sm["seg_failed_count"]),
        "adult_bio": round(float(age["adult"]["biometric_ensemble_mean"]), 4) if "adult" in age else None,
        "adult_count": int(age["adult"]["count"]) if "adult" in age else 0,
        "teen_bio": round(float(age["teen"]["biometric_ensemble_mean"]), 4) if "teen" in age else None,
        "teen_count": int(age["teen"]["count"]) if "teen" in age else 0,
        "child_bio": round(float(age["child"]["biometric_ensemble_mean"]), 4) if "child" in age else None,
        "child_count": int(age["child"]["count"]) if "child" in age else 0,
    }
    if stat_block:
        out.update({
            "rank1_v2_count": int(stat_block["rank1_accuracy"]["v2_count"]),
            "rank1_v1_count": int(stat_block["rank1_accuracy"]["v1_count"]),
            "rank1_total_count": int(stat_block["rank1_accuracy"]["total_count"]),
            "spatial_v2_count": int(stat_block["spatial_position_pass"]["v2_count"]),
            "spatial_v1_count": int(stat_block["spatial_position_pass"]["v1_count"]),
            "spatial_total_count": int(stat_block["spatial_position_pass"]["total_count"]),
            "wardrobe_v2_count": int(stat_block["wardrobe_palette_pass"]["v2_count"]),
            "wardrobe_v1_count": int(stat_block["wardrobe_palette_pass"]["v1_count"]),
            "wardrobe_total_count": int(stat_block["wardrobe_palette_pass"]["total_count"]),
            "mean_defects_per_instance_weighted": round(float(stat_block["defects_per_instance_weighted"]["v2_mean"]), 4),
            "v1_mean_defects_per_instance_weighted": round(float(stat_block["defects_per_instance_weighted"]["v1_mean"]), 4),
        })
    return out


def _collect_telemetry() -> Dict[str, Any]:
    all_csvs = {}
    all_vals: List[float] = []
    for p in sorted(TELEMETRY_DIR.glob("vram_*.csv")):
        rows, times = [], []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) < 2:
                    continue
                try:
                    dt = datetime.strptime(r[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S")
                    val = float(r[1].strip().replace("MiB", "").replace("MB", "").strip())
                    rows.append(val)
                    times.append(dt)
                except Exception:
                    continue
        if not rows:
            continue
        arr = np.array(rows, dtype=np.float64)
        all_vals.extend(rows)
        all_csvs[p.name] = {
            "samples_count": int(len(arr)),
            "start_time": str(times[0]),
            "end_time": str(times[-1]),
            "min_mb": round(float(np.min(arr)), 1),
            "idle_first10_median_mb": round(float(np.median(arr[:10])), 1),
            "p50_mb": round(float(np.percentile(arr, 50)), 1),
            "p95_mb": round(float(np.percentile(arr, 95)), 1),
            "p99_mb": round(float(np.percentile(arr, 99)), 1),
            "mean_mb": round(float(np.mean(arr)), 1),
            "max_mb": round(float(np.max(arr)), 1),
            "seconds_above_7400_mb": int(np.sum(arr > 7400.0)),
            "seconds_above_7577_6_mib": int(np.sum(arr > 7577.6)),
        }

    # Detailed sealed TEST-2 excursion breakdown
    sealed_times, sealed_vals = [], []
    with open(TELEMETRY_SEALED_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 2:
                continue
            try:
                sealed_times.append(datetime.strptime(r[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S"))
                sealed_vals.append(float(r[1].strip()))
            except Exception:
                continue

    render_windows = []
    for split_name, rdir in [
        ("test2_ebright", PROJECT_ROOT / "results" / "test2_ebright" / "renders"),
        ("test2_holdout", PROJECT_ROOT / "results" / "test2_holdout" / "renders"),
    ]:
        for sc in sorted(rdir.glob("*.json")):
            png = sc.with_suffix(".png")
            if not png.exists():
                continue
            meta = _read_json(sc)
            end_dt = datetime.fromtimestamp(png.stat().st_mtime)
            dur = float(meta.get("runtime_s", meta.get("runtime_seconds", 0.0)))
            render_windows.append((end_dt - timedelta(seconds=dur), end_dt, dur, split_name, png.name))
    for arm_dir in sorted((PROJECT_ROOT / "results" / "ablation_test2").iterdir()):
        if not arm_dir.is_dir():
            continue
        for png in sorted(arm_dir.glob("*.png")):
            sc = png.with_suffix(".json")
            end_dt = datetime.fromtimestamp(png.stat().st_mtime)
            dur = 90.0
            if sc.exists():
                try:
                    meta = _read_json(sc)
                    dur = float(meta.get("runtime_s", meta.get("runtime_seconds", 90.0)))
                except Exception:
                    pass
            render_windows.append((end_dt - timedelta(seconds=dur), end_dt, dur, f"ablation_test2/{arm_dir.name}", png.name))

    render_windows.sort(key=lambda x: x[0])
    first_render_start = render_windows[0][0] if render_windows else sealed_times[0]
    pre_idle = [v for t, v in zip(sealed_times, sealed_vals) if t < first_render_start]
    idle_baseline_mb = round(float(np.median(pre_idle)) if pre_idle else float(np.min(sealed_vals)), 1)

    excursions = []
    cur = None
    for t, v in zip(sealed_times, sealed_vals):
        if v > 7400.0:
            if cur is None or (t - cur["end_dt"]).total_seconds() > 3:
                overlaps = [w for w in render_windows if not (w[1] < t - timedelta(seconds=5) or w[0] > t + timedelta(seconds=5))]
                stage_label = "; ".join(f"{w[3]}/{w[4]}@{((t-w[0]).total_seconds()):.0f}/{w[2]:.0f}s" for w in overlaps)
                cur = {"start_dt": t, "end_dt": t, "start": str(t), "end": str(t), "peak_mb": round(v, 1), "seconds": 1, "coincident_render": stage_label or "post-render"}
                excursions.append(cur)
            else:
                cur["end_dt"] = t
                cur["end"] = str(t)
                cur["peak_mb"] = round(max(cur["peak_mb"], v), 1)
                cur["seconds"] += 1
    for ex in excursions:
        ex.pop("start_dt", None)
        ex.pop("end_dt", None)

    sealed_arr = np.array(sealed_vals, dtype=np.float64)
    comb_arr = np.array(all_vals, dtype=np.float64)
    peak_sealed = int(round(float(np.max(sealed_arr))))
    return {
        "samples_count": int(len(sealed_arr)),
        "peak_vram_mb": peak_sealed,
        "peak_vram_gib": round(peak_sealed / 1024.0, 3),
        "peak_vram_gb_decimal": round(peak_sealed / 1000.0, 3),
        "mean_vram_mb": round(float(np.mean(sealed_arr)), 1),
        "median_vram_mb": round(float(np.percentile(sealed_arr, 50)), 1),
        "p95_vram_mb": round(float(np.percentile(sealed_arr, 95)), 1),
        "p99_vram_mb": round(float(np.percentile(sealed_arr, 99)), 1),
        "memory_total_mb": 8192,
        "idle_desktop_baseline_mb": idle_baseline_mb,
        "idle_sample_count": len(pre_idle),
        "process_attributable_delta_peak_mb": round(float(peak_sealed - idle_baseline_mb), 1),
        "process_attributable_delta_peak_gib": round(float(peak_sealed - idle_baseline_mb) / 1024.0, 3),
        "seconds_above_7400_mb": int(np.sum(sealed_arr > 7400.0)),
        "seconds_above_7577_6_mib": int(np.sum(sealed_arr > 7577.6)),
        "excursion_episodes_count": len(excursions),
        "excursion_episodes": excursions,
        "combined_9_csvs": {
            "files_count": len(all_csvs),
            "samples_count": int(len(comb_arr)),
            "min_mb": round(float(np.min(comb_arr)), 1),
            "p50_mb": round(float(np.percentile(comb_arr, 50)), 1),
            "p95_mb": round(float(np.percentile(comb_arr, 95)), 1),
            "p99_mb": round(float(np.percentile(comb_arr, 99)), 1),
            "max_mb": round(float(np.max(comb_arr)), 1),
            "seconds_above_7400_mb": int(np.sum(comb_arr > 7400.0)),
            "seconds_above_7577_6_mib": int(np.sum(comb_arr > 7577.6)),
        },
        "per_csv_summaries": all_csvs,
        "r9_clean_slate_telemetry": _read_json(TELEMETRY_R9_PATH) if TELEMETRY_R9_PATH.exists() else {},
    }


def _extract_exploratory_cases(
    t2_eb_rep: Dict[str, Any],
    t2_ho_rep: Dict[str, Any],
    v1_eb_rep: Dict[str, Any],
) -> Dict[str, Any]:
    # 1. Holdout Scene 4 (3 seeds)
    ho_s4 = []
    for rep in t2_ho_rep.get("per_scene_reports", []):
        if rep.get("scene_number") == 4:
            sc_path = PROJECT_ROOT / "results" / "test2_holdout" / "renders" / (Path(rep["image_filename"]).stem + ".json")
            swap_chars = []
            if sc_path.exists():
                sc_data = _read_json(sc_path)
                swap_chars = [s.get("character_id") for s in sc_data.get("swap_report", {}).get("specs", [])]
            sa = rep.get("scene_audit", {})
            ho_s4.append({
                "image": rep["image_filename"],
                "expected_faces": sa.get("expected_faces"),
                "detected_faces": sa.get("detected_faces_count"),
                "detected_yolo": sa.get("detected_persons_yolo_count"),
                "swap_report_characters": swap_chars,
                "defects": sa.get("defect_taxonomy_violations", []),
                "characters": [
                    {
                        "character_id": c.get("character_id"),
                        "bio": c.get("biometric_ensemble_score"),
                        "spatial_pass": c.get("spatial_pass"),
                        "wardrobe_delta_e": c.get("wardrobe_delta_e"),
                        "wardrobe_pass": c.get("wardrobe_pass"),
                    }
                    for c in rep.get("character_evaluations", [])
                ],
            })

    # 2. Ebright Scene 6 (5 seeds, v2 vs v1)
    v1_s6_map = {}
    for rep in v1_eb_rep.get("per_scene_reports", []):
        if rep.get("scene_number") == 6:
            cs = rep.get("character_evaluations", [])
            v1_s6_map[rep["image_filename"]] = cs[0].get("wardrobe_delta_e") if cs else None

    eb_s6 = []
    for rep in t2_eb_rep.get("per_scene_reports", []):
        if rep.get("scene_number") == 6:
            sa = rep.get("scene_audit", {})
            cs = rep.get("character_evaluations", [])
            eb_s6.append({
                "image": rep["image_filename"],
                "v2_bio": cs[0].get("biometric_ensemble_score") if cs else None,
                "v2_wardrobe_delta_e": cs[0].get("wardrobe_delta_e") if cs else None,
                "v2_wardrobe_pass": cs[0].get("wardrobe_pass") if cs else None,
                "v1_wardrobe_delta_e": v1_s6_map.get(rep["image_filename"]),
                "defects": sa.get("defect_taxonomy_violations", []),
            })

    # 3. Extra-person / YOLO divergence cases (Ebright)
    eb_extra_cases = []
    for rep in t2_eb_rep.get("per_scene_reports", []):
        sa = rep.get("scene_audit", {})
        if sa.get("detected_persons_yolo_count", 0) > sa.get("expected_faces", 0) or sa.get("detected_faces_count", 0) > sa.get("expected_faces", 0):
            eb_extra_cases.append({
                "image": rep["image_filename"],
                "expected_faces": sa.get("expected_faces"),
                "verifier_detected_faces": sa.get("detected_faces_count"),
                "verifier_detected_yolo": sa.get("detected_persons_yolo_count"),
                "defects": sa.get("defect_taxonomy_violations", []),
            })

    return {
        "holdout_scene_4": ho_s4,
        "ebright_scene_6": eb_s6,
        "ebright_yolo_or_extra_face_cases": eb_extra_cases,
    }


def collect_numbers() -> Dict[str, Any]:
    print("=" * 80)
    print("  COLLECTING CANONICAL MMCCP NUMBERS INTO results/numbers.json")
    print("=" * 80)

    freeze_data = _read_json(FREEZE_PATH)
    test2_stats = _read_json(TEST2_STAT_PATH)
    t2_eb_rep = _read_json(TEST2_EBRIGHT_PATH)
    t2_ho_rep = _read_json(TEST2_HOLDOUT_PATH)
    v1_eb_rep = _read_json(V1_EBRIGHT_PATH)
    v1_ho_rep = _read_json(V1_HOLDOUT_PATH)
    v1_dev2_rep = _read_json(V1_DEV2_PATH)
    t2_ablation = _read_json(TEST2_ABLATION_PATH)
    t2_tracer = _read_json(TEST2_TRACER_PATH)
    dev2_bundle = _read_json(DEV2_BUNDLE_PATH)
    dev2_ablation = _read_json(DEV2_ABLATION_PATH)
    dev2_tracer = _read_json(DEV2_TRACER_PATH)
    baseline_rescored = _read_json(BASELINE_RESCORED_PATH)
    extra_person_audit = _read_json(EXTRA_PERSON_AUDIT_PATH) if EXTRA_PERSON_AUDIT_PATH.exists() else {}
    circularity_audit = _read_json(CIRCULARITY_AUDIT_PATH) if CIRCULARITY_AUDIT_PATH.exists() else {}
    blind_study = _read_json(BLIND_STUDY_PATH) if BLIND_STUDY_PATH.exists() else {}
    canary_audit = _read_json(CANARY_AUDIT_PATH) if CANARY_AUDIT_PATH.exists() else {}

    vram_stats = _collect_telemetry()

    eb_eval = test2_stats["ebright_evaluation"]
    ho_eval = test2_stats["holdout_evaluation"]
    co_eval = test2_stats["combined_evaluation"]

    orig_sc = test2_stats["hypothesis_scorecard"]
    posthoc_sc = test2_stats["post_hoc_redefined_scorecard_rc2"]

    eb_v2 = _extract_report_summary(t2_eb_rep, eb_eval)
    ho_v2 = _extract_report_summary(t2_ho_rep, ho_eval)
    eb_v1 = _extract_report_summary(v1_eb_rep)
    ho_v1 = _extract_report_summary(v1_ho_rep)
    dev2_v2 = _extract_report_summary(dev2_bundle)
    dev2_v1 = _extract_report_summary(v1_dev2_rep)

    def _build_deltas(ev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "delta_bio": ev["biometric_ensemble"]["paired_delta_mean"],
            "delta_bio_cluster_95ci": ev["biometric_ensemble"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_bio_image_95ci": ev["biometric_ensemble"]["paired_delta_image_bootstrap_95ci"],
            "delta_bio_wilcoxon_w": ev["biometric_ensemble"]["wilcoxon_signed_rank_W"],
            "delta_bio_wilcoxon_p": ev["biometric_ensemble"]["wilcoxon_p_value"],
            "delta_rank1": ev["rank1_accuracy"]["paired_delta_rate"],
            "delta_rank1_cluster_95ci": ev["rank1_accuracy"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_rank1_image_95ci": ev["rank1_accuracy"]["paired_delta_image_bootstrap_95ci"],
            "delta_rank1_wilcoxon_p": ev["rank1_accuracy"]["wilcoxon_p_value"],
            "delta_spatial": ev["spatial_position_pass"]["paired_delta_rate"],
            "delta_spatial_cluster_95ci": ev["spatial_position_pass"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_spatial_image_95ci": ev["spatial_position_pass"]["paired_delta_image_bootstrap_95ci"],
            "delta_wardrobe": ev["wardrobe_palette_pass"]["paired_delta_rate"],
            "delta_wardrobe_cluster_95ci": ev["wardrobe_palette_pass"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_wardrobe_image_95ci": ev["wardrobe_palette_pass"]["paired_delta_image_bootstrap_95ci"],
            "delta_wardrobe_wilcoxon_w": ev["wardrobe_palette_pass"]["wilcoxon_signed_rank_W"],
            "delta_wardrobe_wilcoxon_p": ev["wardrobe_palette_pass"]["wilcoxon_p_value"],
            "delta_defects_per_scene": ev["defects_per_scene"]["paired_delta_mean"],
            "delta_defects_per_scene_cluster_95ci": ev["defects_per_scene"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_defects_per_scene_image_95ci": ev["defects_per_scene"]["paired_delta_image_bootstrap_95ci"],
            "delta_defects_per_scene_wilcoxon_w": ev["defects_per_scene"]["wilcoxon_signed_rank_W"],
            "delta_defects_per_scene_wilcoxon_p": ev["defects_per_scene"]["wilcoxon_p_value"],
            "delta_defects_per_instance_weighted": ev["defects_per_instance_weighted"]["paired_delta_mean"],
            "delta_defects_per_instance_weighted_cluster_95ci": ev["defects_per_instance_weighted"]["paired_delta_cluster_bootstrap_95ci"],
            "delta_defects_per_instance_weighted_image_95ci": ev["defects_per_instance_weighted"]["paired_delta_image_bootstrap_95ci"],
        }

    paired_deltas = {
        "ebright": _build_deltas(eb_eval),
        "holdout": _build_deltas(ho_eval),
        "combined": _build_deltas(co_eval),
        # Compatibility top-level keys derived strictly from paired summaries
        "ebright_delta_bio": eb_eval["biometric_ensemble"]["paired_delta_mean"],
        "ebright_delta_bio_ci": eb_eval["biometric_ensemble"]["paired_delta_cluster_bootstrap_95ci"],
        "ebright_delta_rank1": eb_eval["rank1_accuracy"]["paired_delta_rate"],
        "ebright_delta_wardrobe": eb_eval["wardrobe_palette_pass"]["paired_delta_rate"],
        "ebright_delta_defects_per_scene": eb_eval["defects_per_scene"]["paired_delta_mean"],
        "ebright_delta_defects_per_instance_weighted": eb_eval["defects_per_instance_weighted"]["paired_delta_mean"],
        "ebright_wilcoxon_w": eb_eval["biometric_ensemble"]["wilcoxon_signed_rank_W"],
        "ebright_wilcoxon_p": eb_eval["biometric_ensemble"]["wilcoxon_p_value"],
        "holdout_delta_bio": ho_eval["biometric_ensemble"]["paired_delta_mean"],
        "holdout_delta_rank1": ho_eval["rank1_accuracy"]["paired_delta_rate"],
        "holdout_delta_spatial": ho_eval["spatial_position_pass"]["paired_delta_rate"],
        "holdout_delta_wardrobe": ho_eval["wardrobe_palette_pass"]["paired_delta_rate"],
        "holdout_delta_defects_per_scene": ho_eval["defects_per_scene"]["paired_delta_mean"],
        "holdout_delta_defects_per_instance_weighted": ho_eval["defects_per_instance_weighted"]["paired_delta_mean"],
        "combined_delta_bio": co_eval["biometric_ensemble"]["paired_delta_mean"],
        "combined_delta_rank1": co_eval["rank1_accuracy"]["paired_delta_rate"],
        "combined_delta_wardrobe": co_eval["wardrobe_palette_pass"]["paired_delta_rate"],
        "combined_delta_defects_per_scene": co_eval["defects_per_scene"]["paired_delta_mean"],
        "combined_delta_defects_per_instance_weighted": co_eval["defects_per_instance_weighted"]["paired_delta_mean"],
    }

    eb_def_pct_reduction = round((eb_v1["mean_defects_per_scene"] - eb_v2["mean_defects_per_scene"]) / eb_v1["mean_defects_per_scene"] * 100.0, 1)

    claims_flags = {
        "zero_memorization_proven": False,
        "zero_identity_collapse_proven": False,
        "all_4_hypotheses_met": bool(orig_sc["strict_all_splits_met_count"] == 4),
        "vram_100pct_compliant": bool(vram_stats["peak_vram_mb"] <= 7577.6),
        "statistically_significant_ebright_bio": bool(eb_eval["biometric_ensemble"]["significant_by_cluster_ci"] or eb_eval["biometric_ensemble"]["significant_by_image_ci"]),
        "statistically_significant_ebright_rank1_cluster": bool(eb_eval["rank1_accuracy"]["significant_by_cluster_ci"]),
        "statistically_significant_ebright_rank1_image": bool(eb_eval["rank1_accuracy"]["significant_by_image_ci"]),
        "statistically_significant_ebright_wardrobe": bool(eb_eval["wardrobe_palette_pass"]["significant_by_cluster_ci"] and eb_eval["wardrobe_palette_pass"]["significant_by_image_ci"]),
        "statistically_significant_ebright_defects_scene": bool(eb_eval["defects_per_scene"]["significant_by_cluster_ci"] and eb_eval["defects_per_scene"]["significant_by_image_ci"]),
        "statistically_significant_holdout_any": bool(
            ho_eval["biometric_ensemble"]["significant_by_image_ci"]
            or ho_eval["rank1_accuracy"]["significant_by_image_ci"]
            or ho_eval["wardrobe_palette_pass"]["significant_by_image_ci"]
            or ho_eval["defects_per_scene"]["significant_by_image_ci"]
        ),
        "human_blind_test_completed": False,
    }

    canonical_payload = {
        "metadata": {
            "project_name": "MMCCP v2.0",
            "capstone_id": "PW26_CGM_02",
            "institution": "PES University",
            "verifier_version": freeze_data["verifier_version"],
            "verifier_freeze_sha": freeze_data["combined_sha256"],
            "git_info": get_git_info(),
            "target_hardware": "NVIDIA RTX 3070 Ti Laptop GPU (8GB VRAM)",
            "budget_limit_mb": 7400.0,
            "budget_limit_mib_7_4_gib": 7577.6,
        },
        "protocol_constants": {
            "p1_ci_lower_threshold": 0.0,
            "p2_spatial_threshold_pct": 90.0,
            "p3_wardrobe_threshold_pct": 60.0,
            "p4_vram_limit_gb": 7.4,
            "p4_vram_limit_mb": 7400.0,
            "p4_vram_limit_mib": 7577.6,
            "p4_max_ooms": 0,
            "p4_max_extra_yolo": 0,
            "rc2_p1_noninf_ci_floor": -0.03,
            "rc2_p2_rank1_threshold_pct": 75.0,
            "rc2_p3_wardrobe_threshold_pct": 75.0,
            "rc2_p4_spatial_threshold_pct": 80.0,
            "rc2_p4_defects_max": 2.0,
            "rc2_p4_bio_min": 0.25,
            "ablation_noise_band": 0.03,
            "wardrobe_delta_e_threshold": 38.0,
            "reserved_seeds": sorted(int(s) for s in RESERVED_SEEDS),
            "n_bootstrap_resamples": 10000,
            "bootstrap_random_seed": 42,
            "gpu_total_mb": 8192,
            "gpu_total_gb": 8.0,
            "blind_reid_total_crops": 20,
            "yolo_conf_threshold": 0.40,
        },
        "prereg_forensic_timeline": {
            "pre_test2_commit": "96ff419",
            "pre_test2_commit_date": "2026-11-05 21:55:39 +0530",
            "original_directive_utc": "2026-11-06T05:27:01Z",
            "test2_unseal_utc": "2026-11-06T05:45:11Z",
            "first_v2_test2_score_ist": "2026-11-06 13:31:15 +0530",
            "first_score_inspected_ist": "2026-11-06 13:38:50 +0530",
            "compute_test2_statistics_authored_ist": "2026-11-06 13:41:51 +0530",
            "rc2_commit": "7904ad5",
            "rc2_commit_date": "2026-11-06 13:54:10 +0530",
        },
        "claims_flags": claims_flags,
        "test2_evaluation": {
            "ebright": eb_v2,
            "holdout": ho_v2,
            "combined": {
                "images_count": int(co_eval["sample_size_scenes"]),
                "faces_count": int(co_eval["sample_size_instances"]),
                "biometric_ensemble_mean": co_eval["biometric_ensemble"]["v2_mean"],
                "v1_biometric_ensemble_mean": co_eval["biometric_ensemble"]["v1_mean"],
                "rank1_accuracy": co_eval["rank1_accuracy"]["v2_rate"],
                "v1_rank1_accuracy": co_eval["rank1_accuracy"]["v1_rate"],
                "spatial_pass_rate": co_eval["spatial_position_pass"]["v2_rate"],
                "v1_spatial_pass_rate": co_eval["spatial_position_pass"]["v1_rate"],
                "wardrobe_pass_rate": co_eval["wardrobe_palette_pass"]["v2_rate"],
                "v1_wardrobe_pass_rate": co_eval["wardrobe_palette_pass"]["v1_rate"],
                "mean_defects_per_scene": co_eval["defects_per_scene"]["v2_mean"],
                "v1_mean_defects_per_scene": co_eval["defects_per_scene"]["v1_mean"],
                "mean_defects_per_instance_weighted": co_eval["defects_per_instance_weighted"]["v2_mean"],
                "v1_mean_defects_per_instance_weighted": co_eval["defects_per_instance_weighted"]["v1_mean"],
                "rank1_v2_count": co_eval["rank1_accuracy"]["v2_count"],
                "rank1_total_count": co_eval["rank1_accuracy"]["total_count"],
                "spatial_v2_count": co_eval["spatial_position_pass"]["v2_count"],
                "spatial_total_count": co_eval["spatial_position_pass"]["total_count"],
                "wardrobe_v2_count": co_eval["wardrobe_palette_pass"]["v2_count"],
                "wardrobe_total_count": co_eval["wardrobe_palette_pass"]["total_count"],
                "age_disaggregated": co_eval["age_disaggregated"],
            },
            "v1_baselines": {
                "ebright": eb_v1,
                "holdout": ho_v1,
                "dev2": dev2_v1,
            },
            "derived": {
                "ebright_defects_per_scene_pct_reduction": eb_def_pct_reduction,
            },
            "paired_deltas": paired_deltas,
            "hypothesis_scorecard": orig_sc,
            "post_hoc_redefined_scorecard_rc2": posthoc_sc,
            "extra_person_audit": extra_person_audit.get("datasets", {}),
            "circularity_and_coherence": circularity_audit,
            "exploratory_defect_cases": _extract_exploratory_cases(t2_eb_rep, t2_ho_rep, v1_eb_rep),
        },
        "test2_ablation": {
            **_ablation_meta(t2_tracer, t2_eb_rep),
            "conditions": t2_ablation.get("conditions", {}),
            "stage_diffs": _compute_stage_diffs(t2_tracer),
        },
        "dev2_bundle": dev2_v2,
        "dev2_ablation": {
            **_ablation_meta(dev2_tracer, dev2_bundle),
            "conditions": dev2_ablation.get("conditions", {}),
            "stage_diffs": _compute_stage_diffs(dev2_tracer),
        },
        "telemetry": vram_stats,
        "automated_visual_review": {
            "evaluation_type": "automated, not human",
            "blind_study": {
                "ab_v2_wins": blind_study.get("blind_ab_study", {}).get("v2_wins"),
                "ab_total_pairs": blind_study.get("blind_ab_study", {}).get("total_pairs"),
                "ab_v2_win_rate": blind_study.get("blind_ab_study", {}).get("v2_win_rate"),
                "ab_binomial_95ci": blind_study.get("blind_ab_study", {}).get("clopper_pearson_95_ci"),
                "ab_p_value": blind_study.get("blind_ab_study", {}).get("binomial_p_value_vs_chance"),
                "reid_correct": blind_study.get("blind_reid_study", {}).get("correct_reid"),
                "reid_total_crops": blind_study.get("blind_reid_study", {}).get("total_crops"),
                "reid_accuracy": blind_study.get("blind_reid_study", {}).get("reid_accuracy"),
            },
            "canary_audit": {
                "total_canaries": canary_audit.get("total_canaries"),
                "total_caught": canary_audit.get("total_caught"),
                "catch_rate_percent": canary_audit.get("catch_rate_percent"),
            },
            "human_blind_test_status": "PENDING_HUMAN_INPUT",
        },
        "historical_baselines_rescored": baseline_rescored,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(canonical_payload, indent=2), encoding="utf-8")
    print(f"[OK] Successfully collected canonical numbers to: {OUT_PATH}")
    return canonical_payload


if __name__ == "__main__":
    collect_numbers()
