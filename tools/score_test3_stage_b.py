"""
MMCCP TEST-3 Evaluation Protocol: Stage B Scoring & Statistical Analysis (tools/score_test3_stage_b.py).

Executes B2 strictly AFTER Stage B generation is finished (zero GPU contention):
- Verifies frozen SHA-256 hashes of evaluator.py, calibration_eval3.json, selected_config.json,
  and verifier_freeze.json before scoring any render.
- Scores all primary completed replicates (rep_00, rep_01, rep_02; 108 renders) with the frozen
  Eval3Evaluator and frozen v1.2 BiometricEnsemble verifier.
- Scores partial replicate (rep_03; 17 renders) SEPARATELY into results/eval3/partial/ and strictly
  excludes it from primary statistical analysis.
- Evaluates both Phase 1 arms:
    1. v2_final (all 3 seeds per scene across 12 supported scenes x R complete replicates)
    2. v2_final_bon3 (deployment-realistic best-of-3 selection across the same 3 seeds; 0 extra gen)
- Runs hierarchical cluster bootstrap (10,000 resamples, fixed RNG seed=42).
- Exports:
    - results/eval3/test3_per_face.csv
    - results/eval3/test3_per_scene.csv
    - results/eval3/test3_summary.json
    (and mirrors them into D:\\5TH SEM\\HACKATHON_eval\\test3_v2_final\\ for Tab 5 viewer).
- Prepares a blinded B3 Visual QA image queue (paths + expected characters ONLY, no scores).
"""
import os
import sys
import json
import math
import time
import shutil
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble, cosine_similarity
from eval3.metrics.evaluator import Eval3Evaluator
from eval3.metrics.calibration import compute_d_prime, score_pair_with_calibration, compute_roc_auc
from eval3.analysis import hierarchical_cluster_bootstrap
from eval3.runner import EVAL_OUTPUT_BASE, get_verifier_freeze_sha, compute_file_sha256

RUN_ID = "test3_v2_final"
EXPECTED_EVALUATOR_SHA = "6cc0b42652cd02ccf03ad821309633cf1e8b11cf284dd5dddab9e338e9a77524"
EXPECTED_CALIBRATION_SHA = "5adaa6fa2234bf702d609a86677130595a222a2751a78e5066244f517f17b0f2"
EXPECTED_VERIFIER_SHA = "3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6"

ORDERED_CHAR_IDS = [
    "young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong",
    "elena_child", "dr_marcus_vance", "clara_thorn",
    "leo_child", "maya_adult", "arthur_elder",
]

STORY_CHAR_MAP = {
    "Story_A": ["young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong"],
    "Story_B": ["elena_child", "dr_marcus_vance", "clara_thorn"],
    "Story_C": ["leo_child", "maya_adult", "arthur_elder"],
}

CHAR_AGE_GROUP = {
    "young_ebright": "child",
    "mother": "adult",
    "teen_ebright": "teen",
    "college_ebright": "adult",
    "james_wong": "adult",
    "elena_child": "child",
    "dr_marcus_vance": "adult",
    "clara_thorn": "adult",
    "leo_child": "child",
    "maya_adult": "adult",
    "arthur_elder": "elder",
}

TIE_HANDLING_CONVENTION = (
    "Gallery ranking uses Python Timsort descending by calibrated ensemble score "
    "(sorted(gallery_scores.items(), key=lambda x: x[1], reverse=True)), preserving "
    "deterministic insertion order of expected_chars on exact ties. Missing/undetected "
    "eligible probe faces or empty other-scene galleries (score == -1.0) count as "
    "incorrect under intention-to-treat (rank = N_gallery + 1, reciprocal rank = 0.0)."
)


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items() if k != "crop_bgr"}
    elif isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def compute_eer_and_tar(genuine: List[float], impostor: List[float]) -> Dict[str, float]:
    """Computes Equal Error Rate (EER) and TAR@FAR=0.05 from genuine and impostor score lists."""
    if not genuine or not impostor:
        return {"d_prime": 0.0, "eer": 0.5, "tar_at_far05": 0.0}
    g = np.array(genuine, dtype=np.float64)
    im = np.array(impostor, dtype=np.float64)
    dp = compute_d_prime(genuine, impostor)

    # TAR @ FAR = 0.05 (threshold at 95th percentile of impostors)
    thresh_far05 = float(np.percentile(im, 95.0))
    tar_at_far05 = float(np.mean(g >= thresh_far05))

    # EER sweep over unique score thresholds
    thresholds = np.unique(np.concatenate([g, im]))
    best_diff = 1.0
    eer = 0.5
    for th in thresholds:
        far = float(np.mean(im >= th))
        frr = float(np.mean(g < th))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = 0.5 * (far + frr)

    return {
        "d_prime": round(float(dp), 4),
        "eer": round(float(eer), 4),
        "tar_at_far05": round(float(tar_at_far05), 4),
    }


def compute_bon3_tier(audit: Dict[str, Any]) -> int:
    """
    Pre-registered v2_final_bon3 tier rule (PREREG_TEST3.md Section 2):
    - Tier 3: Correct person count (YOLO) and all expected faces detected.
    - Tier 2: All faces detected, but extra person detected.
    - Tier 1: Person count matched, but face detector missed a crop.
    - Tier 0: Severe mismatch.
    """
    exp_c = int(audit.get("expected_count", len(audit.get("expected_characters", []))))
    det_c = int(audit.get("detected_faces_count", len(audit.get("detected_faces", []))))
    yolo_c = int(audit.get("yolo_persons_count", 0))

    if yolo_c == exp_c and det_c >= exp_c:
        return 3
    if det_c >= exp_c and yolo_c > exp_c:
        return 2
    if yolo_c == exp_c and det_c < exp_c:
        return 1
    return 0


def extract_anchor_banks(rep_dir: Path, evaluator: Eval3Evaluator) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, List[np.ndarray]]], Dict[str, Dict[str, np.ndarray]]]:
    """Loads certified anchor images and builds v1.2 4-judge embeddings & centroids per character."""
    anchors_dir = rep_dir / "anchors"
    anchor_crops: Dict[str, np.ndarray] = {}
    all_banks: Dict[str, Dict[str, List[np.ndarray]]] = {}
    all_centroids: Dict[str, Dict[str, np.ndarray]] = {}

    for cid in ORDERED_CHAR_IDS:
        sc_p = anchors_dir / f"{cid}_selected_anchor.json"
        if not sc_p.exists():
            continue
        with open(sc_p, "r", encoding="utf-8") as f:
            ameta = json.load(f)
        img_p = ameta.get("image_path", "")
        if not img_p or not Path(img_p).exists():
            continue
        img = cv2.imread(img_p)
        if img is None:
            continue
        anchor_crops[img_p] = img
        faces = evaluator.auditor.detect_faces(img)
        crop = faces[0].get("crop_bgr") if faces and faces[0].get("crop_bgr") is not None else img
        embs = evaluator.extract_face_embeddings(crop)
        all_banks[cid] = {m: [v] for m, v in embs.items()}
        all_centroids[cid] = {m: v for m, v in embs.items()}

    return anchor_crops, all_banks, all_centroids


def audit_replicate_scenes(
    rep_dir: Path,
    evaluator: Eval3Evaluator,
    anchor_crops: Dict[str, np.ndarray],
    all_banks: Dict[str, Dict[str, List[np.ndarray]]],
    all_centroids: Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Audits every rendered scene JSON/PNG in rep_dir/scenes with caching."""
    scenes_dir = rep_dir / "scenes"
    cache_path = scenes_dir / "audits_cache.json"

    if cache_path.exists():
        log(f"  [CACHE] Loading cached audits from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            audits = json.load(f)
        # Convert embeddings back to np.ndarray
        for a in audits:
            for face in a.get("detected_faces", []):
                embs = face.get("embeddings", {})
                face["embeddings"] = {m: np.array(v, dtype=np.float32) for m, v in embs.items()}
        return audits

    sidecar_files = sorted(scenes_dir.glob("rep_*.json"))
    audits: List[Dict[str, Any]] = []
    for idx, sc_file in enumerate(sidecar_files, 1):
        with open(sc_file, "r", encoding="utf-8") as f:
            sc_data = json.load(f)

        render_id = sc_data["render_id"]
        img_p = scenes_dir / f"{render_id}.png"
        exp_chars = sc_data.get("expected_characters", [])
        char_pos = sc_data.get("character_positions", {})

        if sc_data.get("status") != "SUCCESS" or not img_p.exists():
            audit = {
                "image_path": str(img_p),
                "h": 0,
                "w": 0,
                "detected_faces": [],
                "detected_faces_count": 0,
                "expected_characters": exp_chars,
                "expected_count": len(exp_chars),
                "yolo_persons_count": 0,
                "all_expected_present": False,
                "no_extra_person": False,
                "order_correct": False,
                "pm3_correct": False,
                "extra_person_flag": False,
                "torch_res": {},
                "render_failed": True,
            }
        else:
            audit = evaluator.audit_single_scene_image(
                image_path=img_p,
                expected_characters=exp_chars,
                character_positions=char_pos,
                anchor_crops=anchor_crops,
            )
            audit["render_failed"] = False

        # Attach metadata from sidecar
        st_idx = int(sc_data.get("story_idx", 0))
        sc_num = int(sc_data.get("scene_number", 1))
        audit["render_id"] = render_id
        audit["replicate_idx"] = int(sc_data.get("replicate_idx", 0))
        audit["story_idx"] = st_idx
        audit["story_label"] = sc_data.get("story_label", f"Story_{st_idx}")
        audit["story_id"] = sc_data.get("story_id", "")
        audit["scene_idx"] = int(sc_data.get("scene_idx", 0))
        audit["scene_number_story"] = sc_num
        # Unique global scene_number across the 12 supported scenes in the replicate
        audit["scene_number"] = st_idx * 10 + sc_num
        audit["seed_idx"] = int(sc_data.get("seed_idx", 0))
        audit["seed"] = int(sc_data.get("seed", 0))
        audit["elapsed_seconds"] = float(sc_data.get("elapsed_seconds", 0.0))
        audit["retry_count"] = int(sc_data.get("retry_count", 0))
        audit["status"] = sc_data.get("status", "UNKNOWN")

        # Extract IN-LOOP sidecar metrics: buffalo_l anchor sim & wardrobe delta E
        meta = sc_data.get("metadata", {}) or {}
        mem_promos = meta.get("memory_promotions", []) or []
        buffalo_map = {
            mp.get("character_id"): float(mp.get("similarity_to_original", 0.0))
            for mp in mem_promos
            if isinstance(mp, dict) and mp.get("character_id")
        }
        wardrobe_details = (meta.get("wardrobe_report", {}) or {}).get("details", []) or []
        delta_e_map = {
            wd.get("character_id"): float(wd.get("final_delta_e", 0.0))
            for wd in wardrobe_details
            if isinstance(wd, dict) and wd.get("character_id")
        }
        audit["buffalo_l_by_char"] = buffalo_map
        audit["wardrobe_delta_e_by_char"] = delta_e_map
        audit["mean_buffalo_l_sim"] = round(float(np.mean(list(buffalo_map.values()))), 4) if buffalo_map else 0.0
        audit["mean_wardrobe_delta_e"] = round(float(np.mean(list(delta_e_map.values()))), 2) if delta_e_map else 0.0

        # Assign detected faces to expected characters (same layout rule as Eval3Evaluator)
        # and score each assigned face with frozen v1.2 verifier (IN-LOOP)
        s_faces = audit.get("detected_faces", [])
        assigned_pairs: List[Tuple[str, Dict[str, Any]]] = []
        if len(exp_chars) == 1 and len(s_faces) >= 1:
            f_best = sorted(s_faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)[0]
            assigned_pairs.append((exp_chars[0], f_best))
        elif len(exp_chars) == 2 and len(s_faces) >= 2:
            sf = sorted(s_faces, key=lambda f: f["bbox"]["x"])
            assigned_pairs.append((exp_chars[0], sf[0]))
            assigned_pairs.append((exp_chars[1], sf[-1]))

        story_chars = STORY_CHAR_MAP.get(audit["story_label"], ORDERED_CHAR_IDS)
        story_banks = {c: all_banks[c] for c in story_chars if c in all_banks}
        story_centroids = {c: all_centroids[c] for c in story_chars if c in all_centroids}

        for cid, face in assigned_pairs:
            face["assigned_char_id"] = cid
            embs = face.get("embeddings", {})
            age_grp = CHAR_AGE_GROUP.get(cid, "adult")
            age_key = "adult" if age_grp == "elder" else age_grp
            fw = int((face.get("bbox") or {}).get("w", 0))
            if cid in all_banks and embs:
                v12_bank = evaluator.ensemble.score_against_bank(
                    embs, all_banks[cid], all_centroids.get(cid), age_group=age_key, face_width_px=fw
                )
                v12_disc = evaluator.ensemble.compute_discrimination(
                    embs, story_banks, story_centroids, target_char_id=cid
                )
                face["v1_2_ensemble_score"] = float(v12_bank.get("ensemble_score", 0.0))
                face["v1_2_ensemble_passed"] = bool(v12_bank.get("ensemble_passed", False))
                face["v1_2_rank1_vs_anchor"] = bool(v12_disc.get("is_rank1", False))
                face["v1_2_margin"] = float(v12_disc.get("margin", 0.0))
            else:
                face["v1_2_ensemble_score"] = 0.0
                face["v1_2_ensemble_passed"] = False
                face["v1_2_rank1_vs_anchor"] = False
                face["v1_2_margin"] = 0.0
            face["buffalo_l_anchor_sim"] = float(buffalo_map.get(cid, 0.0))
            face["wardrobe_delta_e"] = float(delta_e_map.get(cid, 0.0))

        # Compute bon3 tier & tie-breaker stats
        sharp_vals = [float(f.get("sharpness", 0.0)) for f in s_faces]
        audit["mean_face_sharpness"] = round(float(np.mean(sharp_vals)), 2) if sharp_vals else 0.0
        v12_vals = [float(f.get("v1_2_ensemble_score", 0.0)) for _, f in assigned_pairs]
        audit["mean_v1_2_ensemble_score"] = round(float(np.mean(v12_vals)), 4) if v12_vals else 0.0
        audit["bon3_tier"] = compute_bon3_tier(audit)

        audits.append(audit)
        if idx % 6 == 0 or idx == len(sidecar_files):
            log(f"    Audited {idx}/{len(sidecar_files)} renders in {rep_dir.name} (latest: {render_id}, PM3={audit['pm3_correct']}, tier={audit['bon3_tier']})")

    # Save cache (without crop_bgr; embeddings as float lists)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(audits), f)
    return audits


def select_bon3_audits(audits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set]:
    """
    Applies the pre-registered v2_final_bon3 gate across the 3 seeds per (story_idx, scene_idx):
    Tier DESC -> mean_face_sharpness DESC -> mean_buffalo_l_sim DESC -> lowest seed ASC.
    """
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for a in audits:
        key = (int(a["story_idx"]), int(a["scene_idx"]))
        grouped.setdefault(key, []).append(a)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for key in sorted(grouped.keys()):
        cands = grouped[key]
        best = max(
            cands,
            key=lambda a: (
                int(a.get("bon3_tier", 0)),
                float(a.get("mean_face_sharpness", 0.0)),
                float(a.get("mean_buffalo_l_sim", 0.0)),
                -int(a.get("seed", 0)),
            ),
        )
        selected.append(best)
        selected_ids.add(best["render_id"])
    return selected, selected_ids


def compute_per_char_and_age_breakdown(
    evaluator: Eval3Evaluator,
    rep_audits_list: List[List[Dict[str, Any]]],
    story_char_map: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes per-character and per-age PM2 Rank-1 hit rates across replicates
    using within-story galleries; explicitly reports 'N/A (ineligible)' for single-scene characters.
    """
    char_stats: Dict[str, Dict[str, Any]] = {
        cid: {
            "age_group": CHAR_AGE_GROUP[cid],
            "eligible": False,
            "n_probes": 0,
            "rank1_hits": 0,
            "v1_2_scores": [],
            "buffalo_l_sims": [],
            "sharpnesses": [],
        }
        for cid in ORDERED_CHAR_IDS
    }

    for audits in rep_audits_list:
        for st_label, st_chars in story_char_map.items():
            st_audits = [a for a in audits if a.get("story_label") == st_label]
            # Build faces_by_char & scene sets within this story
            faces_by_char: Dict[str, List[Dict[str, Any]]] = {c: [] for c in st_chars}
            char_scenes: Dict[str, set] = {c: set() for c in st_chars}
            for a in st_audits:
                sc_num = a["scene_number_story"]
                exp = a.get("expected_characters", [])
                for c in exp:
                    if c in char_scenes:
                        char_scenes[c].add(sc_num)
                s_faces = a.get("detected_faces", [])
                if len(exp) == 1 and len(s_faces) >= 1:
                    fb = sorted(s_faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)[0]
                    faces_by_char[exp[0]].append({"scene": sc_num, "render_id": a["render_id"], "face": fb})
                elif len(exp) == 2 and len(s_faces) >= 2:
                    sf = sorted(s_faces, key=lambda f: f["bbox"]["x"])
                    faces_by_char[exp[0]].append({"scene": sc_num, "render_id": a["render_id"], "face": sf[0]})
                    faces_by_char[exp[1]].append({"scene": sc_num, "render_id": a["render_id"], "face": sf[-1]})

            eligible = {c for c, scs in char_scenes.items() if len(scs) >= 2}
            for c in eligible:
                char_stats[c]["eligible"] = True

            # Record auxiliary scores for all detected faces
            for c, recs in faces_by_char.items():
                for rec in recs:
                    f = rec["face"]
                    char_stats[c]["v1_2_scores"].append(float(f.get("v1_2_ensemble_score", 0.0)))
                    char_stats[c]["buffalo_l_sims"].append(float(f.get("buffalo_l_anchor_sim", 0.0)))
                    char_stats[c]["sharpnesses"].append(float(f.get("sharpness", 0.0)))

            # Evaluate eligible probes within story gallery
            for a in st_audits:
                sc_num = a["scene_number_story"]
                rid = a["render_id"]
                for cid in a.get("expected_characters", []):
                    if cid not in eligible:
                        continue
                    char_stats[cid]["n_probes"] += 1
                    probe_rec = next((r for r in faces_by_char[cid] if r["scene"] == sc_num and r["render_id"] == rid), None)
                    if probe_rec is None:
                        continue
                    p_emb = probe_rec["face"]["embeddings"]
                    g_scores = {}
                    for cand in st_chars:
                        c_faces = [r["face"]["embeddings"] for r in faces_by_char[cand] if r["scene"] != sc_num]
                        if c_faces:
                            sims = [score_pair_with_calibration(evaluator.compute_pair_judge_cosines(p_emb, cf), evaluator.calibration) for cf in c_faces]
                            g_scores[cand] = float(np.mean(sims))
                        else:
                            g_scores[cand] = -1.0
                    ranked = [x[0] for x in sorted(g_scores.items(), key=lambda x: x[1], reverse=True)]
                    if cid in ranked and g_scores[cid] > -1.0 and ranked.index(cid) == 0:
                        char_stats[cid]["rank1_hits"] += 1

    per_char_out = {}
    age_buckets: Dict[str, Dict[str, Any]] = {}
    for cid, d in char_stats.items():
        ag = d["age_group"]
        age_buckets.setdefault(ag, {"n_probes": 0, "rank1_hits": 0, "v1_2_scores": [], "eligible_chars": [], "ineligible_chars": []})
        if d["eligible"]:
            pm2_val = round(d["rank1_hits"] / d["n_probes"], 4) if d["n_probes"] > 0 else 0.0
            age_buckets[ag]["n_probes"] += d["n_probes"]
            age_buckets[ag]["rank1_hits"] += d["rank1_hits"]
            age_buckets[ag]["eligible_chars"].append(cid)
        else:
            pm2_val = "N/A (ineligible)"
            age_buckets[ag]["ineligible_chars"].append(cid)
        age_buckets[ag]["v1_2_scores"].extend(d["v1_2_scores"])

        per_char_out[cid] = {
            "age_group": ag,
            "pm2_eligible": d["eligible"],
            "n_eligible_probes": d["n_probes"],
            "rank1_hits": d["rank1_hits"] if d["eligible"] else "N/A (ineligible)",
            "pm2_rank1_within_story": pm2_val,
            "mean_v1_2_ensemble_score_in_loop": round(float(np.mean(d["v1_2_scores"])), 4) if d["v1_2_scores"] else 0.0,
            "mean_buffalo_l_sim_in_loop": round(float(np.mean(d["buffalo_l_sims"])), 4) if d["buffalo_l_sims"] else 0.0,
            "mean_face_sharpness": round(float(np.mean(d["sharpnesses"])), 2) if d["sharpnesses"] else 0.0,
        }

    per_age_out = {}
    for ag, b in age_buckets.items():
        per_age_out[ag] = {
            "eligible_characters": b["eligible_chars"],
            "ineligible_characters": b["ineligible_chars"],
            "n_eligible_probes": b["n_probes"],
            "rank1_hits": b["rank1_hits"],
            "pm2_rank1_within_story": round(b["rank1_hits"] / b["n_probes"], 4) if b["n_probes"] > 0 else "N/A (ineligible)",
            "mean_v1_2_ensemble_score_in_loop": round(float(np.mean(b["v1_2_scores"])), 4) if b["v1_2_scores"] else 0.0,
        }
    return per_char_out, per_age_out


def evaluate_arm(
    arm_name: str,
    evaluator: Eval3Evaluator,
    completed_rep_indices: List[int],
    rep_audits_by_idx: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Evaluates an arm (v2_final or v2_final_bon3) across all completed replicates and per story."""
    full_rep_results: List[Dict[str, Any]] = []
    story_rep_results: Dict[str, List[Dict[str, Any]]] = {"Story_A": [], "Story_B": [], "Story_C": []}
    all_gen: List[float] = []
    all_imp: List[float] = []

    for r_idx in completed_rep_indices:
        audits = rep_audits_by_idx[r_idx]
        # 1. Full-replicate evaluation (11 gallery chars, global unique scene_number)
        full_eval = evaluator.evaluate_replicate(
            replicate_idx=r_idx,
            scene_audits=audits,
            expected_chars=ORDERED_CHAR_IDS,
        )
        sep_stats = compute_eer_and_tar(full_eval["raw_genuine_scores"], full_eval["raw_impostor_scores"])
        full_eval.update(sep_stats)
        all_gen.extend(full_eval["raw_genuine_scores"])
        all_imp.extend(full_eval["raw_impostor_scores"])
        full_eval["scenes"] = [{"render_id": a["render_id"], "pm3_correct": a["pm3_correct"]} for a in audits]

        # 2. Per-story evaluation within this replicate
        per_story_this_rep = {}
        for st_label, st_chars in STORY_CHAR_MAP.items():
            st_audits = []
            for a in audits:
                if a.get("story_label") == st_label:
                    a_copy = dict(a)
                    a_copy["scene_number"] = a["scene_number_story"]
                    st_audits.append(a_copy)
            st_eval = evaluator.evaluate_replicate(
                replicate_idx=r_idx,
                scene_audits=st_audits,
                expected_chars=st_chars,
            )
            st_sep = compute_eer_and_tar(st_eval["raw_genuine_scores"], st_eval["raw_impostor_scores"])
            st_eval.update(st_sep)
            st_eval["scenes"] = [{"render_id": a["render_id"], "pm3_correct": a["pm3_correct"]} for a in st_audits]
            # Strip raw score lists from JSON summary
            st_clean = {k: v for k, v in st_eval.items() if k not in ["raw_genuine_scores", "raw_impostor_scores"]}
            per_story_this_rep[st_label] = st_clean
            story_rep_results[st_label].append(st_clean)

        full_clean = {k: v for k, v in full_eval.items() if k not in ["raw_genuine_scores", "raw_impostor_scores"]}
        full_clean["per_story"] = per_story_this_rep
        full_rep_results.append(full_clean)

    # Hierarchical cluster bootstrap (10,000 resamples, seed=42)
    boot_full = hierarchical_cluster_bootstrap(full_rep_results, n_resamples=10000, seed=42)

    # Per-story aggregated metrics & 10,000-resample bootstrap
    per_story_summary = {}
    for st_label, s_reps in story_rep_results.items():
        s_boot = hierarchical_cluster_bootstrap(s_reps, n_resamples=10000, seed=42)
        per_story_summary[st_label] = {
            "n_gallery_characters": len(STORY_CHAR_MAP[st_label]),
            "chance_floor_pm2": round(1.0 / len(STORY_CHAR_MAP[st_label]), 4),
            "n_expected_per_rep": s_reps[0]["n_expected"],
            "n_eligible_per_rep": s_reps[0]["n_eligible"],
            "n_ineligible_per_rep": s_reps[0]["n_ineligible"],
            "eligible_characters": s_reps[0]["eligible_characters"],
            "ineligible_characters": s_reps[0]["ineligible_characters"],
            "pm1_cross_scene_auc_mean": round(float(np.mean([r["pm1_cross_scene_auc"] for r in s_reps])), 4),
            "pm1_bootstrap": s_boot.get("pm1_cross_scene_auc", {}),
            "pm2_cross_scene_rank1_mean": round(float(np.mean([r["pm2_cross_scene_rank1"] for r in s_reps])), 4),
            "pm2_old_unfiltered_mean": round(float(np.mean([r["pm2_old_unfiltered"] for r in s_reps])), 4),
            "pm2_bootstrap": s_boot.get("pm2_cross_scene_rank1", {}),
            "pm3_scene_correctness_mean": round(float(np.mean([r["pm3_scene_correctness_rate"] for r in s_reps])), 4),
            "pm3_bootstrap": s_boot.get("pm3_scene_correctness_rate", {}),
            "rank3_accuracy_mean": round(float(np.mean([r["rank3_accuracy"] for r in s_reps])), 4),
            "mrr_mean": round(float(np.mean([r["mrr"] for r in s_reps])), 4),
            "per_replicate": [
                {k: v for k, v in r.items() if k != "scenes"} for r in s_reps
            ],
        }

    # Aggregate full-replicate summaries
    pm1_vals = [r["pm1_cross_scene_auc"] for r in full_rep_results]
    pm2_vals = [r["pm2_cross_scene_rank1"] for r in full_rep_results]
    pm2_old_vals = [r["pm2_old_unfiltered"] for r in full_rep_results]
    pm3_vals = [r["pm3_scene_correctness_rate"] for r in full_rep_results]
    rank3_vals = [r["rank3_accuracy"] for r in full_rep_results]
    mrr_vals = [r["mrr"] for r in full_rep_results]
    recall_vals = [r["face_recall"] for r in full_rep_results]
    extra_vals = [r["extra_person_rate"] for r in full_rep_results]
    sharp_vals = [r["mean_face_sharpness"] for r in full_rep_results]

    judge_summary = {
        m: round(float(np.mean([r["judge_aucs"][m] for r in full_rep_results])), 4)
        for m in BiometricEnsemble.JUDGE_MODELS
    }

    pooled_sep = compute_eer_and_tar(all_gen, all_imp)
    per_char_bd, per_age_bd = compute_per_char_and_age_breakdown(
        evaluator, [rep_audits_by_idx[r] for r in completed_rep_indices], STORY_CHAR_MAP
    )

    # Also compute average within-story PM2 across the 3 stories
    within_story_pm2_reps = []
    for i in range(len(completed_rep_indices)):
        st_vals = [story_rep_results[sl][i]["pm2_cross_scene_rank1"] for sl in ["Story_A", "Story_B", "Story_C"]]
        within_story_pm2_reps.append(round(float(np.mean(st_vals)), 4))

    return {
        "arm": arm_name,
        "n_replicates": len(completed_rep_indices),
        "primary_metrics": {
            "pm1_cross_scene_auc": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm1_vals)), 4),
                "bootstrap": boot_full.get("pm1_cross_scene_auc", {}),
                "chance_floor": 0.50,
                "ceiling": 1.00,
            },
            "pm2_cross_scene_rank1": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm2_vals)), 4),
                "pm2_old_unfiltered_mean": round(float(np.mean(pm2_old_vals)), 4),
                "bootstrap": boot_full.get("pm2_cross_scene_rank1", {}),
                "chance_floor": round(1.0 / len(ORDERED_CHAR_IDS), 4),
                "n_gallery_characters": len(ORDERED_CHAR_IDS),
                "n_expected_per_rep": full_rep_results[0]["n_expected"],
                "n_eligible_per_rep": full_rep_results[0]["n_eligible"],
                "n_ineligible_per_rep": full_rep_results[0]["n_ineligible"],
                "within_story_mean_pm2": round(float(np.mean(within_story_pm2_reps)), 4),
                "ceiling": 1.00,
            },
            "pm3_scene_correctness_rate": {
                "loop_label": "OUT-OF-LOOP",
                "empirical_mean": round(float(np.mean(pm3_vals)), 4),
                "bootstrap": boot_full.get("pm3_scene_correctness_rate", {}),
                "chance_floor": 0.00,
                "ceiling": 1.00,
            },
        },
        "secondary_metrics": {
            "rank3_accuracy": {"loop_label": "OUT-OF-LOOP", "mean": round(float(np.mean(rank3_vals)), 4)},
            "mrr": {"loop_label": "OUT-OF-LOOP", "mean": round(float(np.mean(mrr_vals)), 4)},
            "face_recall": {"loop_label": "OUT-OF-LOOP", "mean": round(float(np.mean(recall_vals)), 4)},
            "extra_person_rate": {"loop_label": "OUT-OF-LOOP", "mean": round(float(np.mean(extra_vals)), 4)},
            "mean_face_sharpness": {"loop_label": "OUT-OF-LOOP", "mean": round(float(np.mean(sharp_vals)), 2)},
            "judge_aucs": judge_summary,
            "pooled_biometric_separation": pooled_sep,
            "n_genuine_pairs_total": len(all_gen),
            "n_impostor_pairs_total": len(all_imp),
        },
        "per_story": per_story_summary,
        "per_character": per_char_bd,
        "per_age_group": per_age_bd,
        "per_replicate": [
            {k: v for k, v in r.items() if k != "scenes"} for r in full_rep_results
        ],
    }


def main():
    t_start = time.time()
    log("=" * 72)
    log("MMCCP TEST-3 STAGE B SCORING & BOOTSTRAP (B2)")
    log("=" * 72)

    # 1. Verify frozen hashes before touching any render
    eval_sha = compute_file_sha256(PROJECT_ROOT / "eval3" / "metrics" / "evaluator.py")
    calib_sha = compute_file_sha256(PROJECT_ROOT / "results" / "eval3" / "calibration_eval3.json")
    verif_sha = get_verifier_freeze_sha()
    assert eval_sha == EXPECTED_EVALUATOR_SHA, f"Evaluator SHA mismatch: {eval_sha} != {EXPECTED_EVALUATOR_SHA}"
    assert calib_sha == EXPECTED_CALIBRATION_SHA, f"Calibration SHA mismatch: {calib_sha} != {EXPECTED_CALIBRATION_SHA}"
    assert verif_sha == EXPECTED_VERIFIER_SHA, f"Verifier freeze SHA mismatch: {verif_sha} != {EXPECTED_VERIFIER_SHA}"
    log(f"[FREEZE OK] evaluator={eval_sha[:12]}... calibration={calib_sha[:12]}... verifier={verif_sha[:12]}...")

    run_dir = EVAL_OUTPUT_BASE / RUN_ID
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    completed_rep_keys = manifest.get("completed_replicates", [])
    partial_rep_keys = manifest.get("partial_replicates", [])
    log(f"Completed replicates ({len(completed_rep_keys)}): {completed_rep_keys}")
    log(f"Partial replicates ({len(partial_rep_keys)}): {partial_rep_keys} (EXCLUDED from primary analysis)")

    evaluator = Eval3Evaluator()

    # 2. Audit all completed replicates
    v2_final_by_rep: Dict[int, List[Dict[str, Any]]] = {}
    bon3_by_rep: Dict[int, List[Dict[str, Any]]] = {}
    bon3_selected_ids = set()

    for rep_key in completed_rep_keys:
        r_idx = int(rep_key.split("_")[1])
        rep_dir = run_dir / "replicates" / rep_key
        log(f"\n--- Auditing Completed Replicate {rep_key} ---")
        anchor_crops, all_banks, all_centroids = extract_anchor_banks(rep_dir, evaluator)
        audits = audit_replicate_scenes(rep_dir, evaluator, anchor_crops, all_banks, all_centroids)
        v2_final_by_rep[r_idx] = audits
        bon3_audits, sel_ids = select_bon3_audits(audits)
        bon3_by_rep[r_idx] = bon3_audits
        bon3_selected_ids.update(sel_ids)

    # 3. Audit partial replicate(s) separately (archived in results/eval3/partial/, excluded from primary)
    partial_reports = {}
    partial_dir = PROJECT_ROOT / "results" / "eval3" / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for rep_key in partial_rep_keys:
        r_idx = int(rep_key.split("_")[1])
        rep_dir = run_dir / "replicates" / rep_key
        log(f"\n--- Auditing Partial Replicate {rep_key} (SEPARATE, EXCLUDED FROM PRIMARY) ---")
        anchor_crops, all_banks, all_centroids = extract_anchor_banks(rep_dir, evaluator)
        p_audits = audit_replicate_scenes(rep_dir, evaluator, anchor_crops, all_banks, all_centroids)
        p_pm3 = round(float(np.mean([1.0 if a["pm3_correct"] else 0.0 for a in p_audits])), 4) if p_audits else 0.0
        p_summary = {
            "run_id": RUN_ID,
            "replicate_key": rep_key,
            "excluded_from_primary_analysis": True,
            "reason": "Reached 8.0h generation cap mid-replicate",
            "n_renders_completed": len(p_audits),
            "pm3_scene_correctness_rate_partial": p_pm3,
            "renders": [
                {
                    "render_id": a["render_id"],
                    "story_label": a["story_label"],
                    "scene_number": a["scene_number_story"],
                    "seed": a["seed"],
                    "pm3_correct": a["pm3_correct"],
                    "yolo_persons_count": a["yolo_persons_count"],
                    "detected_faces_count": a["detected_faces_count"],
                    "mean_v1_2_ensemble_score": a["mean_v1_2_ensemble_score"],
                }
                for a in p_audits
            ],
        }
        p_out = partial_dir / f"{RUN_ID}_{rep_key}_partial_scored.json"
        with open(p_out, "w", encoding="utf-8") as f:
            json.dump(p_summary, f, indent=2)
        partial_reports[rep_key] = {
            "n_renders_completed": len(p_audits),
            "pm3_partial": p_pm3,
            "report_path": str(p_out.relative_to(PROJECT_ROOT)),
        }

    # 4. Evaluate both Phase 1 arms across completed replicates
    completed_indices = [int(k.split("_")[1]) for k in completed_rep_keys]
    log("\n--- Computing Primary & Secondary Metrics + 10,000-Resample Hierarchical Bootstrap ---")
    arm_v2_final = evaluate_arm("v2_final", evaluator, completed_indices, v2_final_by_rep)
    arm_bon3 = evaluate_arm("v2_final_bon3", evaluator, completed_indices, bon3_by_rep)

    # 5. Build test3_per_scene.csv and test3_per_face.csv (primary completed replicates only)
    scene_rows: List[Dict[str, Any]] = []
    face_rows: List[Dict[str, Any]] = []
    all_v12_scores: List[float] = []
    all_v12_rank1s: List[float] = []
    all_buffalo_sims: List[float] = []
    all_delta_es: List[float] = []

    for r_idx in completed_indices:
        for a in v2_final_by_rep[r_idx]:
            is_bon3 = a["render_id"] in bon3_selected_ids
            scene_rows.append({
                "run_id": RUN_ID,
                "replicate_idx": a["replicate_idx"],
                "story_idx": a["story_idx"],
                "story_label": a["story_label"],
                "story_id": a["story_id"],
                "scene_idx": a["scene_idx"],
                "scene_number": a["scene_number_story"],
                "seed_idx": a["seed_idx"],
                "seed": a["seed"],
                "render_id": a["render_id"],
                "bon3_selected": is_bon3,
                "bon3_tier": a["bon3_tier"],
                "expected_count": a["expected_count"],
                "yolo_count": a["yolo_persons_count"],
                "faces_detected": a["detected_faces_count"],
                "all_expected_present": a["all_expected_present"],
                "no_extra_person": a["no_extra_person"],
                "order_correct": a["order_correct"],
                "pm3_correct": a["pm3_correct"],
                "extra_person_flag": a["extra_person_flag"],
                "mean_face_sharpness": a["mean_face_sharpness"],
                "mean_v1_2_ensemble_score_in_loop": a["mean_v1_2_ensemble_score"],
                "mean_buffalo_l_sim_in_loop": a["mean_buffalo_l_sim"],
                "mean_wardrobe_delta_e_in_loop": a["mean_wardrobe_delta_e"],
                "render_time_s": a["elapsed_seconds"],
                "retry_count": a["retry_count"],
                "status": a["status"],
            })

            for f in a.get("detected_faces", []):
                cid = f.get("assigned_char_id", "unassigned")
                v12_sc = float(f.get("v1_2_ensemble_score", 0.0))
                v12_r1 = bool(f.get("v1_2_rank1_vs_anchor", False))
                buf_sim = float(f.get("buffalo_l_anchor_sim", 0.0))
                w_de = float(f.get("wardrobe_delta_e", 0.0))
                if cid != "unassigned":
                    all_v12_scores.append(v12_sc)
                    all_v12_rank1s.append(1.0 if v12_r1 else 0.0)
                    if buf_sim > 0:
                        all_buffalo_sims.append(buf_sim)
                    if w_de > 0:
                        all_delta_es.append(w_de)
                bbox = f.get("bbox") or {}
                face_rows.append({
                    "run_id": RUN_ID,
                    "replicate_idx": a["replicate_idx"],
                    "story_label": a["story_label"],
                    "story_id": a["story_id"],
                    "scene_number": a["scene_number_story"],
                    "seed": a["seed"],
                    "render_id": a["render_id"],
                    "bon3_selected": is_bon3,
                    "face_idx": f.get("face_idx", 0),
                    "character_id": cid,
                    "age_group": CHAR_AGE_GROUP.get(cid, "unknown"),
                    "bbox_x": bbox.get("x", 0),
                    "bbox_y": bbox.get("y", 0),
                    "bbox_w": bbox.get("w", 0),
                    "bbox_h": bbox.get("h", 0),
                    "confidence": round(float(f.get("confidence", 0.0)), 4),
                    "sharpness": round(float(f.get("sharpness", 0.0)), 2),
                    "v1_2_ensemble_score_in_loop": round(v12_sc, 4),
                    "v1_2_rank1_vs_anchor_in_loop": v12_r1,
                    "buffalo_l_anchor_sim_in_loop": round(buf_sim, 4),
                    "wardrobe_delta_e_in_loop": round(w_de, 2),
                    "extra_person_flag": a["extra_person_flag"],
                })

    scoring_elapsed_s = round(time.time() - t_start, 2)
    scoring_elapsed_h = round(scoring_elapsed_s / 3600.0, 4)

    # 6. Assemble final test3_summary.json (top-level mirrors v2_final for Tab 5 viewer compatibility)
    in_loop_summary = {
        "loop_label": "IN-LOOP",
        "mean_v1_2_ensemble_score": round(float(np.mean(all_v12_scores)), 4) if all_v12_scores else 0.0,
        "v1_2_rank1_vs_anchor_rate": round(float(np.mean(all_v12_rank1s)), 4) if all_v12_rank1s else 0.0,
        "mean_buffalo_l_anchor_similarity": round(float(np.mean(all_buffalo_sims)), 4) if all_buffalo_sims else 0.0,
        "mean_wardrobe_cielab_delta_e": round(float(np.mean(all_delta_es)), 2) if all_delta_es else 0.0,
    }

    summary = {
        "run_id": RUN_ID,
        "protocol_version": "TEST-3 Phase 1 (v1.2.1)",
        "scored_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scoring_duration_seconds": scoring_elapsed_s,
        "scoring_duration_hours": scoring_elapsed_h,
        "generation_duration_seconds": round(float(manifest["cumulative_generation_seconds"]), 2),
        "generation_duration_hours": round(float(manifest["cumulative_generation_seconds"]) / 3600.0, 3),
        "generation_status": manifest["status"],
        "completed_replicates": completed_rep_keys,
        "partial_replicates_excluded": partial_rep_keys,
        "partial_replicates_summary": partial_reports,
        "frozen_hashes": {
            "evaluator_sha256": eval_sha,
            "calibration_sha256": calib_sha,
            "verifier_freeze_sha256": verif_sha,
            "config_hash": manifest["config_hash"],
            "config_compute_hash": manifest["config_compute_hash"],
        },
        "bootstrap_iterations": 10000,
        "bootstrap_rng_seed": 42,
        "pm2_tie_handling_convention": TIE_HANDLING_CONVENTION,
        "n_replicates": len(completed_rep_keys),
        "total_scenes_evaluated": len(scene_rows),
        "total_faces_evaluated": len(face_rows),
        # Primary top-level keys (v2_final) for direct Tab 5 dashboard compatibility
        "primary_metrics": arm_v2_final["primary_metrics"],
        "secondary_metrics": arm_v2_final["secondary_metrics"],
        "in_loop_metrics_v1_2": in_loop_summary,
        "arms": {
            "v2_final": arm_v2_final,
            "v2_final_bon3": arm_bon3,
        },
        "per_replicate": arm_v2_final["per_replicate"],
    }

    # Export to results/eval3/ AND mirror to D:\5TH SEM\HACKATHON_eval\test3_v2_final\
    out_repo = PROJECT_ROOT / "results" / "eval3"
    out_repo.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(face_rows).to_csv(out_repo / "test3_per_face.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(out_repo / "test3_per_scene.csv", index=False)
    with open(out_repo / "test3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for fn in ["test3_per_face.csv", "test3_per_scene.csv", "test3_summary.json"]:
        shutil.copyfile(out_repo / fn, run_dir / fn)

    # 7. Prepare blinded B3 Visual QA queue (NO scores stored in the blind queue!)
    rng_qa = np.random.RandomState(42)
    all_primary_audits = [a for r in completed_indices for a in v2_final_by_rep[r]]
    rand_indices = sorted(rng_qa.choice(len(all_primary_audits), size=min(10, len(all_primary_audits)), replace=False).tolist())
    random_10 = [all_primary_audits[i] for i in rand_indices]
    flagged_failures = [a for a in all_primary_audits if (not a["pm3_correct"]) or a.get("render_failed", False)]
    weakest_10 = sorted(
        all_primary_audits,
        key=lambda a: (
            1 if a["pm3_correct"] else 0,
            float(a.get("mean_v1_2_ensemble_score", 0.0)),
            float(a.get("mean_face_sharpness", 0.0)),
        ),
    )[:10]

    # Deduplicate by render_id while tracking membership tags (stored ONLY in unblinded key!)
    seen_qa: Dict[str, Dict[str, Any]] = {}
    qa_tags: Dict[str, List[str]] = {}
    for tag, lst in [("random_10", random_10), ("flagged_failure", flagged_failures), ("weakest_10", weakest_10)]:
        for a in lst:
            rid = a["render_id"]
            qa_tags.setdefault(rid, []).append(tag)
            if rid not in seen_qa:
                seen_qa[rid] = {
                    "render_id": rid,
                    "image_path": a["image_path"],
                    "replicate_idx": a["replicate_idx"],
                    "story_label": a["story_label"],
                    "scene_number": a["scene_number_story"],
                    "expected_characters": a["expected_characters"],
                    "character_positions": a.get("character_positions", {}),
                }
    # Deterministic shuffled presentation order for blind inspection
    qa_order = list(seen_qa.keys())
    rng_qa.shuffle(qa_order)
    blind_queue = [seen_qa[rid] for rid in qa_order]
    with open(run_dir / "visual_qa_blind_queue.json", "w", encoding="utf-8") as f:
        json.dump({"n_images": len(blind_queue), "items": blind_queue}, f, indent=2)
    with open(run_dir / "visual_qa_selection_key.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_random_10": len(random_10),
                "n_flagged_failures": len(flagged_failures),
                "n_weakest_10": len(weakest_10),
                "n_unique_images": len(blind_queue),
                "tags_by_render_id": qa_tags,
            },
            f,
            indent=2,
        )

    log("\n" + "=" * 72)
    log(f"B2 SCORING COMPLETE in {scoring_elapsed_s:.1f}s ({scoring_elapsed_h:.4f}h)")
    for arm_k in ["v2_final", "v2_final_bon3"]:
        ad = summary["arms"][arm_k]
        p1 = ad["primary_metrics"]["pm1_cross_scene_auc"]
        p2 = ad["primary_metrics"]["pm2_cross_scene_rank1"]
        p3 = ad["primary_metrics"]["pm3_scene_correctness_rate"]
        log(f"  [{arm_k}]")
        log(f"    PM1 AUC:    {p1['empirical_mean']:.4f}  95% CI [{p1['bootstrap']['ci_95_lower']:.4f}, {p1['bootstrap']['ci_95_upper']:.4f}] (chance 0.50)")
        log(f"    PM2 Rank-1: {p2['empirical_mean']:.4f}  95% CI [{p2['bootstrap']['ci_95_lower']:.4f}, {p2['bootstrap']['ci_95_upper']:.4f}] (chance {p2['chance_floor']:.4f}, within-story mean {p2['within_story_mean_pm2']:.4f})")
        log(f"    PM3 Rate:   {p3['empirical_mean']:.4f}  95% CI [{p3['bootstrap']['ci_95_lower']:.4f}, {p3['bootstrap']['ci_95_upper']:.4f}] (floor 0.00)")
    log(f"  Visual QA blind queue prepared: {len(blind_queue)} unique images (random=10, failures={len(flagged_failures)}, weakest=10)")
    log("=" * 72)


if __name__ == "__main__":
    main()
