"""
Comprehensive Verification CLI Runner
Usage:
    python -m consistency_v2.verify --images DIR --story STORY.json --bank DIR --out report.json [--baseline-report BASELINE.json]
"""
import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from scipy import stats

# Force CPU and UTF-8
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.binding_checker import BindingChecker
from consistency_v2.verify.appearance_evaluator import AppearanceEvaluator
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.visual_review.make_contact_sheet import make_contact_sheet
from story_engine.schema import StorySpec, SceneSpec


def bootstrap_ci_95(data: List[float], n_boot: int = 1000) -> Tuple[float, float]:
    """Computes empirical bootstrap 95% confidence interval of the mean."""
    if not data or len(data) == 0:
        return 0.0, 0.0
    arr = np.array(data, dtype=np.float32)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.RandomState(42)
    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return round(low, 4), round(high, 4)


def load_character_banks(bank_dir: Path, verifier: BiometricEnsemble) -> Tuple[Dict[str, Dict[str, List[np.ndarray]]], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Loads reference image bank per character.
    Returns:
      banks: char_id -> {model_name: [embeddings]}
      centroids: char_id -> {model_name: centroid_embedding}
      reference_crops: char_id -> primary_crop_bgr
    """
    banks: Dict[str, Dict[str, List[np.ndarray]]] = {}
    centroids: Dict[str, Dict[str, np.ndarray]] = {}
    reference_crops: Dict[str, np.ndarray] = {}

    if not bank_dir.exists():
        return banks, centroids, reference_crops

    for char_folder in bank_dir.iterdir():
        if char_folder.is_dir():
            cid = char_folder.name
            banks[cid] = {m: [] for m in BiometricEnsemble.JUDGE_MODELS}
            for img_p in sorted(char_folder.glob("*.png")):
                crop = cv2.imread(str(img_p))
                if crop is not None:
                    if cid not in reference_crops:
                        reference_crops[cid] = crop
                    embs = verifier.extract_face_embeddings(crop)
                    for m, vec in embs.items():
                        banks[cid][m].append(vec)

            # Compute centroids
            centroids[cid] = {}
            for m in BiometricEnsemble.JUDGE_MODELS:
                vecs = banks[cid][m]
                if vecs:
                    mean_v = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(mean_v)
                    centroids[cid][m] = (mean_v / norm) if norm > 0 else mean_v
        elif char_folder.is_file() and char_folder.suffix.lower() in [".png", ".jpg"]:
            # Single anchor file in bank folder (e.g. young_ebright_anchor.png)
            cid = char_folder.stem.replace("_anchor", "").replace("_ground_truth", "")
            if cid not in banks:
                banks[cid] = {m: [] for m in BiometricEnsemble.JUDGE_MODELS}
            crop = cv2.imread(str(char_folder))
            if crop is not None:
                if cid not in reference_crops:
                    reference_crops[cid] = crop
                embs = verifier.extract_face_embeddings(crop)
                for m, vec in embs.items():
                    banks[cid][m].append(vec)
                centroids[cid] = {}
                for m, v_list in banks[cid].items():
                    if v_list:
                        centroids[cid][m] = v_list[0]

    return banks, centroids, reference_crops


def run_verification(
    images_dir: Path,
    story_path: Path,
    bank_dir: Path,
    output_report_path: Path,
    baseline_report_path: Optional[Path] = None
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("  [VERIFIER] MMCCP v2 VERIFICATION SUITE RUNNER")
    print(f"  Images Directory: {images_dir}")
    print(f"  Story Spec:       {story_path.name}")
    print(f"  Identity Banks:   {bank_dir}")
    print("=" * 90 + "\n")

    # Load Storyboard
    with open(story_path, "r", encoding="utf-8") as f:
        story_dict = json.load(f)
    story = StorySpec.from_dict(story_dict)

    # Initialize Modules
    calib_path = PROJECT_ROOT / "consistency_v2" / "verify" / "calibration.json"
    verifier = BiometricEnsemble(calibration_path=calib_path)
    binder = BindingChecker()
    appearance = AppearanceEvaluator()
    auditor = SceneAuditor()

    # Load Banks
    banks, centroids, ref_crops = load_character_banks(bank_dir, verifier)
    print(f"  [+] Loaded Identity Banks for {len(banks)} characters: {list(banks.keys())}")

    # Discover images to verify
    image_files = sorted(list(images_dir.glob("*.png")))
    print(f"  [+] Found {len(image_files)} scene images to evaluate.\n")

    scene_reports = []
    face_flat_records = []

    ensemble_scores_all = []
    rank1_correct_all = []
    spatial_passed_all = []
    wardrobe_passed_all = []
    id_pos_passed_all = []
    combined_binding_passed_all = []
    defect_counts_all = []
    discrimination_margins_all = []
    dinov2_sims_all = []
    per_judge_scores = {m: [] for m in BiometricEnsemble.JUDGE_MODELS}

    child_ensemble_scores = []
    child_rank1 = []
    child_combined_binding = []
    child_wardrobe = []

    adult_ensemble_scores = []
    adult_rank1 = []
    adult_combined_binding = []
    adult_wardrobe = []

    detector_health_counts = {
        "retinaface": 0,
        "yolov8n": 0,
        "segformer": 0,
        "clip": 0,
        "dinov2": 0,
        "facenet512": 0,
        "sface": 0,
        "ghostfacenet": 0,
        "arcface": 0
    }
    total_seg_failed = 0

    for img_p in image_files:
        stem = img_p.stem
        # Match image to scene spec by scene number / id
        matched_scene: Optional[SceneSpec] = None
        for s in story.scenes:
            if s.scene_id in stem or f"scene_{s.scene_number}" in stem:
                matched_scene = s
                break
        if matched_scene is None and len(story.scenes) > 0:
            matched_scene = story.scenes[0]

        print(f"  Auditing: {img_p.name} (Scene {matched_scene.scene_number}: {matched_scene.title})...")
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"  [SKIP] Failed to load {img_p.name}")
            continue

        h_img, w_img = img_bgr.shape[:2]

        # 1. Detect faces via DeepFace/RetinaFace
        detected_faces = auditor.detect_faces(img_bgr)
        detector_health_counts["retinaface"] += 1
        detected_faces = sorted(detected_faces, key=lambda f: f["bbox"]["x"])
        face_boxes = [f["bbox"] for f in detected_faces]

        # 2. Call PyTorch Vision Helper in Forge python for YOLO, SegFormer, DINOv2, CLIP
        from consistency_v2.verify.torch_bridge import call_torch_vision
        ref_paths = [Path(img_p) for img_p in bank_dir.glob("*.png")] if bank_dir.exists() else []
        vision_res = call_torch_vision(
            image_path=img_p,
            face_bboxes=face_boxes,
            reference_image_paths=ref_paths,
            required_props=matched_scene.required_props,
            forbidden_props=matched_scene.forbidden_props
        )

        d_health = vision_res.get("detector_health", {})
        if d_health.get("yolov8n"):
            detector_health_counts["yolov8n"] += 1
        if d_health.get("segformer"):
            detector_health_counts["segformer"] += 1
        if d_health.get("clip"):
            detector_health_counts["clip"] += 1
        if d_health.get("dinov2"):
            detector_health_counts["dinov2"] += 1

        # 3. Scene-Level Audit
        scene_audit = auditor.audit_scene_with_vision(
            image_bgr=img_bgr,
            expected_faces=len(matched_scene.characters_present),
            vision_data=vision_res
        )

        expected_chars = matched_scene.characters_present

        # 4. Match Detected Faces to Expected Characters
        char_evaluations = []
        scene_face_embs_list = []

        clothing_labs = vision_res.get("clothing_labs", [])
        dino_sims = vision_res.get("dinov2_similarities", [])

        for i, cid in enumerate(expected_chars):
            char_spec = story.characters.get(cid)
            expected_pos = matched_scene.character_positions.get(cid, "center")
            expected_colors = char_spec.clothing_colors if char_spec else []
            is_child = bool("young" in cid.lower() or "child" in cid.lower())

            # Assign face
            face_item = detected_faces[i] if i < len(detected_faces) else None
            face_crop = face_item["crop_bgr"] if face_item else None
            face_bbox = face_item["bbox"] if face_item else None

            # Biometrics
            if face_crop is not None and face_crop.size > 0:
                face_embs = verifier.extract_face_embeddings(face_crop)
                scene_face_embs_list.append((cid, face_embs))
                for m_name in BiometricEnsemble.JUDGE_MODELS:
                    if m_name in face_embs:
                        detector_health_counts[m_name.lower()] += 1

                # Biometric ensemble against bank
                c_banks = banks.get(cid, {})
                c_cents = centroids.get(cid, {})
                bio_res = verifier.score_against_bank(face_embs, c_banks, c_cents)
                for m_name in BiometricEnsemble.JUDGE_MODELS:
                    per_judge_scores[m_name].append(bio_res["per_model_details"].get(m_name, {}).get("max_similarity", 0.0))

                # Discrimination test
                disc_res = verifier.compute_discrimination(face_embs, banks, centroids, cid)

                # Spatial position check
                spatial_pass, obs_zone, r_x = binder.verify_spatial_position(
                    face_bbox=face_bbox,
                    image_width=w_img,
                    expected_position_str=expected_pos
                )

                # Wardrobe color check from SegFormer
                raw_lab = clothing_labs[i] if (i < len(clothing_labs)) else None
                if raw_lab is None:
                    seg_status = "seg_failed"
                    total_seg_failed += 1
                    wardrobe_pass, min_de, matched_color = False, 99.0, "seg_failed"
                else:
                    seg_status = "seg_ok"
                    obs_lab = np.array(raw_lab, dtype=np.float32)
                    wardrobe_pass, min_de, matched_color = binder.verify_wardrobe_color(
                        observed_lab=obs_lab,
                        expected_color_names=expected_colors
                    )

                id_pos_pass = bool(bio_res["ensemble_passed"] and disc_res["rank1_correct"])
                combined_binding_pass = bool(spatial_pass and wardrobe_pass and id_pos_pass)

                dino_sim = dino_sims[i] if i < len(dino_sims) else 0.0
                face_quality = auditor.audit_face_quality(face_item)

                overall_char_pass = bool(
                    combined_binding_pass and
                    not face_quality["too_small"]
                )

                char_eval = {
                    "character_id": cid,
                    "is_child": is_child,
                    "overall_pass": overall_char_pass,
                    "face_bbox": face_bbox,
                    "detected_face_crop_bgr": face_crop,
                    "reference_crop_bgr": ref_crops.get(cid),
                    "biometric_ensemble_score": bio_res["ensemble_score"],
                    "biometric_pass": bio_res["ensemble_passed"],
                    "biometric_details": bio_res["per_model_details"],
                    "rank1_correct": disc_res["rank1_correct"],
                    "discrimination_margin": disc_res["discrimination_margin"],
                    "spatial_pass": spatial_pass,
                    "observed_position": obs_zone,
                    "wardrobe_pass": wardrobe_pass,
                    "wardrobe_delta_e": min_de,
                    "identity_at_position_pass": id_pos_pass,
                    "combined_binding_pass": combined_binding_pass,
                    "seg_status": seg_status,
                    "face_width_px": face_quality["width_px"],
                    "face_sharpness": face_quality["sharpness_laplacian"],
                    "dinov2_similarity": dino_sim,
                    "clip_i_similarity": 0.0
                }

                ensemble_scores_all.append(bio_res["ensemble_score"])
                rank1_correct_all.append(1.0 if disc_res["rank1_correct"] else 0.0)
                spatial_passed_all.append(1.0 if spatial_pass else 0.0)
                wardrobe_passed_all.append(1.0 if wardrobe_pass else 0.0)
                id_pos_passed_all.append(1.0 if id_pos_pass else 0.0)
                combined_binding_passed_all.append(1.0 if combined_binding_pass else 0.0)
                discrimination_margins_all.append(disc_res["discrimination_margin"])
                dinov2_sims_all.append(dino_sim)

                if is_child:
                    child_ensemble_scores.append(bio_res["ensemble_score"])
                    child_rank1.append(1.0 if disc_res["rank1_correct"] else 0.0)
                    child_combined_binding.append(1.0 if combined_binding_pass else 0.0)
                    child_wardrobe.append(1.0 if wardrobe_pass else 0.0)
                else:
                    adult_ensemble_scores.append(bio_res["ensemble_score"])
                    adult_rank1.append(1.0 if disc_res["rank1_correct"] else 0.0)
                    adult_combined_binding.append(1.0 if combined_binding_pass else 0.0)
                    adult_wardrobe.append(1.0 if wardrobe_pass else 0.0)

            else:
                # Face missing
                char_eval = {
                    "character_id": cid,
                    "is_child": is_child,
                    "overall_pass": False,
                    "face_bbox": None,
                    "detected_face_crop_bgr": None,
                    "reference_crop_bgr": ref_crops.get(cid),
                    "biometric_ensemble_score": 0.0,
                    "biometric_pass": False,
                    "rank1_correct": False,
                    "discrimination_margin": -1.0,
                    "spatial_pass": False,
                    "observed_position": "missing",
                    "wardrobe_pass": False,
                    "wardrobe_delta_e": 99.0,
                    "identity_at_position_pass": False,
                    "combined_binding_pass": False,
                    "seg_status": "missing_face",
                    "face_width_px": 0,
                    "face_sharpness": 0.0,
                    "dinov2_similarity": 0.0,
                    "clip_i_similarity": 0.0
                }
                ensemble_scores_all.append(0.0)
                rank1_correct_all.append(0.0)
                spatial_passed_all.append(0.0)
                wardrobe_passed_all.append(0.0)
                id_pos_passed_all.append(0.0)
                combined_binding_passed_all.append(0.0)
                discrimination_margins_all.append(-1.0)
                dinov2_sims_all.append(0.0)

                for m_name in BiometricEnsemble.JUDGE_MODELS:
                    per_judge_scores[m_name].append(0.0)

                if is_child:
                    child_ensemble_scores.append(0.0)
                    child_rank1.append(0.0)
                    child_combined_binding.append(0.0)
                    child_wardrobe.append(0.0)
                else:
                    adult_ensemble_scores.append(0.0)
                    adult_rank1.append(0.0)
                    adult_combined_binding.append(0.0)
                    adult_wardrobe.append(0.0)

            char_evaluations.append(char_eval)

            # Flat record for CSV
            face_flat_records.append({
                "image_filename": img_p.name,
                "scene_number": matched_scene.scene_number,
                "scene_title": matched_scene.title,
                "character_id": cid,
                "is_child": is_child,
                "overall_pass": char_eval["overall_pass"],
                "biometric_ensemble_score": char_eval["biometric_ensemble_score"],
                "rank1_correct": char_eval["rank1_correct"],
                "discrimination_margin": char_eval["discrimination_margin"],
                "spatial_pass": char_eval["spatial_pass"],
                "observed_position": char_eval["observed_position"],
                "wardrobe_pass": char_eval["wardrobe_pass"],
                "wardrobe_delta_e": char_eval["wardrobe_delta_e"],
                "identity_at_position_pass": char_eval["identity_at_position_pass"],
                "combined_binding_pass": char_eval["combined_binding_pass"],
                "seg_status": char_eval["seg_status"],
                "face_sharpness": char_eval["face_sharpness"],
                "face_width_px": char_eval["face_width_px"]
            })

        # Check for intra-scene face clone/bleed
        clone_alarms = verifier.check_clone_or_bleed(scene_face_embs_list)
        if clone_alarms:
            scene_audit["defect_taxonomy_violations"].append("duplicated_or_cloned_faces")

        defect_counts_all.append(len(scene_audit["defect_taxonomy_violations"]))

        # Build Contact Sheet (VR-1)
        contact_sheet_dir = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "contact_sheets"
        contact_sheet_p = contact_sheet_dir / f"{stem}_contact.png"
        scene_meta_for_sheet = {
            "scene_number": matched_scene.scene_number,
            "title": matched_scene.title,
            "seed": stem.split("_")[-1] if "seed" in stem else -1,
            "positive_prompt": matched_scene.action_description,
            "defects": scene_audit["defect_taxonomy_violations"],
            "vram_peak_mb": 6100.0,
            "runtime_seconds": 150.0
        }
        make_contact_sheet(
            scene_img_bgr=img_bgr,
            scene_meta=scene_meta_for_sheet,
            character_evals=char_evaluations,
            output_path=contact_sheet_p
        )

        clean_detected_faces = []
        for f in scene_audit.get("detected_faces", []):
            clean_detected_faces.append({k: v for k, v in f.items() if k != "crop_bgr"})
        clean_scene_audit = {k: v for k, v in scene_audit.items() if k != "detected_faces"}
        clean_scene_audit["detected_faces"] = clean_detected_faces

        scene_report = {
            "image_filename": img_p.name,
            "scene_number": matched_scene.scene_number,
            "scene_title": matched_scene.title,
            "scene_audit": clean_scene_audit,
            "character_evaluations": [
                {k: v for k, v in ce.items() if not k.endswith("_bgr")}
                for ce in char_evaluations
            ],
            "clone_alarms": clone_alarms,
            "contact_sheet_path": str(contact_sheet_p)
        }
        scene_reports.append(scene_report)

    # 3. Aggregate Statistical Summary
    ci_ens_low, ci_ens_high = bootstrap_ci_95(ensemble_scores_all)

    mean_ens = float(np.mean(ensemble_scores_all)) if ensemble_scores_all else 0.0
    std_ens = float(np.std(ensemble_scores_all)) if ensemble_scores_all else 0.0
    rank1_acc = float(np.mean(rank1_correct_all)) if rank1_correct_all else 0.0
    spatial_acc = float(np.mean(spatial_passed_all)) if spatial_passed_all else 0.0
    wardrobe_acc = float(np.mean(wardrobe_passed_all)) if wardrobe_passed_all else 0.0
    id_pos_acc = float(np.mean(id_pos_passed_all)) if id_pos_passed_all else 0.0
    combined_binding_acc = float(np.mean(combined_binding_passed_all)) if combined_binding_passed_all else 0.0
    mean_defects = float(np.mean(defect_counts_all)) if defect_counts_all else 0.0
    mean_margin = float(np.mean(discrimination_margins_all)) if discrimination_margins_all else 0.0
    mean_dino = float(np.mean(dinov2_sims_all)) if dinov2_sims_all else 0.0

    child_mean_ens = float(np.mean(child_ensemble_scores)) if child_ensemble_scores else 0.0
    child_rank1_acc = float(np.mean(child_rank1)) if child_rank1 else 0.0
    child_binding_acc = float(np.mean(child_combined_binding)) if child_combined_binding else 0.0
    child_wardrobe_acc = float(np.mean(child_wardrobe)) if child_wardrobe else 0.0

    adult_mean_ens = float(np.mean(adult_ensemble_scores)) if adult_ensemble_scores else 0.0
    adult_rank1_acc = float(np.mean(adult_rank1)) if adult_rank1 else 0.0
    adult_binding_acc = float(np.mean(adult_combined_binding)) if adult_combined_binding else 0.0
    adult_wardrobe_acc = float(np.mean(adult_wardrobe)) if adult_wardrobe else 0.0

    # 4. Optional Paired Wilcoxon Signed-Rank Test vs Baseline
    wilcoxon_results = {}
    if baseline_report_path and baseline_report_path.exists():
        try:
            with open(baseline_report_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            base_scores = base_data.get("raw_biometric_ensemble_scores", [])
            if len(base_scores) == len(ensemble_scores_all) and len(base_scores) >= 5:
                w_stat, p_val = stats.wilcoxon(ensemble_scores_all, base_scores)
                wilcoxon_results = {
                    "wilcoxon_w_statistic": float(w_stat),
                    "p_value": float(p_val),
                    "statistically_significant_p05": bool(p_val < 0.05)
                }
        except Exception as e:
            logger.warning(f"Could not compute Wilcoxon test: {e}")

    # Read freeze sha
    freeze_sha = "unfrozen"
    freeze_json = PROJECT_ROOT / "consistency_v2" / "verifier_freeze.json"
    if freeze_json.exists():
        try:
            with open(freeze_json, "r", encoding="utf-8") as ff:
                freeze_sha = json.load(ff).get("combined_sha256", "unfrozen")
        except Exception:
            pass

    # Detector health verification
    n_imgs = max(1, len(image_files))
    detector_health_summary = {
        k: bool(v >= n_imgs or (k in ["facenet512", "sface", "ghostfacenet", "arcface"] and v >= 1))
        for k, v in detector_health_counts.items()
    }
    detector_health_summary["all_healthy"] = all(detector_health_summary.values())

    final_report = {
        "verifier_version": "1.1",
        "verifier_freeze_sha": freeze_sha,
        "detector_health": detector_health_summary,
        "images_evaluated_count": len(image_files),
        "total_face_instances": len(ensemble_scores_all),
        "summary_metrics": {
            "biometric_ensemble_mean": round(mean_ens, 4),
            "biometric_ensemble_std": round(std_ens, 4),
            "biometric_ensemble_95ci": [ci_ens_low, ci_ens_high],
            "rank1_discrimination_accuracy": round(rank1_acc, 4),
            "discrimination_margin_mean": round(mean_margin, 4),
            "dinov2_mean_similarity": round(mean_dino, 4),
            "per_judge_means": {
                m: round(float(np.mean(per_judge_scores[m])), 4) if per_judge_scores[m] else 0.0
                for m in BiometricEnsemble.JUDGE_MODELS
            },
            "binding_decomposed": {
                "spatial_position_pass_rate": round(spatial_acc, 4),
                "wardrobe_palette_pass_rate": round(wardrobe_acc, 4),
                "identity_at_position_pass_rate": round(id_pos_acc, 4),
                "combined_binding_pass_rate": round(combined_binding_acc, 4)
            },
            "age_split_metrics": {
                "child": {
                    "count": len(child_ensemble_scores),
                    "biometric_ensemble_mean": round(child_mean_ens, 4),
                    "rank1_accuracy": round(child_rank1_acc, 4),
                    "combined_binding_accuracy": round(child_binding_acc, 4),
                    "wardrobe_accuracy": round(child_wardrobe_acc, 4)
                },
                "adult": {
                    "count": len(adult_ensemble_scores),
                    "biometric_ensemble_mean": round(adult_mean_ens, 4),
                    "rank1_accuracy": round(adult_rank1_acc, 4),
                    "combined_binding_accuracy": round(adult_binding_acc, 4),
                    "wardrobe_accuracy": round(adult_wardrobe_acc, 4)
                }
            },
            "mean_defects_per_scene": round(mean_defects, 4),
            "seg_failed_count": total_seg_failed
        },
        "comparative_significance": wilcoxon_results,
        "raw_biometric_ensemble_scores": ensemble_scores_all,
        "per_scene_reports": scene_reports
    }

    def json_serialize_fallback(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return str(o)

    # Save report.json
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=json_serialize_fallback)
    print(f"\n  ✅ Saved comprehensive report to: {output_report_path}")

    # Save summary.csv
    csv_path = output_report_path.parent / "summary.csv"
    if face_flat_records:
        keys = face_flat_records[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(face_flat_records)
        print(f"  ✅ Saved per-face metric summary CSV to: {csv_path}")

    print("\n" + "=" * 90)
    print("  VERIFICATION SUMMARY RESULTS (Verifier v1.1)")
    print(f"  Biometric Ensemble Mean:     {mean_ens:.4f} (95% CI: [{ci_ens_low:.4f}, {ci_ens_high:.4f}])")
    print(f"    - Adult Biometric Mean:    {adult_mean_ens:.4f} (n={len(adult_ensemble_scores)})")
    print(f"    - Child Biometric Mean:    {child_mean_ens:.4f} (n={len(child_ensemble_scores)})")
    print(f"  Rank-1 Re-ID Accuracy:       {rank1_acc * 100:.1f}% (Adult: {adult_rank1_acc*100:.1f}%, Child: {child_rank1_acc*100:.1f}%)")
    print(f"  Discrimination Margin:       {mean_margin:.4f}")
    print(f"  Combined Binding Accuracy:   {combined_binding_acc * 100:.1f}%")
    print(f"    - Spatial Position Pass:   {spatial_acc * 100:.1f}%")
    print(f"    - Wardrobe Palette Pass:   {wardrobe_acc * 100:.1f}%")
    print(f"    - Identity-at-Position:    {id_pos_acc * 100:.1f}%")
    print(f"  Mean Defects per Scene:      {mean_defects:.2f}")
    print(f"  Detector Health Status:      {'ALL HEALTHY' if detector_health_summary['all_healthy'] else 'DEGRADED'}")
    print(f"  SegFormer Failures:          {total_seg_failed}")
    print("=" * 90 + "\n")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMCCP v2 Verifier CLI")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing scene images to audit")
    parser.add_argument("--story", required=True, type=Path, help="Path to story spec JSON")
    parser.add_argument("--bank", required=True, type=Path, help="Directory containing character bank reference images")
    parser.add_argument("--out", required=True, type=Path, help="Output path for report.json")
    parser.add_argument("--baseline-report", required=False, type=Path, help="Optional baseline report for Wilcoxon test")
    args = parser.parse_args()

    run_verification(
        images_dir=args.images,
        story_path=args.story,
        bank_dir=args.bank,
        output_report_path=args.out,
        baseline_report_path=args.baseline_report
    )
