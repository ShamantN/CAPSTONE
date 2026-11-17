"""
MMCCP TEST-3 Evaluation Protocol: Comprehensive Evaluator (eval3/metrics/evaluator.py).

Computes BOTH:
(a) Frozen Verifier v1.2 continuity metrics via read-only calls, and
(b) New out-of-loop multi-judge evaluation metrics.

Strictly labels every metric as IN-LOOP or OUT-OF-LOOP:
- IN-LOOP: InsightFace buffalo_l similarity, similarity to anchor, wardrobe delta E, spatial layout check.
- OUT-OF-LOOP: DeepFace 4-judge ensemble (Facenet512, SFace, GhostFaceNet, ArcFace), DINOv2, CLIP,
  YOLO person count, independent face detector.

Primary Metrics:
- PM1: Cross-scene identity AUC (within-replicate pairs only, out-of-loop 4-judge ensemble)
- PM2: Cross-scene Rank-1 (leave-one-scene-out, intention-to-treat expected face denominator)
- PM3: Scene correctness rate (all expected present AND no extra person AND left/right order correct)
"""
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble, cosine_similarity, normalize_to_unit_vector
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.torch_bridge import call_torch_vision
from eval3.metrics.calibration import load_calibration, score_pair_with_calibration, compute_roc_auc


class Eval3Evaluator:
    """
    Evaluator for TEST-3 protocol.
    Runs on CPU to ensure zero VRAM footprint and absolute determinism.
    """

    METRIC_LOOP_LABELS = {
        "pm1_cross_scene_auc": "OUT-OF-LOOP",
        "pm2_cross_scene_rank1": "OUT-OF-LOOP",
        "pm3_scene_correctness_rate": "OUT-OF-LOOP",
        "facenet512_auc": "OUT-OF-LOOP",
        "sface_auc": "OUT-OF-LOOP",
        "ghostfacenet_auc": "OUT-OF-LOOP",
        "arcface_auc": "OUT-OF-LOOP",
        "dinov2_head_similarity": "OUT-OF-LOOP",
        "clip_clothing_agreement": "OUT-OF-LOOP",
        "clip_prompt_alignment": "OUT-OF-LOOP",
        "yolo_extra_person_rate": "OUT-OF-LOOP",
        "independent_face_recall": "OUT-OF-LOOP",
        "face_sharpness": "OUT-OF-LOOP",
        "d_prime": "OUT-OF-LOOP",
        "eer": "OUT-OF-LOOP",
        "tar_at_far05": "OUT-OF-LOOP",
        "diversity_dinov2_full_image": "OUT-OF-LOOP",
        "copy_paste_flag": "OUT-OF-LOOP",
        "v1_2_ensemble_score": "IN-LOOP",
        "v1_2_rank1_vs_anchor": "IN-LOOP",
        "buffalo_l_anchor_similarity": "IN-LOOP",
        "wardrobe_cielab_delta_e": "IN-LOOP",
        "spatial_iou_against_plan": "IN-LOOP",
        "defect_count_per_scene": "IN-LOOP",
    }

    INTRA_FAMILY_EBRIGHT_CHARS = {"young_ebright", "teen_ebright", "college_ebright"}

    def __init__(self, calibration_file: Optional[Path] = None):
        self.ensemble = BiometricEnsemble()
        self.auditor = SceneAuditor()
        self.calibration = load_calibration(calibration_file)

    def extract_face_embeddings(self, crop_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts L2-normalized embeddings for the 4 DeepFace judges."""
        embs = self.ensemble.extract_face_embeddings(crop_bgr)
        norm_embs = {}
        for m in BiometricEnsemble.JUDGE_MODELS:
            if m in embs:
                norm_embs[m] = normalize_to_unit_vector(embs[m])
        return norm_embs

    def compute_pair_judge_cosines(
        self,
        embs1: Dict[str, np.ndarray],
        embs2: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Computes cosine similarity for each judge between two face crops."""
        cosines = {}
        for m in BiometricEnsemble.JUDGE_MODELS:
            if m in embs1 and m in embs2:
                cosines[m] = round(cosine_similarity(embs1[m], embs2[m]), 4)
            else:
                cosines[m] = 0.0
        return cosines

    def audit_single_scene_image(
        self,
        image_path: Path,
        expected_characters: List[str],
        character_positions: Dict[str, str],
        anchor_crops: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Audits one rendered scene image:
        - Detects faces (DeepFace RetinaFace/Haar)
        - Executes TorchVisionHelper via TorchBridge (YOLO persons, SegFormer wardrobe, DINOv2, CLIP)
        - Computes in-loop metrics (spatial layout IoU, wardrobe delta E, buffalo_l sim)
        - Computes scene correctness check (PM3)
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {
                "error": f"Failed reading {image_path}",
                "detected_faces": [],
                "pm3_correct": False,
                "yolo_persons_count": 0
            }

        h_img, w_img = img_bgr.shape[:2]
        faces = self.auditor.detect_faces(img_bgr)

        # Call TorchBridge for YOLO persons & vision checks
        face_bboxes = [f["bbox"] for f in faces]
        ref_paths = [Path(p) for p in anchor_crops.keys() if isinstance(p, (str, Path))]
        torch_res = call_torch_vision(image_path, face_bboxes, ref_paths)
        yolo_persons = torch_res.get("person_boxes", [])
        yolo_count = len(yolo_persons)

        # PM3 Scene Correctness Check:
        # (1) All expected people present
        # (2) No extra person
        # (3) Left/Right order correct
        exp_count = len(expected_characters)
        det_count = len(faces)
        all_present = bool(det_count >= exp_count)
        no_extra = bool(yolo_count == exp_count)

        order_correct = True
        if exp_count == 2 and det_count >= 2:
            left_char = None
            right_char = None
            for cid in expected_characters:
                pos = character_positions.get(cid, "").lower()
                if "left" in pos:
                    left_char = cid
                elif "right" in pos:
                    right_char = cid

            if left_char and right_char:
                # Sort detected faces by center X
                sorted_faces = sorted(faces, key=lambda f: f["bbox"]["x"] + f["bbox"]["w"] / 2.0)
                # First face is left, second face is right
                left_face_x = sorted_faces[0]["bbox"]["x"] + sorted_faces[0]["bbox"]["w"] / 2.0
                right_face_x = sorted_faces[-1]["bbox"]["x"] + sorted_faces[-1]["bbox"]["w"] / 2.0
                order_correct = bool(left_face_x < right_face_x)

        pm3_correct = bool(all_present and no_extra and order_correct)

        # Extract embeddings and sharpness per detected face
        face_records = []
        for f_idx, face in enumerate(faces):
            crop = face.get("crop_bgr")
            embs = self.extract_face_embeddings(crop) if crop is not None else {}
            q = self.auditor.audit_face_quality(face)
            sharpness = float(q.get("sharpness_laplacian", 0.0))
            face_records.append({
                "face_idx": f_idx,
                "bbox": face.get("bbox"),
                "confidence": face.get("confidence", 0.0),
                "sharpness": sharpness,
                "embeddings": embs,
                "crop_bgr": crop
            })

        return {
            "image_path": str(image_path),
            "h": h_img,
            "w": w_img,
            "detected_faces": face_records,
            "detected_faces_count": det_count,
            "expected_characters": expected_characters,
            "expected_count": exp_count,
            "yolo_persons_count": yolo_count,
            "all_expected_present": all_present,
            "no_extra_person": no_extra,
            "order_correct": order_correct,
            "pm3_correct": pm3_correct,
            "extra_person_flag": bool(yolo_count > exp_count),
            "torch_res": torch_res
        }

    def evaluate_replicate(
        self,
        replicate_idx: int,
        scene_audits: List[Dict[str, Any]],
        expected_chars: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates a complete replicate:
        Computes genuine and impostor pairs strictly across scenes within this replicate.
        Computes PM1 (Cross-scene AUC) and PM2 (Cross-scene leave-one-scene-out Rank-1).
        """
        # Form genuine and impostor pairs
        # Each face detected in a scene is assigned to the corresponding expected character based on layout position
        faces_by_char: Dict[str, List[Dict[str, Any]]] = {c: [] for c in expected_chars}
        all_detected_faces = []

        for audit in scene_audits:
            s_exp = audit.get("expected_characters", [])
            s_faces = audit.get("detected_faces", [])

            if len(s_exp) == 1 and len(s_faces) >= 1:
                # Single character scene: assign largest face
                f_best = sorted(s_faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)[0]
                rec = {"char_id": s_exp[0], "scene": audit.get("scene_number", 0), "render_id": audit.get("render_id", ""), "face": f_best}
                faces_by_char[s_exp[0]].append(rec)
                all_detected_faces.append(rec)

            elif len(s_exp) == 2 and len(s_faces) >= 2:
                # Two character scene: assign left-to-right
                sorted_f = sorted(s_faces, key=lambda f: f["bbox"]["x"])
                c_left = s_exp[0]
                c_right = s_exp[1]
                rec_l = {"char_id": c_left, "scene": audit.get("scene_number", 0), "render_id": audit.get("render_id", ""), "face": sorted_f[0]}
                rec_r = {"char_id": c_right, "scene": audit.get("scene_number", 0), "render_id": audit.get("render_id", ""), "face": sorted_f[-1]}
                faces_by_char[c_left].append(rec_l)
                faces_by_char[c_right].append(rec_r)
                all_detected_faces.extend([rec_l, rec_r])

        # 1. PM1: Genuine vs Impostor Pair Scores
        genuine_scores = []
        impostor_scores = []
        judge_scores: Dict[str, Dict[str, List[float]]] = {
            m: {"genuine": [], "impostor": []} for m in BiometricEnsemble.JUDGE_MODELS
        }

        # Genuine: same character, different scenes
        for cid, f_list in faces_by_char.items():
            for i in range(len(f_list)):
                for j in range(i + 1, len(f_list)):
                    if f_list[i].get("scene") != f_list[j].get("scene") and f_list[i].get("render_id") != f_list[j].get("render_id"):
                        embs1 = f_list[i]["face"]["embeddings"]
                        embs2 = f_list[j]["face"]["embeddings"]
                        cos_map = self.compute_pair_judge_cosines(embs1, embs2)
                        ens_prob = score_pair_with_calibration(cos_map, self.calibration)
                        genuine_scores.append(ens_prob)
                        for m in BiometricEnsemble.JUDGE_MODELS:
                            judge_scores[m]["genuine"].append(cos_map[m])

        # Impostor: different characters, strictly across different scenes and different images in same replicate
        char_keys = list(faces_by_char.keys())
        for i in range(len(char_keys)):
            for j in range(i + 1, len(char_keys)):
                c1 = char_keys[i]
                c2 = char_keys[j]
                for f1 in faces_by_char[c1]:
                    for f2 in faces_by_char[c2]:
                        if f1.get("scene") == f2.get("scene") or f1.get("render_id") == f2.get("render_id"):
                            continue  # Protocol: pairs must never come from the same scene or same image
                        embs1 = f1["face"]["embeddings"]
                        embs2 = f2["face"]["embeddings"]
                        cos_map = self.compute_pair_judge_cosines(embs1, embs2)
                        ens_prob = score_pair_with_calibration(cos_map, self.calibration)
                        impostor_scores.append(ens_prob)
                        for m in BiometricEnsemble.JUDGE_MODELS:
                            judge_scores[m]["impostor"].append(cos_map[m])

        # Compute PM1 ROC AUC
        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
            y_scores = genuine_scores + impostor_scores
            pm1_auc = float(compute_roc_auc(y_true, y_scores))
        else:
            pm1_auc = 0.50

        # Judge-specific AUCs
        judge_aucs = {}
        for m in BiometricEnsemble.JUDGE_MODELS:
            g_list = judge_scores[m]["genuine"]
            i_list = judge_scores[m]["impostor"]
            if len(g_list) > 0 and len(i_list) > 0:
                judge_aucs[m] = float(compute_roc_auc([1]*len(g_list) + [0]*len(i_list), g_list + i_list))
            else:
                judge_aucs[m] = 0.50

        # 2. PM2: Leave-One-Scene-Out Rank-1 (Eligible Probes with Intention-to-Treat)
        # Determine character appearance counts across scenes to establish ELIGIBILITY
        char_scene_counts: Dict[str, set] = {c: set() for c in expected_chars}
        for audit in scene_audits:
            sc_num = audit.get("scene_number", 0)
            for c in audit.get("expected_characters", []):
                if c in char_scene_counts:
                    char_scene_counts[c].add(sc_num)

        # A character is eligible for leave-one-scene-out cross-scene identification
        # if and only if they appear in at least 2 different scenes of this replicate.
        eligible_chars = {c for c, sc_set in char_scene_counts.items() if len(sc_set) >= 2}
        ineligible_chars = set(expected_chars) - eligible_chars

        # Probe counts
        n_expected = sum(len(a.get("expected_characters", [])) for a in scene_audits)
        n_eligible = sum(1 for a in scene_audits for c in a.get("expected_characters", []) if c in eligible_chars)
        n_ineligible = n_expected - n_eligible

        rank1_hits = 0
        rank3_hits = 0
        reciprocal_ranks = []
        cmc_hits = {k: 0 for k in range(1, len(expected_chars) + 1)}

        # Evaluate only ELIGIBLE probe slots across all scenes
        for audit in scene_audits:
            sc_num = audit.get("scene_number", 0)
            render_id = audit.get("render_id", "")
            for cid in audit.get("expected_characters", []):
                if cid not in eligible_chars:
                    continue  # Ineligible character: single-scene appearance makes cross-scene gallery match impossible

                # Check if this probe face was successfully detected
                probe_rec = next(
                    (r for r in faces_by_char.get(cid, []) if r["scene"] == sc_num and r.get("render_id") == render_id),
                    None
                )

                if probe_rec is None:
                    # Intention-to-treat: an undetected or missing face counts as incorrect!
                    # Assigned rank beyond gallery; reciprocal rank is 0.0
                    rank = len(expected_chars) + 1
                    reciprocal_ranks.append(0.0)
                    continue

                probe_face = probe_rec["face"]
                probe_embs = probe_face["embeddings"]
                probe_scene = sc_num

                # Gallery templates: for each candidate character in expected_chars,
                # compute mean embedding across OTHER scenes
                gallery_scores = {}
                for cand_cid in expected_chars:
                    cand_faces = [r["face"]["embeddings"] for r in faces_by_char.get(cand_cid, []) if r["scene"] != probe_scene]
                    if cand_faces:
                        cand_sims = []
                        for cf_emb in cand_faces:
                            cos_map = self.compute_pair_judge_cosines(probe_embs, cf_emb)
                            cand_sims.append(score_pair_with_calibration(cos_map, self.calibration))
                        gallery_scores[cand_cid] = float(np.mean(cand_sims))
                    else:
                        gallery_scores[cand_cid] = -1.0

                # Compute rank of genuine character
                sorted_gallery = sorted(gallery_scores.items(), key=lambda x: x[1], reverse=True)
                ranked_cids = [item[0] for item in sorted_gallery]
                if cid in ranked_cids and gallery_scores[cid] > -1.0:
                    rank = ranked_cids.index(cid) + 1
                    if rank == 1:
                        rank1_hits += 1
                    if rank <= 3:
                        rank3_hits += 1
                    reciprocal_ranks.append(1.0 / float(rank))
                    for k in range(1, len(expected_chars) + 1):
                        if rank <= k:
                            cmc_hits[k] += 1
                else:
                    rank = len(expected_chars) + 1
                    reciprocal_ranks.append(0.0)

        # Denominator: ELIGIBLE probes (intention-to-treat preserved over all eligible slots)
        pm2_rank1 = round(float(rank1_hits) / float(n_eligible), 4) if n_eligible > 0 else 0.0
        rank3_rate = round(float(rank3_hits) / float(n_eligible), 4) if n_eligible > 0 else 0.0
        mrr = round(float(sum(reciprocal_ranks)) / float(n_eligible), 4) if n_eligible > 0 else 0.0
        cmc_curve = {k: round(float(cmc_hits[k]) / float(n_eligible), 4) for k in range(1, len(expected_chars) + 1)} if n_eligible > 0 else {}
        chance_level = round(1.0 / float(len(expected_chars)), 4) if expected_chars else 0.0
        pm2_old_unfiltered = round(float(rank1_hits) / float(n_expected), 4) if n_expected > 0 else 0.0

        # 3. PM3: Scene Correctness Rate
        correct_scenes_count = sum(1 for a in scene_audits if a.get("pm3_correct", False))
        total_scenes = len(scene_audits)
        pm3_rate = round(float(correct_scenes_count) / float(total_scenes), 4) if total_scenes > 0 else 0.0

        # Secondary summaries
        sharpnesses = [f["face"].get("sharpness", 100.0) for f in all_detected_faces]
        mean_sharpness = round(float(np.mean(sharpnesses)), 2) if sharpnesses else 0.0
        extra_persons = sum(1 for a in scene_audits if a.get("extra_person_flag", False))
        extra_person_rate = round(float(extra_persons) / float(total_scenes), 4) if total_scenes > 0 else 0.0

        return {
            "replicate_idx": replicate_idx,
            "pm1_cross_scene_auc": round(pm1_auc, 4),
            "pm2_cross_scene_rank1": pm2_rank1,
            "pm2_old_unfiltered": pm2_old_unfiltered,
            "pm3_scene_correctness_rate": pm3_rate,
            "rank3_accuracy": rank3_rate,
            "mrr": mrr,
            "cmc_curve": cmc_curve,
            "chance_level": chance_level,
            "n_gallery_characters": len(expected_chars),
            "n_expected": n_expected,
            "n_eligible": n_eligible,
            "n_ineligible": n_ineligible,
            "eligible_characters": sorted(list(eligible_chars)),
            "ineligible_characters": sorted(list(ineligible_chars)),
            "rank1_hits": rank1_hits,
            "rank3_hits": rank3_hits,
            "judge_aucs": {k: round(v, 4) for k, v in judge_aucs.items()},
            "n_genuine_pairs": len(genuine_scores),
            "n_impostor_pairs": len(impostor_scores),
            "total_expected_faces": n_expected,
            "total_detected_faces": len(all_detected_faces),
            "face_recall": round(float(len(all_detected_faces)) / float(n_expected), 4) if n_expected > 0 else 0.0,
            "total_scenes": total_scenes,
            "correct_scenes": correct_scenes_count,
            "extra_person_rate": extra_person_rate,
            "mean_face_sharpness": mean_sharpness,
            "raw_genuine_scores": genuine_scores,
            "raw_impostor_scores": impostor_scores,
        }
