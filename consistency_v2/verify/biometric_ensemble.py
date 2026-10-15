"""
Biometric Ensemble Verifier
Breaks evaluation circularity by using independent face recognition models
(Facenet512, SFace, GhostFaceNet, DeepFace-ArcFace) on CPU.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

# Force CPU and UTF-8
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from deepface import DeepFace

logger = logging.getLogger("MMCCP.BiometricEnsemble")


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm == 0.0 or v_norm == 0.0:
        return 0.0
    return float(np.dot(u, v) / (u_norm * v_norm))


def normalize_to_unit_vector(vec: Union[List[float], np.ndarray]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        return arr / norm
    return arr


class BiometricEnsemble:
    """
    Ensemble of independent facial recognition models running on CPU.
    Evaluates face crops against multi-reference identity banks.
    """
    JUDGE_MODELS = ["Facenet512", "SFace", "GhostFaceNet", "ArcFace"]

    # Calibrated default thresholds (cosine similarity)
    DEFAULT_THRESHOLDS = {
        "Facenet512": 0.40,
        "SFace": 0.45,
        "GhostFaceNet": 0.42,
        "ArcFace": 0.48
    }

    def __init__(self, calibration_path: Optional[Path] = None):
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if calibration_path and Path(calibration_path).exists():
            self.load_calibration(Path(calibration_path))

    def load_calibration(self, calib_file: Path):
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "thresholds" in data:
                    self.thresholds.update(data["thresholds"])
        except Exception as e:
            logger.warning(f"Could not load calibration file: {e}")

    def extract_face_embeddings(self, face_img_input: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extracts embeddings for a single face crop across all judge models.
        Input can be file path or BGR numpy array.
        """
        if isinstance(face_img_input, str):
            img_bgr = cv2.imread(face_img_input)
        else:
            img_bgr = face_img_input

        if img_bgr is None or img_bgr.size == 0:
            return {}

        embeddings = {}
        for model_name in self.JUDGE_MODELS:
            try:
                res = DeepFace.represent(
                    img_path=img_bgr,
                    model_name=model_name,
                    detector_backend="skip",
                    enforce_detection=False
                )
                if res and len(res) > 0 and "embedding" in res[0]:
                    embeddings[model_name] = normalize_to_unit_vector(res[0]["embedding"])
            except Exception as e:
                logger.debug(f"DeepFace {model_name} extraction failed: {e}")
        return embeddings

    def score_against_bank(
        self,
        face_embeddings: Dict[str, np.ndarray],
        bank_embeddings: Dict[str, List[np.ndarray]],
        bank_centroids: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Scores extracted face embeddings against a character bank.
        Returns per-model scores, centroid score, max bank score, and calibrated ensemble.
        """
        model_scores = {}
        calibrated_scores = {}

        for model_name in self.JUDGE_MODELS:
            if model_name not in face_embeddings:
                continue

            emb = face_embeddings[model_name]
            thresh = self.thresholds.get(model_name, 0.45)

            # Centroid match
            centroid_sim = 0.0
            if bank_centroids and model_name in bank_centroids:
                centroid_sim = cosine_similarity(emb, bank_centroids[model_name])

            # Max match across bank images
            max_sim = centroid_sim
            if model_name in bank_embeddings and len(bank_embeddings[model_name]) > 0:
                for b_emb in bank_embeddings[model_name]:
                    s = cosine_similarity(emb, b_emb)
                    if s > max_sim:
                        max_sim = s

            model_scores[model_name] = {
                "max_similarity": round(max_sim, 4),
                "centroid_similarity": round(centroid_sim, 4),
                "threshold": thresh,
                "passed": bool(max_sim >= thresh)
            }

            # Map to calibrated [0, 1] scale: 0.5 is threshold, 1.0 is perfect match
            # Calibrated formula: S_cal = 0.5 + 0.5 * (S - thresh) / (1.0 - thresh) if S >= thresh else 0.5 * (S / thresh)
            if max_sim >= thresh:
                cal_val = 0.5 + 0.5 * min(1.0, max(0.0, (max_sim - thresh) / max(0.01, (1.0 - thresh))))
            else:
                cal_val = 0.5 * max(0.0, max_sim / max(0.01, thresh))
            calibrated_scores[model_name] = round(cal_val, 4)

        if calibrated_scores:
            ensemble_score = round(float(np.mean(list(calibrated_scores.values()))), 4)
            pass_count = sum(1 for v in model_scores.values() if v["passed"])
            ensemble_pass = bool(pass_count >= 3)  # At least 3 of 4 judges pass
        else:
            ensemble_score = 0.0
            ensemble_pass = False

        return {
            "ensemble_score": ensemble_score,
            "ensemble_passed": ensemble_pass,
            "calibrated_scores": calibrated_scores,
            "per_model_details": model_scores
        }

    def compute_discrimination(
        self,
        face_embeddings: Dict[str, np.ndarray],
        all_banks: Dict[str, Dict[str, List[np.ndarray]]],
        all_centroids: Dict[str, Dict[str, np.ndarray]],
        target_char_id: str
    ) -> Dict[str, Any]:
        """
        Rank-1 Discrimination Test:
        Compares the face embedding against ALL known character banks.
        Determines whether the intended character is Rank-1 (argmax).
        Computes the margin between the target and runner-up impostor.
        """
        character_scores = {}
        for cid, b_embs in all_banks.items():
            cents = all_centroids.get(cid)
            res = self.score_against_bank(face_embeddings, b_embs, cents)
            character_scores[cid] = res["ensemble_score"]

        if not character_scores:
            return {"rank1_correct": False, "margin": 0.0, "scores": {}}

        sorted_chars = sorted(character_scores.items(), key=lambda x: x[1], reverse=True)
        rank1_id, rank1_score = sorted_chars[0]
        target_score = character_scores.get(target_char_id, 0.0)

        # Runner-up impostor score
        runner_up_score = 0.0
        for cid, sc in sorted_chars:
            if cid != target_char_id:
                runner_up_score = sc
                break

        margin = round(target_score - runner_up_score, 4)
        is_rank1 = bool(rank1_id == target_char_id)

        return {
            "rank1_character": rank1_id,
            "rank1_correct": is_rank1,
            "target_score": target_score,
            "runner_up_score": runner_up_score,
            "discrimination_margin": margin,
            "all_character_scores": character_scores
        }

    @staticmethod
    def check_clone_or_bleed(
        face_embeddings_list: List[Tuple[str, Dict[str, np.ndarray]]],
        clone_threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        """
        Pairwise similarity check between different faces in the SAME scene.
        Detects face cloning / identity bleed if similarity > clone_threshold.
        """
        alarms = []
        n = len(face_embeddings_list)
        for i in range(n):
            c_i, embs_i = face_embeddings_list[i]
            for j in range(i + 1, n):
                c_j, embs_j = face_embeddings_list[j]
                # Compare models present in both
                sims = []
                for m in ["Facenet512", "ArcFace", "GhostFaceNet"]:
                    if m in embs_i and m in embs_j:
                        sims.append(cosine_similarity(embs_i[m], embs_j[m]))
                if sims:
                    mean_cross_sim = float(np.mean(sims))
                    if mean_cross_sim >= clone_threshold:
                        alarms.append({
                            "face_a_character": c_i,
                            "face_b_character": c_j,
                            "cross_similarity": round(mean_cross_sim, 4),
                            "alarm": "CLONE_OR_BLEED_DETECTED",
                            "severity": "CRITICAL"
                        })
        return alarms
