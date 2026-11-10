"""
Candidate Screening and Best-of-N Selector (M9)
Evaluates Pass 1 drafts using fast geometric and demographic heuristics
(face count, face size, spatial position, and defect absence)
to select the best candidate for high-resolution Pass 2 refinement.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("MMCCP.Candidates")


class CandidateScore(NamedTuple):
    candidate_id: int
    image_bgr: np.ndarray
    total_score: float
    face_count: int
    expected_faces: int
    mean_face_size_px: float
    binding_passed: bool
    defect_flags: List[str]


class CandidateScreen:
    """
    Screens Pass 1 drafts with lightweight geometric and defect heuristics.
    """
    def __init__(
        self,
        min_face_size_px: int = 64,
        target_face_size_px: int = 110
    ):
        self.min_face_size_px = min_face_size_px
        self.target_face_size_px = target_face_size_px

    def screen_candidate(
        self,
        candidate_id: int,
        image_bgr: np.ndarray,
        expected_faces: int,
        expected_positions: List[Dict[str, Any]]
    ) -> CandidateScore:
        """
        Fast heuristic evaluation of a single Pass 1 draft.
        """
        h_img, w_img = image_bgr.shape[:2]
        defect_flags = []

        # 1. Blank/Corrupt Image Check
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        if brightness_mean < 15.0 or brightness_mean > 240.0 or brightness_std < 10.0:
            defect_flags.append("extreme_exposure_or_blank")

        # 2. Fast Face Detection (OpenCV Haar Cascade fallback for zero-overhead screening)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))

        detected_count = len(faces)
        face_sizes = []
        spatial_hits = 0

        for (fx, fy, fw, fh) in faces:
            face_sizes.append(max(fw, fh))
            # Check horizontal center
            x_center_ratio = (fx + (fw / 2.0)) / max(1, w_img)

            # Match against expected positions
            for ep in expected_positions:
                pos_str = ep.get("position", "").lower()
                if "left" in pos_str and x_center_ratio <= 0.48:
                    spatial_hits += 1
                    break
                elif "right" in pos_str and x_center_ratio >= 0.52:
                    spatial_hits += 1
                    break
                elif ("center" in pos_str or "middle" in pos_str) and (0.30 <= x_center_ratio <= 0.70):
                    spatial_hits += 1
                    break

        mean_size = float(np.mean(face_sizes)) if face_sizes else 0.0

        if detected_count < expected_faces:
            defect_flags.append(f"missing_faces_{detected_count}_of_{expected_faces}")
        elif detected_count > expected_faces + 1:
            defect_flags.append(f"extra_faces_{detected_count}_of_{expected_faces}")

        if face_sizes and min(face_sizes) < self.min_face_size_px:
            defect_flags.append(f"face_too_small_{min(face_sizes)}px")

        binding_passed = (spatial_hits >= expected_faces) or (expected_faces == 0)

        # Compute heuristic score (0.0 to 1.0)
        # Factor A: Face count match (weight 0.40)
        count_diff = abs(detected_count - expected_faces)
        score_count = max(0.0, 1.0 - (0.45 * count_diff))

        # Factor B: Face size adequacy (weight 0.30)
        size_ratio = min(1.0, mean_size / max(1.0, float(self.target_face_size_px)))
        score_size = size_ratio

        # Factor C: Spatial binding (weight 0.30)
        score_binding = 1.0 if binding_passed else 0.35

        total_score = (0.40 * score_count) + (0.30 * score_size) + (0.30 * score_binding)

        if defect_flags:
            total_score *= 0.60  # Defect penalty

        return CandidateScore(
            candidate_id=candidate_id,
            image_bgr=image_bgr,
            total_score=round(total_score, 4),
            face_count=detected_count,
            expected_faces=expected_faces,
            mean_face_size_px=round(mean_size, 1),
            binding_passed=binding_passed,
            defect_flags=defect_flags
        )

    def select_best_candidates(
        self,
        candidates: List[Tuple[int, np.ndarray]],
        expected_faces: int,
        expected_positions: List[Dict[str, Any]],
        top_k: int = 1
    ) -> List[CandidateScore]:
        """
        Screens a list of (candidate_id, image_bgr) tuples and returns the top K sorted by score.
        """
        scores = []
        for cid, img in candidates:
            sc = self.screen_candidate(cid, img, expected_faces, expected_positions)
            scores.append(sc)

        # Sort descending by total score
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores[:top_k]
