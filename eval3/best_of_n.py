"""
MMCCP TEST-3 Evaluation Protocol: In-Loop Best-of-3 Selector (eval3/best_of_n.py).

Implements deployment-realistic selection rule for v2_final_bon3 across the 3 seeds
rendered for v2_final without requiring any additional generation:
1. Correct person count and no extra person (from YOLOv8n detector)
2. All expected faces detected
3. Highest InsightFace buffalo_l similarity to character anchor(s)
4. Highest face sharpness (Laplacian variance)
5. Lowest seed as tie-breaker
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def select_best_of_three_scene(
    candidates: List[Dict[str, Any]],
    expected_characters: List[str]
) -> Dict[str, Any]:
    """
    Selects best render among 3 seeds according to deployment-realistic rule:
    Each candidate dict contains:
      - seed: int
      - image_bgr: np.ndarray (or None if failed)
      - yolo_person_count: int
      - detected_faces_count: int
      - buffalo_sim_to_anchors: float (mean across expected faces)
      - mean_face_sharpness: float
      - render_status: str ("SUCCESS", "FAILED")
    """
    assert len(candidates) > 0, "No candidates provided to best-of-3 selector!"

    exp_count = len(expected_characters)

    scored_cands = []
    for cand in candidates:
        if cand.get("render_status") != "SUCCESS" or cand.get("image_bgr") is None:
            # Fatal render failure
            tier = 0
            sim = -1.0
            sharp = -1.0
        else:
            yolo_cnt = cand.get("yolo_person_count", 0)
            det_cnt = cand.get("detected_faces_count", 0)
            count_ok = (yolo_cnt == exp_count)
            all_detected = (det_cnt >= exp_count)

            if count_ok and all_detected:
                tier = 3  # Best tier: perfect person count & all faces found
            elif all_detected:
                tier = 2  # Found all faces, but count had extra/missing person
            elif count_ok:
                tier = 1  # Count matched, but face detector missed a crop
            else:
                tier = 0  # Severe mismatch

            sim = float(cand.get("buffalo_sim_to_anchors", 0.0))
            sharp = float(cand.get("mean_face_sharpness", 0.0))

        scored_cands.append({
            "candidate": cand,
            "seed": cand.get("seed", 0),
            "tier": tier,
            "sim": sim,
            "sharp": sharp,
        })

    # Sort: tier DESC, sim DESC, sharp DESC, seed ASC
    sorted_cands = sorted(
        scored_cands,
        key=lambda item: (-item["tier"], -item["sim"], -item["sharp"], item["seed"])
    )

    winner_record = sorted_cands[0]
    best_cand = winner_record["candidate"].copy()
    best_cand["bon3_selection_tier"] = winner_record["tier"]
    best_cand["bon3_selected_sim"] = winner_record["sim"]
    best_cand["bon3_selected_sharp"] = winner_record["sharp"]
    return best_cand


def score_best_of_n_candidates(
    candidates: List[Dict[str, Any]],
    expected_characters_count: int = 1
) -> List[Dict[str, Any]]:
    """
    Ranks candidates by deployment-realistic tiering:
    - Tier 3: perfect person count and all expected faces detected
    - Tier 2: all faces detected but extra person
    - Tier 1: person count matched but face missed
    - Tier 0: severe mismatch
    Secondary sort: sharpness DESC, buffalo_l_sim DESC.
    """
    scored = []
    for cand in candidates:
        yolo_cnt = cand.get("yolo_persons_count", cand.get("yolo_person_count", 0))
        det_cnt = cand.get("detected_faces_count", 0)
        count_ok = (yolo_cnt == expected_characters_count)
        all_det = (det_cnt >= expected_characters_count)

        if count_ok and all_det:
            tier = 3
        elif all_det:
            tier = 2
        elif count_ok:
            tier = 1
        else:
            tier = 0

        sharp = float(cand.get("sharpness", cand.get("mean_face_sharpness", 0.0)))
        sim = float(cand.get("buffalo_l_sim", cand.get("buffalo_sim_to_anchors", 0.0)))
        c_copy = cand.copy()
        c_copy["tier"] = tier
        scored.append((tier, sharp, sim, c_copy))

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return [x[3] for x in scored]
