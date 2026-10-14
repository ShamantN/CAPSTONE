"""
Cross-Scene Identity Memory with Drift Guard (M10)
Dynamically updates character reference banks across narrative scenes
with strict drift guard boundaries to prevent semantic and facial identity drift.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from pathlib import Path
import cv2
import numpy as np

from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble

logger = logging.getLogger("MMCCP.IdentityMemory")


class PromotionDecision(NamedTuple):
    promoted: bool
    character_id: str
    similarity_to_original: float
    centroid_drift: float
    reason: str


class IdentityMemoryManager:
    """
    Evaluates and safely promotes verified face renders from accepted scenes
    into character identity banks while strictly bounding centroid drift.
    """
    def __init__(
        self,
        bank_manager: IdentityBankManager,
        max_bank_size: int = 8,
        min_similarity_to_anchor: float = 0.72,
        max_centroid_drift: float = 0.05,
        min_face_size_px: int = 96
    ):
        self.bank_manager = bank_manager
        self.max_bank_size = max_bank_size
        self.min_similarity_to_anchor = min_similarity_to_anchor
        self.max_centroid_drift = max_centroid_drift
        self.min_face_size_px = min_face_size_px

        # Cache original anchor centroids per character to prevent long-term cumulative drift
        self._original_centroids: Dict[str, Dict[str, np.ndarray]] = {}
        self._init_original_centroids()

    def _init_original_centroids(self):
        """Snapshots the original certified anchor centroids from bank manager."""
        for char_id, meta in self.bank_manager._banks.items():
            centroids = meta.get("centroid_embeddings", {})
            if centroids:
                self._original_centroids[char_id] = {
                    m: np.array(vec, dtype=np.float32)
                    for m, vec in centroids.items()
                }

    def consider_scene_face_promotion(
        self,
        character_id: str,
        scene_image_bgr: np.ndarray,
        face_bbox: Dict[str, int],
        scene_id: int
    ) -> PromotionDecision:
        """
        Evaluates a rendered face for potential bank promotion with drift guard:
        1. Face size >= min_face_size_px (>= 96px).
        2. Sharpness check (Laplacian variance >= 60.0).
        3. Similarity to ORIGINAL certified anchor centroid >= min_similarity_to_anchor (>= 0.72).
        4. Drift Guard: Candidate centroid shift from original anchor centroid <= max_centroid_drift (<= 0.05).
        """
        fx, fy, fw, fh = face_bbox.get("x", 0), face_bbox.get("y", 0), face_bbox.get("w", 0), face_bbox.get("h", 0)
        h_img, w_img = scene_image_bgr.shape[:2]

        # 1. Size check
        face_size = max(fw, fh)
        if face_size < self.min_face_size_px:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=0.0,
                centroid_drift=0.0,
                reason=f"Face size ({face_size}px) below minimum required ({self.min_face_size_px}px)"
            )

        # Crop face
        face_crop = scene_image_bgr[fy:fy+fh, fx:fx+fw]
        if face_crop.size == 0:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=0.0,
                centroid_drift=0.0,
                reason="Invalid empty face crop"
            )

        # 2. Sharpness check
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 60.0:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=0.0,
                centroid_drift=0.0,
                reason=f"Face sharpness ({lap_var:.1f}) below threshold (60.0)"
            )

        # 3. Extract candidate embeddings
        cand_embs = self.bank_manager.ensemble.extract_face_embeddings(face_crop)
        if not cand_embs:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=0.0,
                centroid_drift=0.0,
                reason="Failed to extract facial embeddings from candidate face"
            )

        orig_centroid = self._original_centroids.get(character_id)
        if not orig_centroid:
            # Fallback to current bank centroid if original snapshot wasn't present
            current_meta = self.bank_manager._banks.get(character_id, {})
            current_c = current_meta.get("centroid_embeddings", {})
            orig_centroid = {m: np.array(v, dtype=np.float32) for m, v in current_c.items()}
            self._original_centroids[character_id] = orig_centroid

        if not orig_centroid:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=0.0,
                centroid_drift=0.0,
                reason=f"No certified anchor centroid available for character {character_id}"
            )

        # Compute similarity to original anchor centroid
        sims = []
        for m in BiometricEnsemble.JUDGE_MODELS:
            if m in cand_embs and m in orig_centroid:
                sims.append(float(np.dot(cand_embs[m], orig_centroid[m])))

        sim_to_orig = float(np.mean(sims)) if sims else 0.0

        if sim_to_orig < self.min_similarity_to_anchor:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=round(sim_to_orig, 4),
                centroid_drift=0.0,
                reason=f"Similarity to anchor ({sim_to_orig:.3f}) below threshold ({self.min_similarity_to_anchor:.3f})"
            )

        # 4. DRIFT GUARD CHECK:
        # Simulate candidate centroid update and verify Euclidean distance to original anchor centroid
        current_meta = self.bank_manager._banks.get(character_id, {})
        refs = current_meta.get("references", [])
        num_refs = len(refs)

        # Check bank capacity
        if num_refs >= self.max_bank_size:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=round(sim_to_orig, 4),
                centroid_drift=0.0,
                reason=f"Bank capacity reached ({num_refs}/{self.max_bank_size})"
            )

        drifts = []
        new_centroids = {}
        for m in BiometricEnsemble.JUDGE_MODELS:
            if m in cand_embs and m in orig_centroid:
                orig_vec = orig_centroid[m]
                cand_vec = cand_embs[m]
                # Hypothetical new centroid with new reference added:
                # new_vec = (current_vec * num_refs + cand_vec) / (num_refs + 1)
                curr_c_vec = np.array(current_meta.get("centroid_embeddings", {}).get(m, orig_vec), dtype=np.float32)
                hypothetical_new = (curr_c_vec * num_refs + cand_vec) / float(num_refs + 1)
                hyp_norm = hypothetical_new / max(1e-8, float(np.linalg.norm(hypothetical_new)))

                # Measure L2 drift from the ORIGINAL anchor centroid
                drift_l2 = float(np.linalg.norm(hyp_norm - orig_vec))
                drifts.append(drift_l2)
                new_centroids[m] = hyp_norm.tolist()

        max_observed_drift = float(np.max(drifts)) if drifts else 0.0

        if max_observed_drift > self.max_centroid_drift:
            return PromotionDecision(
                promoted=False,
                character_id=character_id,
                similarity_to_original=round(sim_to_orig, 4),
                centroid_drift=round(max_observed_drift, 4),
                reason=f"Centroid drift ({max_observed_drift:.4f}) exceeds drift guard ceiling ({self.max_centroid_drift:.4f})"
            )

        # 5. ALL CRITERIA PASSED: PROMOTE TO BANK
        char_dir = self.bank_manager.get_bank_dir(character_id)
        promo_filename = f"promo_scene_{scene_id}_face.png"
        promo_path = char_dir / promo_filename
        cv2.imwrite(str(promo_path), face_crop)

        # Append reference
        new_ref_entry = {
            "image_path": str(promo_path),
            "source": f"promoted_scene_{scene_id}",
            "face_width_px": face_size,
            "sharpness_laplacian": round(lap_var, 1),
            "similarity_to_anchor": round(sim_to_orig, 4),
            "estimated_pose": {"yaw": 0.0, "pitch": 0.0}
        }
        refs.append(new_ref_entry)
        current_meta["references"] = refs
        current_meta["centroid_embeddings"] = new_centroids

        # Save metadata
        meta_file = char_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(current_meta, f, indent=2)

        self.bank_manager._banks[character_id] = current_meta

        logger.info(f"Promoted scene {scene_id} face for {character_id} into bank (drift: {max_observed_drift:.4f})")

        return PromotionDecision(
            promoted=True,
            character_id=character_id,
            similarity_to_original=round(sim_to_orig, 4),
            centroid_drift=round(max_observed_drift, 4),
            reason="Promoted successfully with drift guard verification"
        )
