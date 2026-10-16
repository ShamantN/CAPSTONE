"""
MMCCP Consistency v2: Method M12 (Collage-Init Composition)
Solves multi-character resolution starvation and single-character collapse in wide/group shots.
Composes pose-aligned character face/torso reference crops onto the draft canvas at planned
layout coordinates prior to Pass 2 refinement, giving SDXL an explicit spatial and biometric anchor.
"""
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("MMCCP.CollageInit")


@dataclass
class CollageInitConfig:
    enabled: bool = False
    feather_radius: int = 15
    alpha_blend: float = 0.85
    min_face_scale_px: int = 96
    margin_expansion: float = 0.20  # Include 20% hair and chin margin


class CollageInitEngine:
    """
    Engine for compositing character reference crops onto the scene canvas at planned layout positions.
    """

    @staticmethod
    def create_feathered_ellipse_mask(width: int, height: int, feather_radius: int = 15) -> np.ndarray:
        """
        Creates an elliptical alpha mask with Gaussian-feathered boundaries to eliminate hard edges.
        """
        mask = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, height // 2)
        axes = (max(width // 2 - feather_radius, 5), max(height // 2 - feather_radius, 5))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_radius / 2.0)

        return np.clip(mask, 0.0, 1.0)

    @classmethod
    def compose_collage(
        cls,
        canvas_bgr: np.ndarray,
        layout_boxes: List[Any],
        bank_manager: Any,
        config: Optional[CollageInitConfig] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Pastes character reference face crops onto canvas_bgr at layout box coordinates.

        Args:
            canvas_bgr: Base canvas image from Pass 1 draft.
            layout_boxes: List of CharacterLayoutBox objects with target_zone and face_bbox.
            bank_manager: IdentityBankManager instance.
            config: CollageInitConfig instance.

        Returns:
            Tuple of (collaged_bgr, report_dict)
        """
        if config is None:
            config = CollageInitConfig()

        if not config.enabled or not layout_boxes:
            return canvas_bgr, {"collage_applied": False, "num_characters_collaged": 0}

        canvas_h, canvas_w = canvas_bgr.shape[:2]
        result = canvas_bgr.copy()
        report = {
            "collage_applied": True,
            "characters": [],
            "num_characters_collaged": 0
        }

        for box in layout_boxes:
            cid = getattr(box, "character_id", None)
            if not cid:
                continue

            ref_bgr = bank_manager.get_primary_reference_image(cid)
            if ref_bgr is None:
                logger.warning(f"No reference image available for collage-init of {cid}")
                continue

            # Detect face in reference image or use top-center crop
            ref_faces = bank_manager.auditor.detect_faces(ref_bgr) if hasattr(bank_manager, "auditor") else []
            if ref_faces:
                rf_bbox = ref_faces[0]["bbox"]
                rx, ry, rw, rh = rf_bbox["x"], rf_bbox["y"], rf_bbox["w"], rf_bbox["h"]
                # Expand margin for hair and chin
                exp_x = int(rw * config.margin_expansion)
                exp_y = int(rh * config.margin_expansion)
                ry1 = max(0, ry - exp_y)
                ry2 = min(ref_bgr.shape[0], ry + rh + exp_y)
                rx1 = max(0, rx - exp_x)
                rx2 = min(ref_bgr.shape[1], rx + rw + exp_x)
                face_crop = ref_bgr[ry1:ry2, rx1:rx2]
            else:
                # Default centered upper-body crop
                ref_h, ref_w = ref_bgr.shape[:2]
                face_crop = ref_bgr[int(ref_h * 0.1):int(ref_h * 0.6), int(ref_w * 0.25):int(ref_w * 0.75)]

            if face_crop.size == 0:
                continue

            # Target position from layout box
            fb = getattr(box, "face_bbox", None)
            if not fb:
                # Estimate from body bbox
                tx1, ty1, tx2, ty2 = getattr(box, "target_box_px", (0, 0, canvas_w, canvas_h))
                tw = tx2 - tx1
                th = ty2 - ty1
                target_w = max(int(tw * 0.40), config.min_face_scale_px)
                target_h = max(int(th * 0.30), config.min_face_scale_px)
                target_x = tx1 + (tw - target_w) // 2
                target_y = ty1 + int(th * 0.05)
            else:
                target_x = fb.get("x", 0)
                target_y = fb.get("y", 0)
                target_w = max(fb.get("w", config.min_face_scale_px), config.min_face_scale_px)
                target_h = max(fb.get("h", config.min_face_scale_px), config.min_face_scale_px)

            # Clamp target within canvas boundaries
            target_x = max(0, min(canvas_w - target_w, target_x))
            target_y = max(0, min(canvas_h - target_h, target_y))

            # Resize face crop to target dimension
            resized_crop = cv2.resize(face_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # Generate feathered alpha mask
            mask = cls.create_feathered_ellipse_mask(target_w, target_h, config.feather_radius)
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * config.alpha_blend

            # Blend onto canvas
            canvas_roi = result[target_y:target_y + target_h, target_x:target_x + target_w].astype(np.float32)
            crop_float = resized_crop.astype(np.float32)
            blended = (crop_float * mask_3d + canvas_roi * (1.0 - mask_3d)).astype(np.uint8)
            result[target_y:target_y + target_h, target_x:target_x + target_w] = blended

            report["characters"].append({
                "character_id": cid,
                "target_bbox": {"x": target_x, "y": target_y, "w": target_w, "h": target_h},
                "source_crop_shape": face_crop.shape[:2]
            })

        report["num_characters_collaged"] = len(report["characters"])
        return result, report
