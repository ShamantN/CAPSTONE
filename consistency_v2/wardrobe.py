"""
Wardrobe and Palette Lock (M7)
Uses SegFormer semantic segmentation to extract clothing, hair, and skin masks,
evaluates color consistency against character specifications and identity banks in CIELAB space,
and applies bounded palette repairs to eliminate wardrobe color drift while strictly protecting facial identity.
"""
import os
import sys
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("MMCCP.WardrobeLock")

FORGE_PYTHON = Path(r"C:\Stable-Diffusion\system\python\python.exe")

# Standard CIELAB reference colors matching the verification suite
REFERENCE_LAB_COLORS = {
    "yellow": np.array([85.0, 5.0, 80.0], dtype=np.float32),
    "mustard yellow": np.array([75.0, 10.0, 70.0], dtype=np.float32),
    "navy blue": np.array([22.0, 5.0, -32.0], dtype=np.float32),
    "pale blue": np.array([78.0, -8.0, -20.0], dtype=np.float32),
    "blue": np.array([45.0, 10.0, -50.0], dtype=np.float32),
    "dark green": np.array([35.0, -25.0, 15.0], dtype=np.float32),
    "green": np.array([45.0, -35.0, 25.0], dtype=np.float32),
    "crimson red": np.array([38.0, 55.0, 28.0], dtype=np.float32),
    "red": np.array([50.0, 65.0, 45.0], dtype=np.float32),
    "black": np.array([12.0, 0.0, 0.0], dtype=np.float32),
    "charcoal": np.array([25.0, 0.0, 0.0], dtype=np.float32),
    "brown": np.array([40.0, 18.0, 25.0], dtype=np.float32),
    "emerald green": np.array([48.0, -42.0, 18.0], dtype=np.float32),
    "indigo blue": np.array([28.0, 15.0, -42.0], dtype=np.float32),
    "white": np.array([95.0, 0.0, 0.0], dtype=np.float32),
    "grey": np.array([50.0, 0.0, 0.0], dtype=np.float32),
}


def delta_e_cielab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Euclidean distance in CIELAB color space (CIE76 Delta E)."""
    return float(np.linalg.norm(lab1 - lab2))


def bgr_to_standard_cielab(bgr: np.ndarray) -> np.ndarray:
    """Converts OpenCV BGR image to standard CIELAB (L* 0..100, a* -128..127, b* -128..127)."""
    lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_std = lab_cv[:, :, 0] * (100.0 / 255.0)
    a_std = lab_cv[:, :, 1] - 128.0
    b_std = lab_cv[:, :, 2] - 128.0
    return np.stack([l_std, a_std, b_std], axis=-1)


def standard_cielab_to_bgr(lab_std: np.ndarray) -> np.ndarray:
    """Converts standard CIELAB image back to OpenCV BGR uint8."""
    l_cv = np.clip(lab_std[:, :, 0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
    a_cv = np.clip(lab_std[:, :, 1] + 128.0, 0, 255).astype(np.uint8)
    b_cv = np.clip(lab_std[:, :, 2] + 128.0, 0, 255).astype(np.uint8)
    lab_cv = np.stack([l_cv, a_cv, b_cv], axis=-1)
    return cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)


class WardrobeLock:
    """
    Measures and locks character wardrobe palette across scenes.
    """
    def __init__(self, delta_e_threshold: float = 38.0):
        self.delta_e_threshold = delta_e_threshold
        self._segformer_model = None
        self._segformer_proc = None
        self._checked_env = False

    def _ensure_segformer(self):
        if self._checked_env:
            return
        self._checked_env = True
        try:
            import numpy as _np
            # Forge python uses numpy 1.x where torch C-extensions are binary compatible
            if _np.__version__.startswith("1."):
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                logger.info("Initializing SegFormer semantic clothes segmentation on CPU...")
                self._segformer_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
                self._segformer_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
                self._segformer_model.eval()
            else:
                self._segformer_proc = None
                self._segformer_model = None
        except Exception:
            self._segformer_proc = None
            self._segformer_model = None

    def segment_person_regions(
        self,
        image_bgr: np.ndarray,
        face_bbox: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Segments person upper body below face into clothing, skin, and hair masks on full image canvas.
        Returns dict of binary boolean masks: {"clothing": mask, "face_exclusion": mask}
        """
        self._ensure_segformer()
        h_img, w_img = image_bgr.shape[:2]
        fx, fy, fw, fh = face_bbox.get("x", 0), face_bbox.get("y", 0), face_bbox.get("w", 0), face_bbox.get("h", 0)

        # Full image boolean masks
        clothing_full = np.zeros((h_img, w_img), dtype=bool)
        face_exclusion = np.zeros((h_img, w_img), dtype=bool)

        # Define face exclusion zone with 15% safety padding
        pad_x = int(0.15 * fw)
        pad_y = int(0.15 * fh)
        fe_x1 = max(0, fx - pad_x)
        fe_y1 = max(0, fy - pad_y)
        fe_x2 = min(w_img, fx + fw + pad_x)
        fe_y2 = min(h_img, fy + fh + pad_y)
        face_exclusion[fe_y1:fe_y2, fe_x1:fe_x2] = True

        # Region of interest: torso below face
        y1 = min(h_img, fy + int(0.7 * fh))
        y2 = min(h_img, fy + int(3.5 * fh))
        x1 = max(0, fx - int(0.6 * fw))
        x2 = min(w_img, fx + int(1.6 * fw))

        if (y2 - y1) < 20 or (x2 - x1) < 20:
            return {"clothing": clothing_full, "face_exclusion": face_exclusion}

        torso_bgr = image_bgr[y1:y2, x1:x2]
        torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)

        if self._segformer_model is not None:
            import torch
            from PIL import Image
            pil_torso = Image.fromarray(torso_rgb)
            inputs = self._segformer_proc(images=pil_torso, return_tensors="pt")
            with torch.no_grad():
                outputs = self._segformer_model(**inputs)
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(torso_rgb.shape[0], torso_rgb.shape[1]),
                mode="bilinear",
                align_corners=False
            )
            pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy()

            # Class 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Coat/Suit
            torso_clothing = (
                (pred_mask == 4) | (pred_mask == 5) | (pred_mask == 6) |
                (pred_mask == 7) | (pred_mask == 8)
            )
        else:
            # Fallback: central torso patch
            th, tw = torso_rgb.shape[:2]
            torso_clothing = np.zeros((th, tw), dtype=bool)
            torso_clothing[int(th * 0.15):int(th * 0.85), int(tw * 0.15):int(tw * 0.85)] = True

        if np.sum(torso_clothing) < 50:
            th, tw = torso_rgb.shape[:2]
            torso_clothing = np.zeros((th, tw), dtype=bool)
            torso_clothing[int(th * 0.15):int(th * 0.85), int(tw * 0.15):int(tw * 0.85)] = True

        # Map torso clothing back to full image
        clothing_full[y1:y2, x1:x2] = torso_clothing
        # Strictly exclude face exclusion zone
        clothing_full[face_exclusion] = False

        return {
            "clothing": clothing_full,
            "face_exclusion": face_exclusion
        }

    def measure_clothing_lab(
        self,
        image_bgr: np.ndarray,
        clothing_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculates median CIELAB color of the clothing region."""
        if not np.any(clothing_mask):
            return None
        lab_std = bgr_to_standard_cielab(image_bgr)
        sel_l = lab_std[:, :, 0][clothing_mask]
        sel_a = lab_std[:, :, 1][clothing_mask]
        sel_b = lab_std[:, :, 2][clothing_mask]

        if len(sel_l) == 0:
            return None

        return np.array([
            float(np.median(sel_l)),
            float(np.median(sel_a)),
            float(np.median(sel_b))
        ], dtype=np.float32)

    def get_target_lab(
        self,
        expected_color_names: List[str],
        bank_wardrobe_lab: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Resolves target CIELAB color from bank reference or expected color names."""
        if bank_wardrobe_lab is not None:
            return np.array(bank_wardrobe_lab, dtype=np.float32)

        for c_name in expected_color_names:
            clean = c_name.lower().strip()
            if clean in REFERENCE_LAB_COLORS:
                return REFERENCE_LAB_COLORS[clean]
            for k, v in REFERENCE_LAB_COLORS.items():
                if k in clean or clean in k:
                    return v

        return None

    def apply_bounded_cielab_repair(
        self,
        image_bgr: np.ndarray,
        clothing_mask: np.ndarray,
        target_lab: np.ndarray,
        blend_strength: float = 0.65
    ) -> np.ndarray:
        """
        Applies smooth, bounded color transfer in CIELAB space on clothing pixels,
        strictly leaving unmasked and face pixels bitwise identical.
        blend_strength: 0.0 (no change) to 1.0 (full target palette).
        """
        if not np.any(clothing_mask) or target_lab is None:
            return image_bgr.copy()

        # Gaussian blur the clothing mask edges to avoid harsh boundaries
        mask_uint8 = (clothing_mask.astype(np.uint8) * 255)
        blurred_mask = cv2.GaussianBlur(mask_uint8, (9, 9), 3).astype(np.float32) / 255.0

        lab_std = bgr_to_standard_cielab(image_bgr)
        target_l, target_a, target_b = target_lab[0], target_lab[1], target_lab[2]

        curr_a = lab_std[:, :, 1]
        curr_b = lab_std[:, :, 2]
        curr_l = lab_std[:, :, 0]

        obs_l_median = np.median(curr_l[clothing_mask]) if np.any(clothing_mask) else target_l
        l_offset = (target_l - obs_l_median) * 0.4

        new_l = np.clip(curr_l + l_offset, 5.0, 95.0)
        new_a = curr_a * (1.0 - blend_strength) + target_a * blend_strength
        new_b = curr_b * (1.0 - blend_strength) + target_b * blend_strength

        alpha = blurred_mask[:, :, np.newaxis]
        target_img_lab = np.stack([new_l, new_a, new_b], axis=-1)
        final_lab = lab_std * (1.0 - alpha) + target_img_lab * alpha

        repaired_bgr = standard_cielab_to_bgr(final_lab)

        outside = (blurred_mask == 0.0)
        repaired_bgr[outside] = image_bgr[outside]

        return repaired_bgr

    def lock_scene_wardrobe(
        self,
        image_bgr: np.ndarray,
        faces_info: List[Dict[str, Any]],
        character_wardrobes: Dict[str, Dict[str, Any]],
        blend_strength: float = 0.65
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes all characters in scene:
        - Segments clothing per character
        - Measures CIELAB color and Delta E
        - If Delta E > threshold, repairs clothing palette
        - Verifies face regions remain byte-identical
        Returns (repaired_image_bgr, repair_report)
        """
        output_bgr = image_bgr.copy()
        report = {
            "characters_checked": len(faces_info),
            "repairs_applied": 0,
            "details": []
        }

        for face_info in faces_info:
            char_id = face_info.get("character_id", "unknown")
            bbox = face_info.get("bbox", {})
            wardrobe_info = character_wardrobes.get(char_id, {})
            expected_colors = wardrobe_info.get("clothing_colors", [])
            bank_lab = wardrobe_info.get("bank_wardrobe_lab")

            target_lab = self.get_target_lab(expected_colors, bank_lab)
            if target_lab is None:
                continue

            masks = self.segment_person_regions(output_bgr, bbox)
            clothing_mask = masks["clothing"]

            obs_lab = self.measure_clothing_lab(output_bgr, clothing_mask)
            if obs_lab is None:
                continue

            initial_de = delta_e_cielab(obs_lab, target_lab)
            char_report = {
                "character_id": char_id,
                "initial_lab": [round(float(x), 2) for x in obs_lab],
                "target_lab": [round(float(x), 2) for x in target_lab],
                "initial_delta_e": round(initial_de, 2),
                "repaired": False,
                "final_delta_e": round(initial_de, 2)
            }

            if initial_de > self.delta_e_threshold:
                fx, fy, fw, fh = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
                face_crop_before = output_bgr[fy:fy+fh, fx:fx+fw].copy()

                output_bgr = self.apply_bounded_cielab_repair(
                    image_bgr=output_bgr,
                    clothing_mask=clothing_mask,
                    target_lab=target_lab,
                    blend_strength=blend_strength
                )

                face_crop_after = output_bgr[fy:fy+fh, fx:fx+fw]
                pixel_diff = np.max(np.abs(face_crop_before.astype(int) - face_crop_after.astype(int)))
                if pixel_diff > 0:
                    output_bgr[fy:fy+fh, fx:fx+fw] = face_crop_before
                    char_report["face_preserved_strictly"] = True
                else:
                    char_report["face_preserved_strictly"] = True

                new_obs_lab = self.measure_clothing_lab(output_bgr, clothing_mask)
                if new_obs_lab is not None:
                    final_de = delta_e_cielab(new_obs_lab, target_lab)
                    char_report["final_delta_e"] = round(final_de, 2)
                    char_report["final_lab"] = [round(float(x), 2) for x in new_obs_lab]

                char_report["repaired"] = True
                report["repairs_applied"] += 1

            report["details"].append(char_report)

        return output_bgr, report
