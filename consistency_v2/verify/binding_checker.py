"""
Spatial and Wardrobe Binding Checker
Verifies character-to-position binding and person-to-wardrobe binding using
spatial geometry and SegFormer semantic clothing segmentation on CPU.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger("MMCCP.BindingChecker")

# Standard CIELAB reference colors for storyboard wardrobe palettes
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
    "indigo blue": np.array([28.0, 15.0, -42.0], dtype=np.float32)
}


def delta_e_cielab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Euclidean distance in CIELAB color space (CIE76 Delta E)."""
    return float(np.linalg.norm(lab1 - lab2))


class BindingChecker:
    """
    Checks spatial position and wardrobe binding for detected characters in scenes.
    """
    def __init__(self):
        self._segformer_model = None
        self._segformer_proc = None

    def _ensure_segformer(self):
        if self._segformer_model is None:
            try:
                import torch
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                logger.info("Loading SegFormer model on CPU...")
                self._segformer_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
                self._segformer_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
                self._segformer_model.eval()
            except Exception as e:
                logger.error(f"Failed to load SegFormer: {e}")

    def verify_spatial_position(
        self,
        face_bbox: Dict[str, int],
        image_width: int,
        expected_position_str: str
    ) -> Tuple[bool, str, float]:
        """
        Verifies if face center falls into expected horizontal zone.
        expected_position_str: e.g. "sitting on the left", "center", "sitting on the right"
        Returns (passed, observed_zone, center_ratio)
        """
        x = face_bbox.get("x", 0)
        w = face_bbox.get("w", 0)
        x_center = x + (w / 2.0)
        r_x = x_center / max(1, image_width)

        if r_x < 0.42:
            observed_zone = "left"
        elif r_x > 0.58:
            observed_zone = "right"
        else:
            observed_zone = "center"

        pos_lower = expected_position_str.lower()
        if "left" in pos_lower:
            expected_zone = "left"
            # Allow left or left-leaning center
            passed = (r_x <= 0.48)
        elif "right" in pos_lower:
            expected_zone = "right"
            # Allow right or right-leaning center
            passed = (r_x >= 0.52)
        elif "center" in pos_lower or "middle" in pos_lower:
            expected_zone = "center"
            # Allow middle band
            passed = (0.28 <= r_x <= 0.72)
        else:
            # Undefined position: pass by default
            expected_zone = "any"
            passed = True

        return passed, observed_zone, round(r_x, 3)

    def extract_clothing_color_lab(
        self,
        image_bgr: np.ndarray,
        face_bbox: Dict[str, int]
    ) -> Optional[np.ndarray]:
        """
        Uses SegFormer to isolate clothing directly below the face bbox and extracts median CIELAB color.
        """
        self._ensure_segformer()
        if self._segformer_model is None or image_bgr is None:
            return None

        h_img, w_img = image_bgr.shape[:2]
        fx, fy, fw, fh = face_bbox.get("x", 0), face_bbox.get("y", 0), face_bbox.get("w", 0), face_bbox.get("h", 0)

        # Region of interest: upper torso below face
        y1 = min(h_img, fy + int(0.7 * fh))
        y2 = min(h_img, fy + int(3.5 * fh))
        x1 = max(0, fx - int(0.6 * fw))
        x2 = min(w_img, fx + int(1.6 * fw))

        if (y2 - y1) < 20 or (x2 - x1) < 20:
            return None

        torso_bgr = image_bgr[y1:y2, x1:x2]
        torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)

        import torch
        from PIL import Image
        pil_torso = Image.fromarray(torso_rgb)

        clothing_pixels = None
        seg_status = "seg_failed"
        try:
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

            # SegFormer classes: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 17: Scarf
            cloth_candidates = (
                (pred_mask == 4) | (pred_mask == 5) | (pred_mask == 6) |
                (pred_mask == 7) | (pred_mask == 17)
            )
            seg_pixels = int(np.sum(cloth_candidates))
            if seg_pixels >= 50:
                clothing_pixels = cloth_candidates
                seg_status = "ok"
            else:
                seg_status = "seg_failed"
        except Exception as e:
            logger.warning(f"SegFormer clothing segmentation failed: {e}")
            seg_status = "seg_failed"

        # STRICT: Zero central-torso fallback! On failure, return (None, 'seg_failed')
        if clothing_pixels is None or seg_status == "seg_failed":
            return None, "seg_failed"

        torso_lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Standardize OpenCV LAB to standard CIELAB (L*: 0-100, a*: -128..127, b*: -128..127)
        l_std = torso_lab[:, :, 0] * (100.0 / 255.0)
        a_std = torso_lab[:, :, 1] - 128.0
        b_std = torso_lab[:, :, 2] - 128.0

        selected_l = l_std[clothing_pixels]
        selected_a = a_std[clothing_pixels]
        selected_b = b_std[clothing_pixels]

        if len(selected_l) == 0:
            return None, "seg_failed"

        median_lab = np.array([
            float(np.median(selected_l)),
            float(np.median(selected_a)),
            float(np.median(selected_b))
        ], dtype=np.float32)

        return median_lab, "ok"

    def verify_wardrobe_color(
        self,
        observed_lab: Optional[np.ndarray],
        expected_color_names: List[str],
        max_delta_e: float = 38.0
    ) -> Tuple[bool, float, str]:
        """
        Compares observed clothing LAB color against expected color names.
        Returns (passed, min_delta_e, closest_color_name)
        """
        if observed_lab is None or not expected_color_names:
            return True, 0.0, "unknown"

        min_de = float("inf")
        best_name = "none"

        for c_name in expected_color_names:
            c_clean = c_name.lower().strip()
            # Find nearest reference LAB
            ref_lab = REFERENCE_LAB_COLORS.get(c_clean)
            if ref_lab is None:
                # Partial match search
                for k, v in REFERENCE_LAB_COLORS.items():
                    if k in c_clean or c_clean in k:
                        ref_lab = v
                        break
            if ref_lab is not None:
                de = delta_e_cielab(observed_lab, ref_lab)
                if de < min_de:
                    min_de = de
                    best_name = c_clean

        passed = (min_de <= max_delta_e)
        return passed, round(min_de, 2), best_name

    def verify_binding(
        self,
        image_bgr: np.ndarray,
        face_bbox: Dict[str, int],
        character_id: str,
        expected_position_str: str,
        expected_clothing_colors: List[str]
    ) -> Dict[str, Any]:
        """
        Executes complete spatial and wardrobe binding check for one face.
        """
        h_img, w_img = image_bgr.shape[:2]
        spatial_pass, obs_zone, r_x = self.verify_spatial_position(
            face_bbox=face_bbox,
            image_width=w_img,
            expected_position_str=expected_position_str
        )

        obs_lab, seg_status = self.extract_clothing_color_lab(image_bgr, face_bbox)
        if seg_status == "seg_failed" or obs_lab is None:
            wardrobe_pass = False
            min_de = 999.0
            matched_color = "seg_failed"
        else:
            wardrobe_pass, min_de, matched_color = self.verify_wardrobe_color(
                observed_lab=obs_lab,
                expected_color_names=expected_clothing_colors
            )

        overall_binding_pass = bool(spatial_pass and wardrobe_pass)
        reasons = []
        if not spatial_pass:
            reasons.append(f"Spatial mismatch: expected '{expected_position_str}', observed in {obs_zone} (x={r_x})")
        if seg_status == "seg_failed":
            reasons.append("Wardrobe segmentation failed: fewer than 50 clothing pixels detected (zero-fallback protocol)")
        elif not wardrobe_pass:
            reasons.append(f"Wardrobe color mismatch: Delta E {min_de} exceeds tolerance for {expected_clothing_colors}")

        return {
            "character_id": character_id,
            "overall_binding_pass": overall_binding_pass,
            "overall_binding_passed": overall_binding_pass,
            "spatial_passed": spatial_pass,
            "wardrobe_passed": wardrobe_pass,
            "clothing_delta_e": min_de,
            "seg_status": seg_status,
            "spatial_check": {
                "passed": spatial_pass,
                "expected": expected_position_str,
                "observed_zone": obs_zone,
                "center_x_ratio": r_x
            },
            "wardrobe_check": {
                "passed": wardrobe_pass,
                "expected_colors": expected_clothing_colors,
                "matched_reference": matched_color,
                "delta_e": min_de,
                "seg_status": seg_status,
                "observed_lab": obs_lab.tolist() if obs_lab is not None else None
            },
            "failure_reasons": reasons
        }
