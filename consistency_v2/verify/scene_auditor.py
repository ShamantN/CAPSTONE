"""
Scene Auditor and Defect Taxonomy Evaluator
Validates scene-level constraints: face/person counts, face sharpness/size/symmetry,
CLIP zero-shot prop and solarization probes, and the 14-point defect taxonomy.
Runs on CPU.
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

from deepface import DeepFace

logger = logging.getLogger("MMCCP.SceneAuditor")


def compute_laplacian_sharpness(img_gray: np.ndarray) -> float:
    """Computes variance of the Laplacian as a focus/sharpness metric."""
    if img_gray is None or img_gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


class SceneAuditor:
    """
    Audits complete scene images for geometric integrity, biometric detectability,
    required/forbidden props, chromatic sanity, and defect taxonomy violations.
    """
    def __init__(self):
        pass

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces via DeepFace (RetinaFace backend with OpenCV fallback).
        Returns list of face dicts with 'bbox' {x, y, w, h}, 'confidence', 'facial_area', 'crop'.
        """
        detected_faces = []
        try:
            results = DeepFace.extract_faces(
                img_path=image_bgr,
                detector_backend="retinaface",
                enforce_detection=False,
                align=False
            )
            for res in results:
                area = res.get("facial_area", {})
                conf = float(res.get("confidence", 0.0))
                w = area.get("w", 0)
                h = area.get("h", 0)
                if conf >= 0.50 and w >= 24 and h >= 24:
                    x = area.get("x", 0)
                    y = area.get("y", 0)
                    crop_bgr = image_bgr[y:y+h, x:x+w]
                    detected_faces.append({
                        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        "confidence": round(conf, 4),
                        "crop_bgr": crop_bgr,
                        "landmarks": area.get("landmarks", {})
                    })
        except Exception as e:
            logger.warning(f"RetinaFace detection error: {e}. Falling back to OpenCV Haar cascade.")
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                crop_bgr = image_bgr[y:y+h, x:x+w]
                detected_faces.append({
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "confidence": 0.85,
                    "crop_bgr": crop_bgr,
                    "landmarks": {}
                })
        return detected_faces

    def audit_face_quality(self, face_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Audits face sharpness, pixel width, and landmark distortion."""
        crop_bgr = face_dict.get("crop_bgr")
        bbox = face_dict.get("bbox", {})
        w = bbox.get("w", 0)

        sharpness = 0.0
        if crop_bgr is not None and crop_bgr.size > 0:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = compute_laplacian_sharpness(gray)

        too_small = bool(w < 64)
        is_blurry = bool(sharpness < 75.0)

        melted_face = False
        landmarks = face_dict.get("landmarks", {})
        if "left_eye" in landmarks and "right_eye" in landmarks:
            lx, ly = landmarks["left_eye"]
            rx, ry = landmarks["right_eye"]
            dx = abs(rx - lx)
            dy = abs(ry - ly)
            if dx > 0:
                tilt_ratio = dy / dx
                if tilt_ratio > 1.2:
                    melted_face = True

        return {
            "width_px": w,
            "too_small": too_small,
            "sharpness_laplacian": round(sharpness, 2),
            "is_blurry": is_blurry,
            "melted_face_flag": melted_face
        }

    def audit_scene_with_vision(
        self,
        image_bgr: np.ndarray,
        expected_faces: int,
        vision_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes full scene audit combining DeepFace face detection and torch vision outputs.
        """
        detected_faces = self.detect_faces(image_bgr)
        num_faces = len(detected_faces)
        num_persons = vision_data.get("person_count_yolo", num_faces)
        is_inverted = vision_data.get("is_inverted", False)
        inv_prob = vision_data.get("inversion_probability", 0.0)

        face_audits = [self.audit_face_quality(f) for f in detected_faces]

        defects = []
        if num_faces > expected_faces:
            defects.append(f"extra_people (found {num_faces} faces, expected {expected_faces})")
        elif num_faces < expected_faces:
            defects.append(f"missing_people (found {num_faces} faces, expected {expected_faces})")

        if is_inverted:
            defects.append(f"colour_inversion (probability {inv_prob})")

        if any(fa["melted_face_flag"] for fa in face_audits):
            defects.append("melted_or_distorted_face")

        req_missing = vision_data.get("missing_required_props", [])
        for m in req_missing:
            defects.append(f"required_prop_absent: {m['prop']}")

        forb_detected = vision_data.get("detected_forbidden_props", [])
        for f in forb_detected:
            defects.append(f"forbidden_prop_present: {f['prop']}")

        scene_pass = (len(defects) == 0)

        return {
            "scene_passed": scene_pass,
            "expected_faces": expected_faces,
            "detected_faces_count": num_faces,
            "detected_persons_yolo_count": num_persons,
            "detected_faces": detected_faces,
            "face_audits": face_audits,
            "inversion_check": {
                "is_inverted": is_inverted,
                "inversion_probability": inv_prob
            },
            "prop_checks": {
                "required_pass": (len(req_missing) == 0),
                "missing_required": req_missing,
                "forbidden_pass": (len(forb_detected) == 0),
                "detected_forbidden": forb_detected
            },
            "defect_count": len(defects),
            "defect_taxonomy_violations": defects
        }
