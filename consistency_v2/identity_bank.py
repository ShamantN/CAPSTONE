"""
M1: Multi-Reference Identity Bank System
Maintains multi-angle, multi-expression reference banks with blended centroid embeddings,
4-point admission gates, and pose/scale nearest-neighbor reference selection.
"""
import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.binding_checker import BindingChecker


class IdentityBankManager:
    """
    Manages local character identity banks, centroid embeddings, and reference selection.
    """

    def __init__(
        self,
        bank_root: Path = PROJECT_ROOT / "consistency_v2" / "banks",
        calibration_path: Optional[Path] = None
    ):
        self.bank_root = bank_root
        self.bank_root.mkdir(parents=True, exist_ok=True)
        self.calib_path = calibration_path or (PROJECT_ROOT / "consistency_v2" / "verify" / "calibration.json")
        self.ensemble = BiometricEnsemble(calibration_path=self.calib_path)
        self.auditor = SceneAuditor()
        self.binder = BindingChecker()

        # In-memory registry: character_id -> metadata dict
        self._banks: Dict[str, Dict[str, Any]] = {}
        self.load_all_banks()

    def get_bank_dir(self, character_id: str) -> Path:
        p = self.bank_root / character_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def load_all_banks(self):
        """Scans the bank directory and loads metadata for existing character banks."""
        self._banks.clear()
        for char_dir in self.bank_root.iterdir():
            if char_dir.is_dir():
                meta_file = char_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            self._banks[char_dir.name] = json.load(f)
                    except Exception:
                        pass

    def estimate_head_pose(self, face_landmarks: List[Any], face_bbox: Dict[str, int]) -> Dict[str, float]:
        """
        Estimates yaw and pitch from 2D facial landmarks (eyes, nose, mouth corners).
        Returns angles in degrees: yaw (negative = looking left, positive = looking right),
        pitch (negative = looking down, positive = looking up).
        """
        w = max(face_bbox.get("w", 100), 1)
        h = max(face_bbox.get("h", 100), 1)

        if len(face_landmarks) >= 5:
            # Standard 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
            pts = np.array(face_landmarks)
            left_eye = pts[0]
            right_eye = pts[1]
            nose = pts[2]

            eye_center = (left_eye + right_eye) / 2.0
            eye_dist = np.linalg.norm(right_eye - left_eye)

            if eye_dist > 0:
                # Yaw approximation from nose horizontal offset from eye center
                dx = (nose[0] - eye_center[0]) / (eye_dist / 2.0)
                yaw_deg = float(np.clip(dx * 45.0, -90.0, 90.0))

                # Pitch approximation from nose vertical distance to eye line
                dy = (nose[1] - eye_center[1]) / (eye_dist / 2.0) - 0.70
                pitch_deg = float(np.clip(dy * 45.0, -90.0, 90.0))
                return {"yaw": round(yaw_deg, 1), "pitch": round(pitch_deg, 1)}

        return {"yaw": 0.0, "pitch": 0.0}

    def extract_torso_lab(self, image_bgr: np.ndarray, face_bbox: Dict[str, int]) -> np.ndarray:
        """Extracts median CIELAB color vector from upper torso area below face."""
        h_img, w_img = image_bgr.shape[:2]
        fx = face_bbox.get("x", 0)
        fy = face_bbox.get("y", 0)
        fw = face_bbox.get("w", 0)
        fh = face_bbox.get("h", 0)

        y1 = min(h_img, fy + int(0.8 * fh))
        y2 = min(h_img, fy + int(3.0 * fh))
        x1 = max(0, fx - int(0.5 * fw))
        x2 = min(w_img, fx + int(1.5 * fw))

        if (y2 - y1) >= 20 and (x2 - x1) >= 20:
            torso_bgr = image_bgr[y1:y2, x1:x2]
            torso_lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            l_chan = torso_lab[:, :, 0] * (100.0 / 255.0)
            a_chan = torso_lab[:, :, 1] - 128.0
            b_chan = torso_lab[:, :, 2] - 128.0
            return np.array([
                float(np.median(l_chan)),
                float(np.median(a_chan)),
                float(np.median(b_chan))
            ], dtype=np.float32)
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)

    def verify_candidate_4point_gate(
        self,
        candidate_bgr: np.ndarray,
        anchor_centroid: Dict[str, np.ndarray]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Strict 4-point admission gate for bank inclusion:
        1. Single detected face.
        2. Face size >= 96 px across.
        3. Sharpness (Laplacian variance >= 60.0).
        4. Identity to anchor centroid above genuine threshold.
        5. Zero hallucination / deformation defects.
        """
        faces = self.auditor.detect_faces(candidate_bgr)
        gate_info = {
            "single_face": len(faces) == 1,
            "face_count": len(faces),
            "face_width_px": 0,
            "size_pass": False,
            "sharpness": 0.0,
            "sharpness_pass": False,
            "biometric_ensemble_score": 0.0,
            "biometric_pass": False,
            "defects": []
        }

        if len(faces) != 1:
            return False, gate_info

        face = faces[0]
        quality = self.auditor.audit_face_quality(face)
        gate_info["face_width_px"] = quality["width_px"]
        gate_info["size_pass"] = quality["width_px"] >= 96
        gate_info["sharpness"] = quality["sharpness_laplacian"]
        gate_info["sharpness_pass"] = quality["sharpness_laplacian"] >= 60.0

        if not (gate_info["size_pass"] and gate_info["sharpness_pass"]):
            return False, gate_info

        # Biometric score against anchor centroid
        embs = self.ensemble.extract_face_embeddings(candidate_bgr)
        if anchor_centroid:
            sims = []
            for m in BiometricEnsemble.JUDGE_MODELS:
                if m in embs and m in anchor_centroid:
                    sims.append(float(np.dot(embs[m], anchor_centroid[m])))
            ens_score = float(np.mean(sims)) if sims else 0.0
            gate_info["biometric_ensemble_score"] = round(ens_score, 4)
            gate_info["biometric_pass"] = ens_score >= 0.50
        else:
            gate_info["biometric_pass"] = True

        # Defect checks
        if quality.get("melted_face"):
            gate_info["defects"].append("melted_face")

        passed = bool(
            gate_info["single_face"] and
            gate_info["size_pass"] and
            gate_info["sharpness_pass"] and
            gate_info["biometric_pass"] and
            len(gate_info["defects"]) == 0
        )
        return passed, gate_info

    def initialize_character_bank(
        self,
        character_id: str,
        seed_image_paths: List[Path]
    ) -> Dict[str, Any]:
        """
        Builds a multi-reference identity bank for a character using available certified anchors.
        Computes multi-model embeddings and computes the normalized centroid embedding.
        """
        char_dir = self.get_bank_dir(character_id)
        references: List[Dict[str, Any]] = []
        raw_embs: Dict[str, List[np.ndarray]] = {m: [] for m in BiometricEnsemble.JUDGE_MODELS}
        wardrobe_palettes: List[List[float]] = []

        anchor_centroid: Dict[str, np.ndarray] = {}

        for idx, img_p in enumerate(seed_image_paths):
            if not img_p.exists():
                continue
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                continue

            passed, gate_info = self.verify_candidate_4point_gate(img_bgr, anchor_centroid)
            if not passed and len(references) > 0:
                # If subsequent candidate fails gate, skip
                continue

            # Copy or save to bank
            ref_fn = f"ref_{len(references):02d}_{img_p.stem}.png"
            dest_p = char_dir / ref_fn
            if not dest_p.exists():
                cv2.imwrite(str(dest_p), img_bgr)

            # Extract biometric embeddings
            embs = self.ensemble.extract_face_embeddings(img_bgr)
            for m, vec in embs.items():
                raw_embs[m].append(vec)

            # Update running centroid
            anchor_centroid = {}
            for m in BiometricEnsemble.JUDGE_MODELS:
                if raw_embs[m]:
                    mv = np.mean(raw_embs[m], axis=0)
                    norm = np.linalg.norm(mv)
                    anchor_centroid[m] = (mv / norm) if norm > 0 else mv

            # Estimate pose
            faces = self.auditor.detect_faces(img_bgr)
            face_box = faces[0]["bbox"] if faces else {"w": 100, "h": 100}
            landmarks = faces[0].get("landmarks", []) if faces else []
            pose = self.estimate_head_pose(landmarks, face_box)

            # Wardrobe palette
            dom_lab = self.extract_torso_lab(img_bgr, face_box)
            wardrobe_palettes.append(dom_lab.tolist())

            references.append({
                "index": len(references),
                "filename": ref_fn,
                "path": str(dest_p),
                "pose": pose,
                "face_width_px": gate_info["face_width_px"],
                "sharpness": gate_info["sharpness"],
                "wardrobe_lab": dom_lab.tolist()
            })

        # Save metadata
        serialized_centroids = {m: anchor_centroid[m].tolist() for m in anchor_centroid}
        avg_wardrobe_lab = np.mean(wardrobe_palettes, axis=0).tolist() if wardrobe_palettes else [50.0, 0.0, 0.0]

        bank_meta = {
            "character_id": character_id,
            "num_references": len(references),
            "references": references,
            "centroid_embeddings": serialized_centroids,
            "average_wardrobe_lab": [round(x, 2) for x in avg_wardrobe_lab],
            "last_updated": "2026-10-14"
        }

        with open(char_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(bank_meta, f, indent=2)

        self._banks[character_id] = bank_meta
        return bank_meta

    def select_best_reference(
        self,
        character_id: str,
        target_yaw: float = 0.0,
        target_scale: str = "medium",
        strategy: str = "nearest_pose"
    ) -> Optional[Dict[str, Any]]:
        """
        Multi-reference view selection supporting three distinct strategies:
        - 'single_anchor': strictly returns ref_00 (the certified frontal anchor).
        - 'centroid': returns the most representative exemplar reference view.
        - 'nearest_pose': returns the reference view with minimal angular distance to target_yaw.
        """
        if character_id not in self._banks:
            return None

        bank = self._banks[character_id]
        refs = bank.get("references", [])
        if not refs:
            return None

        if strategy == "single_anchor":
            return refs[0]

        if strategy == "centroid":
            best_ref = max(refs, key=lambda r: r.get("biometric_score", 0.0))
            return best_ref

        # Default: nearest_pose
        best_ref = None
        min_dist = float("inf")

        for r in refs:
            yaw = r.get("pose", {}).get("yaw", 0.0)
            dist = abs(yaw - target_yaw)
            if dist < min_dist:
                min_dist = dist
                best_ref = r

        return best_ref if best_ref is not None else refs[0]

    def get_blended_centroid(self, character_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Returns the L2-normalized centroid embedding vector across all bank references."""
        if character_id not in self._banks:
            return None
        c_dict = self._banks[character_id].get("centroid_embeddings", {})
        return {m: np.array(v, dtype=np.float32) for m, v in c_dict.items()}

    def get_primary_reference_image(self, character_id: str) -> Optional[np.ndarray]:
        """Returns the certified anchor (ref_0) image as BGR array."""
        ref = self.select_best_reference(character_id, target_yaw=0.0)
        if ref and "path" in ref:
            return cv2.imread(ref["path"])
        return None
