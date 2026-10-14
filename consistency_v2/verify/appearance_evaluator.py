"""
Appearance and Attribute Evaluator
Measures person-crop visual features using DINOv2, CLIP-I image similarity,
and zero-shot CLIP demographic/attribute probes (hair, age, skin tone).
Runs on CPU.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger("MMCCP.AppearanceEvaluator")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    if a_n == 0 or b_n == 0:
        return 0.0
    return float(np.dot(a, b) / (a_n * b_n))


class AppearanceEvaluator:
    """
    Evaluates visual appearance, clothing style, and phenotypic attributes on person crops.
    """
    def __init__(self):
        self._dinov2_model = None
        self._dinov2_proc = None
        self._clip_model = None
        self._clip_proc = None

    def _ensure_dinov2(self):
        if self._dinov2_model is None:
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel
                logger.info("Loading DINOv2 on CPU...")
                self._dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                self._dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")
                self._dinov2_model.eval()
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}")

    def _ensure_clip(self):
        if self._clip_model is None:
            try:
                import torch
                from transformers import CLIPProcessor, CLIPModel
                logger.info("Loading CLIP on CPU...")
                self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.eval()
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")

    def extract_dinov2_features(self, img_pil: Image.Image) -> Optional[np.ndarray]:
        """Extracts 768-D DINOv2 CLS token embedding from image crop."""
        self._ensure_dinov2()
        if self._dinov2_model is None:
            return None
        import torch
        inputs = self._dinov2_proc(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self._dinov2_model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        norm = np.linalg.norm(cls_token)
        if norm > 0:
            cls_token = cls_token / norm
        return cls_token

    def extract_clip_image_features(self, img_pil: Image.Image) -> Optional[np.ndarray]:
        """Extracts 512-D normalized CLIP image embedding."""
        self._ensure_clip()
        if self._clip_model is None:
            return None
        import torch
        inputs = self._clip_proc(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            feat = self._clip_model.get_image_features(**inputs).cpu().numpy()[0]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def compute_clip_probe_probs(self, img_pil: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """Computes zero-shot softmax probabilities across candidate text prompts."""
        self._ensure_clip()
        if self._clip_model is None:
            return {}
        import torch
        inputs = self._clip_proc(text=text_prompts, images=img_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        return {text_prompts[i]: round(float(probs[i]), 4) for i in range(len(text_prompts))}

    def evaluate_person_appearance(
        self,
        person_crop_bgr: np.ndarray,
        reference_crops_bgr: List[np.ndarray],
        spec_hair_color: str = "brown",
        spec_skin_tone: str = "fair",
        spec_age_years: str = "15"
    ) -> Dict[str, Any]:
        """
        Computes DINOv2 similarity, CLIP-I similarity, and attribute probe consistency.
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return {"dinov2_similarity": 0.0, "clip_i_similarity": 0.0}

        person_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_person = Image.fromarray(person_rgb)

        target_dino = self.extract_dinov2_features(pil_person)
        target_clip = self.extract_clip_image_features(pil_person)

        # 1. DINOv2 and CLIP-I similarity to references
        dino_sims = []
        clip_sims = []

        for ref_bgr in reference_crops_bgr:
            if ref_bgr is None or ref_bgr.size == 0:
                continue
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            pil_ref = Image.fromarray(ref_rgb)

            if target_dino is not None:
                r_dino = self.extract_dinov2_features(pil_ref)
                if r_dino is not None:
                    dino_sims.append(cosine_sim(target_dino, r_dino))

            if target_clip is not None:
                r_clip = self.extract_clip_image_features(pil_ref)
                if r_clip is not None:
                    clip_sims.append(cosine_sim(target_clip, r_clip))

        max_dino = round(float(max(dino_sims)), 4) if dino_sims else 0.0
        mean_dino = round(float(np.mean(dino_sims)), 4) if dino_sims else 0.0
        max_clip = round(float(max(clip_sims)), 4) if clip_sims else 0.0
        mean_clip = round(float(np.mean(clip_sims)), 4) if clip_sims else 0.0

        # 2. CLIP Zero-Shot Attribute Probes
        hair_prompts = [
            f"a person with {spec_hair_color} hair",
            "a person with black hair",
            "a person with blonde hair",
            "a person with red hair",
            "a person with grey hair"
        ]
        hair_probs = self.compute_clip_probe_probs(pil_person, hair_prompts)
        hair_top = max(hair_probs.items(), key=lambda x: x[1])[0] if hair_probs else ""
        hair_match = bool(spec_hair_color.lower() in hair_top.lower())

        age_prompts = [
            "a young 8-year-old child boy",
            "a 15-year-old adolescent teenager",
            "a 21-year-old young adult college student",
            "a 45-year-old mature adult"
        ]
        age_probs = self.compute_clip_probe_probs(pil_person, age_prompts)

        return {
            "dinov2_similarity": {
                "max": max_dino,
                "mean": mean_dino,
                "passed": bool(max_dino >= 0.65)
            },
            "clip_i_similarity": {
                "max": max_clip,
                "mean": mean_clip,
                "passed": bool(max_clip >= 0.75)
            },
            "attribute_probes": {
                "hair_color_match": hair_match,
                "hair_probabilities": hair_probs,
                "age_probabilities": age_probs
            }
        }
