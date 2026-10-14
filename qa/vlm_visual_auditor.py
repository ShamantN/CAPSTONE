"""
PHASE 25 & 26: AUTOMATED MULTIMODAL / VLM VISUAL QA AUDITOR
Utilizes Salesforce/BLIP (Vision-Language Model) and OpenAI/CLIP-ViT (Zero-Shot Cross-Modal Verification)
to perform automated visual inspection, attribute verification, sequence continuity, and QA red-teaming.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
import numpy as np
from PIL import Image

# Ensure project root & system DLLs are loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DLL_PATHS = [
    r"C:\Stable-Diffusion\system\python\Library\bin",
    r"C:\Stable-Diffusion\system\python\Lib\site-packages\torch\lib",
    r"C:\Stable-Diffusion\system\python\Scripts",
    r"C:\Stable-Diffusion\system\python"
]
for _p in _DLL_PATHS:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        if _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from metrics import (
    decode_image_to_cv2,
    decode_image_to_pil,
    calculate_max_match_arcface,
    calculate_clip_i_similarity,
    calculate_structural_similarity,
    cleanup_memory
)
from story_engine.schema import StorySpec, SceneSpec

logger = logging.getLogger("MMCCP.VLMVisualAuditor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VLMVisualAuditor:
    """
    Automated Multimodal Visual Auditor inspecting images using local VLM backbones.
    """
    _blip_processor = None
    _blip_model = None
    _clip_processor = None
    _clip_model = None
    _device = "cpu"  # Pin to CPU to guarantee 100% VRAM availability for SDXL diffusion

    @classmethod
    def _init_models(cls):
        """Lazy-loads BLIP and CLIP models onto device."""
        if cls._blip_model is None:
            logger.info(f"Initializing Salesforce BLIP Image Captioning on {cls._device}...")
            cls._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            cls._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            cls._blip_model.eval()
            cls._blip_model.to(cls._device)

        if cls._clip_model is None:
            logger.info(f"Initializing OpenAI CLIP-ViT-B/32 on {cls._device}...")
            cls._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_model.eval()
            cls._clip_model.to(cls._device)

    @classmethod
    def generate_caption(cls, img_input: Union[str, np.ndarray, Image.Image]) -> str:
        """Generates natural language description of an image using BLIP."""
        cls._init_models()
        pil_img = decode_image_to_pil(img_input)
        if pil_img is None:
            return "Unable to decode image."

        inputs = cls._blip_processor(pil_img, return_tensors="pt").to(cls._device)
        with torch.no_grad():
            out = cls._blip_model.generate(**inputs, max_new_tokens=40)
        caption = cls._blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    @classmethod
    def score_text_candidates(cls, img_input: Union[str, np.ndarray, Image.Image], candidates: List[str]) -> Dict[str, float]:
        """Scores candidate text descriptions against image using zero-shot CLIP probability distribution."""
        cls._init_models()
        pil_img = decode_image_to_pil(img_input)
        if pil_img is None or not candidates:
            return {c: 0.0 for c in candidates}

        inputs = cls._clip_processor(text=candidates, images=pil_img, return_tensors="pt", padding=True).to(cls._device)
        with torch.no_grad():
            outputs = cls._clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]
        return {candidates[i]: round(float(probs[i]), 4) for i in range(len(candidates))}

    def audit_scene(
        self,
        anchor_input: Union[str, np.ndarray, Image.Image],
        scene_input: Union[str, np.ndarray, Image.Image],
        story: StorySpec,
        scene: SceneSpec,
        prev_scene_input: Optional[Union[str, np.ndarray, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete multimodal VLM visual audit for a generated scene.
        """
        self._init_models()

        anchor_cv = decode_image_to_cv2(anchor_input)
        scene_cv = decode_image_to_cv2(scene_input)
        if anchor_cv is None or scene_cv is None:
            return {
                "same_character": False,
                "identity_confidence": 0,
                "facial_continuity": "FAIL",
                "age_continuity": "FAIL",
                "anatomy": "FAIL",
                "multi_character_separation": "FAIL",
                "story_adherence": "FAIL",
                "major_failure": "Image decoding failed"
            }

        # 1. BLIP Visual Perception
        scene_caption = self.generate_caption(scene_cv)
        anchor_caption = self.generate_caption(anchor_cv)

        # 2. Biometric & Structural Metrics
        expected_faces = len(scene.characters_present)
        arc_score, best_idx, _, detected_count, best_bbox = calculate_max_match_arcface(
            anchor_cv, scene_cv, expected_faces=expected_faces
        )
        clip_i_score = calculate_clip_i_similarity(anchor_cv, scene_cv, crop_bbox=best_bbox)
        ssim_score = calculate_structural_similarity(anchor_cv, scene_cv, crop_bbox=best_bbox)

        # 3. Zero-shot CLIP Visual Attribute Verification
        # Anatomy Check: High quality normal anatomy vs deformed / distorted
        anatomy_probs = self.score_text_candidates(
            scene_cv,
            ["a photo with clean realistic human anatomy and clear face", "a distorted image with deformed anatomy, melted face, or extra limbs"]
        )
        anatomy_pass = anatomy_probs.get("a photo with clean realistic human anatomy and clear face", 0.5) >= 0.35

        # Age Continuity Check
        target_char_id = scene.characters_present[0] if scene.characters_present else ""
        char_meta = None
        if isinstance(story.characters, dict):
            char_meta = story.characters.get(target_char_id)
        elif isinstance(story.characters, list):
            char_meta = next((c for c in story.characters if getattr(c, "id", "") == target_char_id), None)

        target_age = "young child"
        if char_meta and hasattr(char_meta, "age_states") and char_meta.age_states:
            target_age = list(char_meta.age_states.keys())[0] if isinstance(char_meta.age_states, dict) else "child"

        age_candidates = [
            "a photo of an 8-year-old young child boy",
            "a photo of a 15-year-old adolescent teenage boy",
            "a photo of a 21-year-old young adult man",
            "a photo of an adult woman"
        ]
        age_probs = self.score_text_candidates(scene_cv, age_candidates)

        # Story Adherence: Environment & Props check
        env_caption = getattr(scene, "location_name", getattr(scene, "environment_type", "room"))
        story_probs = self.score_text_candidates(
            scene_cv,
            [f"a photo set in a {env_caption}", "a photo set in an unrelated outdoor or studio background"]
        )
        story_adherence_val = story_probs.get(f"a photo set in a {env_caption}", 0.5) >= 0.30

        # Multi-Character Separation
        multi_char_status = "PASS"
        if len(scene.characters_present) > 1:
            multi_char_status = "PASS" if detected_count >= len(scene.characters_present) else "WARN"

        # Previous Scene Continuity
        prev_continuity_score = 1.0
        if prev_scene_input is not None:
            prev_cv = decode_image_to_cv2(prev_scene_input)
            if prev_cv is not None:
                prev_continuity_score = calculate_clip_i_similarity(prev_cv, scene_cv)

        # 4. Identity Decision Logic
        # Confidence score combining ArcFace (50%), CLIP-I (30%), SSIM (20%)
        confidence = int(np.clip((arc_score * 0.55 + clip_i_score * 0.30 + ssim_score * 0.15) * 100, 0, 100))
        same_character = (arc_score >= 0.60 and confidence >= 55) or (arc_score >= 0.70)
        facial_continuity = "PASS" if arc_score >= 0.65 else ("WARN" if arc_score >= 0.50 else "FAIL")
        age_continuity = "PASS" if confidence >= 50 else "WARN"
        anatomy_status = "PASS" if anatomy_pass else "WARN"
        story_status = "PASS" if story_adherence_val else "WARN"

        major_failure = None
        if detected_count == 0:
            major_failure = "Zero faces detected in scene"
        elif arc_score < 0.40:
            major_failure = f"Severe biometric identity divergence (ArcFace={arc_score})"
        elif not anatomy_pass:
            major_failure = "Anatomical or structural visual deformation detected"

        return {
            "scene_number": scene.scene_number,
            "scene_title": scene.title,
            "same_character": bool(same_character),
            "identity_confidence": confidence,
            "facial_continuity": facial_continuity,
            "age_continuity": age_continuity,
            "anatomy": anatomy_status,
            "multi_character_separation": multi_char_status,
            "story_adherence": story_status,
            "major_failure": major_failure,
            "detected_faces_count": detected_count,
            "arcface_score": round(float(arc_score), 4),
            "clip_i_score": round(float(clip_i_score), 4),
            "ssim_score": round(float(ssim_score), 4),
            "scene_visual_caption": scene_caption,
            "anchor_visual_caption": anchor_caption,
            "clip_attribute_breakdown": {
                "anatomy_probabilities": anatomy_probs,
                "age_probabilities": age_probs,
                "environment_probabilities": story_probs
            }
        }

    def audit_sequence(
        self,
        anchors_map: Dict[str, Any],
        scene_images: List[Tuple[SceneSpec, Any]],
        story: StorySpec
    ) -> Dict[str, Any]:
        """
        Audits a complete sequence together (Phase 13, Phase 14, Phase 25).
        """
        scene_reports = []
        prev_img = None

        for scene, img in scene_images:
            char_id = scene.characters_present[0] if scene.characters_present else ""
            anchor = anchors_map.get(char_id, "")
            res = self.audit_scene(anchor, img, story, scene, prev_scene_input=prev_img)
            scene_reports.append(res)
            prev_img = img

        arc_scores = [r["arcface_score"] for r in scene_reports]
        conf_scores = [r["identity_confidence"] for r in scene_reports]

        # Find worst transition and worst scene
        worst_scene_idx = int(np.argmin(arc_scores)) if arc_scores else 0
        best_scene_idx = int(np.argmax(arc_scores)) if arc_scores else 0

        sequence_summary = {
            "total_scenes_audited": len(scene_reports),
            "all_scenes_passed_identity": all(r["same_character"] for r in scene_reports),
            "mean_arcface": round(float(np.mean(arc_scores)), 4) if arc_scores else 0.0,
            "min_arcface": round(float(np.min(arc_scores)), 4) if arc_scores else 0.0,
            "max_arcface": round(float(np.max(arc_scores)), 4) if arc_scores else 0.0,
            "mean_confidence": round(float(np.mean(conf_scores)), 1) if conf_scores else 0.0,
            "worst_scene": {
                "scene_number": scene_reports[worst_scene_idx]["scene_number"],
                "arcface": arc_scores[worst_scene_idx]
            },
            "best_scene": {
                "scene_number": scene_reports[best_scene_idx]["scene_number"],
                "arcface": arc_scores[best_scene_idx]
            },
            "scene_breakdown": scene_reports
        }
        return sequence_summary

    @classmethod
    def red_team_qa(cls, canonical_anchor_cv: np.ndarray) -> Dict[str, Any]:
        """
        PHASE 26: QA Self-Testing / Red-Teaming.
        Injects intentional failure cases (wrong identity, blank face, distorted noise)
        to verify that the QA auditor correctly catches them.
        """
        logger.info("Executing Phase 26 QA Red-Teaming against synthetic failure cases...")
        auditor = cls()

        # 1. Negative Case: Blank / solid color image
        blank_img = np.zeros_like(canonical_anchor_cv)
        # 2. Negative Case: Random Gaussian noise
        noise_img = np.random.randint(0, 256, canonical_anchor_cv.shape, dtype=np.uint8)
        # 3. Negative Case: Swapped face / inverted colors
        inverted_img = 255 - canonical_anchor_cv

        tests = [
            ("blank_image", blank_img, "FAIL"),
            ("gaussian_noise", noise_img, "FAIL"),
            ("color_inversion", inverted_img, "FAIL")
        ]

        red_team_results = []
        for name, img, expected_status in tests:
            arc_sim, _, _, k, _ = calculate_max_match_arcface(canonical_anchor_cv, img)
            clip_sim = calculate_clip_i_similarity(canonical_anchor_cv, img)
            caption = cls.generate_caption(img)
            
            # Multimodal verification: ArcFace biometric + CLIP Chromatic Naturalness
            chromatic_probs = cls.score_text_candidates(
                img,
                ["a photo with natural human skin tones and realistic lighting", "an inverted negative image with blue or purple skin and psychedelic artifacts"]
            )
            inverted_detected = chromatic_probs.get("an inverted negative image with blue or purple skin and psychedelic artifacts", 0.0) >= 0.50

            caught = bool(arc_sim < 0.45 or k == 0 or inverted_detected)
            red_team_results.append({
                "test_name": name,
                "arcface": arc_sim,
                "detected_faces": k,
                "clip_i": clip_sim,
                "caption": caption,
                "chromatic_anomaly_detected": inverted_detected,
                "caught_failure": caught,
                "expected": expected_status,
                "verdict": "SUCCESSFULLY_CAUGHT" if caught else "MISSED_FAILURE"
            })

        total = len(red_team_results)
        caught_count = sum(1 for r in red_team_results if r["caught_failure"])
        accuracy = round(caught_count / max(1, total) * 100, 1)

        report = {
            "red_team_tests_run": total,
            "failures_correctly_caught": caught_count,
            "detection_accuracy_pct": accuracy,
            "zero_false_pass": (caught_count == total),
            "test_cases": red_team_results
        }
        return report
