"""
STAGE 26: STORY-DRIVEN PROGRAMMATIC QA AUDITOR
Evaluates any arbitrary story using its structured StorySpec and SceneSpec QA rules.
"""
import os, sys, json, time, math, logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from metrics import (
    decode_image_to_cv2,
    detect_all_scene_faces,
    calculate_max_match_arcface,
    cleanup_memory
)

logger = logging.getLogger("MMCCP.ProgrammaticAuditor")

_clip_model = None
_clip_processor = None

def get_clip_engine():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
            _clip_model.to("cpu")
        except Exception as e:
            logger.warning(f"CLIP loader exception: {e}")
    return _clip_model, _clip_processor

def classify_crop_clip(crop_cv: np.ndarray, candidate_labels: List[str]) -> Dict[str, float]:
    model, processor = get_clip_engine()
    if model is None or crop_cv is None or crop_cv.size == 0 or not candidate_labels:
        return {label: 1.0 / max(1, len(candidate_labels)) for label in candidate_labels}

    try:
        import torch
        rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        device = next(model.parameters()).device

        inputs = processor(text=candidate_labels, images=pil_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1).cpu().numpy()[0]

        return {label: round(float(prob), 4) for label, prob in zip(candidate_labels, probs)}
    except Exception as e:
        logger.warning(f"CLIP zero-shot classification failed: {e}")
        return {label: 1.0 / max(1, len(candidate_labels)) for label in candidate_labels}


class ProgrammaticAuditor:
    """
    Evaluates generated scenes against story-specific and generalized computer vision criteria.
    """

    def __init__(self, registry: Optional[CharacterRegistry] = None):
        self.registry = registry or CharacterRegistry()
        self.results: List[Dict[str, Any]] = []

    def record_test(self, test_id: str, scene_num: int, category: str, passed: bool, description: str, value: Any = None, threshold: Any = None):
        entry = {
            "test_id": test_id,
            "scene": scene_num,
            "category": category,
            "passed": passed,
            "description": description,
            "measured_value": value,
            "threshold": threshold
        }
        self.results.append(entry)
        status = "[PASS]" if passed else "[FAIL]"
        val_str = f" (val={value})" if value is not None else ""
        print(f"  {status:6s} | {test_id:8s} | Scene {scene_num} | {category:10s} | {description[:65]}{val_str}")

    def audit_scene(self, story: StorySpec, scene: SceneSpec, img_path: Path) -> bool:
        """Audits a single scene image according to its SceneSpec QA criteria."""
        s_num = scene.scene_number
        if not img_path.exists():
            self.record_test(f"F{s_num}.0", s_num, "File", False, f"File {img_path.name} not found")
            return False

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            self.record_test(f"F{s_num}.1", s_num, "File", False, f"Failed to decode image {img_path.name}")
            return False

        h, w = img_cv.shape[:2]
        char_ids = scene.characters_present
        expected_faces = len(char_ids)

        faces = detect_all_scene_faces(img_cv, expected_faces=max(1, expected_faces))
        face_count = len(faces)

        # 1. Spatial & Face Count Test
        if expected_faces > 0:
            count_pass = face_count >= expected_faces
            self.record_test(
                f"S{s_num}.1", s_num, "Spatial",
                count_pass,
                f"Face count expected {expected_faces}, detected {face_count}",
                face_count, expected_faces
            )

            # Check left-right spatial separation for multi-person scenes
            if expected_faces >= 2 and face_count >= 2:
                faces_sorted = sorted(faces, key=lambda f: f.get("facial_area", {}).get("x", 0))
                left_x = (faces_sorted[0]["facial_area"]["x"] + faces_sorted[0]["facial_area"]["w"] / 2) / w
                right_x = (faces_sorted[1]["facial_area"]["x"] + faces_sorted[1]["facial_area"]["w"] / 2) / w
                spatial_pass = left_x < 0.50 and right_x > 0.50
                self.record_test(
                    f"S{s_num}.2", s_num, "Spatial",
                    spatial_pass,
                    f"Multi-person spatial separation: Left x={left_x:.2f}, Right x={right_x:.2f}"
                )

        # 2. Biometric Identity Tests (ArcFace)
        for idx, c_id in enumerate(char_ids):
            age_key = scene.character_age_states.get(c_id)
            anchor_b64 = self.registry.get_anchor_b64(c_id, age_key)
            if anchor_b64:
                sim, best_idx, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv, expected_faces=expected_faces)
                char_name = story.characters[c_id].name if c_id in story.characters else c_id
                # Minimum acceptable threshold for cross-scene ArcFace
                bio_pass = sim >= 0.45
                self.record_test(
                    f"B{s_num}.{idx+1}", s_num, "Biometric",
                    bio_pass,
                    f"{char_name} ArcFace identity similarity",
                    round(sim, 4), 0.45
                )

        # 3. Prop Validation Tests (CLIP Zero-Shot)
        if scene.required_props:
            labels = scene.required_props + ["an empty room", "irrelevant objects"]
            scores = classify_crop_clip(img_cv, labels)
            for p_idx, req_prop in enumerate(scene.required_props):
                prop_score = scores.get(req_prop, 0.0)
                prop_pass = prop_score > 0.20 or any(prop_score > scores.get(other, 0.0) for other in ["an empty room", "irrelevant objects"])
                self.record_test(
                    f"P{s_num}.{p_idx+1}", s_num, "Prop",
                    prop_pass,
                    f"Required prop '{req_prop}' verification (score={prop_score:.3f})"
                )

        # 4. Forbidden Prop Rejection Tests
        if scene.forbidden_props:
            labels = scene.forbidden_props + ["a normal photo", "correct scene objects"]
            scores = classify_crop_clip(img_cv, labels)
            for fp_idx, forb_prop in enumerate(scene.forbidden_props[:2]):
                f_score = scores.get(forb_prop, 0.0)
                f_pass = f_score < 0.25
                self.record_test(
                    f"FP{s_num}.{fp_idx+1}", s_num, "ForbiddenProp",
                    f_pass,
                    f"Rejection of forbidden artifact '{forb_prop}' (score={f_score:.3f})"
                )

        return True

    def run_story_audit(self, story: StorySpec, image_dir: Path = Path("edu_outputs/images"), prefix: str = "scene_") -> Dict[str, Any]:
        """Audits all scenes in a story."""
        self.results = []
        print(f"\n{'='*90}")
        print(f"  EXECUTING PROGRAMMATIC QA AUDIT: '{story.title}'")
        print(f"{'='*90}")

        for scene in story.scenes:
            img_p = image_dir / f"{prefix}{scene.scene_number}.png"
            if not img_p.exists():
                img_p = image_dir / f"SOTA_scene_{scene.scene_number}.png"
            print(f"\n  --- Auditing Scene {scene.scene_number}: {scene.title} ({img_p.name}) ---")
            self.audit_scene(story, scene, img_p)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        pass_rate = round(passed / total * 100, 1) if total > 0 else 0.0

        summary = {
            "story_id": story.story_id,
            "title": story.title,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "results": self.results
        }

        print("\n" + "=" * 90)
        print(f"  QA AUDIT SUMMARY: {passed} / {total} Passed ({pass_rate}%)")
        print("=" * 90)
        return summary
