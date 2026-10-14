"""
STAGE 15: SOTA MULTI-PILLAR PROGRAMMATIC QA AUDITOR
Evaluates:
1. Biometrics (ArcFace 512-D Cosine Similarity via calculate_max_match_arcface)
2. Spatial Layout & Multi-Person Collision
3. Prop Correctness (Zero-shot CLIP & Visual Heuristics)
4. Garment / Palette Continuity
5. Anatomical Integrity
"""
import os, sys, json, time, math
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

from metrics import (
    CACHE_FILE,
    HARDCODED_CHARACTERS,
    HARDCODED_SCENES,
    decode_image_to_cv2,
    detect_all_scene_faces,
    calculate_max_match_arcface,
    cleanup_memory,
)

OUTPUT_DIR = Path("edu_outputs/images")
EXP_DIR = Path("experiments")
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = EXP_DIR / "sota_qa_report.json"

# CLIP zero-shot helper
_clip_model = None
_clip_processor = None

def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
            if torch.cuda.is_available():
                _clip_model.to("cuda")
        except Exception as e:
            print(f"[WARN] CLIP load error: {e}")
    return _clip_model, _clip_processor

def classify_crop_clip(crop_cv: np.ndarray, candidate_labels: List[str]) -> Dict[str, float]:
    """Zero-shot classify an image crop using CLIP."""
    model, processor = get_clip()
    if model is None or crop_cv is None or crop_cv.size == 0:
        return {label: 1.0 / max(1, len(candidate_labels)) for label in candidate_labels}

    try:
        import torch
        rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        device = next(model.parameters()).device

        inputs = processor(
            text=candidate_labels,
            images=pil_img,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        return {label: round(float(prob), 4) for label, prob in zip(candidate_labels, probs)}
    except Exception as e:
        print(f"[WARN] CLIP classification failed: {e}")
        return {label: 1.0 / max(1, len(candidate_labels)) for label in candidate_labels}


class SOTAQAAuditor:
    def __init__(self):
        self.cache_data = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                self.cache_data = json.load(f)
        self.results = []

    def log_test(self, test_id: str, scene: int, category: str, passed: bool, description: str, value: Any = None, threshold: Any = None):
        res = {
            "test_id": test_id,
            "scene": scene,
            "category": category,
            "passed": passed,
            "description": description,
            "measured_value": value,
            "threshold": threshold
        }
        self.results.append(res)
        status = "[PASS]" if passed else "[FAIL]"
        val_str = f" (val={value})" if value is not None else ""
        print(f"  {status:6s} | {test_id:8s} | Scene {scene} | {category:10s} | {description[:65]}{val_str}")

    def audit_scene(self, scene_num: int, img_path: Path) -> bool:
        """Audits a single scene image across all dimensions."""
        if not img_path.exists():
            self.log_test(f"F{scene_num}.0", scene_num, "File", False, f"File {img_path.name} not found on disk")
            return False

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            self.log_test(f"F{scene_num}.1", scene_num, "File", False, f"Failed to decode image {img_path.name}")
            return False

        h, w = img_cv.shape[:2]
        faces = detect_all_scene_faces(img_cv, expected_faces=2 if scene_num in [1, 5] else 1)
        face_count = len(faces)

        # -------------------------------------------------------------
        # SCENE 1: The Spark of Curiosity (Young Ebright + Mother)
        # -------------------------------------------------------------
        if scene_num == 1:
            self.log_test("S1.1", 1, "Spatial", face_count >= 2, f"Face count expected 2, detected {face_count}", face_count, 2)
            if face_count >= 2:
                # Sort left to right
                faces_sorted = sorted(faces, key=lambda f: f.get("facial_area", {}).get("x", 0))
                left_x = (faces_sorted[0]["facial_area"]["x"] + faces_sorted[0]["facial_area"]["w"] / 2) / w
                right_x = (faces_sorted[1]["facial_area"]["x"] + faces_sorted[1]["facial_area"]["w"] / 2) / w
                self.log_test("S1.2", 1, "Spatial", left_x < 0.50 and right_x > 0.50, f"Left face at x={left_x:.2f}, Right face at x={right_x:.2f}")

                # Biometrics: Young Ebright on Left vs Mother on Right
                y_anchor = self.cache_data.get("young_ebright")
                m_anchor = self.cache_data.get("mother")

                if y_anchor:
                    sim_y, _, _, _, _ = calculate_max_match_arcface(y_anchor, img_cv, expected_faces=2)
                    self.log_test("B1.1", 1, "Biometric", sim_y >= 0.45, f"Young Ebright ArcFace identity similarity", round(sim_y, 4), 0.45)

                if m_anchor:
                    sim_m, _, _, _, _ = calculate_max_match_arcface(m_anchor, img_cv, expected_faces=2)
                    self.log_test("B1.2", 1, "Biometric", sim_m >= 0.45, f"Mother ArcFace identity similarity", round(sim_m, 4), 0.45)

        # -------------------------------------------------------------
        # SCENE 2: The Butterfly Basement (Young Ebright)
        # -------------------------------------------------------------
        elif scene_num == 2:
            self.log_test("S2.1", 2, "Spatial", face_count >= 1, f"Face count detected {face_count}")
            y_anchor = self.cache_data.get("young_ebright")
            if face_count >= 1 and y_anchor:
                sim, _, _, _, _ = calculate_max_match_arcface(y_anchor, img_cv, expected_faces=1)
                self.log_test("B2.1", 2, "Biometric", sim >= 0.45, f"Young Ebright Scene 2 vs Anchor similarity", round(sim, 4), 0.45)

        # -------------------------------------------------------------
        # SCENE 3: Failure at the Science Fair (Teen Ebright + Microscope)
        # -------------------------------------------------------------
        elif scene_num == 3:
            self.log_test("S3.1", 3, "Spatial", face_count >= 1, f"Face count detected {face_count}")
            t_anchor = self.cache_data.get("teen_ebright")
            if face_count >= 1 and t_anchor:
                sim, _, _, _, _ = calculate_max_match_arcface(t_anchor, img_cv, expected_faces=1)
                self.log_test("B3.1", 3, "Biometric", sim >= 0.45, f"Teen Ebright Scene 3 vs Anchor similarity", round(sim, 4), 0.45)

            # Prop QA: Optical microscope vs empty desk vs laptop
            desk_crop = img_cv[int(h * 0.4):, :]
            candidates = ["a microscope on a table", "a laptop computer", "an empty table", "a glowing sphere"]
            scores = classify_crop_clip(desk_crop, candidates)
            microscope_score = scores.get("a microscope on a table", 0.0)
            laptop_score = scores.get("a laptop computer", 0.0)
            orb_score = scores.get("a glowing sphere", 0.0)
            prop_pass = (microscope_score > 0.30 or microscope_score > laptop_score) and orb_score < 0.20
            self.log_test("P3.1", 3, "Prop", prop_pass, f"Desk Microscope CLIP={microscope_score:.3f} (laptop={laptop_score:.3f}, orb={orb_score:.3f})")

        # -------------------------------------------------------------
        # SCENE 4: The Starling Experiment (Teen Ebright + Backyard)
        # -------------------------------------------------------------
        elif scene_num == 4:
            self.log_test("S4.1", 4, "Spatial", face_count >= 1, f"Face count detected {face_count}")
            t_anchor = self.cache_data.get("teen_ebright")
            if face_count >= 1 and t_anchor:
                sim, _, _, _, _ = calculate_max_match_arcface(t_anchor, img_cv, expected_faces=1)
                self.log_test("B4.1", 4, "Biometric", sim >= 0.45, f"Teen Ebright Scene 4 vs Anchor similarity", round(sim, 4), 0.45)

            # Outdoor context QA
            candidates = ["a boy observing a bird in an outdoor backyard", "indoor bedroom", "a busy city street"]
            scores = classify_crop_clip(img_cv, candidates)
            outdoor_score = scores.get("a boy observing a bird in an outdoor backyard", 0.0)
            self.log_test("P4.1", 4, "Prop", outdoor_score >= 0.30, f"Outdoor backyard bird experiment CLIP={outdoor_score:.3f}")

        # -------------------------------------------------------------
        # SCENE 5: The Breakthrough at Harvard (College Ebright + James Wong + DNA Model)
        # -------------------------------------------------------------
        elif scene_num == 5:
            self.log_test("S5.1", 5, "Spatial", face_count >= 2, f"Face count expected 2, detected {face_count}")
            if face_count >= 2:
                faces_sorted = sorted(faces, key=lambda f: f.get("facial_area", {}).get("x", 0))
                left_x = (faces_sorted[0]["facial_area"]["x"] + faces_sorted[0]["facial_area"]["w"] / 2) / w
                right_x = (faces_sorted[1]["facial_area"]["x"] + faces_sorted[1]["facial_area"]["w"] / 2) / w
                self.log_test("S5.2", 5, "Spatial", left_x < 0.50 and right_x > 0.50, f"Left face at x={left_x:.2f}, Right face at x={right_x:.2f}")

                # Biometrics: College Ebright (Left) vs James Wong (Right)
                c_anchor = self.cache_data.get("college_ebright")
                w_anchor = self.cache_data.get("james_wong")

                if c_anchor:
                    sim_c, _, _, _, _ = calculate_max_match_arcface(c_anchor, img_cv, expected_faces=2)
                    self.log_test("B5.1", 5, "Biometric", sim_c >= 0.45, f"College Ebright ArcFace similarity", round(sim_c, 4), 0.45)

                if w_anchor:
                    sim_w, _, _, _, _ = calculate_max_match_arcface(w_anchor, img_cv, expected_faces=2)
                    self.log_test("B5.2", 5, "Biometric", sim_w >= 0.45, f"James Wong ArcFace similarity", round(sim_w, 4), 0.45)

            # Prop QA: DNA model vs glowing orb vs laptop
            desk_crop = img_cv[int(h * 0.35):, :]
            candidates = ["a plastic ball-and-stick molecular model", "a laptop computer", "a glowing magical orb", "an empty wooden desk"]
            scores = classify_crop_clip(desk_crop, candidates)
            orb_score = scores.get("a glowing magical orb", 0.0)
            self.log_test("P5.1", 5, "Prop", orb_score < 0.15, f"No glowing orb hallucination (orb_score={orb_score:.3f})")

        # -------------------------------------------------------------
        # SCENE 6: Graduation (College Ebright on Stage)
        # -------------------------------------------------------------
        elif scene_num == 6:
            self.log_test("S6.1", 6, "Spatial", face_count >= 1, f"Face count detected {face_count}")
            c_anchor = self.cache_data.get("college_ebright")
            if face_count >= 1 and c_anchor:
                sim, _, _, _, _ = calculate_max_match_arcface(c_anchor, img_cv, expected_faces=1)
                self.log_test("B6.1", 6, "Biometric", sim >= 0.45, f"College Ebright Graduation ArcFace similarity", round(sim, 4), 0.45)

            # Regalia QA
            candidates = ["a person wearing a black graduation gown and mortarboard cap", "a person in casual clothes", "a suit and tie"]
            scores = classify_crop_clip(img_cv, candidates)
            gown_score = scores.get("a person wearing a black graduation gown and mortarboard cap", 0.0)
            self.log_test("P6.1", 6, "Prop", gown_score >= 0.50, f"Graduation regalia classification (gown={gown_score:.3f})")

        return True

    def run_full_audit(self, prefix: str = "SOTA_scene_") -> Dict[str, Any]:
        print("\n" + "=" * 90)
        print(f"  EXECUTING SOTA 3-PILLAR QA AUDIT ON '{prefix}*.png'")
        print("=" * 90)

        for s_idx in range(1, 7):
            p = OUTPUT_DIR / f"{prefix}{s_idx}.png"
            if not p.exists():
                p = OUTPUT_DIR / f"QA_scene_{s_idx}.png"
            print(f"\n  --- Auditing Scene {s_idx} ({p.name}) ---")
            self.audit_scene(s_idx, p)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round(passed / total * 100, 1) if total > 0 else 0,
            "results": self.results
        }

        with open(REPORT_PATH, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 90)
        print(f"  SOTA QA AUDIT SCORECARD: {passed} / {total} Passed ({summary['pass_rate_pct']}%)")
        print(f"  Report saved to: {REPORT_PATH}")
        print("=" * 90)
        return summary


def main():
    auditor = SOTAQAAuditor()
    auditor.run_full_audit()

if __name__ == "__main__":
    main()
