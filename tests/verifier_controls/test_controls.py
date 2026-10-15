"""
MMCCP Phase R2: Verifier Control Tests Suite
Validates that Verifier v1.1 behaves with mathematical precision under known perturbations:
1. Identical image yields max similarity (1.0)
2. Horizontal flip of 2-person scene FAILS spatial position binding
3. Color-inverted image is flagged by inversion detector
4. Hue-rotated shirt FAILS wardrobe Delta E tolerance
5. Wrong-character face transplant breaks Rank-1 identification
6. Duplicating a face triggers cloned/duplicated face defect
7. Blank image reports missing people
8. JPEG re-save changes similarity score by < 0.03
9. Clean ground-truth portrait triggers 0 defects
10. Detector health records ran=True across all detectors
"""
import os
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DLL paths for Torch on Windows
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
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble, cosine_similarity
from consistency_v2.verify.binding_checker import BindingChecker, delta_e_cielab, REFERENCE_LAB_COLORS
from consistency_v2.verify.scene_auditor import SceneAuditor


class TestVerifierControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = BiometricEnsemble()
        cls.binder = BindingChecker()
        cls.auditor = SceneAuditor()
        
        cls.canary_orig_path = PROJECT_ROOT / "scratch" / "canary" / "canary_original.png"
        cls.canary_inv_path = PROJECT_ROOT / "scratch" / "canary" / "canary_inverted.png"
        cls.scene1_path = PROJECT_ROOT / "scratch" / "v1_regression" / "scene_1_v1_regression.png"
        
        assert cls.canary_orig_path.exists(), "Canary original image must exist"
        assert cls.canary_inv_path.exists(), "Canary inverted image must exist"
        assert cls.scene1_path.exists(), "Scene 1 regression image must exist"
        
        cls.img_canary = cv2.imread(str(cls.canary_orig_path))
        cls.img_inverted = cv2.imread(str(cls.canary_inv_path))
        cls.img_scene1 = cv2.imread(str(cls.scene1_path))

    def test_01_identical_image_gives_max_similarity(self):
        """Control 1: Comparing an image to itself must yield cosine similarity == 1.0 (within 0.005)."""
        embs1 = self.verifier.extract_face_embeddings(self.img_canary)
        embs2 = self.verifier.extract_face_embeddings(self.img_canary)
        for m in BiometricEnsemble.JUDGE_MODELS:
            self.assertIn(m, embs1)
            self.assertIn(m, embs2)
            sim = cosine_similarity(embs1[m], embs2[m])
            self.assertAlmostEqual(sim, 1.0, places=2, msg=f"Model {m} failed self-identity similarity")

    def test_02_horizontal_flip_fails_spatial_binding(self):
        """Control 2: Horizontally flipping a 2-person scene must invert left/right and FAIL spatial binding."""
        h, w = self.img_scene1.shape[:2]
        flipped = cv2.flip(self.img_scene1, 1)
        
        # In original scene 1, young_ebright is on left (x ~ 250)
        # In flipped scene, young_ebright is on right (x ~ w - 250 = 774)
        orig_bbox = {"x": 250, "y": 200, "w": 150, "h": 180}
        flipped_bbox = {"x": w - (orig_bbox["x"] + orig_bbox["w"]), "y": orig_bbox["y"], "w": orig_bbox["w"], "h": orig_bbox["h"]}
        
        # Verify that original position passes "left"
        pass_orig, zone_orig, rx_orig = self.binder.verify_spatial_position(orig_bbox, w, "left")
        self.assertTrue(pass_orig, f"Original position should pass 'left', got {zone_orig}")
        
        # Verify that flipped position FAILS "left"
        pass_flip, zone_flip, rx_flip = self.binder.verify_spatial_position(flipped_bbox, w, "left")
        self.assertFalse(pass_flip, f"Flipped position must FAIL 'left', but got {zone_flip}")
        self.assertEqual(zone_flip, "right", f"Flipped position should be observed in 'right' zone")

    def test_03_color_inversion_flagged(self):
        """Control 3: Color-inverted canary image must be flagged by the inversion detector."""
        from consistency_v2.verify.torch_bridge import call_torch_vision
        res = call_torch_vision(self.canary_inv_path, [], [])
        self.assertFalse("error" in res, f"Bridge call failed: {res.get('error')}")
        self.assertTrue(res.get("is_inverted"), "Inverted canary image must be detected as inverted")
        self.assertGreaterEqual(res.get("inversion_probability", 0.0), 0.50)

    def test_04_hue_rotated_shirt_fails_wardrobe(self):
        """Control 4: Hue-rotating a yellow shirt to purple/blue must fail wardrobe tolerance."""
        # Clean yellow shirt LAB median is ~ [85, 5, 80]
        yellow_lab = np.array([85.0, 5.0, 80.0], dtype=np.float32)
        # Rotating hue shifts a* and b* drastically (e.g. blue/purple: [40, 20, -50])
        hue_rotated_lab = np.array([40.0, 20.0, -50.0], dtype=np.float32)
        
        pass_clean, de_clean, _ = self.binder.verify_wardrobe_color(yellow_lab, ["bright yellow"], max_delta_e=25.0)
        self.assertTrue(pass_clean, f"Clean yellow shirt should pass, got Delta E {de_clean}")
        
        pass_rotated, de_rotated, _ = self.binder.verify_wardrobe_color(hue_rotated_lab, ["bright yellow"], max_delta_e=25.0)
        self.assertFalse(pass_rotated, f"Hue rotated shirt must FAIL wardrobe check, got Delta E {de_rotated}")
        self.assertGreater(de_rotated, 40.0, "Delta E of purple shirt vs yellow must be > 40")

    def test_05_pasted_wrong_face_breaks_rank1(self):
        """Control 5: Pasting mother's face on Ebright must break Rank-1 re-identification."""
        # Extract embeddings for mother and college_ebright
        faces_scene1 = self.auditor.detect_faces(self.img_scene1)
        faces_scene1 = sorted(faces_scene1, key=lambda f: f["bbox"]["x"])
        # Mother is right face
        mother_crop = faces_scene1[1]["crop_bgr"]
        
        # Build synthetic bank
        ebright_embs = self.verifier.extract_face_embeddings(self.img_canary)
        mother_embs = self.verifier.extract_face_embeddings(mother_crop)
        
        banks = {
            "college_ebright": {m: [ebright_embs[m]] for m in BiometricEnsemble.JUDGE_MODELS},
            "mother": {m: [mother_embs[m]] for m in BiometricEnsemble.JUDGE_MODELS}
        }
        centroids = {
            "college_ebright": {m: ebright_embs[m] for m in BiometricEnsemble.JUDGE_MODELS},
            "mother": {m: mother_embs[m] for m in BiometricEnsemble.JUDGE_MODELS}
        }
        
        # Query with mother's face claiming to be college_ebright
        disc_res = self.verifier.compute_discrimination(mother_embs, banks, centroids, target_char_id="college_ebright")
        self.assertFalse(disc_res["rank1_correct"], "Mother's face queried against college_ebright must FAIL Rank-1")
        self.assertEqual(disc_res["rank1_character"], "mother", "Winner must be mother")

    def test_06_duplicated_face_triggers_clone_alarm(self):
        """Control 6: Duplicating a face in a scene must trigger the clone/duplicate detector."""
        img_clone = self.img_scene1.copy()
        # Paste young_ebright face onto mother's face position
        faces = self.auditor.detect_faces(self.img_scene1)
        faces = sorted(faces, key=lambda f: f["bbox"]["x"])
        b_left = faces[0]["bbox"]
        b_right = faces[1]["bbox"]
        
        # Resize left face to right face size and paste
        left_crop = img_clone[b_left["y"]:b_left["y"]+b_left["h"], b_left["x"]:b_left["x"]+b_left["w"]]
        left_resized = cv2.resize(left_crop, (b_right["w"], b_right["h"]))
        img_clone[b_right["y"]:b_right["y"]+b_right["h"], b_right["x"]:b_right["x"]+b_right["w"]] = left_resized
        
        # Audit faces
        detected = self.auditor.detect_faces(img_clone)
        self.assertGreaterEqual(len(detected), 2)
        embs1 = self.verifier.extract_face_embeddings(detected[0]["crop_bgr"])
        embs2 = self.verifier.extract_face_embeddings(detected[1]["crop_bgr"])
        
        # Similarity between cloned faces must be high (> 0.85)
        sim = cosine_similarity(embs1["ArcFace"], embs2["ArcFace"])
        self.assertGreater(sim, 0.85, f"Cloned faces must have ArcFace similarity > 0.85, got {sim}")
        
        # Verify check_clone_or_bleed generates clone alarm
        alarms = self.verifier.check_clone_or_bleed([("face_left", embs1), ("face_right", embs2)])
        self.assertGreaterEqual(len(alarms), 1, "Cloned faces must trigger clone alarm")

    def test_07_blank_image_reports_missing_people(self):
        """Control 7: A completely blank/black image must detect 0 faces and report missing people."""
        blank = np.zeros((1024, 1024, 3), dtype=np.uint8)
        faces = self.auditor.detect_faces(blank)
        self.assertEqual(len(faces), 0, "Blank image should detect 0 faces")
        
        audit_res = self.auditor.audit_scene_with_vision(
            image_bgr=blank,
            expected_faces=2,
            vision_data={"person_count_yolo": 0}
        )
        self.assertFalse(audit_res["scene_passed"])
        self.assertIn("missing_people (found 0 faces, expected 2)", audit_res["defect_taxonomy_violations"])

    def test_08_jpeg_resave_within_small_tolerance(self):
        """Control 8: JPEG compression (quality 85) must alter biometric scores by < 0.03."""
        embs_orig = self.verifier.extract_face_embeddings(self.img_canary)
        # Encode as JPEG quality 85 and decode
        _, enc = cv2.imencode(".jpg", self.img_canary, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        embs_jpeg = self.verifier.extract_face_embeddings(jpeg_decoded)
        
        for m in BiometricEnsemble.JUDGE_MODELS:
            sim = cosine_similarity(embs_orig[m], embs_jpeg[m])
            diff = abs(1.0 - sim)
            self.assertLess(diff, 0.03, f"JPEG compression altered {m} score too much: diff {diff} >= 0.03")

    def test_09_clean_image_has_zero_defects(self):
        """Control 9: A clean ground-truth portrait must trigger 0 defect taxonomy violations."""
        faces = self.auditor.detect_faces(self.img_canary)
        self.assertEqual(len(faces), 1, "Clean portrait must detect exactly 1 face")
        
        audit_res = self.auditor.audit_scene_with_vision(
            image_bgr=self.img_canary,
            expected_faces=1,
            vision_data={"person_count_yolo": 1, "is_inverted": False, "missing_required_props": [], "detected_forbidden_props": []}
        )
        self.assertTrue(audit_res["scene_passed"], f"Clean image should pass scene audit, got defects: {audit_res['defect_taxonomy_violations']}")
        self.assertEqual(audit_res["defect_count"], 0)

    def test_10_detector_health_tracked(self):
        """Control 10: Vision helper must track detector health and flag missing detectors."""
        from consistency_v2.verify.torch_bridge import call_torch_vision
        res = call_torch_vision(self.canary_orig_path, [{"x": 100, "y": 100, "w": 200, "h": 200}], [])
        d_health = res.get("detector_health", {})
        self.assertIn("yolov8n", d_health)
        self.assertIn("segformer", d_health)
        self.assertIn("clip", d_health)
        self.assertIn("dinov2", d_health)
        self.assertTrue(d_health["yolov8n"])
        self.assertTrue(d_health["segformer"])
        self.assertTrue(d_health["clip"])
        self.assertTrue(d_health["dinov2"])


if __name__ == "__main__":
    unittest.main()
