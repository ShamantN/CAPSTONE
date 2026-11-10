"""
Unit tests for Cross-Scene Identity Memory with Drift Guard (M10)
"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from consistency_v2.identity_memory import IdentityMemoryManager, PromotionDecision


class TestIdentityMemory(unittest.TestCase):
    def setUp(self):
        # Mock Bank Manager
        self.mock_bank_manager = MagicMock()
        self.mock_bank_manager._banks = {
            "char_1": {
                "centroid_embeddings": {
                    "Facenet512": [1.0, 0.0],
                    "ArcFace": [1.0, 0.0],
                    "SFace": [1.0, 0.0],
                    "GhostFaceNet": [1.0, 0.0]
                },
                "references": []
            }
        }
        self.mock_bank_manager.get_bank_dir.return_value = Path("scratch")
        self.memory = IdentityMemoryManager(
            bank_manager=self.mock_bank_manager,
            max_bank_size=8,
            min_similarity_to_anchor=0.72,
            max_centroid_drift=0.05,
            min_face_size_px=96
        )

    def test_reject_small_face(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = {"x": 10, "y": 10, "w": 60, "h": 60}  # 60px < 96px
        res = self.memory.consider_scene_face_promotion("char_1", img, bbox, scene_id=1)
        self.assertFalse(res.promoted)
        self.assertIn("below minimum required", res.reason)

    def test_reject_blurry_face(self):
        # Uniform color has 0 laplacian variance
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[10:130, 10:130] = (120, 120, 120)
        bbox = {"x": 10, "y": 10, "w": 120, "h": 120}  # 120px > 96px
        res = self.memory.consider_scene_face_promotion("char_1", img, bbox, scene_id=1)
        self.assertFalse(res.promoted)
        self.assertIn("sharpness", res.reason)

    def test_reject_drift_guard(self):
        # Create textured patch for sharpness
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = {"x": 10, "y": 10, "w": 120, "h": 120}

        # Mock embeddings to return a vector with large angle / shift from [1.0, 0.0]
        # e.g. [0.75, 0.66] -> dot product with [1, 0] is 0.75 (> 0.72 pass similarity),
        # but single candidate addition shifts centroid norm distance significantly (> 0.05)
        shifted_vec = np.array([0.74, 0.67], dtype=np.float32)
        shifted_vec /= np.linalg.norm(shifted_vec)

        self.mock_bank_manager.ensemble.extract_face_embeddings.return_value = {
            "Facenet512": shifted_vec,
            "ArcFace": shifted_vec,
            "SFace": shifted_vec,
            "GhostFaceNet": shifted_vec
        }

        res = self.memory.consider_scene_face_promotion("char_1", img, bbox, scene_id=1)
        self.assertFalse(res.promoted)
        self.assertIn("drift", res.reason.lower())


if __name__ == "__main__":
    unittest.main()
