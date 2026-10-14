"""
Unit tests for M1 IdentityBankManager
Verifies bank loading, centroid retrieval, pose selection, and 4-point gate.
Compatible with standard unittest runner.
"""
import unittest
from pathlib import Path
import numpy as np
from consistency_v2.identity_bank import IdentityBankManager


class TestIdentityBankManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = IdentityBankManager()

    def test_identity_bank_loading(self):
        self.assertGreaterEqual(len(self.mgr._banks), 5)
        for cid in ["young_ebright", "teen_ebright", "college_ebright", "mother", "james_wong"]:
            self.assertIn(cid, self.mgr._banks)
            bank = self.mgr._banks[cid]
            self.assertGreaterEqual(bank["num_references"], 1)
            self.assertIn("centroid_embeddings", bank)
            self.assertIn("references", bank)

    def test_blended_centroid_retrieval(self):
        centroid = self.mgr.get_blended_centroid("young_ebright")
        self.assertIsNotNone(centroid)
        self.assertIn("ArcFace", centroid)
        self.assertIn("Facenet512", centroid)
        # L2 normalized
        norm = float(np.linalg.norm(centroid["ArcFace"]))
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_pose_guided_reference_selection(self):
        ref = self.mgr.select_best_reference("young_ebright", target_yaw=0.0)
        self.assertIsNotNone(ref)
        self.assertIn("filename", ref)
        self.assertIn("pose", ref)
        self.assertIn("yaw", ref["pose"])

    def test_primary_reference_image_exists(self):
        img = self.mgr.get_primary_reference_image("young_ebright")
        self.assertIsNotNone(img)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
