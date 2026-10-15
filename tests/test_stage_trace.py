"""
Unit tests for MMCCP consistency_v2.stage_trace module.
Tests StageTracer instrumentation, region pixel-diff computations,
and StageNoOpError enforcement.
"""
import unittest
import numpy as np
import tempfile
import json
from pathlib import Path

from consistency_v2.stage_trace import (
    StageTracer,
    StageNoOpError,
    compute_image_sha256,
    compute_region_pixel_diff
)


class TestStageTrace(unittest.TestCase):

    def test_compute_image_sha256_deterministic(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img3 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        sha1 = compute_image_sha256(img1)
        sha2 = compute_image_sha256(img2)
        sha3 = compute_image_sha256(img3)

        self.assertEqual(len(sha1), 64)
        self.assertEqual(sha1, sha2)
        self.assertNotEqual(sha1, sha3)

    def test_compute_region_pixel_diff(self):
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((100, 100, 3), dtype=np.uint8)
        # Identical
        self.assertEqual(compute_region_pixel_diff(img_a, img_b), 0.0)

        # Whole image diff of 10
        img_c = np.ones((100, 100, 3), dtype=np.uint8) * 10
        diff_whole = compute_region_pixel_diff(img_a, img_c)
        self.assertAlmostEqual(diff_whole, 10.0, places=4)

        # Bounded diff: only change [10:30, 10:30]
        img_d = np.zeros((100, 100, 3), dtype=np.uint8)
        img_d[10:30, 10:30] = 50
        bboxes = [{"x": 10, "y": 10, "w": 20, "h": 20}]
        diff_region = compute_region_pixel_diff(img_a, img_d, bboxes=bboxes)
        self.assertAlmostEqual(diff_region, 50.0, places=4)

        # Region outside bounding box has no change in tested area
        outside_bboxes = [{"x": 50, "y": 50, "w": 20, "h": 20}]
        diff_outside = compute_region_pixel_diff(img_a, img_d, bboxes=outside_bboxes)
        self.assertEqual(diff_outside, 0.0)

    def test_stage_tracer_execution_and_noop_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_root = Path(tmpdir)
            tracer = StageTracer(trace_id="test_run", enabled=True, trace_root=trace_root, epsilon_diff=0.5)

            img0 = np.zeros((100, 100, 3), dtype=np.uint8)
            tracer.record_stage("stage_0_init", img0)

            # Stage 1: real modification -> should pass enforce_active
            img1 = img0.copy()
            img1[20:40, 20:40] = 120
            target_boxes = [{"x": 20, "y": 20, "w": 20, "h": 20}]
            rec1 = tracer.record_stage("stage_1_active", img1, target_bboxes=target_boxes, enforce_active=True)
            self.assertEqual(rec1["stage_index"], 2)
            self.assertGreater(rec1["pixel_diff_from_previous"], 10.0)

            # Stage 2: no modification when enforce_active=True -> should raise StageNoOpError
            img2 = img1.copy()
            with self.assertRaises(StageNoOpError):
                tracer.record_stage("stage_2_noop", img2, target_bboxes=target_boxes, enforce_active=True)

            # Stage 3: skipped stage
            tracer.record_skipped_stage("stage_3_disabled", reason="feature flag off")

            # Finalize
            rep = tracer.finalize()
            self.assertEqual(rep["trace_id"], "test_run")
            self.assertEqual(rep["total_stages"], 3)
            self.assertTrue((trace_root / "test_run" / "trace_report.json").exists())


if __name__ == "__main__":
    unittest.main()
