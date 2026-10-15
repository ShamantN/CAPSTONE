"""
MMCCP Consistency Pipeline v2: Stage Tracer and Assertion Instrumentation
Records intermediate images, per-stage biometric scores, and pixel-diff metrics across all pipeline stages.
Enforces StageNoOpError whenever an enabled stage fails to alter its target region.
"""
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StageNoOpError(RuntimeError):
    """Raised when an enabled pipeline stage fails to modify its designated region."""
    pass


def compute_image_sha256(img_bgr: np.ndarray) -> str:
    """Computes SHA256 of PNG-encoded image buffer."""
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        return "encode_failed"
    return hashlib.sha256(buf.tobytes()).hexdigest()


def compute_region_pixel_diff(
    img_a: np.ndarray,
    img_b: np.ndarray,
    bboxes: Optional[List[Dict[str, int]]] = None
) -> float:
    """
    Computes mean absolute pixel difference between two images.
    If bboxes are provided, computes diff only within the union of bounding boxes.
    """
    if img_a.shape != img_b.shape:
        return 255.0

    diff = cv2.absdiff(img_a, img_b)

    if not bboxes:
        return float(np.mean(diff))

    h, w = img_a.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        bx = max(0, int(b.get("x", 0)))
        by = max(0, int(b.get("y", 0)))
        bw = max(1, int(b.get("w", 1)))
        bh = max(1, int(b.get("h", 1)))
        mask[by:min(h, by + bh), bx:min(w, bx + bw)] = 255

    num_pixels = np.count_nonzero(mask)
    if num_pixels == 0:
        return float(np.mean(diff))

    diff_region = cv2.bitwise_and(diff, diff, mask=mask)
    return float(np.sum(diff_region) / (num_pixels * 3.0))


class StageTracer:
    """
    Instruments pipeline_v2 execution, records per-stage states, and verifies stage efficacy.
    """

    def __init__(
        self,
        trace_id: str,
        enabled: bool = True,
        trace_root: Optional[Path] = None,
        epsilon_diff: float = 0.5
    ):
        self.trace_id = trace_id
        self.enabled = enabled
        self.trace_root = trace_root or (PROJECT_ROOT / "results" / "traces")
        self.trace_dir = self.trace_root / self.trace_id
        self.epsilon_diff = epsilon_diff

        self.stages: List[Dict[str, Any]] = []
        self._last_image: Optional[np.ndarray] = None

        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def record_stage(
        self,
        stage_name: str,
        image_bgr: np.ndarray,
        target_bboxes: Optional[List[Dict[str, int]]] = None,
        stage_metadata: Optional[Dict[str, Any]] = None,
        enforce_active: bool = False
    ) -> Dict[str, Any]:
        """
        Records an intermediate pipeline stage.
        If enforce_active=True, asserts pixel change in target region > epsilon_diff.
        """
        t_now = time.time()
        sha = compute_image_sha256(image_bgr)

        pixel_diff = 0.0
        if self._last_image is not None:
            pixel_diff = compute_region_pixel_diff(self._last_image, image_bgr, target_bboxes)

        if enforce_active and self._last_image is not None:
            if pixel_diff < self.epsilon_diff:
                raise StageNoOpError(
                    f"StageNoOpError in stage '{stage_name}': "
                    f"Stage was enabled but produced negligible pixel modification "
                    f"(mean diff = {pixel_diff:.4f} < epsilon {self.epsilon_diff}). "
                    f"Target bboxes: {target_bboxes}"
                )

        stage_record = {
            "stage_index": len(self.stages) + 1,
            "stage_name": stage_name,
            "timestamp": t_now,
            "sha256": sha,
            "pixel_diff_from_previous": round(pixel_diff, 4),
            "target_bboxes": target_bboxes or [],
            "metadata": stage_metadata or {}
        }

        # Save stage image to disk if tracing is enabled
        if self.enabled:
            fname = f"{len(self.stages) + 1:02d}_{stage_name}.png"
            out_path = self.trace_dir / fname
            cv2.imwrite(str(out_path), image_bgr)
            stage_record["image_file"] = fname

        self.stages.append(stage_record)
        self._last_image = image_bgr.copy()
        return stage_record

    def record_skipped_stage(self, stage_name: str, reason: str):
        """Records a disabled or intentionally bypassed stage."""
        stage_record = {
            "stage_index": len(self.stages) + 1,
            "stage_name": stage_name,
            "status": "SKIPPED",
            "reason": reason,
            "timestamp": time.time(),
            "pixel_diff_from_previous": 0.0
        }
        self.stages.append(stage_record)

    def finalize(self) -> Dict[str, Any]:
        """Saves final trace report JSON."""
        report = {
            "trace_id": self.trace_id,
            "total_stages": len(self.stages),
            "stages": self.stages
        }
        if self.enabled:
            rep_path = self.trace_dir / "trace_report.json"
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return report
