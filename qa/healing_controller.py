"""
STAGE 30: AUTONOMOUS HEALING CONTROLLER
Detects specific generation failure modes and automatically tunes parameters to heal failed scenes.
"""
import os, sys, json, time, logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

from story_engine.schema import StorySpec, SceneSpec
from generation.generation_orchestrator import GenerationOrchestrator
from qa.programmatic_auditor import ProgrammaticAuditor

logger = logging.getLogger("MMCCP.HealingController")


class HealingController:
    """
    Automated diagnosis and self-healing loop for story diffusion scenes.
    """

    def __init__(self, orchestrator: GenerationOrchestrator, auditor: ProgrammaticAuditor, max_retries: int = 3):
        self.orchestrator = orchestrator
        self.auditor = auditor
        self.max_retries = max_retries
        self.healing_history: List[Dict[str, Any]] = []

    def heal_scene(
        self,
        story: StorySpec,
        scene: SceneSpec,
        output_dir: Path = Path("edu_outputs/images"),
        prefix: str = "scene_"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any], bool]:
        """
        Executes generation, audits result, and automatically retries with diagnosed parameter adjustments if needed.
        """
        img_p = output_dir / f"{prefix}{scene.scene_number}.png"
        json_p = output_dir / f"{prefix}{scene.scene_number}.json"

        # Attempt 1: Standard Generation
        img_cv, meta = self.orchestrator.generate_scene(story=story, scene=scene)
        if img_cv is not None:
            cv2.imwrite(str(img_p), img_cv)
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        # Audit initial render
        self.auditor.results = []
        self.auditor.audit_scene(story, scene, img_p)
        failed_tests = [t for t in self.auditor.results if not t["passed"]]

        if not failed_tests:
            logger.info(f"Scene {scene.scene_number} passed all QA tests on Attempt 1.")
            return img_cv, meta, True

        # Healing Loop
        attempt = 1
        cfg_scale = 5.5
        steps1 = 25
        enable_openpose = False

        while failed_tests and attempt < self.max_retries:
            attempt += 1
            logger.info(f"[HEALING] Scene {scene.scene_number} Attempt {attempt} of {self.max_retries} | Fixing {len(failed_tests)} failures.")

            # Diagnose failure types
            has_spatial_fail = any(t["category"] == "Spatial" for t in failed_tests)
            has_bio_fail = any(t["category"] == "Biometric" for t in failed_tests)
            has_prop_fail = any(t["category"] == "Prop" for t in failed_tests)

            # Apply targeted parameter adjustments
            if has_spatial_fail or len(scene.characters_present) >= 2:
                enable_openpose = True
                cfg_scale = 6.0
                steps1 = 28

            if has_bio_fail:
                # Boost demographic tokens in prompt builder
                cfg_scale = 6.5
                steps1 = 30

            # Re-generate with healed parameters
            img_cv, meta = self.orchestrator.generate_scene(
                story=story,
                scene=scene,
                steps_pass1=steps1,
                cfg_scale=cfg_scale,
                enable_openpose=enable_openpose
            )

            if img_cv is not None:
                cv2.imwrite(str(img_p), img_cv)
                with open(json_p, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            # Re-audit
            self.auditor.results = []
            self.auditor.audit_scene(story, scene, img_p)
            failed_tests = [t for t in self.auditor.results if not t["passed"]]

            self.healing_history.append({
                "scene_number": scene.scene_number,
                "attempt": attempt,
                "remaining_failures": len(failed_tests),
                "adjusted_cfg": cfg_scale,
                "adjusted_steps": steps1,
                "openpose_enabled": enable_openpose
            })

        success = len(failed_tests) == 0
        return img_cv, meta, success
