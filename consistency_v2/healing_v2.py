"""
Diagnosis-Driven Healing Controller v2 (M9)
Replaces naive seed bumping with a systematic diagnostic remedy table.
Maps specific verifier failure modes to targeted parameter adjustments and surgical inpainting passes.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import cv2
import numpy as np

logger = logging.getLogger("MMCCP.HealingV2")


@dataclass
class HealingAction:
    remedy_id: str
    description: str
    target_component: str
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    re_render_required: bool = False
    inpaint_pass_required: bool = False


@dataclass
class HealingStepLog:
    step_number: int
    diagnosis: str
    action_taken: str
    initial_score: float
    resulting_score: float
    success: bool


class HealingControllerV2:
    """
    Coordinates targeted remediation based on deterministic diagnostic signals.
    """
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def diagnose_failure(
        self,
        verifier_report: Dict[str, Any],
        face_bboxes: List[Dict[str, Any]]
    ) -> Optional[HealingAction]:
        """
        Diagnoses primary failure mode from verifier report and returns targeted remedy.
        """
        # 1. Check for Clone / Bleed across characters
        scene_audits = verifier_report.get("scene_audit", {})
        defects = scene_audits.get("defects_detected", [])
        if "cloned_or_bleed_faces" in defects or scene_audits.get("duplicate_faces", False):
            return HealingAction(
                remedy_id="ISOLATE_REGIONAL_CONDITIONING",
                description="Clone/bleed detected. Strengthening regional spatial separation and reducing global blend.",
                target_component="regional",
                parameter_overrides={
                    "regional_global_weight": 0.25,
                    "ip_adapter_plus_weight": 0.45
                },
                re_render_required=True
            )

        # 2. Check for Extra Limbs / Duplicated People
        if "extra_limbs" in defects or scene_audits.get("extra_persons", False):
            return HealingAction(
                remedy_id="STRENGTHEN_POSE_AND_NEGATIVE_CONSTRAINTS",
                description="Extra persons or limbs detected. Increasing OpenPose guidance weight and negative prompt filters.",
                target_component="layout_pose",
                parameter_overrides={
                    "openpose_weight": 0.85,
                    "extra_negative_prompt": "extra limbs, malformed limbs, duplicate person, twin"
                },
                re_render_required=True
            )

        # 3. Check for Wardrobe Color Discrepancy
        binding_results = verifier_report.get("binding_check", {}).get("per_character", {})
        for char_id, b_info in binding_results.items():
            if not b_info.get("wardrobe_pass", True) or b_info.get("delta_e", 0.0) > 38.0:
                return HealingAction(
                    remedy_id="WARDROBE_CIELAB_REPAIR",
                    description=f"Wardrobe color mismatch on {char_id} (Delta E {b_info.get('delta_e', 0.0)}). Applying bounded CIELAB color lock.",
                    target_component="wardrobe",
                    parameter_overrides={
                        "target_character": char_id,
                        "blend_strength": 0.75
                    },
                    inpaint_pass_required=False  # Direct bounded post-process
                )

        # 4. Check for Spatial Binding Failure (face in wrong zone)
        for char_id, b_info in binding_results.items():
            if not b_info.get("spatial_pass", True):
                return HealingAction(
                    remedy_id="RELAYOUT_SPATIAL_ZONES",
                    description=f"Character {char_id} placed in incorrect spatial zone. Adjusting OpenPose skeleton offset.",
                    target_component="layout_pose",
                    parameter_overrides={
                        "relayout_character": char_id,
                        "reseed": True
                    },
                    re_render_required=True
                )

        # 5. Check for Low Identity Similarity
        bio_results = verifier_report.get("biometrics", {}).get("per_character", {})
        for char_id, m_info in bio_results.items():
            mean_sim = m_info.get("ensemble_mean", 1.0)
            if mean_sim < 0.60:
                # Find face size
                c_bbox = next((f.get("bbox", {}) for f in face_bboxes if f.get("character_id") == char_id), {})
                f_size = max(c_bbox.get("w", 0), c_bbox.get("h", 0))

                if f_size < 96:
                    # Small face -> Boost Detailer
                    return HealingAction(
                        remedy_id="BOOST_FACE_DETAILER",
                        description=f"Low identity on small face ({f_size}px) for {char_id}. Increasing detailer steps and denoise.",
                        target_component="face_detailer",
                        parameter_overrides={
                            "face_detailer_steps": 25,
                            "face_detailer_denoise": 0.42,
                            "target_character": char_id
                        },
                        inpaint_pass_required=True
                    )
                else:
                    # Large face -> Boost Transplant & Units
                    return HealingAction(
                        remedy_id="BOOST_IDENTITY_TRANSPLANT",
                        description=f"Low identity on large face ({f_size}px) for {char_id}. Boosting CodeFormer fidelity and swap weight.",
                        target_component="face_swap",
                        parameter_overrides={
                            "codeformer_fidelity": 0.85,
                            "target_character": char_id
                        },
                        inpaint_pass_required=False
                    )

        # No critical defects requiring healing
        return None

    def execute_healing_loop(
        self,
        initial_image_bgr: np.ndarray,
        initial_report: Dict[str, Any],
        face_bboxes: List[Dict[str, Any]],
        wardrobe_handler: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
        detailer_handler: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
        re_render_handler: Optional[Callable[[Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], List[HealingStepLog]]:
        """
        Executes targeted healing iterations up to max_retries.
        Returns (healed_image_bgr, final_report, healing_history).
        """
        current_image = initial_image_bgr.copy()
        current_report = initial_report
        best_image = initial_image_bgr.copy()
        best_report = initial_report
        best_score = self._extract_overall_score(initial_report)

        history: List[HealingStepLog] = []

        for step in range(1, self.max_retries + 1):
            action = self.diagnose_failure(current_report, face_bboxes)
            if action is None:
                logger.info(f"Step {step}: No defects requiring healing. Exiting loop.")
                break

            logger.info(f"Healing Step {step}: Firing remedy {action.remedy_id} - {action.description}")

            repaired_image = None
            if action.remedy_id == "WARDROBE_CIELAB_REPAIR" and wardrobe_handler:
                repaired_image = wardrobe_handler(current_image, action.parameter_overrides)
            elif action.remedy_id == "BOOST_FACE_DETAILER" and detailer_handler:
                repaired_image = detailer_handler(current_image, action.parameter_overrides)
            elif action.re_render_required and re_render_handler:
                repaired_image, new_report = re_render_handler(action.parameter_overrides)
                current_report = new_report

            if repaired_image is not None:
                current_image = repaired_image

            new_score = self._extract_overall_score(current_report)
            success = (new_score >= best_score)

            history.append(HealingStepLog(
                step_number=step,
                diagnosis=action.remedy_id,
                action_taken=action.description,
                initial_score=round(best_score, 4),
                resulting_score=round(new_score, 4),
                success=success
            ))

            if success:
                best_score = new_score
                best_image = current_image.copy()
                best_report = current_report
            else:
                # Revert to best known state
                current_image = best_image.copy()

        return best_image, best_report, history

    @staticmethod
    def _extract_overall_score(report: Dict[str, Any]) -> float:
        """Extracts scalar composite quality score from verifier report."""
        if not report:
            return 0.0
        # Average ensemble biometric score
        bio = report.get("biometrics", {})
        mean_bio = float(bio.get("mean_ensemble", 0.50))

        # Binding score
        binding = report.get("binding_check", {})
        binding_pass_rate = float(binding.get("pass_rate", 0.50))

        # Defect penalty
        defects = len(report.get("scene_audit", {}).get("defects_detected", []))
        defect_penalty = max(0.0, 1.0 - (0.20 * defects))

        return float((0.50 * mean_bio) + (0.30 * binding_pass_rate) + (0.20 * defect_penalty))
