"""
STAGE 10: REAL INSTANTID CONDITIONER
Actively integrates InstantID SDXL ControlNet model and InsightFace keypoints.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from identity.identity_conditioner import IdentityConditioner
from experiments.controlnet_adapter import ControlSpec
from story_engine.scene_planner import SceneExecutionPlan


class InstantIDConditioner(IdentityConditioner):
    """
    Native InstantID SDXL Adapter with InsightFace facial landmark keypoints.
    """

    def __init__(self, model_name: str = "instantid_xl [c5c25a50]", module: str = "instant_id_face_keypoints"):
        self.model_name = model_name
        self.module = module

    def get_adapter_name(self) -> str:
        return "InstantID-SDXL"

    def is_available(self) -> bool:
        p = Path(r"C:\Stable-Diffusion\webui\models\ControlNet\instantid_xl.safetensors")
        return p.exists()

    def build_control_specs(self, plan: SceneExecutionPlan) -> List[ControlSpec]:
        if not plan.character_anchors:
            return []

        specs = []
        # Multi-character or single character reference anchor
        ref_img = plan.character_anchors[0] if len(plan.character_anchors) == 1 else plan.reactor_composite_b64
        specs.append(ControlSpec(
            name="instantid",
            model=self.model_name,
            module=self.module,
            image_b64=ref_img,
            weight=0.85 if len(plan.character_anchors) == 1 else 0.40,
            guidance_start=0.0,
            guidance_end=0.85,
            control_mode="ControlNet is more important"
        ))
        return specs
