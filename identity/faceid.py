"""
STAGE 11: REAL FACEID PLUS V2 CONDITIONER
Actively integrates IP-Adapter FaceID Plus v2 SDXL model and LoRA.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from identity.identity_conditioner import IdentityConditioner
from experiments.controlnet_adapter import ControlSpec
from story_engine.scene_planner import SceneExecutionPlan


class FaceIDPlusV2Conditioner(IdentityConditioner):
    """
    Native FaceID Plus v2 SDXL Adapter with face recognition embeddings and companion LoRA.
    """

    def __init__(self, model_name: str = "ip-adapter-faceid-plusv2_sdxl [187cb962]", module: str = "InsightFace+CLIP-H (IPAdapter)"):
        self.model_name = model_name
        self.module = module

    def get_adapter_name(self) -> str:
        return "FaceID-Plus-v2"

    def is_available(self) -> bool:
        p = Path(r"C:\Stable-Diffusion\webui\models\ControlNet\ip-adapter-faceid-plusv2_sdxl.bin")
        return p.exists()

    def build_control_specs(self, plan: SceneExecutionPlan) -> List[ControlSpec]:
        if not plan.character_anchors:
            return []

        specs = []
        ref_img = plan.character_anchors[0] if len(plan.character_anchors) == 1 else plan.reactor_composite_b64
        specs.append(ControlSpec(
            name="faceid_plus_v2",
            model=self.model_name,
            module=self.module,
            image_b64=ref_img,
            weight=0.85 if len(plan.character_anchors) == 1 else 0.40,
            guidance_start=0.0,
            guidance_end=0.80
        ))
        return specs
