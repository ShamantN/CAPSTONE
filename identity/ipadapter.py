"""
STAGE 9 & 10: CONCRETE IDENTITY ADAPTERS
Implements IP-Adapter SDXL, InstantID, and FaceID adapters with exact vision encoder matching.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from identity.identity_conditioner import IdentityConditioner
from experiments.controlnet_adapter import ControlSpec
from story_engine.scene_planner import SceneExecutionPlan


class IPAdapterSDXLConditioner(IdentityConditioner):
    """
    Primary SOTA Identity Adapter using SDXL IP-Adapter and OpenCLIP-ViT-bigG cross-attention guidance.
    Uses CLIP-ViT-bigG (embedding dim 1280) for ip-adapter_xl (shape [8192, 1280]).
    """

    def __init__(self, model_name: str = "ip-adapter_xl [af81326a]", module: Optional[str] = None):
        self.model_name = model_name
        # Match module to model architecture to prevent projection shape mismatch
        if module is not None:
            self.module = module
        elif "vit-h" in model_name.lower():
            self.module = "CLIP-ViT-H (IPAdapter)"
        else:
            self.module = "CLIP-ViT-bigG (IPAdapter)"

    def get_adapter_name(self) -> str:
        return "IP-Adapter-SDXL"

    def is_available(self) -> bool:
        p = Path(r"C:\Stable-Diffusion\webui\models\ControlNet\ip-adapter_xl.safetensors")
        return p.exists()

    def build_control_specs(self, plan: SceneExecutionPlan) -> List[ControlSpec]:
        if not plan.character_anchors:
            return []

        specs = []
        if len(plan.character_anchors) == 1:
            specs.append(ControlSpec(
                name="ip-adapter",
                model=self.model_name,
                module=self.module,
                image_b64=plan.character_anchors[0],
                weight=plan.ip_adapter_weight,
                guidance_start=0.0,
                guidance_end=0.85
            ))
        else:
            # Multi-character composite
            specs.append(ControlSpec(
                name="ip-adapter",
                model=self.model_name,
                module=self.module,
                image_b64=plan.reactor_composite_b64,
                weight=plan.ip_adapter_weight,
                guidance_start=0.0,
                guidance_end=0.60
            ))
        return specs
