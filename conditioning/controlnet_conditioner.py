"""
STAGE 13 & 14: CONTROLNET CONDITIONER
Manages multi-unit ControlNet dispatch for OpenPose SDXL, Depth SDXL, Canny, and IP-Adapter.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from experiments.controlnet_adapter import ControlSpec, ControlNetAdapter


class ControlNetConditioner:
    """
    Multi-unit ControlNet orchestrator for SDXL diffusion pipelines.
    """

    AVAILABLE_MODELS = {
        "openpose": "openpose_xl [f4251cb4]",
        "ip_adapter": "ip-adapter_xl [af81326a]",
        "depth": "depth_xl [4a0891d4]",
        "instantid": "instantid_xl [c5c25a50]",
        "faceid": "ip-adapter-faceid-plusv2_sdxl [187cb962]"
    }

    MODULE_MAP = {
        "openpose": "dw_openpose_full",
        "ip_adapter": "CLIP-ViT-bigG (IPAdapter)",
        "depth": "depth_anything_v2",
        "instantid": "instant_id_face_keypoints",
        "faceid": "InsightFace+CLIP-H (IPAdapter)"
    }

    @classmethod
    def is_model_on_disk(cls, model_key: str) -> bool:
        fn_map = {
            "openpose": r"C:\Stable-Diffusion\webui\models\ControlNet\openpose_xl.safetensors",
            "ip_adapter": r"C:\Stable-Diffusion\webui\models\ControlNet\ip-adapter_xl.safetensors",
            "depth": r"C:\Stable-Diffusion\webui\models\ControlNet\depth_xl.safetensors",
            "instantid": r"C:\Stable-Diffusion\webui\models\ControlNet\instantid_xl.safetensors",
            "faceid": r"C:\Stable-Diffusion\webui\models\ControlNet\ip-adapter-faceid-plusv2_sdxl.bin"
        }
        p = fn_map.get(model_key)
        return Path(p).exists() if p else False

    @classmethod
    def create_openpose_spec(
        cls,
        pose_image_b64: Optional[str] = None,
        weight: float = 0.70,
        module: Optional[str] = None
    ) -> Optional[ControlSpec]:
        """Creates an OpenPose ControlSpec. Returns None if no valid input image is provided (fixing D1)."""
        if not cls.is_model_on_disk("openpose") or not pose_image_b64:
            return None
        # If pre-rendered skeleton from LayoutPlanner, use module="None" (capitalized in Forge)
        chosen_module = module if module is not None else ("None" if pose_image_b64 else cls.MODULE_MAP["openpose"])
        return ControlSpec(
            name="openpose",
            model=cls.AVAILABLE_MODELS["openpose"],
            module=chosen_module,
            image_b64=pose_image_b64,
            weight=weight,
            guidance_start=0.0,
            guidance_end=0.75,
            control_mode="Balanced"
        )

    @classmethod
    def create_depth_spec(
        cls,
        depth_image_b64: Optional[str] = None,
        weight: float = 0.65,
        module: Optional[str] = None
    ) -> Optional[ControlSpec]:
        """Creates a Depth ControlSpec. Returns None if no valid input image is provided (fixing D1)."""
        if not cls.is_model_on_disk("depth") or not depth_image_b64:
            return None
        chosen_module = module if module is not None else ("None" if depth_image_b64 else cls.MODULE_MAP["depth"])
        return ControlSpec(
            name="depth",
            model=cls.AVAILABLE_MODELS["depth"],
            module=chosen_module,
            image_b64=depth_image_b64,
            weight=weight,
            guidance_start=0.0,
            guidance_end=0.70,
            control_mode="Balanced"
        )

    @classmethod
    def build_payload(cls, specs: List[ControlSpec]) -> Dict[str, Any]:
        """Converts active ControlSpecs into alwayson_scripts dictionary."""
        valid_specs = [s for s in specs if s is not None]
        return ControlNetAdapter.build_alwayson_payload(valid_specs)
