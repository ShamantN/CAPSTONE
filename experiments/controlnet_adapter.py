"""
STAGE 5: CONTROLNET & NATIVE IDENTITY CONDITIONING ADAPTER
Translates internal high-level ControlSpec definitions into the exact SD Forge ControlNet API payload.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ControlSpec:
    name: str
    model: str
    module: str = "none"
    image_b64: Optional[str] = None
    weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    control_mode: str = "Balanced"  # "Balanced", "My prompt is more important", "ControlNet is more important"
    resize_mode: str = "Crop and Resize"  # "Just Resize", "Crop and Resize", "Resize and Fill"
    pixel_perfect: bool = True
    processor_res: int = 512
    threshold_a: float = 0.5
    threshold_b: float = 0.5

class ControlNetAdapter:
    """
    Standardizes ControlNet and IP-Adapter payloads for SD Forge API across SDXL architectures.
    """

    CONTROL_MODE_MAP = {
        "Balanced": 0,
        "My prompt is more important": 1,
        "ControlNet is more important": 2
    }

    RESIZE_MODE_MAP = {
        "Just Resize": 0,
        "Crop and Resize": 1,
        "Resize and Fill": 2
    }

    @staticmethod
    def to_forge_unit(spec: ControlSpec) -> Dict[str, Any]:
        """Converts a ControlSpec into a Forge ControlNet unit dictionary."""
        return {
            "enabled": True,
            "module": spec.module,
            "model": spec.model,
            "weight": spec.weight,
            "image": spec.image_b64,
            "input_image": spec.image_b64,
            "guidance_start": spec.guidance_start,
            "guidance_end": spec.guidance_end,
            "control_mode": spec.control_mode,
            "resize_mode": spec.resize_mode,
            "pixel_perfect": spec.pixel_perfect,
            "processor_res": spec.processor_res,
            "threshold_a": spec.threshold_a,
            "threshold_b": spec.threshold_b
        }

    @staticmethod
    def build_alwayson_payload(specs: List[ControlSpec]) -> Dict[str, Any]:
        """Builds alwayson_scripts dictionary entry for Forge ControlNet."""
        if not specs:
            return {}
        units = [ControlNetAdapter.to_forge_unit(s) for s in specs]
        return {
            "controlnet": {
                "args": units
            }
        }

if __name__ == "__main__":
    spec = ControlSpec(
        name="ip-adapter",
        model="ip-adapter_xl [af81326a]",
        module="InsightFace+CLIP-H (IPAdapter)",
        weight=0.85
    )
    p = ControlNetAdapter.build_alwayson_payload([spec])
    print("Sample ControlNet Payload Built Successfully:")
    import json
    print(json.dumps(p, indent=2))
