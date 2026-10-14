"""
STAGE 17: REFINEMENT CONTROLLER
Manages secondary high-resolution face restoration (ReActor + CodeFormer / Localized Inpainting).
"""
from typing import Dict, Any


class RefinementController:
    """
    Manages secondary high-resolution face restoration (ReActor + CodeFormer / Localized Inpainting).
    """

    @staticmethod
    def build_reactor_payload(
        source_image_b64: str,
        face_index_str: str = "0",
        codeformer_weight: float = 0.80
    ) -> Dict[str, Any]:
        """Constructs deterministic 30-arg Forge ReActor payload."""
        clean_b64 = source_image_b64.split(",")[-1] if "," in source_image_b64 else source_image_b64
        reactor_args = [
            clean_b64,           # 0: source image base64
            True,                # 1: enable
            face_index_str,      # 2: source face index
            face_index_str,      # 3: target face index
            "inswapper_128.onnx",# 4: model
            "CodeFormer",        # 5: restorer
            1.0,                 # 6: restore visibility
            False,               # 7: restore face -> upscale
            "None",              # 8: upscaler name
            1.0,                 # 9: upscaler scale
            1.0,                 # 10: upscaler visibility
            False,               # 11: swap in source
            True,                # 12: swap in generated
            0,                   # 13: log level
            0,                   # 14: gender source (0: Any)
            0,                   # 15: gender target (0: Any)
            False,               # 16: save original
            codeformer_weight,   # 17: codeformer weight
            False,               # 18: source hash check
            False,               # 19: target hash check
            "CUDA",              # 20: device
            True,                # 21: mask correction
            0,                   # 22: Select Source: 0 = IMAGE
            "",                  # 23: face model filename
            "",                  # 24: source folder path
            None,                # 25: extra
            False,               # 26: random select (deterministic)
            True,                # 27: force upscale
            0.5,                 # 28: detection threshold
            0                    # 29: max faces
        ]
        return {"reactor": {"args": reactor_args}}
