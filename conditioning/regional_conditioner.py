"""
STAGE 15, 16, 17, 18: REGIONAL, LORA, REFINEMENT, AND CROSS-FRAME CONDITIONERS
"""
import os, sys, json, logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests

logger = logging.getLogger("MMCCP.Conditioning")


class RegionalConditioner:
    """
    Manages regional spatial column isolation and prompt partitioning.
    """

    @staticmethod
    def format_column_prompt(prompts_per_column: List[str]) -> str:
        """Joins regional column prompts with SD native BREAK syntax."""
        return " BREAK ".join(prompts_per_column)

    @staticmethod
    def get_regional_matrix(num_regions: int) -> Dict[str, Any]:
        """Returns standard column ratio metadata."""
        return {
            "regional_active": num_regions > 1,
            "num_regions": num_regions,
            "mode": "Columns",
            "ratios": ",".join(["1"] * num_regions),
            "base_ratio": 0.20
        }


class LoRARegistry:
    """
    Discovers, validates, and dynamically injects LoRA tokens into prompt strings.
    """

    def __init__(self, lora_dir: Path = Path(r"C:\Stable-Diffusion\webui\models\Lora")):
        self.lora_dir = lora_dir
        self.available_loras: Dict[str, Path] = {}
        self.scan_loras()

    def scan_loras(self):
        if self.lora_dir.exists():
            for f in self.lora_dir.glob("*.safetensors"):
                self.available_loras[f.stem] = f

    def build_lora_prompt_string(self, loras_to_apply: Dict[str, float]) -> str:
        """Returns string like '<lora:realism_xl:0.6> <lora:detail_xl:0.4>'."""
        tokens = []
        for name, weight in loras_to_apply.items():
            if name in self.available_loras:
                tokens.append(f"<lora:{name}:{weight}>")
        return " ".join(tokens)


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


class CrossFrameConditioner:
    """
    Maintains cross-frame character visual DNA continuity and state tracking across consecutive scenes.
    """

    def __init__(self):
        self.frame_history: List[Dict[str, Any]] = []

    def record_frame(self, scene_number: int, image_b64: str, metadata: Dict[str, Any]):
        self.frame_history.append({
            "scene_number": scene_number,
            "image_b64": image_b64,
            "metadata": metadata
        })

    def get_previous_frame_anchor(self, character_id: str) -> Optional[str]:
        """Returns the most recent successfully generated frame containing this character."""
        for frame in reversed(self.frame_history):
            chars = frame["metadata"].get("characters", [])
            if character_id in chars:
                return frame["image_b64"]
        return None
