"""
STAGE 16: LORA REGISTRY
"""
from typing import Dict, Any
from pathlib import Path


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
