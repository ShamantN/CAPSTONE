"""
Reference-Only Attention Injection (M8)
Constructs ControlNet Reference-Only units (reference_only or reference_adain+attn)
as a training-free approximation of cross-image self-attention sharing (ConsiStory / StoryDiffusion).
Injects self-attention features from certified reference images in Pass 1.
"""
import base64
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("MMCCP.ReferenceUnit")


class ReferenceOnlyUnitBuilder:
    """
    Constructs ControlNet payload units for Reference-Only self-attention injection.
    """
    def __init__(
        self,
        module: str = "reference_only",
        default_weight: float = 0.40,
        default_guidance_end: float = 0.70,
        style_fidelity: float = 0.50
    ):
        self.module = module  # "reference_only" or "reference_adain+attn"
        self.default_weight = default_weight
        self.default_guidance_end = default_guidance_end
        self.style_fidelity = style_fidelity

    @staticmethod
    def encode_image_b64(image_input: Union[str, Path, np.ndarray]) -> str:
        """Converts file path or numpy BGR array to base64 data string."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Reference image not found: {p}")
            with open(p, "rb") as f:
                raw_bytes = f.read()
            return f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
        elif isinstance(image_input, np.ndarray):
            success, buffer = cv2.imencode(".png", image_input)
            if not success:
                raise ValueError("Failed to encode numpy array to PNG")
            return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def build_unit(
        self,
        reference_image: Union[str, Path, np.ndarray],
        weight: Optional[float] = None,
        guidance_end: Optional[float] = None,
        style_fidelity: Optional[float] = None,
        effective_region_mask: Optional[Union[str, Path, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Builds a single ControlNet Reference-Only unit dictionary.
        """
        w = self.default_weight if weight is None else weight
        gend = self.default_guidance_end if guidance_end is None else guidance_end
        sf = self.style_fidelity if style_fidelity is None else style_fidelity

        ref_b64 = self.encode_image_b64(reference_image)

        unit = {
            "enabled": True,
            "module": self.module,
            "model": "None",
            "weight": float(w),
            "input_image": ref_b64,
            "control_mode": "Balanced",
            "pixel_perfect": True,
            "guidance_start": 0.0,
            "guidance_end": float(gend),
            "threshold_a": float(sf)
        }

        if effective_region_mask is not None:
            unit["effective_region_mask"] = self.encode_image_b64(effective_region_mask)

        return unit

    def build_scene_reference_units(
        self,
        characters: List[str],
        identity_banks: Dict[str, Any],
        layout_masks: Optional[Dict[str, np.ndarray]] = None,
        weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds reference units for characters present in the scene.
        In multi-character scenes, attaches effective_region_mask to prevent attention bleeding.
        """
        units = []
        for char_id in characters:
            bank = identity_banks.get(char_id)
            if bank is None:
                continue

            ref_img = None
            if hasattr(bank, "references") and bank.references:
                # Pick highest quality reference
                best_ref = max(bank.references, key=lambda r: getattr(r, "sharpness_laplacian", 0.0))
                ref_img = best_ref.image_path
            elif hasattr(bank, "anchor_path") and bank.anchor_path:
                ref_img = bank.anchor_path
            elif isinstance(bank, dict):
                ref_img = bank.get("anchor_path") or (bank.get("references", [{}])[0].get("image_path") if bank.get("references") else None)

            if ref_img is None:
                continue

            reg_mask = None
            if layout_masks and char_id in layout_masks:
                reg_mask = layout_masks[char_id]

            unit = self.build_unit(
                reference_image=ref_img,
                weight=weight,
                effective_region_mask=reg_mask
            )
            units.append(unit)

        return units
