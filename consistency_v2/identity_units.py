"""
MMCCP Consistency v2: Method M4 (Identity Units Done Right)
Configures high-fidelity identity conditioning using IP-Adapter Plus-Face SDXL
and dual-unit InstantID (InsightFace embedding + keypoints ControlNet) fed from Identity Bank references.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import base64
from io import BytesIO
from PIL import Image
import logging

from experiments.controlnet_adapter import ControlSpec
from consistency_v2.identity_bank import IdentityBankManager

logger = logging.getLogger("MMCCP.IdentityUnits")


@dataclass
class IdentityUnitConfig:
    """Configuration for native SDXL identity units in ControlNet."""
    enable_ip_adapter_plus_face: bool = True
    ip_adapter_plus_face_model: str = "ip-adapter-plus-face_sdxl_vit-h [368cf551]"
    ip_adapter_plus_face_module: str = "CLIP-ViT-H (IPAdapter)"
    ip_adapter_plus_face_weight: float = 0.60
    ip_adapter_plus_face_guidance_start: float = 0.0
    ip_adapter_plus_face_guidance_end: float = 0.85

    enable_instantid: bool = False  # Enabled in quality mode or when strong structural face lock is required
    instantid_ip_model: str = "ip-adapter_instant_id_sdxl [eb2d3ec0]"
    instantid_ip_module: str = "InsightFace (InstantID)"
    instantid_ip_weight: float = 0.65
    instantid_kp_model: str = "control_instant_id_sdxl [c5c25a50]"
    instantid_kp_module: str = "instant_id_face_keypoints"
    instantid_kp_weight: float = 0.50
    instantid_guidance_end: float = 0.80

    control_mode: str = "Balanced"  # "Balanced", "My prompt is more important", "ControlNet is more important"


class IdentityUnitManager:
    """
    Constructs calibrated ControlSpec instances for SDXL identity conditioning.
    Retrieves viewpoint-aligned reference face images from the Identity Bank.
    """

    @classmethod
    def image_to_base64(cls, image_path: str) -> Optional[str]:
        """Loads an image from path and returns base64 PNG string."""
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                buf = BytesIO()
                img_rgb.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    @classmethod
    def build_identity_control_specs(
        cls,
        character_ids: List[str],
        bank_manager: IdentityBankManager,
        target_yaw: float = 0.0,
        config: Optional[IdentityUnitConfig] = None
    ) -> List[ControlSpec]:
        """
        Builds calibrated identity ControlSpecs for all specified characters.

        Args:
            character_ids: List of character identifiers present in the scene.
            bank_manager: Loaded IdentityBankManager instance.
            target_yaw: Target head yaw angle for reference selection.
            config: IdentityUnitConfig instance (defaults to standard settings).

        Returns:
            List of ControlSpec objects ready for Forge ControlNet payload building.
        """
        if config is None:
            config = IdentityUnitConfig()

        specs: List[ControlSpec] = []

        for cid in character_ids:
            best_ref = bank_manager.select_best_reference(cid, target_yaw=target_yaw)
            if not best_ref or "path" not in best_ref:
                logger.warning(f"No identity bank reference available for character: {cid}")
                continue

            ref_b64 = cls.image_to_base64(best_ref["path"])
            if not ref_b64:
                continue

            # 1. IP-Adapter Plus-Face SDXL unit (High-fidelity feature cross-attention)
            if config.enable_ip_adapter_plus_face:
                specs.append(ControlSpec(
                    name=f"ip_adapter_plus_face_{cid}",
                    model=config.ip_adapter_plus_face_model,
                    module=config.ip_adapter_plus_face_module,
                    image_b64=ref_b64,
                    weight=config.ip_adapter_plus_face_weight,
                    guidance_start=config.ip_adapter_plus_face_guidance_start,
                    guidance_end=config.ip_adapter_plus_face_guidance_end,
                    control_mode=config.control_mode
                ))
                logger.info(
                    f"Built IP-Adapter Plus-Face unit for {cid} (weight={config.ip_adapter_plus_face_weight})"
                )

            # 2. Dual-Unit InstantID (Embedding unit + Keypoints unit)
            if config.enable_instantid:
                # Unit A: InsightFace embedding unit
                specs.append(ControlSpec(
                    name=f"instantid_ip_{cid}",
                    model=config.instantid_ip_model,
                    module=config.instantid_ip_module,
                    image_b64=ref_b64,
                    weight=config.instantid_ip_weight,
                    guidance_start=0.0,
                    guidance_end=config.instantid_guidance_end,
                    control_mode=config.control_mode
                ))

                # Unit B: InstantID facial keypoints unit
                specs.append(ControlSpec(
                    name=f"instantid_kp_{cid}",
                    model=config.instantid_kp_model,
                    module=config.instantid_kp_module,
                    image_b64=ref_b64,
                    weight=config.instantid_kp_weight,
                    guidance_start=0.0,
                    guidance_end=config.instantid_guidance_end,
                    control_mode=config.control_mode
                ))
                logger.info(
                    f"Built dual InstantID units for {cid} (ip_weight={config.instantid_ip_weight}, "
                    f"kp_weight={config.instantid_kp_weight})"
                )

        return specs
