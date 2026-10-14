"""
MMCCP Consistency v2: Method M6 (Identity-Conditioned Face Detailer)
Applies layout-aware high-resolution inpainting to small or low-fidelity faces
using character-specific prompt tokens, identity units, and post-swap CodeFormer restoration.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np
import base64
import logging
import requests
from io import BytesIO
from PIL import Image

from consistency_v2.layout_planner import CharacterLayoutBox
from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.identity_units import IdentityUnitManager, IdentityUnitConfig
from consistency_v2.face_swap import DirectFaceSwapper
from consistency_v2.verify.scene_auditor import SceneAuditor
from experiments.controlnet_adapter import ControlNetAdapter

logger = logging.getLogger("MMCCP.FaceDetailer")


@dataclass
class FaceDetailerConfig:
    """Configuration for identity-conditioned face detailing."""
    enabled: bool = True
    face_size_threshold: int = 120  # Apply detailing to faces with width < 120 px
    padding_px: int = 48            # Padding around face crop for inpaint_full_res
    denoising_strength: float = 0.35 # Denoising strength for subtle feature refinement
    steps: int = 20
    cfg_scale: float = 5.0
    mask_blur_radius: int = 21
    apply_swap_after_detail: bool = True
    codeformer_fidelity: float = 0.65
    codeformer_visibility: float = 0.75


class FaceDetailer:
    """
    Automated, spatial-binding-aware face detailing pipeline.
    Resolves latent resolution bottlenecks for small faces in wide and medium shots.
    """

    def __init__(self, forge_api_url: str = "http://127.0.0.1:7860"):
        self.forge_api_url = forge_api_url.rstrip("/")
        self.auditor = SceneAuditor()

    def build_face_mask(self, img_shape: Tuple[int, int, int], bbox: Dict[str, int], blur_radius: int = 21) -> np.ndarray:
        """Constructs a smooth elliptical mask around the face and hair region."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = int(bbox["x"] + bbox["w"] / 2.0)
        cy = int(bbox["y"] + bbox["h"] / 2.0)
        axes = (int(bbox["w"] * 0.75), int(bbox["h"] * 0.90))

        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

        # Apply Gaussian blur for seamless inpaint boundary
        k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        blurred = cv2.GaussianBlur(mask, (k, k), 0)
        return blurred

    def match_face_to_layout_box(self, face_bbox: Dict[str, int], layout_boxes: List[CharacterLayoutBox], img_width: int) -> Optional[CharacterLayoutBox]:
        """Binds a detected face to the planned character box based on horizontal centroid."""
        face_cx = face_bbox["x"] + face_bbox["w"] / 2.0

        for box in layout_boxes:
            norm_region = box.normalized_region
            # Normalized bounds: [y1, y2, x1, x2]
            x_min = norm_region[2] * img_width
            x_max = norm_region[3] * img_width
            if x_min - 20 <= face_cx <= x_max + 20:
                return box

        # Fallback to closest box
        if layout_boxes:
            return min(layout_boxes, key=lambda b: abs((b.normalized_region[2] + b.normalized_region[3]) / 2.0 * img_width - face_cx))
        return None

    def detail_image(
        self,
        image_bgr: np.ndarray,
        character_prompts: Dict[str, str],
        layout_boxes: List[CharacterLayoutBox],
        bank_manager: IdentityBankManager,
        config: Optional[FaceDetailerConfig] = None
    ) -> np.ndarray:
        """
        Detects faces, applies full-res inpaint detailing to qualifying faces,
        and executes identity transplant restoration.
        """
        if config is None:
            config = FaceDetailerConfig()

        if not config.enabled:
            return image_bgr

        detected_faces = self.auditor.detect_faces(image_bgr)
        h, w = image_bgr.shape[:2]
        current_img = image_bgr.copy()

        logger.info(f"FaceDetailer: Found {len(detected_faces)} faces across canvas ({w}x{h}).")

        swap_specs: List[Dict[str, str]] = []

        for face in detected_faces:
            fb = face["bbox"]
            face_w = fb["w"]
            matched_box = self.match_face_to_layout_box(fb, layout_boxes, w)
            if not matched_box:
                continue

            cid = matched_box.character_id
            best_ref = bank_manager.select_best_reference(cid)
            if best_ref and "path" in best_ref:
                swap_specs.append({
                    "character_id": cid,
                    "target_zone": matched_box.target_zone,
                    "source_path": best_ref["path"]
                })

            # Check size threshold
            if face_w < config.face_size_threshold:
                logger.info(f"Detailing face for {cid} (width={face_w}px < {config.face_size_threshold}px)...")

                mask = self.build_face_mask(current_img.shape, fb, config.mask_blur_radius)

                # Encode image and mask to base64
                _, img_enc = cv2.imencode(".png", current_img)
                img_b64 = base64.b64encode(img_enc).decode("utf-8")
                _, mask_enc = cv2.imencode(".png", mask)
                mask_b64 = base64.b64encode(mask_enc).decode("utf-8")

                # Build character identity unit
                id_cfg = IdentityUnitConfig(ip_adapter_plus_face_weight=0.55)
                id_specs = IdentityUnitManager.build_identity_control_specs([cid], bank_manager, config=id_cfg)
                cnet_unit = ControlNetAdapter.to_forge_unit(id_specs[0]) if id_specs else None

                alwayson = {}
                if cnet_unit:
                    alwayson["controlnet"] = {"args": [cnet_unit]}

                char_prompt = character_prompts.get(cid, "a detailed portrait")
                detail_prompt = f"cinematic close-up portrait of {char_prompt}, highly detailed facial features, natural skin texture, 8k"
                neg_prompt = "blurry, deformed, bad eyes, cartoon, 3d, duplicate, watermark"

                payload = {
                    "init_images": [img_b64],
                    "mask": mask_b64,
                    "inpaint_full_res": True,
                    "inpaint_full_res_padding": config.padding_px,
                    "inpainting_fill": 1,  # original
                    "denoising_strength": config.denoising_strength,
                    "prompt": detail_prompt,
                    "negative_prompt": neg_prompt,
                    "steps": config.steps,
                    "width": w,
                    "height": h,
                    "cfg_scale": config.cfg_scale,
                    "sampler_name": "DPM++ 2M",
                    "scheduler": "Karras",
                    "alwayson_scripts": alwayson
                }

                try:
                    resp = requests.post(f"{self.forge_api_url}/sdapi/v1/img2img", json=payload, timeout=180)
                    if resp.status_code == 200:
                        data = resp.json()
                        if "images" in data and data["images"]:
                            dec = base64.b64decode(data["images"][0])
                            arr = np.frombuffer(dec, dtype=np.uint8)
                            current_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            logger.info(f"Successfully detailed face for {cid}.")
                except Exception as e:
                    logger.warning(f"Inpainting pass failed for {cid}: {e}")

        # Post-detail identity transplant & CodeFormer restoration
        if config.apply_swap_after_detail and swap_specs:
            logger.info("Executing post-detail identity transplant and CodeFormer restoration...")
            temp_in = "consistency_v2/banks/_temp_pre_swap.png"
            temp_out = "consistency_v2/banks/_temp_post_swap.png"
            cv2.imwrite(temp_in, current_img)

            success = DirectFaceSwapper.execute_face_swap(
                target_image_path=temp_in,
                swap_specs=swap_specs,
                out_image_path=temp_out,
                codeformer_fidelity=config.codeformer_fidelity,
                codeformer_visibility=config.codeformer_visibility,
                forge_api_url=self.forge_api_url
            )
            if success and os.path.exists(temp_out):
                current_img = cv2.imread(temp_out)

            for p in [temp_in, temp_out]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        return current_img
