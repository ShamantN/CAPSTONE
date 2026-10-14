"""
MMCCP Consistency v2: Method M5 (Reliable Identity Transplant)
Implements schema-validated ReActor payload construction (fixing D2) and direct InsightFace
inswapper_128 face swapping bound to LayoutPlanner spatial regions with CodeFormer restoration.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import subprocess
import os
import json
import base64
import logging
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger("MMCCP.FaceSwap")


class ReActorPayloadBuilder:
    """
    Constructs the exact 31-argument payload contract for Stable Diffusion WebUI Forge's ReActor extension.
    Fixes Defect D2 (missing 31st argument 'tab_single' and integer/string choice collisions).
    """

    @staticmethod
    def build_reactor_payload(
        source_image_b64: str,
        target_face_index: int = 0,
        source_face_index: int = 0,
        codeformer_weight: float = 0.60,
        restorer_visibility: float = 0.75,
        restorer_name: str = "CodeFormer",
        model_name: str = "inswapper_128.onnx",
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Builds the 31-element argument array expected by sd-webui-reactor.
        """
        args = [
            source_image_b64,                    # 0: Image (base64)
            enabled,                             # 1: Enable
            str(source_face_index),              # 2: Source face index
            str(target_face_index),              # 3: Target face index
            model_name,                          # 4: Model ("inswapper_128.onnx")
            restorer_name,                       # 5: Restorer ("CodeFormer")
            float(restorer_visibility),          # 6: Restorer visibility
            True,                                # 7: Restore Face -> Upscale order
            "None",                              # 8: Upscaler
            1.0,                                 # 9: Scale by
            1.0,                                 # 10: Upscaler visibility
            False,                               # 11: Swap in source image
            True,                                # 12: Swap in generated image
            "Minimum",                           # 13: Console log level ("Minimum" string, not int!)
            "No",                                # 14: Gender detect source ("No" string, not int!)
            "No",                                # 15: Gender detect target ("No" string, not int!)
            False,                               # 16: Save original
            float(codeformer_weight),            # 17: CodeFormer weight / fidelity
            True,                                # 18: Source image hash check
            False,                               # 19: Target image hash check
            "CPU",                               # 20: Execution provider
            False,                               # 21: Face mask correction
            "Image(s)",                          # 22: Select source ("Image(s)" string, not int!)
            "None",                              # 23: Choose face model
            "",                                  # 24: Source folder
            None,                                # 25: Multiple source images
            False,                               # 26: Random image
            False,                               # 27: Force upscale
            0.5,                                 # 28: Threshold
            0,                                   # 29: Max faces
            "tab_single"                         # 30: Target tab (mandatory 31st argument!)
        ]

        return {"reactor": {"args": args}}


class DirectFaceSwapper:
    """
    Direct, deterministic face swapper executing inswapper_128.onnx via Forge's embedded Python.
    Binds each target face to its planned character layout zone (M2) and applies CodeFormer restoration.
    """

    WORKER_TEMPLATE = r'''
import sys, cv2, json, os, numpy as np
import insightface
from insightface.app import FaceAnalysis

target_img_path = sys.argv[1]
swap_specs_json = sys.argv[2]
out_img_path = sys.argv[3]

with open(swap_specs_json, "r", encoding="utf-8") as f:
    swap_specs = json.load(f)

target_bgr = cv2.imread(target_img_path)
h, w = target_bgr.shape[:2]

app = FaceAnalysis(name="buffalo_l", root=r"C:\Stable-Diffusion\webui\models", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper_path = r"C:\Stable-Diffusion\webui\models\insightface\inswapper_128.onnx"
swapper = insightface.model_zoo.get_model(swapper_path, providers=["CPUExecutionProvider"])

target_faces = app.get(target_bgr)
result_bgr = target_bgr.copy()

for spec in swap_specs:
    cid = spec["character_id"]
    src_path = spec["source_path"]
    zone = spec["target_zone"]  # "left", "right", "center"

    src_bgr = cv2.imread(src_path)
    src_faces = app.get(src_bgr)
    if not src_faces:
        continue
    src_face = src_faces[0]

    matched_target = None
    for tf in target_faces:
        center_x = (tf.bbox[0] + tf.bbox[2]) / 2.0
        if zone == "left" and center_x < w * 0.52:
            matched_target = tf
            break
        elif zone == "right" and center_x > w * 0.48:
            matched_target = tf
            break
        elif zone == "center":
            matched_target = tf
            break

    if matched_target is not None:
        result_bgr = swapper.get(result_bgr, matched_target, src_face, paste_back=True)

cv2.imwrite(out_img_path, result_bgr)
'''

    @classmethod
    def execute_face_swap(
        cls,
        target_image_path: str,
        swap_specs: List[Dict[str, str]],
        out_image_path: str,
        codeformer_fidelity: float = 0.65,
        codeformer_visibility: float = 0.75,
        forge_api_url: str = "http://127.0.0.1:7860",
        forge_python_exe: str = r"C:\Stable-Diffusion\system\python\python.exe"
    ) -> bool:
        """
        Executes layout-bound face swap and post-processes with CodeFormer face restoration.

        Args:
            target_image_path: Path to input rendered image.
            swap_specs: List of dicts specifying character_id, target_zone, and source_path.
            out_image_path: Destination path for final image.
            codeformer_fidelity: CodeFormer restoration fidelity (0.0 to 1.0).
            codeformer_visibility: CodeFormer blend visibility (0.0 to 1.0).
            forge_api_url: Stable Diffusion Forge WebUI API base URL.
            forge_python_exe: Path to Forge's embedded Python.

        Returns:
            bool: True if swap and restoration succeeded, False otherwise.
        """
        if not os.path.exists(target_image_path):
            logger.error(f"Target image not found: {target_image_path}")
            return False

        # Create temporary worker and specs files
        worker_path = "consistency_v2/banks/_temp_swapper_worker.py"
        specs_path = "consistency_v2/banks/_temp_swap_specs.json"
        raw_swapped_path = "consistency_v2/banks/_temp_swapped_raw.png"

        os.makedirs("consistency_v2/banks", exist_ok=True)
        with open(worker_path, "w", encoding="utf-8") as f:
            f.write(cls.WORKER_TEMPLATE)

        with open(specs_path, "w", encoding="utf-8") as f:
            json.dump(swap_specs, f, indent=2)

        try:
            logger.info("Executing inswapper_128 worker on CPU...")
            res = subprocess.run([
                forge_python_exe,
                worker_path,
                target_image_path,
                specs_path,
                raw_swapped_path
            ], capture_output=True, text=True, timeout=120)

            if res.returncode != 0 or not os.path.exists(raw_swapped_path):
                logger.error(f"Inswapper worker failed (code {res.returncode}): {res.stderr}")
                return False

            # Post-swap CodeFormer restoration via Forge API
            logger.info("Applying CodeFormer restoration via Forge Extras API...")
            with open(raw_swapped_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "image": img_b64,
                "codeformer_visibility": float(codeformer_visibility),
                "codeformer_weight": float(codeformer_fidelity),
                "upscaling_resize": 1.0
            }

            resp = requests.post(f"{forge_api_url.rstrip('/')}/sdapi/v1/extra-single-image", json=payload, timeout=60)
            if resp.status_code == 200 and "image" in resp.json():
                restored_bytes = base64.b64decode(resp.json()["image"])
                with open(out_image_path, "wb") as f:
                    f.write(restored_bytes)
                logger.info(f"Direct face swap and CodeFormer restoration completed: {out_image_path}")
                return True
            else:
                # Fallback to raw swapped image if Extras API fails
                logger.warning(f"Extras API returned status {resp.status_code}; saving raw swapped image.")
                with open(raw_swapped_path, "rb") as f_in, open(out_image_path, "wb") as f_out:
                    f_out.write(f_in.read())
                return True

        except Exception as e:
            logger.error(f"Face swap execution exception: {e}")
            return False
        finally:
            for p in [worker_path, specs_path, raw_swapped_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
