"""
STAGE 6 & 12: SOTA NATIVE U-NET CONDITIONING GENERATOR
Implements:
- Native IP-Adapter cross-attention reference conditioning via ControlNet API
- Native U-Net FreeU + PAG + Dynamic CFG-Fix attention hooks
- Spatial Column Regional Prompt Isolation
- Camera-Distance Adaptive Denoising Scheduler
- Deterministic JSON metadata logging
"""
import os, sys, json, time, base64, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from metrics import (
    SD_MODELS,
    SD_OPTIONS_ENDPOINT,
    SD_TXT2IMG_ENDPOINT,
    SD_IMG2IMG_ENDPOINT,
    set_sd_model,
    decode_image_to_pil,
    decode_image_to_cv2,
    encode_image_to_base64,
    cleanup_memory,
    create_composite_face_image,
)

from experiments.controlnet_adapter import ControlNetAdapter, ControlSpec
from experiments.memory_utils import vram_guard, get_vram_telemetry

logger = logging.getLogger("MMCCP.SOTAGenerator")
OUTPUT_DIR = Path(r"D:\5TH SEM\HACKATHON\edu_outputs\images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SOTAGenerator:
    def __init__(self, sd_checkpoint: str = SD_MODELS["Juggernaut"]):
        self.sd_checkpoint = sd_checkpoint
        set_sd_model(self.sd_checkpoint)

    def render_scene(
        self,
        scene: Dict[str, Any],
        characters_db: List[Dict[str, Any]],
        environments_db: List[Dict[str, Any]],
        cache_data: Dict[str, str],
        style: str = "Cinematic Realistic",
        steps: int = 25,
        ip_adapter_weight: float = 0.85,
        cfg_scale: float = 5.5,
        seed: int = -1
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Executes native SDXL conditioning generation with IP-Adapter, FreeU, PAG, and Regional Prompting.
        """
        s_num = scene["scene_number"]
        s_title = scene["title"]
        char_ids = scene.get("characters", [])
        shot_type = scene.get("cinematic_prompt", {}).get("shot_type", "Medium shot")

        # 1. Determine Aspect Ratio
        is_multichar = len(char_ids) >= 2 or "two-shot" in shot_type.lower() or "wide" in shot_type.lower()
        width = 1216 if is_multichar else 1024
        height = 832 if is_multichar else 1024

        # 2. Build Master Regional Isolated Prompt
        from app import build_regional_isolated_prompt
        master_prompt, regional_meta = build_regional_isolated_prompt(scene, characters_db, environments_db, style)

        # 3. Scene-specific negative prompt guards
        neg_prompt = "blurry, deformed, bad anatomy, bad hands, missing fingers, extra fingers, ugly, watermark, text, signature"
        if s_num == 1:
            neg_prompt += ", solo, 1person, single person, male alone, woman alone, beard on child, adult boy"
        elif s_num == 2:
            neg_prompt += ", teenager, adult man, tall adult, mature face, beard, stubble"
        elif s_num == 3:
            neg_prompt += ", empty desk, empty table, bare table, hands alone on desk, laptop, computer, classroom without microscope, glowing box"
        elif s_num == 5:
            neg_prompt += ", solo, single person, 1person, close up portrait, headshot, laptop, computer, macbook, screen, monitor, glowing orb, glowing ball, crystal ball, magic sphere, yellow ball, orb, lamp, bare desk, Asian on left, Caucasian on right, identical faces, twin, clone"

        # 4. Resolve Identity Reference Anchors
        char_anchors = [cache_data.get(c_id) for c_id in char_ids if cache_data.get(c_id)]
        
        # Build ControlNet IP-Adapter specifications
        control_specs = []
        if len(char_anchors) == 1:
            control_specs.append(ControlSpec(
                name="ip-adapter",
                model="ip-adapter_xl [af81326a]",
                module="InsightFace+CLIP-H (IPAdapter)",
                image_b64=char_anchors[0],
                weight=ip_adapter_weight,
                guidance_start=0.0,
                guidance_end=0.85
            ))
            reactor_b64 = char_anchors[0]
            face_index_str = "0"
        elif len(char_anchors) >= 2:
            composite_b64 = create_composite_face_image(char_anchors)
            # Use light IP-Adapter weight to prevent cross-character identity bleed in dual-ethnic shots
            control_specs.append(ControlSpec(
                name="ip-adapter",
                model="ip-adapter_xl [af81326a]",
                module="InsightFace+CLIP-H (IPAdapter)",
                image_b64=composite_b64,
                weight=0.35,
                guidance_start=0.0,
                guidance_end=0.60
            ))
            reactor_b64 = composite_b64
            face_index_str = "0,1"
        else:
            reactor_b64 = None
            face_index_str = "0"

        cnet_payload = ControlNetAdapter.build_alwayson_payload(control_specs)

        # 5. Core Native U-Net Scripts (FreeU, PAG, CFG-Fix)
        freeu_args = [True, 1.3, 1.4, 0.9, 0.2, 0.0, 1.0]
        pag_args = [True, 3.0, 0.0, 0.0, 1.0]
        cfg_fix_args = [True, 7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, "enable", "MEAN", "AD", 1.0]

        alwayson_scripts = {
            "freeu integrated (sd 1.x, sd 2.x, sdxl)": {"args": freeu_args},
            "perturbedattentionguidance integrated": {"args": pag_args},
            "dynamicthresholding (cfg-fix) integrated": {"args": cfg_fix_args}
        }
        if cnet_payload.get("controlnet"):
            alwayson_scripts["controlnet"] = cnet_payload["controlnet"]

        # 6. Pass 1: TXT2IMG Structural Draft
        txt2img_payload = {
            "prompt": master_prompt,
            "negative_prompt": neg_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "seed": seed,
            "alwayson_scripts": alwayson_scripts
        }

        t0 = time.time()
        with vram_guard(threshold_mb=7400.0, scene_tag=f"Scene {s_num}") as mem_start:
            resp1 = requests.post(SD_TXT2IMG_ENDPOINT, json=txt2img_payload, timeout=300)
            if resp1.status_code != 200:
                logger.error(f"Pass 1 Failed: {resp1.status_code} {resp1.text}")
                return None, {}

            draft_images = resp1.json().get("images", [])
            if not draft_images:
                return None, {}

            draft_b64 = draft_images[0]
            info_json = json.loads(resp1.json().get("info", "{}"))
            actual_seed = info_json.get("seed", seed)

            # Pass 2: Adaptive Denoising & High-Resolution Polish
            from app import compute_adaptive_denoising_schedule
            adaptive_denoise = compute_adaptive_denoising_schedule(shot_type, base_denoise=0.48)

            # Add secondary ReActor biometric lock
            if reactor_b64:
                reactor_args = [
                    reactor_b64, True, face_index_str, face_index_str, "0",
                    "CodeFormer", 0.75, 1, "0", "0",
                    1, 1, 0, 0, 0.95,
                    0.95, False, True, "None", 1,
                    1, 0, None, None, 0,
                    0, 0.5, 0, 0, 0
                ]
                alwayson_scripts["reactor"] = {"args": reactor_args}

            img2img_payload = {
                "prompt": master_prompt,
                "negative_prompt": neg_prompt,
                "init_images": [draft_b64],
                "denoising_strength": adaptive_denoise,
                "steps": 15,
                "width": width,
                "height": height,
                "cfg_scale": cfg_scale,
                "sampler_name": "DPM++ 2M",
                "scheduler": "Karras",
                "seed": actual_seed,
                "alwayson_scripts": alwayson_scripts
            }

            resp2 = requests.post(SD_IMG2IMG_ENDPOINT, json=img2img_payload, timeout=300)
            final_images = resp2.json().get("images", []) if resp2.status_code == 200 else draft_images
            final_cv = decode_image_to_cv2(final_images[0])

        elapsed = time.time() - t0
        mem_end = get_vram_telemetry()

        metadata = {
            "scene_id": scene.get("scene_id", f"scene_{s_num}"),
            "scene_number": s_num,
            "title": s_title,
            "model_checkpoint": self.sd_checkpoint,
            "sampler": "DPM++ 2M Karras",
            "steps_pass1": steps,
            "steps_pass2": 15,
            "denoising_strength": adaptive_denoise,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": actual_seed,
            "positive_prompt": master_prompt,
            "negative_prompt": neg_prompt,
            "native_conditioning": {
                "ip_adapter_model": "ip-adapter_xl [af81326a]",
                "ip_adapter_weight": ip_adapter_weight,
                "freeu_active": True,
                "pag_active": True,
                "cfg_fix_active": True,
                "regional_isolation": regional_meta
            },
            "runtime_s": round(elapsed, 2),
            "peak_vram_mb": mem_end.get("vram_peak_allocated_mb", 0.0)
        }

        return final_cv, metadata


def generate_and_save_scene(scene_num: int, prefix: str = "SOTA_scene_") -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    from metrics import HARDCODED_SCENES, HARDCODED_CHARACTERS, HARDCODED_ENVIRONMENTS, CACHE_FILE
    
    scene = next((s for s in HARDCODED_SCENES if s["scene_number"] == scene_num), None)
    if not scene:
        raise ValueError(f"Scene {scene_num} not found in HARDCODED_SCENES")

    cache_data = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)

    generator = SOTAGenerator()
    img_cv, metadata = generator.render_scene(
        scene=scene,
        characters_db=HARDCODED_CHARACTERS,
        environments_db=HARDCODED_ENVIRONMENTS,
        cache_data=cache_data
    )

    if img_cv is not None:
        out_png = OUTPUT_DIR / f"{prefix}{scene_num}.png"
        out_json = OUTPUT_DIR / f"{prefix}{scene_num}.json"

        cv2.imwrite(str(out_png), img_cv)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[+] Saved {out_png.name} ({img_cv.shape[1]}x{img_cv.shape[0]}) and {out_json.name} in {metadata['runtime_s']}s")

    return img_cv, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=1, help="Scene number to render")
    args = parser.parse_args()
    generate_and_save_scene(args.scene)
