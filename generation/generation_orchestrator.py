"""
STAGE 22: CANONICAL GENERATION ORCHESTRATOR
Unified SOTA story-driven diffusion engine executing native identity conditioning, ControlNet, and regional isolation.
"""
import os, sys, json, time, logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from story_engine.scene_planner import ScenePlanner, SceneExecutionPlan
from identity.ipadapter import IPAdapterSDXLConditioner
from identity.instantid import InstantIDConditioner
from identity.faceid import FaceIDPlusV2Conditioner
from conditioning.controlnet_conditioner import ControlNetConditioner
from conditioning.refinement import RefinementController
from generation.forge_client import ForgeClient
from experiments.memory_utils import vram_guard, get_vram_telemetry
from metrics import decode_image_to_cv2, decode_image_to_pil, SD_MODELS

logger = logging.getLogger("MMCCP.GenerationOrchestrator")


class GenerationOrchestrator:
    """
    Canonical, story-agnostic diffusion orchestrator executing the validated Phase 8.1 SOTA pipeline.
    """

    def __init__(
        self,
        checkpoint_name: str = SD_MODELS["Juggernaut"],
        identity_engine: str = "ip_adapter",  # "ip_adapter", "instantid", "faceid"
        base_url: str = "http://127.0.0.1:7860"
    ):
        self.checkpoint_name = checkpoint_name
        self.identity_engine_type = identity_engine
        self.client = ForgeClient(base_url=base_url)
        self.registry = CharacterRegistry()

        # Initialize identity conditioner
        if identity_engine == "instantid":
            self.identity_conditioner = InstantIDConditioner()
        elif identity_engine == "faceid":
            self.identity_conditioner = FaceIDPlusV2Conditioner()
        else:
            self.identity_conditioner = IPAdapterSDXLConditioner()

    def generate_scene(
        self,
        story: StorySpec,
        scene: SceneSpec,
        steps_pass1: int = 25,
        steps_pass2: int = 15,
        cfg_scale: float = 5.5,
        seed: int = -1,
        enable_openpose: bool = False,
        enable_depth: bool = False,
        style_override: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Executes complete two-pass SOTA generation for an arbitrary scene.
        """
        t0 = time.time()

        # 1. Plan Scene
        plan: SceneExecutionPlan = ScenePlanner.plan_scene(
            story=story,
            scene=scene,
            registry=self.registry,
            style=style_override
        )

        # 2. Build ControlNet Specs (Identity + Optional OpenPose / Depth)
        control_specs = self.identity_conditioner.build_control_specs(plan)

        if enable_openpose:
            op_spec = ControlNetConditioner.create_openpose_spec()
            if op_spec:
                control_specs.append(op_spec)

        if enable_depth:
            depth_spec = ControlNetConditioner.create_depth_spec()
            if depth_spec:
                control_specs.append(depth_spec)

        cnet_payload = ControlNetConditioner.build_payload(control_specs)

        # 3. Core Native U-Net Attention Mechanisms
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

        # 4. Pass 1: TXT2IMG Structural Draft
        txt2img_payload = {
            "prompt": plan.positive_prompt,
            "negative_prompt": plan.negative_prompt,
            "steps": steps_pass1,
            "width": plan.width,
            "height": plan.height,
            "cfg_scale": cfg_scale,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "seed": seed,
            "alwayson_scripts": alwayson_scripts
        }

        with vram_guard(threshold_mb=7400.0, scene_tag=f"Scene {scene.scene_number}"):
            draft_imgs, info1 = self.client.txt2img(txt2img_payload)
            if not draft_imgs:
                logger.error("Pass 1 draft generation failed!")
                return None, {"error": "Pass 1 failed"}

            draft_b64 = draft_imgs[0]
            actual_seed = info1.get("seed", seed)

            # 5. Pass 2: Adaptive Denoising IMG2IMG with ReActor + CodeFormer
            if plan.reactor_composite_b64:
                reactor_payload = RefinementController.build_reactor_payload(
                    source_image_b64=plan.reactor_composite_b64,
                    face_index_str=plan.face_routing_index,
                    codeformer_weight=0.80
                )
                alwayson_scripts["reactor"] = reactor_payload["reactor"]

            pass2_scripts = alwayson_scripts.copy()
            # In Pass 2 img2img, do not rerun spatial pose/depth preprocessors; retain only identity unit
            if "controlnet" in pass2_scripts and len(pass2_scripts["controlnet"].get("args", [])) > 1:
                pass2_scripts["controlnet"] = {"args": [pass2_scripts["controlnet"]["args"][0]]}

            img2img_payload = {
                "prompt": plan.positive_prompt,
                "negative_prompt": plan.negative_prompt,
                "init_images": [draft_b64],
                "denoising_strength": plan.adaptive_denoise_strength,
                "steps": steps_pass2,
                "width": plan.width,
                "height": plan.height,
                "cfg_scale": cfg_scale,
                "sampler_name": "DPM++ 2M",
                "scheduler": "Karras",
                "seed": actual_seed,
                "alwayson_scripts": pass2_scripts
            }

            final_imgs, info2 = self.client.img2img(img2img_payload)
            result_b64 = final_imgs[0] if final_imgs else draft_b64
            final_cv = decode_image_to_cv2(result_b64)

        elapsed = time.time() - t0
        mem_end = get_vram_telemetry()

        metadata = {
            "story_id": story.story_id,
            "scene_id": scene.scene_id,
            "scene_number": scene.scene_number,
            "title": scene.title,
            "model_checkpoint": self.checkpoint_name,
            "identity_engine": self.identity_conditioner.get_adapter_name(),
            "sampler": "DPM++ 2M Karras",
            "steps_pass1": steps_pass1,
            "steps_pass2": steps_pass2,
            "denoising_strength": plan.adaptive_denoise_strength,
            "cfg_scale": cfg_scale,
            "width": plan.width,
            "height": plan.height,
            "seed": actual_seed,
            "positive_prompt": plan.positive_prompt,
            "negative_prompt": plan.negative_prompt,
            "native_conditioning": {
                "identity_adapter": self.identity_conditioner.get_adapter_name(),
                "openpose_active": enable_openpose,
                "depth_active": enable_depth,
                "freeu_active": True,
                "pag_active": True,
                "cfg_fix_active": True,
                "regional_isolation": plan.regional_meta
            },
            "runtime_s": round(elapsed, 2),
            "peak_vram_mb": mem_end.get("vram_peak_allocated_mb", 0.0)
        }

        return final_cv, metadata

    def generate_story(
        self,
        story: StorySpec,
        output_dir: Path = Path("edu_outputs/images"),
        prefix: str = "scene_"
    ) -> List[Tuple[Optional[np.ndarray], Dict[str, Any]]]:
        """
        Generates all scenes in a story sequentially with metadata logging.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        print(f"\n{'='*90}")
        print(f"  GENERATING STORY: '{story.title}' ({len(story.scenes)} scenes)")
        print(f"{'='*90}")

        for scene in story.scenes:
            print(f"\n  >>> Rendering Scene {scene.scene_number}: {scene.title}")
            img_cv, meta = self.generate_scene(story=story, scene=scene)
            results.append((img_cv, meta))

            if img_cv is not None:
                img_path = output_dir / f"{prefix}{scene.scene_number}.png"
                json_path = output_dir / f"{prefix}{scene.scene_number}.json"
                cv2.imwrite(str(img_path), img_cv)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                print(f"      [SAVED] {img_path.name} ({img_cv.shape[1]}x{img_cv.shape[0]}) in {meta['runtime_s']}s")

        return results
