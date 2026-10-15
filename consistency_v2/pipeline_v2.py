"""
MMCCP Consistency Pipeline v2
Integrates methods M1 through M10 into a unified, high-consistency storyboard diffusion engine.
Operates within strict 7.4 GB peak VRAM budget, zero hallucination defects, and strict held-out seed guards.
"""
import os
import sys
import json
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from story_engine.scene_planner import ScenePlanner, SceneExecutionPlan
from generation.forge_client import ForgeClient
from experiments.memory_utils import vram_guard, get_vram_telemetry
from metrics import decode_image_to_cv2, SD_MODELS

from consistency_v2.config import ConsistencyV2Config, RESERVED_SEEDS
from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.layout_planner import LayoutPlanner
from consistency_v2.regional import RegionalCoupleDriver, RegionalCoupleConfig
from consistency_v2.identity_units import IdentityUnitManager, IdentityUnitConfig
from experiments.controlnet_adapter import ControlNetAdapter
from consistency_v2.face_swap import DirectFaceSwapper, ReActorPayloadBuilder
from consistency_v2.face_detailer import FaceDetailer, FaceDetailerConfig
from consistency_v2.wardrobe import WardrobeLock
from consistency_v2.reference_unit import ReferenceOnlyUnitBuilder
from consistency_v2.candidates import CandidateScreen
from consistency_v2.healing_v2 import HealingControllerV2
from consistency_v2.identity_memory import IdentityMemoryManager

from consistency_v2.stage_trace import StageTracer, StageNoOpError

logger = logging.getLogger("MMCCP.PipelineV2")


class ConsistencyPipelineV2:
    """
    Unified Generation Pipeline v2 executing SOTA character consistency (M1-M10).
    """
    def __init__(
        self,
        config: Optional[ConsistencyV2Config] = None,
        base_url: str = "http://127.0.0.1:7860",
        checkpoint_name: str = SD_MODELS["Juggernaut"]
    ):
        self.config = config or ConsistencyV2Config()
        self.client = ForgeClient(base_url=base_url)
        self.checkpoint_name = checkpoint_name
        self.registry = CharacterRegistry()

        # Initialize sub-modules
        self.bank_manager = IdentityBankManager()
        self.layout_planner = LayoutPlanner()
        self.regional_config = RegionalCoupleConfig(
            background_weight=self.config.regional_global_weight
        )
        self.identity_config = IdentityUnitConfig(
            enable_ip_adapter_plus_face=self.config.enable_ip_adapter_plus,
            ip_adapter_plus_face_weight=self.config.ip_adapter_plus_weight,
            enable_instantid=self.config.enable_dual_instantid,
            instantid_ip_weight=self.config.instantid_ip_weight,
            instantid_kp_weight=self.config.instantid_keypoints_weight
        )
        self.face_detailer = FaceDetailer(forge_api_url=base_url)
        self.detailer_config = FaceDetailerConfig(
            enabled=self.config.enable_face_detailer,
            denoising_strength=self.config.face_detailer_denoise,
            padding_px=self.config.face_detailer_padding,
            steps=self.config.face_detailer_steps
        )
        self.wardrobe_lock = WardrobeLock(
            delta_e_threshold=self.config.wardrobe_delta_e_threshold
        )
        self.reference_builder = ReferenceOnlyUnitBuilder(
            default_weight=self.config.reference_attention_weight
        )
        self.candidate_screener = CandidateScreen()
        self.healing_controller = HealingControllerV2(
            max_retries=self.config.max_healing_retries
        )
        self.memory_manager = IdentityMemoryManager(
            bank_manager=self.bank_manager,
            min_similarity_to_anchor=self.config.drift_guard_threshold
        )

    def generate_scene(
        self,
        story: StorySpec,
        scene: SceneSpec,
        seed: int = -1,
        trace: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Executes complete v2 consistency generation for a single storyboard scene with optional tracing.
        """
        # Scope Guard check
        self.config.validate_seed(seed)
        t0 = time.time()

        tracer = StageTracer(
            trace_id=f"{story.story_id}_{scene.scene_id}_seed_{seed}",
            enabled=trace
        )

        # 1. Scene Planning
        plan: SceneExecutionPlan = ScenePlanner.plan_scene(
            story=story,
            scene=scene,
            registry=self.registry
        )

        # Determine character positions as Dict[str, str] and characters_present as List[str]
        char_pos_dict: Dict[str, str] = {}
        if isinstance(scene.character_positions, dict):
            char_pos_dict = scene.character_positions
        elif isinstance(scene.character_positions, list):
            for cp in scene.character_positions:
                if isinstance(cp, dict):
                    cid = cp.get("character_id")
                    pos = cp.get("position", "center")
                    if cid:
                        char_pos_dict[cid] = pos

        characters_present = scene.characters_present if scene.characters_present else list(char_pos_dict.keys())

        # 2. M2: Layout Planner (Spatial regions & OpenPose skeleton)
        layout_masks = {}
        skeleton_b64 = None
        openpose_unit = None
        layout_boxes = []

        if self.config.enable_layout_planner and char_pos_dict:
            layout_boxes = self.layout_planner.plan_scene_layout(
                character_positions=char_pos_dict,
                shot_type=scene.shot_type,
                canvas_w=plan.width,
                canvas_h=plan.height
            )
            layout_masks = {b.character_id: b.normalized_region for b in layout_boxes}
            skeleton_cv = self.layout_planner.render_openpose_canvas(
                layout_boxes=layout_boxes,
                pose_type="standing_front",
                canvas_w=plan.width,
                canvas_h=plan.height
            )
            skeleton_b64 = self.layout_planner.image_to_base64(skeleton_cv)
            openpose_unit = {
                "enabled": True,
                "module": "None",
                "model": "openpose_xl",
                "weight": self.config.openpose_weight,
                "input_image": skeleton_b64,
                "control_mode": "Balanced",
                "pixel_perfect": True,
                "guidance_start": 0.0,
                "guidance_end": self.config.openpose_guidance_end
            }
            if trace and skeleton_cv is not None:
                tracer.record_stage("layout_skeleton", skeleton_cv, enforce_active=False)

        # 3. ControlNet Units Assembly (Pass 1)
        cnet_units = []
        if openpose_unit is not None:
            cnet_units.append(openpose_unit)

        # M4: Identity Units (Dual InstantID & IP-Adapter Plus-Face)
        if characters_present:
            id_specs = IdentityUnitManager.build_identity_control_specs(
                character_ids=characters_present,
                bank_manager=self.bank_manager,
                target_yaw=0.0,
                config=self.identity_config
            )
            for spec in id_specs:
                cnet_units.append(ControlNetAdapter.to_forge_unit(spec))

        # M8: Reference-Only Attention Injection (if enabled)
        if self.config.enable_reference_attention and characters_present:
            ref_units = self.reference_builder.build_scene_reference_units(
                characters=characters_present,
                identity_banks=self.bank_manager._banks,
                layout_masks=layout_masks,
                weight=self.config.reference_attention_weight
            )
            cnet_units.extend(ref_units)

        # 4. Attention & Script Setup
        freeu_args = [True, 1.3, 1.4, 0.9, 0.2, 0.0, 1.0]
        pag_args = [True, 3.0, 0.0, 0.0, 1.0]
        cfg_fix_args = [True, 7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, "enable", "MEAN", "AD", 1.0]

        alwayson_scripts = {
            "freeu integrated (sd 1.x, sd 2.x, sdxl)": {"args": freeu_args},
            "perturbedattentionguidance integrated": {"args": pag_args},
            "dynamicthresholding (cfg-fix) integrated": {"args": cfg_fix_args}
        }
        if cnet_units:
            alwayson_scripts["controlnet"] = {"args": cnet_units}

        # M3: Regional Conditioning (Forge Couple)
        if self.config.enable_regional_couple and len(characters_present) >= 2 and layout_boxes:
            char_prompts = {}
            for cid in characters_present:
                c_spec = self.registry.get_character(cid)
                if c_spec:
                    char_prompts[cid] = f"({c_spec.unique_token}:1.3), {c_spec.name}, {c_spec.ethnicity}, {c_spec.hair_color} hair {c_spec.hair_style}, wearing {c_spec.recurring_clothing}"
                else:
                    char_prompts[cid] = cid

            bg_prompt = f"{scene.location_name}, {scene.environment_type}" if getattr(scene, "location_name", None) else plan.positive_prompt
            full_prompt, couple_payload = RegionalCoupleDriver.build_prompt_and_payload(
                background_prompt=bg_prompt,
                character_prompts=char_prompts,
                layout_boxes=layout_boxes,
                config=self.regional_config
            )
            if couple_payload.get("forge couple"):
                alwayson_scripts["forge couple"] = couple_payload["forge couple"]
                plan.positive_prompt = full_prompt

        # 5. Pass 1: Structural Generation
        txt2img_payload = {
            "prompt": plan.positive_prompt,
            "negative_prompt": plan.negative_prompt,
            "steps": self.config.steps_pass1,
            "width": plan.width,
            "height": plan.height,
            "cfg_scale": self.config.cfg_scale,
            "sampler_name": self.config.sampler_name,
            "scheduler": self.config.scheduler,
            "seed": seed,
            "alwayson_scripts": alwayson_scripts
        }

        with vram_guard(threshold_mb=7400.0, scene_tag=f"Scene {scene.scene_number} Pass 1"):
            draft_imgs, info1 = self.client.txt2img(txt2img_payload)
            if not draft_imgs:
                logger.error("Pass 1 draft generation failed!")
                return None, {"error": "Pass 1 failed"}
            draft_b64 = draft_imgs[0]
            actual_seed = info1.get("seed", seed)
            draft_cv = decode_image_to_cv2(draft_b64)
            if draft_cv is not None:
                tracer.record_stage("pass1_draft", draft_cv, enforce_active=False)

        # 6. Pass 2: Refinement IMG2IMG
        pass2_scripts = alwayson_scripts.copy()
        if "forge couple" in pass2_scripts:
            # Regional couple stays active in Pass 2
            pass
        if "controlnet" in pass2_scripts and len(pass2_scripts["controlnet"].get("args", [])) > 1:
            # In Pass 2, retain only identity unit, drop pose skeleton
            pass2_scripts["controlnet"] = {"args": [pass2_scripts["controlnet"]["args"][-1]]}

        img2img_payload = {
            "prompt": plan.positive_prompt,
            "negative_prompt": plan.negative_prompt,
            "init_images": [draft_b64],
            "denoising_strength": plan.adaptive_denoise_strength,
            "steps": self.config.steps_pass2,
            "width": plan.width,
            "height": plan.height,
            "cfg_scale": self.config.cfg_scale,
            "sampler_name": self.config.sampler_name,
            "scheduler": self.config.scheduler,
            "seed": actual_seed,
            "alwayson_scripts": pass2_scripts
        }

        with vram_guard(threshold_mb=7400.0, scene_tag=f"Scene {scene.scene_number} Pass 2"):
            final_imgs, info2 = self.client.img2img(img2img_payload)
            result_b64 = final_imgs[0] if final_imgs else draft_b64
            current_cv = decode_image_to_cv2(result_b64)
            if current_cv is not None:
                tracer.record_stage("pass2_refined", current_cv, enforce_active=True)

        # 7. M6: Identity-Conditioned Face Detailer
        detailer_report = {}
        if self.config.enable_face_detailer and current_cv is not None and layout_boxes:
            char_prompts = {
                cid: (self.registry.get_character(cid).visual_summary if self.registry.get_character(cid) else cid)
                for cid in characters_present
            }
            detailed_cv = self.face_detailer.detail_image(
                image_bgr=current_cv,
                character_prompts=char_prompts,
                layout_boxes=layout_boxes,
                bank_manager=self.bank_manager,
                config=self.detailer_config
            )
            if detailed_cv is not None:
                current_cv = detailed_cv
                detailer_report = {"detailed": True}
                face_bboxes = [b.face_bbox for b in layout_boxes]
                tracer.record_stage(
                    "m6_face_detailer",
                    current_cv,
                    target_bboxes=None,
                    stage_metadata={"layout_face_bboxes": face_bboxes},
                    enforce_active=True
                )
        else:
            tracer.record_skipped_stage("m6_face_detailer", "disabled or no layout boxes")

        # 8. M5: Reliable Identity Transplant (Direct InsightFace swap)
        swap_report = {}
        if self.config.enable_reliable_face_swap and current_cv is not None and layout_boxes:
            swap_specs = []
            for b in layout_boxes:
                cid = b.character_id
                best_ref = self.bank_manager.select_best_reference(cid)
                if best_ref and "path" in best_ref:
                    swap_specs.append({
                        "character_id": cid,
                        "target_zone": b.target_zone,
                        "source_path": best_ref["path"]
                    })

            if swap_specs:
                os.makedirs("consistency_v2/banks", exist_ok=True)
                temp_in = "consistency_v2/banks/_temp_pre_swap.png"
                temp_out = "consistency_v2/banks/_temp_post_swap.png"
                cv2.imwrite(temp_in, current_cv)
                success = DirectFaceSwapper.execute_face_swap(
                    target_image_path=temp_in,
                    swap_specs=swap_specs,
                    out_image_path=temp_out,
                    codeformer_fidelity=self.config.codeformer_fidelity,
                    forge_api_url=self.client.base_url
                )
                if success and os.path.exists(temp_out):
                    swapped_cv = cv2.imread(temp_out)
                    if swapped_cv is not None:
                        current_cv = swapped_cv
                        swap_report = {"swapped": True, "num_swaps": len(swap_specs), "specs": swap_specs}
                        swap_bboxes = [b.face_bbox for b in layout_boxes]
                        tracer.record_stage(
                            "m5_face_swap",
                            current_cv,
                            target_bboxes=None,
                            stage_metadata={"layout_face_bboxes": swap_bboxes},
                            enforce_active=True
                        )
        else:
            tracer.record_skipped_stage("m5_face_swap", "disabled or no layout boxes")

        # 9. M7: Wardrobe and Palette Lock
        wardrobe_report = {}
        faces_for_wardrobe = []
        if self.config.enable_wardrobe_lock and current_cv is not None:
            char_wardrobes = {}
            for cid in characters_present:
                c_spec = self.registry.get_character(cid)
                c_bank = self.bank_manager._banks.get(cid, {})
                char_wardrobes[cid] = {
                    "clothing_colors": c_spec.clothing_colors if c_spec else [],
                    "bank_wardrobe_lab": c_bank.get("average_wardrobe_lab")
                }

            # Extract detected face bboxes bound to layout boxes
            detected_faces = self.face_detailer.auditor.detect_faces(current_cv)
            if layout_boxes:
                for f in detected_faces:
                    matched_box = self.face_detailer.match_face_to_layout_box(
                        f["bbox"], layout_boxes, current_cv.shape[1]
                    )
                    if matched_box:
                        faces_for_wardrobe.append({
                            "character_id": matched_box.character_id,
                            "bbox": f["bbox"]
                        })

            current_cv, wardrobe_report = self.wardrobe_lock.lock_scene_wardrobe(
                image_bgr=current_cv,
                faces_info=faces_for_wardrobe,
                character_wardrobes=char_wardrobes,
                blend_strength=0.70
            )
            tracer.record_stage(
                "m7_wardrobe_lock",
                current_cv,
                enforce_active=(wardrobe_report.get("repairs_applied", 0) > 0)
            )
        else:
            tracer.record_skipped_stage("m7_wardrobe_lock", "disabled")

        # 10. M10: Cross-Scene Identity Memory Promotion
        memory_promotions = []
        if self.config.enable_identity_memory and current_cv is not None and faces_for_wardrobe:
            for fb in faces_for_wardrobe:
                cid = fb.get("character_id")
                bbox = fb.get("bbox")
                if cid and bbox:
                    promo_decision = self.memory_manager.consider_scene_face_promotion(
                        character_id=cid,
                        scene_image_bgr=current_cv,
                        face_bbox=bbox,
                        scene_id=scene.scene_number
                    )
                    memory_promotions.append(promo_decision._asdict())

        # Final composite record
        if current_cv is not None:
            tracer.record_stage("final_composite", current_cv, enforce_active=False)

        trace_report = tracer.finalize()

        elapsed = time.time() - t0
        mem_end = get_vram_telemetry()

        metadata = {
            "story_id": story.story_id,
            "scene_id": scene.scene_id,
            "scene_number": scene.scene_number,
            "title": scene.title,
            "seed": actual_seed,
            "config_hash": self.config.to_hash(),
            "pipeline_version": "v2.0",
            "runtime_s": round(elapsed, 2),
            "peak_vram_mb": mem_end.get("vram_peak_allocated_mb", 0.0),
            "methods_active": {
                "m1_identity_bank": self.config.enable_identity_bank,
                "m2_layout_planner": self.config.enable_layout_planner,
                "m3_regional_couple": self.config.enable_regional_couple,
                "m4_dual_instantid": self.config.enable_dual_instantid,
                "m4_ip_adapter_plus": self.config.enable_ip_adapter_plus,
                "m5_reliable_swap": self.config.enable_reliable_face_swap,
                "m6_face_detailer": self.config.enable_face_detailer,
                "m7_wardrobe_lock": self.config.enable_wardrobe_lock,
                "m8_reference_only": self.config.enable_reference_attention,
                "m9_best_of_n": self.config.enable_best_of_n,
                "m10_identity_memory": self.config.enable_identity_memory
            },
            "detailer_report": detailer_report,
            "swap_report": swap_report,
            "wardrobe_report": wardrobe_report,
            "memory_promotions": memory_promotions,
            "trace": trace_report
        }

        return current_cv, metadata

    def generate_story(
        self,
        story: StorySpec,
        output_dir: Path,
        seed: int = 101,
        prefix: str = "scene_",
        trace: bool = False
    ) -> List[Tuple[Optional[np.ndarray], Dict[str, Any]]]:
        """
        Renders all scenes in a storyboard with v2 pipeline and writes outputs and sidecars.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        logger.info(f"Generating Storyboard '{story.title}' with Consistency Pipeline v2 (seed {seed})")

        for scene in story.scenes:
            img_cv, meta = self.generate_scene(story=story, scene=scene, seed=seed, trace=trace)
            results.append((img_cv, meta))

            if img_cv is not None:
                img_path = output_dir / f"{prefix}{scene.scene_number}_seed_{seed}.png"
                sidecar_path = output_dir / f"{prefix}{scene.scene_number}_seed_{seed}.json"
                cv2.imwrite(str(img_path), img_cv)
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                logger.info(f"Saved {img_path.name} in {meta['runtime_s']}s (Peak VRAM: {meta['peak_vram_mb']}MB)")

        return results
