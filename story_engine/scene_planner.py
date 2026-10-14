"""
STAGE 21: SCENE PLANNER
Plans generation parameters, resolves character anchors, and configures conditioning requirements per scene.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry


@dataclass
class SceneExecutionPlan:
    scene_number: int
    scene_id: str
    title: str
    width: int
    height: int
    positive_prompt: str
    negative_prompt: str
    regional_meta: Dict[str, Any]
    character_anchors: List[str]
    adaptive_denoise_strength: float
    ip_adapter_weight: float
    face_routing_index: str
    reactor_composite_b64: Optional[str]
    control_modes: List[str]  # e.g. ["ip_adapter", "openpose"]
    qa_requirements: Dict[str, Any]


class ScenePlanner:
    """
    Translates abstract SceneSpecs into concrete SceneExecutionPlans ready for diffusion dispatch.
    """

    @staticmethod
    def compute_adaptive_denoising(shot_type: str, base_denoise: float = 0.48) -> float:
        """Dynamically computes Pass 2 img2img denoising schedule based on camera distance."""
        st = shot_type.lower()
        if "extreme close-up" in st or "macro" in st:
            return round(base_denoise * 0.70, 3)  # ~0.33
        elif "close-up" in st or "portrait" in st or "hero" in st:
            return round(base_denoise * 0.80, 3)  # ~0.38
        elif "medium close-up" in st:
            return round(base_denoise * 0.90, 3)  # ~0.43
        elif "medium shot" in st or "two-shot" in st:
            return round(base_denoise * 1.05, 3)  # ~0.50
        elif "wide shot" in st or "full shot" in st or "long shot" in st:
            return round(base_denoise * 1.15, 3)  # ~0.55
        elif "extreme wide shot" in st:
            return round(base_denoise * 1.25, 3)  # ~0.60
        return base_denoise

    @classmethod
    def plan_scene(
        cls,
        story: StorySpec,
        scene: SceneSpec,
        registry: CharacterRegistry,
        style: Optional[str] = None
    ) -> SceneExecutionPlan:
        from story_engine.prompt_builder import PromptBuilder
        from metrics import create_composite_face_image

        pos_prompt, neg_prompt, reg_meta, (w, h) = PromptBuilder.build_scene_prompts(
            story=story,
            scene=scene,
            registry=registry,
            style=style
        )

        char_ids = scene.characters_present
        char_anchors: List[str] = []
        for c_id in char_ids:
            age_key = scene.character_age_states.get(c_id)
            anchor = registry.get_anchor_b64(c_id, age_key)
            if anchor:
                char_anchors.append(anchor)

        denoise = cls.compute_adaptive_denoising(scene.shot_type)

        # Multi-character vs Solo conditioning parameters
        control_modes = ["ip_adapter"]
        if len(char_anchors) >= 2:
            ip_weight = 0.35  # Reduced to avoid cross-character ethnic bleed
            face_routing = "0,1"
            reactor_b64 = create_composite_face_image(char_anchors)
        elif len(char_anchors) == 1:
            ip_weight = 0.85
            face_routing = "0"
            reactor_b64 = char_anchors[0]
        else:
            ip_weight = 0.0
            face_routing = "0"
            reactor_b64 = None
            control_modes = []

        return SceneExecutionPlan(
            scene_number=scene.scene_number,
            scene_id=scene.scene_id,
            title=scene.title,
            width=w,
            height=h,
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            regional_meta=reg_meta,
            character_anchors=char_anchors,
            adaptive_denoise_strength=denoise,
            ip_adapter_weight=ip_weight,
            face_routing_index=face_routing,
            reactor_composite_b64=reactor_b64,
            control_modes=control_modes,
            qa_requirements=scene.qa_requirements
        )
