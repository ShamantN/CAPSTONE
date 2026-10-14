"""
STAGE 20: GENERALIZED PROMPT BUILDER
Dynamically constructs positive, negative, and regional prompt strings from StorySpec and SceneSpec.
"""
from typing import Tuple, Dict, Any, List, Optional
from story_engine.schema import StorySpec, SceneSpec, CharacterSpec, AgeState
from story_engine.character_registry import CharacterRegistry


class PromptBuilder:
    """
    Builds structured, demographic-locked, and regionally-isolated prompts without hardcoded story dependencies.
    """

    @staticmethod
    def build_scene_prompts(
        story: StorySpec,
        scene: SceneSpec,
        registry: CharacterRegistry,
        style: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any], Tuple[int, int]]:
        """
        Returns (master_positive_prompt, master_negative_prompt, regional_meta, (width, height)).
        """
        active_style = style or story.visual_style
        char_ids = scene.characters_present
        num_chars = len(char_ids)

        # 1. Determine Aspect Ratio (Widescreen for multi-person, 1:1 for solo)
        shot_lower = scene.shot_type.lower()
        is_widescreen = num_chars >= 2 or "two-shot" in shot_lower or "wide" in shot_lower
        width = 1216 if is_widescreen else 1024
        height = 832 if is_widescreen else 1024

        # 2. Build Environmental Description
        env_desc = f"{scene.location_name}, {scene.environment_type} environment, {scene.lighting}"
        shot_info = f"{scene.shot_type}, {scene.camera_angle}"

        # 3. Build Character Conditioning Blocks
        if num_chars == 0:
            # Environment / Abstract scene
            master_pos = f"({active_style}:1.2), masterpiece, cinematic lighting, 8k resolution, {scene.action_description} BREAK Background: {env_desc}, {shot_info}."
            regional_meta = {"regional_active": False}
        elif num_chars == 1:
            # Single Character Scene
            c_id = char_ids[0]
            char = story.characters.get(c_id)
            if char:
                age_key = scene.character_age_states.get(c_id, "primary")
                age_state = char.age_states.get(age_key)
                age_val = age_state.age_years if age_state else "25"
                stage_val = age_state.stage_name if age_state else "adult"

                # Check for scene-specific outfit override vs character default
                outfit = scene.character_outfits.get(c_id)
                if not outfit:
                    outfit = age_state.clothing_description if age_state and age_state.clothing_description else char.recurring_clothing

                demo_str = f"((strictly {char.ethnicity}:1.45)), ({char.skin_tone} skin tone:1.35), ({char.face_shape} face:1.25)"
                color_tokens = " and ".join([f"({col} garment:1.30)" for col in char.clothing_colors[:2]]) if char.clothing_colors else ""

                char_desc = (
                    f"(({char.unique_token}:1.4)), centered in frame, {demo_str}, {age_val} years old {stage_val} {char.gender}, "
                    f"wearing ({outfit}:1.30), {color_tokens}, {char.hair_color} hair ({char.hair_style}), clean face, detailed eyes"
                )
            else:
                char_desc = f"a person in the scene"

            master_pos = f"({active_style}:1.2), masterpiece, cinematic lighting, 8k resolution, highly detailed, {scene.action_description} BREAK {char_desc} BREAK Background: {env_desc}, {shot_info}."
            regional_meta = {"regional_active": False}

        else:
            # Multi-Character Scene: Regional Spatial Column Isolation with BREAK tokens
            regional_blocks = []
            column_ratios = []

            for idx, c_id in enumerate(char_ids):
                char = story.characters.get(c_id)
                if char:
                    # Determine position from scene spec or default left/right
                    pos_str = scene.character_positions.get(c_id)
                    if not pos_str:
                        pos_str = "sitting on the left, left column" if idx == 0 else "sitting on the right, right column"
                    else:
                        pos_str = f"{pos_str}, {'left' if idx==0 else 'right'} column"

                    age_key = scene.character_age_states.get(c_id, "primary")
                    age_state = char.age_states.get(age_key)
                    age_val = age_state.age_years if age_state else "25"
                    stage_val = age_state.stage_name if age_state else "adult"

                    outfit = scene.character_outfits.get(c_id)
                    if not outfit:
                        outfit = age_state.clothing_description if age_state and age_state.clothing_description else char.recurring_clothing

                    demo_str = f"((strictly {char.ethnicity}:1.45)), ({char.skin_tone} skin tone:1.35), ({char.face_shape} face:1.25)"
                    color_tokens = " and ".join([f"({col} garment:1.30)" for col in char.clothing_colors[:2]]) if char.clothing_colors else ""

                    c_desc = (
                        f"(({char.unique_token}:1.4)), {pos_str}, {demo_str}, {age_val} years old {stage_val} {char.gender}, "
                        f"wearing ({outfit}:1.30), {color_tokens}, {char.hair_color} hair, clean face, detailed eyes"
                    )
                    regional_blocks.append(c_desc)
                    column_ratios.append("1")

            regional_char_string = " BREAK ".join(regional_blocks)
            master_pos = f"({active_style}:1.2), masterpiece, cinematic lighting, 8k resolution, highly detailed, {scene.action_description} BREAK {regional_char_string} BREAK Background: {env_desc}, {shot_info}."
            regional_meta = {
                "regional_active": True,
                "num_regions": num_chars,
                "mode": "Columns",
                "ratios": ",".join(column_ratios),
                "base_ratio": 0.20
            }

        # 4. Master Negative Prompt Construction
        neg_parts = [story.global_negative_prompt]
        if scene.negative_prompt_additions:
            neg_parts.append(scene.negative_prompt_additions)
        if scene.forbidden_props:
            neg_parts.append(", ".join([f"no {p}" for p in scene.forbidden_props]))

        # Multi-person specific negative guards
        if num_chars >= 2:
            neg_parts.append("solo, 1person, single person alone, identical faces, twin, clone, demographic bleed, merged bodies")

        master_neg = ", ".join([p for p in neg_parts if p.strip()])

        return master_pos, master_neg, regional_meta, (width, height)
