"""
MMCCP Consistency v2: Method M3 (True Regional Conditioning)
Drives Forge Couple extension via API to eliminate character-to-character cross-attention bleed
and bind character phenotypes and wardrobes to designated spatial columns/boxes.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from consistency_v2.layout_planner import CharacterLayoutBox

logger = logging.getLogger("MMCCP.Regional")


@dataclass
class RegionalCoupleConfig:
    """Configuration for Forge Couple spatial attention conditioning."""
    enabled: bool = True
    compatibility: bool = True
    mode: str = "Basic"  # "Basic" or "Advanced"
    direction: str = "Horizontal"  # "Horizontal" or "Vertical"
    background: str = "First Line"  # "First Line", "Last Line", or "None"
    background_weight: float = 0.45  # Weight of shared scene background (0.1 to 1.0)
    separator: str = "\n"
    syntax: str = "{ }"
    debug: bool = False
    include_def_in_prompt: bool = True


class RegionalCoupleDriver:
    """
    Translates character layout boxes and character-specific prompts into Forge Couple API payloads.
    Provides strict spatial attention isolation between distinct characters.
    """

    @classmethod
    def build_prompt_and_payload(
        cls,
        background_prompt: str,
        character_prompts: Dict[str, str],
        layout_boxes: List[CharacterLayoutBox],
        config: Optional[RegionalCoupleConfig] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Builds the partitioned multi-line prompt and Forge Couple alwayson_scripts dictionary.

        Args:
            background_prompt: Shared environment, lighting, art style, and quality tokens.
            character_prompts: Mapping of character_id to per-character prompt (phenotype + clothing).
            layout_boxes: Planned spatial layout boxes from LayoutPlanner (M2).
            config: RegionalCoupleConfig instance (defaults to standard settings).

        Returns:
            Tuple of:
              - full_prompt (str): multi-line prompt formatted for Forge Couple
              - alwayson_scripts_entry (dict): payload fragment to merge into alwayson_scripts
        """
        if config is None:
            config = RegionalCoupleConfig()

        active_chars = [b.character_id for b in layout_boxes if b.character_id in character_prompts]

        # Solo character or disabled -> no regional coupling required
        if not config.enabled or len(active_chars) <= 1:
            char_text = " ".join([character_prompts[c] for c in active_chars if c in character_prompts])
            combined = f"{background_prompt}, {char_text}" if char_text else background_prompt
            return combined.strip(", "), {}

        # Sort layout boxes horizontally from left to right
        def get_x_center(box: CharacterLayoutBox) -> float:
            if box.normalized_region and len(box.normalized_region) >= 4:
                return (box.normalized_region[2] + box.normalized_region[3]) / 2.0
            return float(box.body_bbox.get("x", 0))

        sorted_boxes = sorted(
            [b for b in layout_boxes if b.character_id in character_prompts],
            key=get_x_center
        )

        # Build prompt lines: Line 0 = background, Line 1..N = character regions
        prompt_lines = [background_prompt.strip()]
        mapping_boxes = []

        for b in sorted_boxes:
            c_prompt = character_prompts[b.character_id].strip()
            prompt_lines.append(c_prompt)
            # Normalized box: [x_min, x_max, y_min, y_max, weight]
            norm_box = [
                round(b.normalized_region[2], 3),  # x1
                round(b.normalized_region[3], 3),  # x2
                round(b.normalized_region[0], 3),  # y1
                round(b.normalized_region[1], 3),  # y2
                1.0                                # weight
            ]
            mapping_boxes.append(norm_box)

        full_prompt = config.separator.join(prompt_lines)

        # 17-element argument contract for sd-forge-couple
        fc_args = [
            True,                          # 0: Enable
            config.compatibility,          # 1: Compatibility
            config.mode,                   # 2: Region Assignment ("Basic" or "Advanced")
            config.separator,              # 3: Couple Separator ("\n")
            config.direction,              # 4: Tile Direction ("Horizontal")
            config.background,             # 5: Global Effect ("First Line")
            float(config.background_weight), # 6: Global Effect Weight (0.45)
            mapping_boxes if config.mode == "Advanced" else [
                [0.0, 0.5, 0.0, 1.0, 1.0], [0.5, 1.0, 0.0, 1.0, 1.0]
            ],                             # 7: Mapping
            config.syntax,                 # 8: Syntax ("{ }")
            config.debug,                  # 9: Debug (False)
            config.include_def_in_prompt,  # 10: Include Definitions in Prompt (True)
            None, None, None, None, None, None  # 11-16: Extra mask slots
        ]

        logger.info(
            f"RegionalCoupleDriver: Configured {len(sorted_boxes)} regional tiles in {config.mode} mode "
            f"with background weight {config.background_weight}."
        )

        return full_prompt, {"forge couple": {"args": fc_args}}
