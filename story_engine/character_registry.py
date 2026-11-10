"""
STAGE 7: CHARACTER REGISTRY
Maintains canonical references, age-state references, and ArcFace/InsightFace embeddings for arbitrary characters.
"""
import os, sys, json, logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import CharacterSpec, AgeState, StorySpec
from metrics import (
    decode_image_to_cv2,
    encode_image_to_base64,
    calculate_max_match_arcface,
    CACHE_FILE
)

logger = logging.getLogger("MMCCP.CharacterRegistry")


class CharacterRegistry:
    """
    Centralized registry managing character references, age-specific anchors, and biometric embeddings.
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache_data: Dict[str, str] = {}  # key -> base64 string
        self.load_cache()

    def load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load character cache: {e}")
                self.cache_data = {}

    def save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save character cache: {e}")

    def get_anchor_b64(self, char_id: str, age_state_key: Optional[str] = None) -> Optional[str]:
        """Retrieves base64 anchor image for a character and optional age state."""
        key = f"{char_id}_{age_state_key}" if age_state_key else char_id
        if key in self.cache_data:
            return self.cache_data[key]
        if char_id in self.cache_data:
            return self.cache_data[char_id]
        return None

    def register_anchor(self, char_id: str, image_b64: str, age_state_key: Optional[str] = None):
        """Registers a base64 anchor image."""
        key = f"{char_id}_{age_state_key}" if age_state_key else char_id
        self.cache_data[key] = image_b64
        self.save_cache()

    def get_character(self, char_id: str) -> Optional[CharacterSpec]:
        """Retrieves CharacterSpec for a given character ID from storyboards."""
        if not hasattr(self, "_char_specs"):
            self._char_specs: Dict[str, CharacterSpec] = {}
            for s_file in [
                Path(r"D:\5TH SEM\HACKATHON\storyboards\ebright.json"),
                Path(r"D:\5TH SEM\HACKATHON\storyboards\generalization_benchmark.json")
            ]:
                if s_file.exists():
                    try:
                        with open(s_file, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        for cid, cdata in d.get("characters", {}).items():
                            if isinstance(cdata, dict):
                                self._char_specs[cid] = CharacterSpec.from_dict(cdata)
                    except Exception:
                        pass
        return self._char_specs.get(char_id)

    def build_anchor_prompt(self, char: CharacterSpec, age_state: Optional[AgeState] = None, style: str = "Cinematic Realistic") -> Tuple[str, str]:
        """
        Generates strict demographic prompt and negative prompt for synthesizing a visual anchor.
        """
        age = age_state.age_years if age_state else "25"
        stage = age_state.stage_name if age_state else "adult"
        hair_style = age_state.hairstyle if age_state and age_state.hairstyle else char.hair_style
        clothing = age_state.clothing_description if age_state and age_state.clothing_description else char.recurring_clothing

        demo = f"((strictly {char.ethnicity}:1.45)), ({char.skin_tone} skin tone:1.35), ({char.face_shape} face shape:1.25), ({char.eye_color} eyes:1.2)"
        hair = f"({char.hair_color} hair, {hair_style}:1.3)"
        outfit = f"wearing ({clothing}:1.3)" if clothing else "wearing casual clothes"

        pos_prompt = (
            f"({style}:1.2), masterpiece, 8k resolution, raw photo, highly detailed, "
            f"a close-up portrait of {char.name}, {age} years old {stage} {char.gender}, "
            f"{demo}, {hair}, {outfit}, neutral white studio background, studio portrait lighting, sharp focus, clean skin texture."
        )

        neg_prompt = (
            f"blurry, deformed, bad anatomy, bad hands, missing fingers, extra fingers, ugly, watermark, text, signature, "
            f"opposite gender, wrong ethnicity, cartoon, illustration, low quality"
        )

        # Age-specific negative prompt constraints
        try:
            age_int = int(age)
            if age_int < 12:
                neg_prompt += ", teenager, adult, mature face, beard, mustache, stubble, deep voice, wrinkles, clean-shaven man"
            elif age_int < 18:
                neg_prompt += ", child, toddler, old man, mature adult, full beard, mustache, wrinkles"
            else:
                neg_prompt += ", child, baby, boy, kid, toddler, adolescent"
        except Exception:
            pass

        return pos_prompt, neg_prompt


if __name__ == "__main__":
    reg = CharacterRegistry()
    print("Character Registry Initialized. Cached keys count:", len(reg.cache_data))
