"""
STAGE 5: GENERALIZED STORY SCHEMA
Typed dataclasses defining universal schema for stories, characters, age-states, scenes, and props.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class AgeState:
    age_years: str
    stage_name: str
    appearance_modifications: List[str] = field(default_factory=list)
    hairstyle: str = ""
    clothing_description: str = ""
    reference_image_b64: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgeState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CharacterSpec:
    id: str
    name: str
    unique_token: str
    gender: str
    ethnicity: str
    skin_tone: str
    face_shape: str
    hair_color: str
    hair_style: str
    eye_color: str
    build: str = "average"
    distinctive_features: List[str] = field(default_factory=list)
    recurring_clothing: str = ""
    clothing_colors: List[str] = field(default_factory=list)
    age_states: Dict[str, AgeState] = field(default_factory=dict)
    canonical_reference_b64: Optional[str] = None
    canonical_embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["age_states"] = {k: v.to_dict() if isinstance(v, AgeState) else v for k, v in self.age_states.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterSpec":
        raw = data.copy()
        raw_age_states = raw.get("age_states", {})
        age_states = {}
        for k, v in raw_age_states.items():
            age_states[k] = AgeState.from_dict(v) if isinstance(v, dict) else v
        raw["age_states"] = age_states
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class PropSpec:
    id: str
    name: str
    description: str
    category: str = "tool"  # tool, furniture, natural, scientific, regalia
    clip_prompt: str = ""
    reference_b64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SceneSpec:
    scene_id: str
    scene_number: int
    title: str
    location_name: str
    environment_type: str  # indoor, outdoor, lab, stage, etc.
    lighting: str
    shot_type: str  # Close-up, Medium shot, Wide two-shot, Over-the-shoulder, Hero shot
    camera_angle: str  # Eye level, Low angle, High angle
    action_description: str
    characters_present: List[str] = field(default_factory=list)
    character_positions: Dict[str, str] = field(default_factory=dict)  # {"char_id": "left", "char_id2": "right"}
    character_age_states: Dict[str, str] = field(default_factory=dict)  # {"char_id": "child"}
    character_outfits: Dict[str, str] = field(default_factory=dict)  # {"char_id": "custom outfit for scene"}
    required_props: List[str] = field(default_factory=list)
    forbidden_props: List[str] = field(default_factory=list)
    negative_prompt_additions: str = ""
    qa_requirements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StorySpec:
    story_id: str
    title: str
    visual_style: str = "Cinematic Realistic"
    global_negative_prompt: str = "blurry, deformed, bad anatomy, bad hands, missing fingers, extra fingers, ugly, watermark, text, signature"
    characters: Dict[str, CharacterSpec] = field(default_factory=dict)
    scenes: List[SceneSpec] = field(default_factory=list)
    props: Dict[str, PropSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "visual_style": self.visual_style,
            "global_negative_prompt": self.global_negative_prompt,
            "characters": {k: v.to_dict() if isinstance(v, CharacterSpec) else v for k, v in self.characters.items()},
            "scenes": [s.to_dict() if isinstance(s, SceneSpec) else s for s in self.scenes],
            "props": {k: v.to_dict() if isinstance(v, PropSpec) else v for k, v in self.props.items()},
            "metadata": self.metadata
        }

    def save_json(self, file_path: str):
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "StorySpec":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorySpec":
        raw = data.copy()
        raw_chars = raw.get("characters", {})
        chars = {}
        for k, v in raw_chars.items():
            chars[k] = CharacterSpec.from_dict(v) if isinstance(v, dict) else v

        raw_scenes = raw.get("scenes", [])
        scenes = [SceneSpec.from_dict(s) if isinstance(s, dict) else s for s in raw_scenes]

        raw_props = raw.get("props", {})
        props = {}
        for k, v in raw_props.items():
            props[k] = PropSpec.from_dict(v) if isinstance(v, dict) else v

        return cls(
            story_id=raw.get("story_id", "story_default"),
            title=raw.get("title", "Untitled Story"),
            visual_style=raw.get("visual_style", "Cinematic Realistic"),
            global_negative_prompt=raw.get("global_negative_prompt", "blurry, deformed, bad anatomy, bad hands, ugly"),
            characters=chars,
            scenes=scenes,
            props=props,
            metadata=raw.get("metadata", {})
        )
