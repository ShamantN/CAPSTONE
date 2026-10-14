"""
STAGE 6: LLM STORY ANALYZER
Parses raw narrative text into structured, validated StorySpec objects using Gemini API / heuristic NLP parser.
"""
import os, sys, json, re, logging
from typing import Dict, Any, Optional
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

from story_engine.schema import StorySpec, CharacterSpec, AgeState, SceneSpec, PropSpec

logger = logging.getLogger("MMCCP.StoryAnalyzer")


class StoryAnalyzer:
    """
    Extracts structured StorySpec representations from arbitrary narrative text.
    Uses Gemini API if API key is configured; otherwise uses deterministic NLP heuristic rule-engine.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def analyze_story_text(self, story_text: str, story_id: str = "custom_story", title: str = "Custom Story") -> StorySpec:
        """
        Main entry point transforming raw narrative text into a structured StorySpec.
        """
        # 1. Attempt LLM-based extraction if Gemini API key exists
        if self.api_key:
            try:
                spec = self._analyze_with_gemini(story_text, story_id, title)
                if spec:
                    return spec
            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to heuristic parser: {e}")

        # 2. Deterministic Structured Fallback Parser
        return self._heuristic_parse(story_text, story_id, title)

    def _analyze_with_gemini(self, text: str, story_id: str, title: str) -> Optional[StorySpec]:
        """Calls Gemini API with structured JSON output schema."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            system_instruction = (
                "You are an expert cinematic storyboard and character consistency analyst. "
                "Extract structured characters, age transitions, scenes, and props from the following story text. "
                "Output strictly valid JSON complying with the provided StorySpec schema."
            )

            prompt = f"{system_instruction}\n\nSTORY TEXT:\n{text}\n\nReturn ONLY raw JSON."
            response = model.generate_content(prompt)
            clean_json = response.text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(clean_json)
            data["story_id"] = story_id
            data["title"] = title
            return StorySpec.from_dict(data)
        except Exception as e:
            logger.error(f"Gemini API parse exception: {e}")
            return None

    def _heuristic_parse(self, text: str, story_id: str, title: str) -> StorySpec:
        """Rule-based NLP parser extracting scene boundaries and character references."""
        # Split by paragraph or scene indicators
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        characters: Dict[str, CharacterSpec] = {}
        scenes: list[SceneSpec] = []
        props: Dict[str, PropSpec] = {}

        # Default character detection heuristics
        char_names = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text))
        ignored = {"The", "In", "On", "At", "After", "Before", "When", "Then", "Later", "During", "Scene", "Chapter"}
        filtered_names = [n for n in char_names if n not in ignored and len(n) > 2][:4]

        for idx, name in enumerate(filtered_names):
            c_id = name.lower().replace(" ", "_")
            characters[c_id] = CharacterSpec(
                id=c_id,
                name=name,
                unique_token=f"{name.replace(' ', '')}_TOK{idx}",
                gender="male" if "he" in text.lower() else "female",
                ethnicity="Caucasian",
                skin_tone="fair",
                face_shape="oval",
                hair_color="brown",
                hair_style="short neat",
                eye_color="brown",
                distinctive_features=["defined facial features"],
                recurring_clothing="casual clothing",
                clothing_colors=["navy blue"],
                age_states={
                    "primary": AgeState(age_years="25", stage_name="adult", hairstyle="short neat", clothing_description="casual clothing")
                }
            )

        for s_idx, para in enumerate(paragraphs, start=1):
            present = [c_id for c_id, c in characters.items() if c.name.lower() in para.lower()]
            if not present and characters:
                present = [list(characters.keys())[0]]

            shot_type = "Wide two-shot" if len(present) >= 2 else "Medium shot"
            env_type = "outdoor" if any(w in para.lower() for w in ["yard", "outside", "forest", "street", "park"]) else "indoor"

            positions = {}
            if len(present) == 1:
                positions[present[0]] = "center"
            elif len(present) >= 2:
                positions[present[0]] = "left"
                positions[present[1]] = "right"

            scenes.append(SceneSpec(
                scene_id=f"scene_{s_idx}",
                scene_number=s_idx,
                title=f"Scene {s_idx}",
                location_name="General Location",
                environment_type=env_type,
                lighting="Cinematic natural lighting",
                shot_type=shot_type,
                camera_angle="Eye level",
                action_description=para,
                characters_present=present,
                character_positions=positions,
                character_age_states={c: "primary" for c in present},
                required_props=[],
                forbidden_props=["blurry", "extra limbs"],
                qa_requirements={"min_face_count": len(present)}
            ))

        return StorySpec(
            story_id=story_id,
            title=title,
            visual_style="Cinematic Realistic",
            characters=characters,
            scenes=scenes,
            props=props
        )


if __name__ == "__main__":
    sample_text = (
        "Young Oliver is sitting in the library reading an ancient map.\n\n"
        "Years later, Oliver, now an experienced explorer, travels across the desert with his guide Maya."
    )
    analyzer = StoryAnalyzer()
    spec = analyzer.analyze_story_text(sample_text, story_id="explorer_story", title="The Explorer's Journey")
    print("Sample StorySpec generated successfully:")
    print(json.dumps(spec.to_dict(), indent=2))
