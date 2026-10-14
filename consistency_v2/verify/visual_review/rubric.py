"""
Visual Review Rubric Schema and Structured Verdict Template
Used by the agent and human evaluators to record rigorous, non-cherry-picked visual review verdicts.
"""
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


def create_blank_visual_review_verdict(
    image_filename: str,
    scene_id: str,
    scene_title: str,
    expected_characters: List[str]
) -> Dict[str, Any]:
    """Generates an empty, structured visual review rubric matching VR-2."""
    char_rubrics = {}
    for cid in expected_characters:
        char_rubrics[cid] = {
            "character_id": cid,
            "present": "yes",  # "yes" | "no" | "uncertain"
            "position_correct": "yes",  # "yes" | "no" | "uncertain"
            "observed_position_description": "",
            "same_person_as_reference": "yes",  # "yes" | "no" | "uncertain"
            "facial_features_compared": {
                "face_shape_match": True,
                "eye_color_gaze_match": True,
                "nose_structure_match": True,
                "hair_color_style_match": True,
                "skin_tone_match": True
            },
            "age_plausible": "yes",
            "clothing_matches_spec": "yes",
            "distinguishing_features_visible": "yes",
            "qualitative_assessment": ""
        }

    scene_defects = {
        "extra_people": False,
        "missing_people": False,
        "duplicated_cloned_faces": False,
        "merged_faces": False,
        "extra_missing_fused_limbs_fingers": False,
        "wrong_number_of_hands": False,
        "melted_distorted_faces": False,
        "text_watermark_logo_artifacts": False,
        "wrong_clothing_color": False,
        "wrong_age_ethnicity": False,
        "color_inversion_solarization": False,
        "forbidden_prop_present": False,
        "required_prop_absent": False,
        "visible_inpaint_seam": False
    }

    return {
        "review_version": "2.0",
        "image_filename": image_filename,
        "scene_id": scene_id,
        "scene_title": scene_title,
        "literal_scene_description": "",  # Describe first what you literally see
        "character_evaluations": char_rubrics,
        "defect_taxonomy_checklist": scene_defects,
        "props_observed": [],
        "overall_visual_verdict": "PASS",  # "PASS" | "FAIL" | "BORDERLINE"
        "confidence_score": 1.0,  # 0.0 to 1.0
        "notes_on_disagreements_with_numeric_scores": ""
    }


def save_visual_review(verdict: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
