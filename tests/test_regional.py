"""
Unit tests for Method M3: True Regional Conditioning (RegionalCoupleDriver).
Uses Python standard library unittest.
"""
import unittest
from consistency_v2.layout_planner import CharacterLayoutBox
from consistency_v2.regional import RegionalCoupleConfig, RegionalCoupleDriver


class TestRegionalCoupleDriver(unittest.TestCase):

    def test_single_character_bypasses_regional_coupling(self):
        bg = "a vintage 1950s kitchen, bright morning light"
        char_prompts = {"young_ebright": "a young boy in a white t-shirt"}
        boxes = [
            CharacterLayoutBox(
                character_id="young_ebright",
                target_zone="center",
                face_bbox={"x": 450, "y": 200, "w": 120, "h": 144},
                body_bbox={"x": 350, "y": 250, "w": 320, "h": 700},
                normalized_region=[0.0, 1.0, 0.25, 0.75],
                scale_category="medium shot"
            )
        ]

        prompt, scripts = RegionalCoupleDriver.build_prompt_and_payload(bg, char_prompts, boxes)
        self.assertIn("vintage 1950s kitchen", prompt)
        self.assertIn("white t-shirt", prompt)
        self.assertEqual(scripts, {})

    def test_two_character_basic_coupling_payload(self):
        bg = "cinematic 1950s living room, sunlight through window"
        char_prompts = {
            "mother": "a middle-aged woman in a yellow floral dress",
            "young_ebright": "a 10-year-old boy in blue shorts"
        }
        # Pass out of order to verify horizontal sorting
        boxes = [
            CharacterLayoutBox(
                character_id="mother",
                target_zone="right",
                face_bbox={"x": 677, "y": 194, "w": 120, "h": 144},
                body_bbox={"x": 580, "y": 250, "w": 300, "h": 650},
                normalized_region=[0.0, 1.0, 0.48, 1.0],
                scale_category="medium shot"
            ),
            CharacterLayoutBox(
                character_id="young_ebright",
                target_zone="left",
                face_bbox={"x": 226, "y": 194, "w": 120, "h": 144},
                body_bbox={"x": 130, "y": 250, "w": 300, "h": 650},
                normalized_region=[0.0, 1.0, 0.0, 0.52],
                scale_category="medium shot"
            )
        ]

        prompt, scripts = RegionalCoupleDriver.build_prompt_and_payload(bg, char_prompts, boxes)

        lines = prompt.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], bg)
        self.assertIn("boy in blue shorts", lines[1])  # Left character must come first!
        self.assertIn("woman in a yellow floral dress", lines[2])  # Right character second!

        self.assertIn("forge couple", scripts)
        args = scripts["forge couple"]["args"]
        self.assertEqual(len(args), 17)
        self.assertIs(args[0], True)   # Enable
        self.assertIs(args[1], True)   # Compatibility
        self.assertEqual(args[2], "Basic")
        self.assertEqual(args[3], "\n")
        self.assertEqual(args[4], "Horizontal")
        self.assertEqual(args[5], "First Line")
        self.assertEqual(args[6], 0.45)
        self.assertEqual(args[8], "{ }")

    def test_advanced_mapping_mode(self):
        bg = "exterior street scene"
        char_prompts = {
            "char_a": "character A on left",
            "char_b": "character B on right"
        }
        boxes = [
            CharacterLayoutBox(
                character_id="char_a",
                target_zone="left",
                face_bbox={"x": 200, "y": 200, "w": 100, "h": 100},
                body_bbox={"x": 100, "y": 200, "w": 300, "h": 600},
                normalized_region=[0.0, 1.0, 0.0, 0.5],
                scale_category="medium shot"
            ),
            CharacterLayoutBox(
                character_id="char_b",
                target_zone="right",
                face_bbox={"x": 700, "y": 200, "w": 100, "h": 100},
                body_bbox={"x": 600, "y": 200, "w": 300, "h": 600},
                normalized_region=[0.0, 1.0, 0.5, 1.0],
                scale_category="medium shot"
            )
        ]
        cfg = RegionalCoupleConfig(mode="Advanced", background_weight=0.5)
        prompt, scripts = RegionalCoupleDriver.build_prompt_and_payload(bg, char_prompts, boxes, config=cfg)

        self.assertIn("forge couple", scripts)
        args = scripts["forge couple"]["args"]
        self.assertEqual(args[2], "Advanced")
        mapping = args[7]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[0], [0.0, 0.5, 0.0, 1.0, 1.0])
        self.assertEqual(mapping[1], [0.5, 1.0, 0.0, 1.0, 1.0])

    def test_disabled_config_returns_unsplit(self):
        bg = "laboratory"
        char_prompts = {"char_a": "prompt a", "char_b": "prompt b"}
        boxes = [
            CharacterLayoutBox("char_a", "left", {}, {}, [0, 1, 0, 0.5], "medium"),
            CharacterLayoutBox("char_b", "right", {}, {}, [0, 1, 0.5, 1], "medium")
        ]
        cfg = RegionalCoupleConfig(enabled=False)
        prompt, scripts = RegionalCoupleDriver.build_prompt_and_payload(bg, char_prompts, boxes, config=cfg)
        self.assertEqual(scripts, {})
        self.assertIn("prompt a", prompt)
        self.assertIn("prompt b", prompt)


if __name__ == "__main__":
    unittest.main()
