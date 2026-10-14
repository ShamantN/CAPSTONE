"""
PHASE 8.4: CRITICAL TEST ANCHOR MANAGER
Enforces independent fresh temporary test anchors and immutable experiment isolation.
"""
import os, sys, json, shutil, logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from metrics import decode_image_to_cv2, encode_image_to_base64

logger = logging.getLogger("MMCCP.TestAnchorManager")


class TestAnchorManager:
    """
    Manages immutable canonical anchors and isolated per-test runtime anchor caches.
    """
    CANONICAL_DIR = Path("cache/canonical")
    TEST_RUNTIME_DIR = Path("cache/test_runtime")
    EXPERIMENTS_DIR = Path("experiments/phase8_4_tests")

    @classmethod
    def setup_canonical_anchors(cls):
        """Initializes canonical anchor repository from StorySpecs."""
        cls.CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
        canonical_json = cls.CANONICAL_DIR / "canonical_anchors.json"

        # Check if already established
        if canonical_json.exists():
            with open(canonical_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data) >= 5:
                return data

        print("[TestAnchorManager] Initializing Canonical Anchor Repository...")
        from story_engine.character_registry import CharacterRegistry
        from story_engine.schema import StorySpec

        # Generate canonical base anchors using SDXL txt2img if not present
        registry = CharacterRegistry(cache_file=canonical_json)
        
        # Load Ebright storyboard
        ebright_story = StorySpec.load_json("storyboards/ebright.json")
        for char_id, char_spec in ebright_story.characters.items():
            for age_key, age_state in char_spec.age_states.items():
                k = f"{char_id}_{age_key}" if age_key != "primary" else char_id
                if not registry.get_anchor_b64(char_id, age_key):
                    pos_prompt, neg_prompt = registry.build_anchor_prompt(char_spec, age_state, ebright_story.visual_style)
                    payload = {
                        "prompt": pos_prompt,
                        "negative_prompt": neg_prompt,
                        "steps": 25,
                        "width": 1024,
                        "height": 1024,
                        "cfg_scale": 6.0,
                        "sampler_name": "DPM++ 2M",
                        "scheduler": "Karras",
                        "seed": 42 + len(registry.cache_data)
                    }
                    import requests
                    try:
                        r = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload, timeout=120)
                        if r.status_code == 200:
                            b64 = r.json()["images"][0]
                            registry.register_anchor(char_id, b64, age_key if age_key != "primary" else None)
                            # Save PNG
                            img_cv = decode_image_to_cv2(b64)
                            cv2.imwrite(str(cls.CANONICAL_DIR / f"{k}.png"), img_cv)
                            print(f"  [CANONICAL ANCHOR CREATED] {k}")
                    except Exception as e:
                        logger.error(f"Failed to generate canonical anchor {k}: {e}")

        # Load Chrono-Archivists storyboard
        chrono_story = StorySpec.load_json("storyboards/generalization_benchmark.json")
        for char_id, char_spec in chrono_story.characters.items():
            for age_key, age_state in char_spec.age_states.items():
                k = f"{char_id}_{age_key}" if age_key != "primary" else char_id
                if not registry.get_anchor_b64(char_id, age_key):
                    pos_prompt, neg_prompt = registry.build_anchor_prompt(char_spec, age_state, chrono_story.visual_style)
                    payload = {
                        "prompt": pos_prompt,
                        "negative_prompt": neg_prompt,
                        "steps": 25,
                        "width": 1024,
                        "height": 1024,
                        "cfg_scale": 6.0,
                        "sampler_name": "DPM++ 2M",
                        "scheduler": "Karras",
                        "seed": 100 + len(registry.cache_data)
                    }
                    import requests
                    try:
                        r = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload, timeout=120)
                        if r.status_code == 200:
                            b64 = r.json()["images"][0]
                            registry.register_anchor(char_id, b64, age_key if age_key != "primary" else None)
                            img_cv = decode_image_to_cv2(b64)
                            cv2.imwrite(str(cls.CANONICAL_DIR / f"{k}.png"), img_cv)
                            print(f"  [CANONICAL ANCHOR CREATED] {k}")
                    except Exception as e:
                        logger.error(f"Failed to generate canonical anchor {k}: {e}")

        return registry.cache_data

    @classmethod
    def reset_fresh_test_anchors(cls, category: str, test_name: str) -> Tuple[Path, Dict[str, str]]:
        """
        1. Completely deletes previous test_runtime cache.
        2. Copies fresh anchors from canonical repository.
        3. Creates isolated output directory under experiments/phase8_4_tests/{category}/{test_name}.
        """
        # Step 1: Wipe test runtime cache
        if cls.TEST_RUNTIME_DIR.exists():
            shutil.rmtree(cls.TEST_RUNTIME_DIR, ignore_errors=True)
        cls.TEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        # Step 2: Ensure canonical exists and load
        canonical_data = cls.setup_canonical_anchors()

        # Step 3: Write fresh runtime cache
        runtime_json = cls.TEST_RUNTIME_DIR / "test_anchors.json"
        with open(runtime_json, "w", encoding="utf-8") as f:
            json.dump(canonical_data, f, indent=2)

        # Step 4: Create immutable experiment directory
        exp_dir = cls.EXPERIMENTS_DIR / category / test_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[TestAnchorManager] Fresh Test Anchors Initialized for '{category}/{test_name}'")
        print(f"  Runtime Cache: {runtime_json}")
        print(f"  Experiment Directory: {exp_dir}")

        return exp_dir, canonical_data
