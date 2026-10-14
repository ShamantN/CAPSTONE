"""
Baseline v1 Batch Runner
Executes canonical v1 GenerationOrchestrator over DEV-FULL and TEST splits with sidecar metadata logging.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DLL_PATHS = [
    r"C:\Stable-Diffusion\system\python\Library\bin",
    r"C:\Stable-Diffusion\system\python\Lib\site-packages\torch\lib",
    r"C:\Stable-Diffusion\system\python\Scripts",
    r"C:\Stable-Diffusion\system\python"
]
for _p in _DLL_PATHS:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec, CharacterSpec
from story_engine.character_registry import CharacterRegistry
from generation.generation_orchestrator import GenerationOrchestrator
from metrics import SD_MODELS, CACHE_FILE


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(PROJECT_ROOT))
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_peak_vram_mb() -> float:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_storyboard(json_path: Path) -> StorySpec:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return StorySpec.from_dict(data)


def run_baseline_split(
    story_path: Path,
    split_name: str,
    seeds: List[int],
    output_dir: Path,
    max_scenes: Optional[int] = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    story = load_storyboard(story_path)
    git_sha = get_git_commit_sha()

    orchestrator = GenerationOrchestrator(checkpoint_name=SD_MODELS["Juggernaut"])

    # Ensure anchors are loaded
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            for k, v in cdata.items():
                if v:
                    orchestrator.registry.register_anchor(k, v)

    scenes_to_run = story.scenes if max_scenes is None else story.scenes[:max_scenes]
    print(f"\n{'='*85}")
    print(f"  RUNNING BASELINE v1: {story.title} | Split: {split_name} | Seeds: {seeds}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*85}\n")

    for scene in scenes_to_run:
        for seed in seeds:
            img_filename = f"{scene.scene_id}_seed_{seed}.png"
            json_filename = f"{scene.scene_id}_seed_{seed}.json"
            img_path = output_dir / img_filename
            json_path = output_dir / json_filename

            if img_path.exists() and json_path.exists():
                print(f"  [SKIP] Scene {scene.scene_number} (Seed {seed}) already exists at {img_filename}")
                continue

            print(f"  >>> Rendering Scene {scene.scene_number} ({scene.title}) with Seed {seed}...")
            t0 = time.time()
            vram_start = get_gpu_peak_vram_mb()

            img_cv, meta = orchestrator.generate_scene(
                story=story,
                scene=scene,
                steps_pass1=25,
                steps_pass2=15,
                cfg_scale=5.5,
                seed=seed
            )
            elapsed = time.time() - t0
            vram_end = get_gpu_peak_vram_mb()

            if img_cv is not None:
                cv2.imwrite(str(img_path), img_cv)

                cfg_dict = {
                    "pipeline_version": "v1",
                    "checkpoint": SD_MODELS["Juggernaut"],
                    "steps_pass1": 25,
                    "steps_pass2": 15,
                    "cfg_scale": 5.5,
                    "sampler": "DPM++ 2M Karras",
                    "seed": seed
                }

                sidecar = {
                    "pipeline_version": "v1",
                    "git_commit_sha": git_sha,
                    "config_hash": compute_config_hash(cfg_dict),
                    "story_id": story.story_id,
                    "scene_id": scene.scene_id,
                    "scene_number": scene.scene_number,
                    "title": scene.title,
                    "seed": seed,
                    "actual_seed": meta.get("seed", seed),
                    "split": split_name,
                    "runtime_seconds": round(elapsed, 2),
                    "vram_start_mb": vram_start,
                    "vram_peak_mb": max(vram_end, meta.get("peak_vram_mb", 0.0)),
                    "image_path": str(img_path),
                    "characters_present": scene.characters_present,
                    "character_positions": scene.character_positions,
                    "positive_prompt": meta.get("positive_prompt", ""),
                    "negative_prompt": meta.get("negative_prompt", ""),
                    "generation_metadata": meta
                }

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)

                print(f"      [OK] Saved {img_filename} in {elapsed:.1f}s (Peak VRAM: {sidecar['vram_peak_mb']} MB)")
            else:
                print(f"      [ERROR] Failed to generate Scene {scene.scene_number} (Seed {seed})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline v1 Split Runner")
    parser.add_argument("--split", choices=["dev_fast", "dev_full", "test", "smoke"], default="smoke")
    args = parser.parse_args()

    ebright_path = PROJECT_ROOT / "storyboards" / "ebright.json"
    gen_path = PROJECT_ROOT / "storyboards" / "generalization_benchmark.json"

    if args.split == "smoke":
        run_baseline_split(
            story_path=ebright_path,
            split_name="smoke",
            seeds=[11],
            output_dir=PROJECT_ROOT / "results" / "baseline_v1" / "smoke",
            max_scenes=1
        )
    elif args.split == "dev_full":
        run_baseline_split(
            story_path=ebright_path,
            split_name="dev_full",
            seeds=[11, 23, 37],
            output_dir=PROJECT_ROOT / "results" / "baseline_v1" / "dev_full"
        )
    elif args.split == "test":
        # 1. Ebright on seeds 101, 202, 303, 404, 505
        run_baseline_split(
            story_path=ebright_path,
            split_name="test_ebright",
            seeds=[101, 202, 303, 404, 505],
            output_dir=PROJECT_ROOT / "results" / "baseline_v1" / "test_ebright"
        )
        # 2. Generalization benchmark on seeds 101, 202, 303
        run_baseline_split(
            story_path=gen_path,
            split_name="test_generalization",
            seeds=[101, 202, 303],
            output_dir=PROJECT_ROOT / "results" / "baseline_v1" / "test_generalization"
        )
