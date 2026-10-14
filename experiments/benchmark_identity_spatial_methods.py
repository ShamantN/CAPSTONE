"""
STAGE 8, 10, 12: EMPIRICAL METHOD A/B/C BENCHMARK RUNNER
Directly compares:
1. Generic IP-Adapter SDXL
2. InstantID SDXL
3. FaceID Plus v2 SDXL
4. OpenPose SDXL + IP-Adapter
5. Depth SDXL + IP-Adapter
6. LoRA-conditioned generation
"""
import os, sys, json, time, logging
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from story_engine.character_registry import CharacterRegistry
from generation.generation_orchestrator import GenerationOrchestrator
from qa.programmatic_auditor import ProgrammaticAuditor
from metrics import calculate_max_match_arcface, decode_image_to_cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MMCCP.MethodBenchmark")

OUTPUT_DIR = Path("edu_outputs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_identity_method_benchmark():
    story = StorySpec.load_json("storyboards/ebright.json")
    test_scene = story.scenes[1]  # Scene 2: Young Ebright in Butterfly Basement (Solo)
    anchor_b64 = CharacterRegistry().get_anchor_b64("young_ebright")

    configs = [
        {"id": "ip_adapter", "name": "Generic IP-Adapter SDXL", "identity": "ip_adapter", "openpose": False, "depth": False, "lora": None},
        {"id": "instantid", "name": "Native InstantID SDXL", "identity": "instantid", "openpose": False, "depth": False, "lora": None},
        {"id": "faceid_plus_v2", "name": "FaceID Plus v2 SDXL", "identity": "faceid", "openpose": False, "depth": False, "lora": "ip-adapter-faceid-plusv2_sdxl_lora"},
        {"id": "openpose_ip", "name": "ControlNet OpenPose + IP-Adapter", "identity": "ip_adapter", "openpose": True, "depth": False, "lora": None},
        {"id": "depth_ip", "name": "ControlNet Depth + IP-Adapter", "identity": "ip_adapter", "openpose": False, "depth": True, "lora": None},
        {"id": "lora_lightning", "name": "SDXL Lightning LoRA + IP-Adapter", "identity": "ip_adapter", "openpose": False, "depth": False, "lora": "sdxl_lightning_8step_lora"}
    ]

    results = []

    print("\n" + "=" * 90)
    print("  RUNNING EMPIRICAL SOTA METHOD COMPARISON BENCHMARK")
    print("=" * 90)

    for cfg in configs:
        print(f"\n  >>> Evaluating Configuration: {cfg['name']}")
        orchestrator = GenerationOrchestrator(identity_engine=cfg["identity"])
        
        # Inject LoRA into prompt builder if configured
        pos_prompt_override = None
        if cfg["lora"]:
            pos_prompt_override = f"<lora:{cfg['lora']}:0.7> "

        t0 = time.time()
        img_cv, meta = orchestrator.generate_scene(
            story=story,
            scene=test_scene,
            steps_pass1=25,
            steps_pass2=15,
            enable_openpose=cfg["openpose"],
            enable_depth=cfg["depth"],
            seed=42
        )
        elapsed = time.time() - t0

        arcface_score = 0.0
        if img_cv is not None and anchor_b64:
            sim, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_cv)
            arcface_score = round(float(sim), 4)

            out_img = OUTPUT_DIR / f"exp_method_{cfg['id']}.png"
            out_json = OUTPUT_DIR / f"exp_method_{cfg['id']}.json"
            cv2.imwrite(str(out_img), img_cv)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        entry = {
            "config_id": cfg["id"],
            "name": cfg["name"],
            "identity_engine": cfg["identity"],
            "openpose": cfg["openpose"],
            "depth": cfg["depth"],
            "lora": cfg["lora"],
            "arcface_similarity": arcface_score,
            "runtime_s": round(elapsed, 2),
            "peak_vram_mb": meta.get("peak_vram_mb", 0.0),
            "status": "PASS" if arcface_score >= 0.45 else "LOW_SCORE"
        }
        results.append(entry)
        print(f"      [RESULT] ArcFace = {arcface_score} | Runtime = {elapsed:.1f}s | VRAM = {entry['peak_vram_mb']}MB | Status = {entry['status']}")

    with open("experiments/phase8_2_method_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": results}, f, indent=2)

    print("\n" + "=" * 90)
    print("  METHOD BENCHMARKS COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    run_identity_method_benchmark()
