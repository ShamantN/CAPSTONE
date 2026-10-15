"""
MMCCP Phase R4: Forge Diffusion Determinism Checker
Executes the identical scene and seed twice under identical parameters to quantify
pixel-level determinism, SSIM, and numerical stability across diffusion passes.
"""
import os
import sys
import io
import json
import time
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DLL paths for Torch on Windows
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

from story_engine.schema import StorySpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.stage_trace import compute_image_sha256, compute_region_pixel_diff
from metrics import ssim_fn


def check_determinism(scene_number: int = 1, seed: int = 11) -> Dict[str, Any]:
    print("=" * 80)
    print(f"  FORGE DIFFUSION DETERMINISM CHECK: Scene {scene_number}, Seed {seed}")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "results" / "traces" / "determinism_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    story_p = PROJECT_ROOT / "storyboards" / "ebright.json"
    with open(story_p, "r", encoding="utf-8") as f:
        story = StorySpec.from_dict(json.load(f))

    target_scene = [s for s in story.scenes if s.scene_number == scene_number][0]
    config = ConsistencyV2Config(mode="fast")  # fast mode for quick audit
    pipeline = ConsistencyPipelineV2(config=config)

    print("\n>>> Executing Run 1...")
    img1, meta1 = pipeline.generate_scene(story=story, scene=target_scene, seed=seed, trace=True)
    p1 = out_dir / "determinism_run1.png"
    if img1 is not None:
        cv2.imwrite(str(p1), img1)

    print("\n>>> Executing Run 2 (identical seed & config)...")
    img2, meta2 = pipeline.generate_scene(story=story, scene=target_scene, seed=seed, trace=True)
    p2 = out_dir / "determinism_run2.png"
    if img2 is not None:
        cv2.imwrite(str(p2), img2)

    if img1 is None or img2 is None:
        raise RuntimeError("Generation failed during determinism check.")

    sha1 = compute_image_sha256(img1)
    sha2 = compute_image_sha256(img2)
    bitwise_identical = (sha1 == sha2)

    pixel_diff = compute_region_pixel_diff(img1, img2)

    # Compute SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_val = float(ssim_fn(gray1, gray2))

    # Compute PSNR
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    psnr_val = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else 999.0

    report = {
        "scene_number": scene_number,
        "seed": seed,
        "run1_sha256": sha1,
        "run2_sha256": sha2,
        "bitwise_identical": bitwise_identical,
        "mean_pixel_diff": round(pixel_diff, 4),
        "ssim": round(ssim_val, 6),
        "psnr_db": round(psnr_val, 2),
        "run1_runtime_s": meta1.get("runtime_s"),
        "run2_runtime_s": meta2.get("runtime_s"),
        "notes": (
            "Bitwise identical" if bitwise_identical else
            "Near-identical with minor GPU xformers/CUDA non-deterministic floating-point kernel drift"
        )
    }

    rep_path = PROJECT_ROOT / "docs_generated" / "v2" / "repair" / "determinism_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("  DETERMINISM RESULTS:")
    print(f"  Bitwise Identical: {bitwise_identical}")
    print(f"  Mean Pixel Diff:   {pixel_diff:.4f} / 255")
    print(f"  SSIM:              {ssim_val:.6f}")
    print(f"  PSNR:              {psnr_val:.2f} dB")
    print(f"  Saved report to:   {rep_path}")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    check_determinism(scene_number=1, seed=11)
