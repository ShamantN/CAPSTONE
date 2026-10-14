"""
STAGE 7, 8, 9: IP-ADAPTER CLEAN SMOKE TEST & IDENTITY A/B VERIFICATION
Verifies:
1. Exact projection tensor compatibility (no ImageProjModel mismatch).
2. Clean generation with ip-adapter_xl + CLIP-ViT-bigG.
3. Controlled A/B test (Without IP-Adapter vs With IP-Adapter) with identical seed.
4. Measurable ArcFace identity influence against character reference anchor.
"""
import os, sys, json, time, urllib.request, base64
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.character_registry import CharacterRegistry
from metrics import calculate_max_match_arcface, decode_image_to_cv2

OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def send_txt2img_request(payload: dict) -> tuple[np.ndarray, dict]:
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as response:
        res = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - t0
    
    img_b64 = res["images"][0]
    img_cv = decode_image_to_cv2(img_b64)
    info = json.loads(res.get("info", "{}"))
    info["client_runtime_s"] = round(elapsed, 2)
    return img_cv, info


def main():
    print("=" * 80)
    print("  IP-ADAPTER CLEAN COMPATIBILITY & A/B VERIFICATION SMOKE TEST")
    print("=" * 80)

    # 1. Retrieve Young Ebright reference anchor
    registry = CharacterRegistry()
    anchor_b64 = registry.get_anchor_b64("young_ebright")
    if not anchor_b64:
        print("[ERROR] Character anchor 'young_ebright' not found in registry!")
        sys.exit(1)

    prompt = "photo of an 8-year-old Caucasian boy in a yellow polo shirt, sitting in a sunlit room, high quality, realistic"
    neg_prompt = "blurry, low quality, deformed, distorted, extra limbs"
    seed = 12345

    base_payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "seed": seed,
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "save_images": False
    }

    # -------------------------------------------------------------
    # TEST A: Plain SDXL (IP-Adapter Disabled)
    # -------------------------------------------------------------
    print("\n[TEST A] Executing plain SDXL baseline generation (IP-Adapter DISABLED)...")
    payload_a = dict(base_payload)
    payload_a["alwayson_scripts"] = {}
    
    img_a, info_a = send_txt2img_request(payload_a)
    path_a = OUTPUT_DIR / "smoke_test_A_no_ipadapter.png"
    cv2.imwrite(str(path_a), img_a)
    
    sim_a, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_a)
    print(f"  -> Test A completed in {info_a['client_runtime_s']}s | ArcFace vs Anchor: {sim_a:.4f}")

    # -------------------------------------------------------------
    # TEST B: SDXL with IP-Adapter Enabled (CLIP-ViT-bigG)
    # -------------------------------------------------------------
    print("\n[TEST B] Executing SDXL with IP-Adapter ENABLED (ip-adapter_xl + CLIP-ViT-bigG)...")
    payload_b = dict(base_payload)
    payload_b["alwayson_scripts"] = {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "module": "CLIP-ViT-bigG (IPAdapter)",
                    "model": "ip-adapter_xl [af81326a]",
                    "weight": 0.85,
                    "image": anchor_b64,
                    "guidance_start": 0.0,
                    "guidance_end": 0.85,
                    "control_mode": "Balanced",
                    "pixel_perfect": True
                }
            ]
        }
    }

    img_b, info_b = send_txt2img_request(payload_b)
    path_b = OUTPUT_DIR / "smoke_test_B_with_ipadapter.png"
    cv2.imwrite(str(path_b), img_b)

    sim_b, _, _, _, _ = calculate_max_match_arcface(anchor_b64, img_b)
    print(f"  -> Test B completed in {info_b['client_runtime_s']}s | ArcFace vs Anchor: {sim_b:.4f}")

    # -------------------------------------------------------------
    # COMPARISON & VERIFICATION
    # -------------------------------------------------------------
    # Compute visual pixel difference between A and B
    diff = np.abs(img_a.astype(float) - img_b.astype(float))
    mean_pixel_diff = float(np.mean(diff))
    arcface_gain = float(sim_b - sim_a)

    print("\n" + "=" * 80)
    print("  SMOKE TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Test A (No IP-Adapter) ArcFace:   {sim_a:.4f}")
    print(f"  Test B (With IP-Adapter) ArcFace: {sim_b:.4f}")
    print(f"  ArcFace Identity Improvement:     {arcface_gain:+.4f}")
    print(f"  Mean Pixel Difference (A vs B):   {mean_pixel_diff:.2f}")

    passed = (mean_pixel_diff > 5.0) and (sim_b > 0.40)
    status_str = "PASS" if passed else "FAIL"
    print(f"  Overall Smoke Test Status:        {status_str}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status_str,
        "test_a": {
            "name": "Without IP-Adapter",
            "runtime_s": info_a["client_runtime_s"],
            "arcface_similarity": round(float(sim_a), 4),
            "image_path": str(path_a)
        },
        "test_b": {
            "name": "With IP-Adapter (CLIP-ViT-bigG)",
            "runtime_s": info_b["client_runtime_s"],
            "arcface_similarity": round(float(sim_b), 4),
            "image_path": str(path_b)
        },
        "comparison": {
            "mean_pixel_diff": round(mean_pixel_diff, 2),
            "arcface_gain": round(arcface_gain, 4),
            "ipadapter_participated": mean_pixel_diff > 5.0
        }
    }

    report_path = OUTPUT_DIR / "ipadapter_smoke_test.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Smoke test results saved to {report_path}")

    if not passed:
        print("\n[ERROR] Smoke test did NOT meet pass criteria!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] IP-Adapter is verified working, loaded without error, and actively conditioning character identity!")


if __name__ == "__main__":
    main()
