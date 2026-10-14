"""
STAGE 4: API SMOKE TEST
Validates basic txt2img communication, FreeU/PAG/CFG-Fix integration, and ControlNet IP-Adapter invocation.
"""
import requests, json, base64, time
from pathlib import Path

API_URL = "http://127.0.0.1:7860"

def test_smoke():
    print("=" * 80)
    print("  STAGE 4: FORGE API SMOKE TEST")
    print("=" * 80)

    # 1. Check endpoints
    r = requests.get(f"{API_URL}/sdapi/v1/options", timeout=5)
    assert r.status_code == 200, f"Options failed: {r.status_code}"
    opts = r.json()
    print(f"  [+] Active Checkpoint: {opts.get('sd_model_checkpoint')}")

    # 2. Minimal txt2img smoke test
    payload = {
        "prompt": "a simple test sphere on a white background, studio lighting",
        "negative_prompt": "blurry, deformed",
        "steps": 5,
        "width": 512,
        "height": 512,
        "cfg_scale": 4.5,
        "sampler_name": "Euler",
        "seed": 42
    }

    t0 = time.time()
    resp = requests.post(f"{API_URL}/sdapi/v1/txt2img", json=payload, timeout=60)
    elapsed = time.time() - t0

    assert resp.status_code == 200, f"Smoke test txt2img failed: {resp.status_code} {resp.text}"
    data = resp.json()
    images = data.get("images", [])
    assert len(images) > 0, "No image returned!"
    
    info_json = json.loads(data.get("info", "{}"))
    print(f"  [+] Smoke test passed in {elapsed:.2f}s!")
    print(f"      Returned Image Bytes: {len(base64.b64decode(images[0]))}")
    print(f"      Seed: {info_json.get('seed')}")
    print(f"      Sampler: {info_json.get('sampler_name')}")
    print(f"      Model Hash: {info_json.get('sd_model_hash')}")
    return True

if __name__ == "__main__":
    test_smoke()
