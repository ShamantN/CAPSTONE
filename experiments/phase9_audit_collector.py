"""
PHASE 0: Current-State Forensic Audit Script
Gathers complete system configuration, extension inventory, model catalog, and file hashes.
"""
import sys, os, json, hashlib
from pathlib import Path

root = Path(r"d:\5TH SEM\HACKATHON")
forge_root = Path(r"C:\Stable-Diffusion")

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]

audit = {
    "python_version": sys.version,
    "source_files": {},
    "forge_extensions": {},
    "forge_builtin_extensions": [],
    "models": {},
    "production_stack": {
        "base_model": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "primary_sampler": "DPM++ 2M Karras",
        "steps": {"pass1": 25, "pass2": 15},
        "resolution": {"solo": "1024x1024", "multi": "1216x832"},
        "cfg_scale": 5.5,
        "attention_mods": ["FreeU", "PAG", "DynamicThresholding (CFG-Fix)"],
        "refinement": "ReActor (inswapper_128.onnx) + CodeFormer (0.80)"
    },
    "qa_thresholds": {
        "arcface_min": 0.45,
        "clip_semantic_min": 0.20,
        "forbidden_prop_max": 0.25,
        "spatial_left_max": 0.50,
        "spatial_right_min": 0.50
    }
}

key_files = [
    "app.py", "metrics.py",
    "identity/ipadapter.py", "identity/instantid.py", "identity/faceid.py", "identity/identity_conditioner.py",
    "conditioning/controlnet_conditioner.py", "conditioning/regional_conditioner.py", "conditioning/lora_registry.py",
    "conditioning/refinement.py", "conditioning/crossframe_conditioner.py",
    "generation/forge_client.py", "generation/generation_orchestrator.py",
    "story_engine/schema.py", "story_engine/character_registry.py", "story_engine/prompt_builder.py", "story_engine/scene_planner.py",
    "qa/healing_controller.py", "qa/programmatic_auditor.py", "qa/test_anchor_manager.py"
]

for kf in key_files:
    p = root / kf
    if p.exists():
        audit["source_files"][kf] = {
            "size": p.stat().st_size,
            "sha256": file_hash(p)
        }

ext_dir = forge_root / "webui" / "extensions"
if ext_dir.exists():
    for d in ext_dir.iterdir():
        if d.is_dir():
            audit["forge_extensions"][d.name] = "installed"

builtin_ext_dir = forge_root / "webui" / "extensions-builtin"
if builtin_ext_dir.exists():
    audit["forge_builtin_extensions"] = [d.name for d in builtin_ext_dir.iterdir() if d.is_dir()]

models_dir = forge_root / "webui" / "models"
if models_dir.exists():
    for sub in ["ControlNet", "Lora", "Stable-diffusion", "insightface", "Codeformer", "reactor"]:
        sd = models_dir / sub
        if sd.exists():
            for f in sd.rglob("*"):
                if f.is_file() and f.suffix in [".safetensors", ".bin", ".onnx", ".pth"]:
                    audit["models"][f"{sub}/{f.relative_to(sd)}"] = {
                        "size": f.stat().st_size,
                        "path": str(f)
                    }

out_dir = root / "experiments" / "phase9_baseline"
out_dir.mkdir(parents=True, exist_ok=True)
manifest_path = out_dir / "baseline_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print(f"Successfully wrote baseline manifest to {manifest_path}")
print(f"Key source files tracked: {len(audit['source_files'])}")
print(f"Models tracked: {len(audit['models'])}")
print(f"Forge extensions: {list(audit['forge_extensions'].keys())}")
print(f"Forge built-in extensions: {len(audit['forge_builtin_extensions'])}")
