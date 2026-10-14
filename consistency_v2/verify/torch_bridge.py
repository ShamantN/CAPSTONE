"""
Bridge to execute PyTorch / Transformers / Ultralytics vision models in Forge's Python environment.
Prevents DLL and C-extension collisions with project environments.
"""
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

FORGE_PYTHON = Path(r"C:\Stable-Diffusion\system\python\python.exe")
HELPER_SCRIPT = Path(__file__).resolve().parent / "torch_vision_helper.py"


def call_torch_vision(
    image_path: Path,
    face_bboxes: List[Dict[str, int]],
    reference_image_paths: List[Path],
    required_props: Optional[List[str]] = None,
    forbidden_props: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes YOLO, SegFormer, DINOv2, and CLIP in Forge's python environment and returns parsed results.
    """
    if required_props is None:
        required_props = []
    if forbidden_props is None:
        forbidden_props = []

    cfg_data = {
        "image_path": str(image_path),
        "face_bboxes": face_bboxes,
        "reference_image_paths": [str(p) for p in reference_image_paths],
        "required_props": required_props,
        "forbidden_props": forbidden_props
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix="_cfg.json", delete=False, encoding="utf-8") as f_cfg:
        json.dump(cfg_data, f_cfg, indent=2)
        cfg_path = Path(f_cfg.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix="_out.json", delete=False, encoding="utf-8") as f_out:
        out_path = Path(f_out.name)

    try:
        cmd = [
            str(FORGE_PYTHON),
            str(HELPER_SCRIPT),
            "--config", str(cfg_path),
            "--out", str(out_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Helper script did not produce output file"}
    except Exception as e:
        return {"error": f"Failed to execute vision helper: {e}"}
    finally:
        try:
            if cfg_path.exists():
                cfg_path.unlink()
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
