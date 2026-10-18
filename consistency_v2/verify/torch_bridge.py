"""
Bridge to execute PyTorch / Transformers / Ultralytics vision models in Forge's Python environment.
Prevents DLL and C-extension collisions with project environments.
Includes persistent worker pool for sub-second per-image inference without cold-start overhead.
"""
import os
import sys
import json
import atexit
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MMCCP.TorchBridge")

FORGE_PYTHON = Path(r"C:\Stable-Diffusion\system\python\python.exe")
HELPER_SCRIPT = Path(__file__).resolve().parent / "torch_vision_helper.py"

_WORKER_PROC: Optional[subprocess.Popen] = None


def _get_worker() -> Optional[subprocess.Popen]:
    global _WORKER_PROC
    if _WORKER_PROC is not None and _WORKER_PROC.poll() is None:
        return _WORKER_PROC

    try:
        _WORKER_PROC = subprocess.Popen(
            [str(FORGE_PYTHON), str(HELPER_SCRIPT), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        ready = _WORKER_PROC.stdout.readline().strip()
        if ready == "READY":
            return _WORKER_PROC
        else:
            logger.warning(f"Vision worker unexpected handshake: {ready}")
            _WORKER_PROC.terminate()
            _WORKER_PROC = None
    except Exception as e:
        logger.warning(f"Could not spawn persistent vision worker: {e}")
        _WORKER_PROC = None
    return _WORKER_PROC


def _shutdown_worker():
    global _WORKER_PROC
    if _WORKER_PROC is not None and _WORKER_PROC.poll() is None:
        try:
            _WORKER_PROC.stdin.write("SHUTDOWN\n")
            _WORKER_PROC.stdin.flush()
            _WORKER_PROC.wait(timeout=2)
        except Exception:
            try:
                _WORKER_PROC.terminate()
            except Exception:
                pass
        _WORKER_PROC = None


atexit.register(_shutdown_worker)


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

    # Fast path: persistent worker
    worker = _get_worker()
    if worker is not None and worker.poll() is None:
        try:
            worker.stdin.write(json.dumps(cfg_data) + "\n")
            worker.stdin.flush()
            resp_line = worker.stdout.readline()
            if resp_line:
                return json.loads(resp_line.strip())
        except Exception as e:
            logger.warning(f"Worker IPC error: {e}, falling back to one-shot invocation")
            _shutdown_worker()

    # Fallback path: one-shot temp files
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)
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
