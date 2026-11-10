"""
MMCCP Preflight Setup Verification Tool (Windows / Linux / Google Colab).
Validates OS, Python version, git, disk/RAM, NVIDIA GPU + VRAM, PyTorch/CUDA,
key packages vs requirements-lock.txt, Forge API & ControlNet/scripts, model weights
& sizes (with optional SHA-256 verification), frozen verifier integrity, and split guards.

No heavy ML imports are performed at module import time.
Usage:
    python tools/verify_setup.py --mode new-machine --fix-hints
    python tools/verify_setup.py --mode colab --fix-hints
    python tools/verify_setup.py --verify-sha256
    python tools/verify_setup.py --skip-forge --json
"""
import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.runtime_paths import (
    configure_windows_forge_env,
    get_forge_dir,
    get_forge_models_dir,
    get_forge_python,
    get_forge_url,
    get_project_root,
)

# Exact ground-truth model weights from verified C:\Stable-Diffusion & caches
WEIGHT_MANIFEST: List[Dict[str, Any]] = [
    {
        "name": "Juggernaut-XL v9 RunDiffusionPhoto v2",
        "rel_path": "Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "root": "forge_models",
        "required": True,
        "auto_downloaded": False,
        "size_bytes": 7105348188,
        "sha256": "c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca",
        "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    },
    {
        "name": "ControlNet OpenPose XL (OpenPoseXL2)",
        "rel_path": "ControlNet/openpose_xl.safetensors",
        "root": "forge_models",
        "required": True,
        "auto_downloaded": False,
        "size_bytes": 5004167829,
        "sha256": "5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92",
        "url": "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors",
    },
    {
        "name": "IP-Adapter Plus-Face SDXL (ViT-H)",
        "rel_path": "ControlNet/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "root": "forge_models",
        "required": True,
        "auto_downloaded": False,
        "size_bytes": 847517512,
        "sha256": "677ad8860204f7d0bfba12d29e6c31ded9beefdf3e4bbd102518357d31a292c1",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    },
    {
        "name": "InsightFace Swapper (inswapper_128.onnx)",
        "rel_path": "insightface/inswapper_128.onnx",
        "root": "forge_models",
        "required": True,
        "auto_downloaded": False,
        "size_bytes": 554253681,
        "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
    },
    {
        "name": "CodeFormer Restoration (codeformer-v0.1.0.pth)",
        "rel_path": "Codeformer/codeformer-v0.1.0.pth",
        "root": "forge_models",
        "required": True,
        "auto_downloaded": True,
        "size_bytes": 376637898,
        "sha256": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    },
    {
        "name": "InsightFace buffalo_l ArcFace (w600k_r50.onnx)",
        "rel_path": "insightface/models/buffalo_l/w600k_r50.onnx",
        "alt_paths": ["models/buffalo_l/w600k_r50.onnx"],
        "root": "forge_models",
        "required": True,
        "auto_downloaded": True,
        "size_bytes": 174383860,
        "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    },
    {
        "name": "CLIP-ViT-H-14 Image Encoder (IP-Adapter)",
        "rel_path": "ControlNetPreprocessor/CLIP-ViT-H-14.safetensors",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": True,
        "size_bytes": 2528373448,
        "sha256": "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    },
    {
        "name": "CLIP-ViT-bigG Image Encoder (SDXL IP-Adapter)",
        "rel_path": "ControlNetPreprocessor/CLIP-ViT-bigG.safetensors",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": True,
        "size_bytes": 3689912664,
        "sha256": "657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d",
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
    },
    {
        "name": "YOLOv8n Person Detector (yolov8n.pt)",
        "rel_path": "yolov8n.pt",
        "root": "project_root",
        "required": True,
        "auto_downloaded": True,
        "size_bytes": 6549796,
        "sha256": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    },
    {
        "name": "Optional Checkpoint: Animagine XL 3.1",
        "rel_path": "Stable-diffusion/animagine-xl-3.1.safetensors",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": False,
        "size_bytes": 6938325776,
        "sha256": "e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0",
        "url": "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors",
    },
    {
        "name": "Optional ControlNet: Depth XL",
        "rel_path": "ControlNet/depth_xl.safetensors",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": False,
        "size_bytes": 5004167860,
        "sha256": "b760b1ed26970c7c60abbcef08e7238bb9428bee510a405aff239e06a1471945",
        "url": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    },
    {
        "name": "Optional ControlNet: InstantID SDXL",
        "rel_path": "ControlNet/control_instant_id_sdxl.safetensors",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": False,
        "size_bytes": 2502139136,
        "sha256": "c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05",
        "url": "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors",
    },
    {
        "name": "Optional ControlNet: InstantID IP-Adapter",
        "rel_path": "ControlNet/ip-adapter_instant_id_sdxl.bin",
        "root": "forge_models",
        "required": False,
        "auto_downloaded": False,
        "size_bytes": 1691134141,
        "sha256": "02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1",
        "url": "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin",
    },
]

CORE_PACKAGES = [
    "numpy",
    "opencv-python",
    "Pillow",
    "scikit-image",
    "scipy",
    "requests",
    "pydantic",
    "streamlit",
    "deepface",
    "tf-keras",
    "tensorflow",
    "retina-face",
    "transformers",
    "ultralytics",
    "torch",
    "torchvision",
]


@dataclass
class CheckResult:
    name: str
    status: str  # PASS, WARN, FAIL
    detail: str
    fix_hint: str = ""


def parse_lock_file(lock_path: Path) -> Dict[str, str]:
    """Parses requirements-lock.txt into {lowercase_name: pinned_version}."""
    locks: Dict[str, str] = {}
    if not lock_path.exists():
        return locks
    raw = lock_path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "==" in s:
            name, ver = s.split("==", 1)
            locks[name.strip().lower()] = ver.strip()
        elif " @ " in s:
            name, url = s.split(" @ ", 1)
            locks[name.strip().lower()] = url.strip()
    return locks


def check_python_version(mode: str, version_info: Optional[Tuple[int, int, int]] = None) -> CheckResult:
    vi = version_info or sys.version_info[:3]
    ver_str = f"{vi[0]}.{vi[1]}.{vi[2]}"
    if (vi[0], vi[1]) < (3, 10):
        return CheckResult(
            "Python Version",
            "FAIL",
            f"Python {ver_str} detected (< 3.10).",
            "Install Python 3.10.6 (tested) or 3.10+ and recreate your virtualenv."
        )
    if mode == "colab" and vi[:2] in ((3, 10), (3, 11), (3, 12)):
        return CheckResult("Python Version", "PASS", f"Python {ver_str} (supported on Colab).")
    if vi[:2] != (3, 10):
        return CheckResult(
            "Python Version",
            "WARN",
            f"Python {ver_str} detected (tested release baseline is 3.10.6).",
            "If you encounter binary wheel mismatches (tf-keras/deepface/torch), create venv with Python 3.10.6."
        )
    return CheckResult("Python Version", "PASS", f"Python {ver_str} (matches tested 3.10.x).")


def check_git_binary() -> CheckResult:
    git_exe = shutil.which("git")
    if not git_exe:
        return CheckResult("Git CLI", "FAIL", "git executable not found in PATH.", "Install Git for Windows or apt install git.")
    try:
        res = subprocess.run([git_exe, "--version"], capture_output=True, text=True, timeout=5)
        return CheckResult("Git CLI", "PASS", res.stdout.strip() or git_exe)
    except Exception as e:
        return CheckResult("Git CLI", "FAIL", f"Failed running git: {e}", "Reinstall Git and ensure it is on PATH.")


def get_host_ram_gb() -> float:
    if sys.platform == "win32" and hasattr(ctypes, "windll"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            pass
    return 0.0


def check_disk_and_ram(target_path: Path, mode: str) -> List[CheckResult]:
    results = []
    usage = shutil.disk_usage(str(target_path if target_path.exists() else PROJECT_ROOT))
    free_gb = usage.free / (1024 ** 3)
    min_warn = 45.0 if mode == "new-machine" else 20.0
    if free_gb < 5.0:
        results.append(CheckResult("Disk Free Space", "FAIL", f"{free_gb:.1f} GB free at {target_path}.", "Free at least 45 GB (local models + outputs) before running SDXL."))
    elif free_gb < min_warn:
        results.append(CheckResult("Disk Free Space", "WARN", f"{free_gb:.1f} GB free (< {min_warn:.0f} GB recommended for full model set).", "Required weights + Forge take ~22-25 GB; optional weights bring total to ~45 GB."))
    else:
        results.append(CheckResult("Disk Free Space", "PASS", f"{free_gb:.1f} GB free."))

    ram_gb = get_host_ram_gb()
    if ram_gb <= 0:
        results.append(CheckResult("System RAM", "WARN", "Could not read physical RAM total."))
    elif ram_gb < 12.0:
        results.append(CheckResult("System RAM", "WARN", f"{ram_gb:.1f} GB total RAM (< 16 GB recommended for CPU DeepFace/SegFormer + Forge swap).", "Close unused apps; SDXL offloads heavily to host RAM on 8 GB GPUs."))
    else:
        results.append(CheckResult("System RAM", "PASS", f"{ram_gb:.1f} GB physical RAM."))
    return results


def check_gpu_and_torch(mode: str) -> List[CheckResult]:
    results = []
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        results.append(CheckResult(
            "NVIDIA GPU (nvidia-smi)",
            "FAIL",
            "nvidia-smi not found.",
            "Install NVIDIA Game Ready / Studio Driver (550+ recommended) or enable GPU runtime in Colab (Runtime -> Change runtime type -> T4 GPU)."
        ))
    else:
        try:
            res = subprocess.run(
                [nvsmi, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8
            )
            if res.returncode != 0 or not res.stdout.strip():
                results.append(CheckResult("NVIDIA GPU (nvidia-smi)", "FAIL", f"nvidia-smi error: {res.stderr.strip()}", "Check NVIDIA driver installation or Colab GPU quota."))
            else:
                first_line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                name, drv = parts[0], parts[1]
                tot_mib = float(parts[2])
                used_mib = float(parts[3])
                free_mib = float(parts[4])
                msg = f"{name} | Driver {drv} | {tot_mib:.0f} MiB total ({used_mib:.0f} MiB used, {free_mib:.0f} MiB free)"
                if tot_mib < 7600:
                    results.append(CheckResult("NVIDIA GPU (nvidia-smi)", "FAIL", msg, "Minimum 8 GB VRAM GPU (e.g. RTX 3070 Ti 8GB or Colab T4 16GB) required for SDXL pipeline."))
                elif mode == "new-machine" and used_mib > 1400 and tot_mib <= 8500:
                    results.append(CheckResult("NVIDIA GPU (nvidia-smi)", "WARN", msg, "Close browsers/GPU desktop apps so idle VRAM <= 1000 MiB before running diffusion to stay near the 7.4 GB budget."))
                else:
                    results.append(CheckResult("NVIDIA GPU (nvidia-smi)", "PASS", msg))
        except Exception as e:
            results.append(CheckResult("NVIDIA GPU (nvidia-smi)", "FAIL", f"Failed querying nvidia-smi: {e}", "Ensure NVIDIA driver is loaded."))

    # Import torch lazily only inside try/except
    try:
        if sys.platform == "win32":
            configure_windows_forge_env()
        import torch  # noqa: E402
        cuda_ok = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "CPU-only"
        status = "PASS" if cuda_ok else "WARN"
        hint = "" if cuda_ok else (
            "If running tests on CPU (`CUDA_VISIBLE_DEVICES=-1`), WARN is normal; for GPU PyTorch on Windows run: "
            "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"
        )
        results.append(CheckResult("PyTorch / CUDA", status, f"torch {torch.__version__} | cuda_available={cuda_ok} | torch.version.cuda={cuda_ver}", hint))
    except ImportError:
        results.append(CheckResult(
            "PyTorch / CUDA",
            "FAIL",
            "torch cannot be imported from active Python or Forge site-packages.",
            "Install PyTorch: pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"
        ))
    return results


def check_packages(lock_path: Path, mode: str) -> List[CheckResult]:
    if sys.platform == "win32":
        configure_windows_forge_env()
    locks = parse_lock_file(lock_path)
    missing: List[str] = []
    drifted: List[str] = []
    numpy_major_bad = False
    found_versions: Dict[str, str] = {}

    for pkg in CORE_PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
            found_versions[pkg] = ver
            if pkg.lower() == "numpy" and not ver.startswith("1."):
                numpy_major_bad = True
            locked = locks.get(pkg.lower())
            if locked and "http" not in locked and ver != locked:
                drifted.append(f"{pkg} (got {ver}, lock {locked})")
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)

    results = []
    if missing:
        results.append(CheckResult(
            "Python Packages",
            "FAIL",
            f"Missing {len(missing)} required packages: {', '.join(missing)}",
            "Run: pip install -r requirements.txt (on Windows install torch cu121 first; on Colab avoid downgrading preinstalled torch)."
        ))
    elif numpy_major_bad:
        results.append(CheckResult(
            "Python Packages",
            "FAIL",
            f"numpy {found_versions.get('numpy')} installed, but numpy 1.x (<2.0) is required by PyTorch 2.3.1 & SegFormer WardrobeLock.",
            'Run: pip install "numpy>=1.26.0,<2.0.0"'
        ))
    elif drifted:
        results.append(CheckResult(
            "Python Packages",
            "PASS" if mode == "colab" else "WARN",
            f"All {len(CORE_PACKAGES)} core packages importable; version diffs vs lock: {len(drifted)} ({'; '.join(drifted[:4])}...)",
            "Optional: align exact versions from requirements-lock.txt on Windows."
        ))
    else:
        results.append(CheckResult("Python Packages", "PASS", f"All {len(CORE_PACKAGES)} core packages match lock/compatible versions."))
    return results


def _http_get_json(url: str, timeout: int = 5) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_forge_api(forge_url: Optional[str] = None, skip_forge: bool = False) -> List[CheckResult]:
    base_url = (forge_url or get_forge_url()).rstrip("/")
    if skip_forge:
        return [CheckResult("Forge API", "WARN", f"Skipped (--skip-forge). Target URL: {base_url}")]

    results = []
    try:
        opts = _http_get_json(f"{base_url}/sdapi/v1/options", timeout=5)
        ckpt = opts.get("sd_model_checkpoint", "UNKNOWN")
        mem = opts.get("forge_inference_memory", "unset")
        xl_mb = opts.get("xl_GPU_MB", "unset")
        ckpt_ok = "juggernaut" in str(ckpt).lower() or "animagine" in str(ckpt).lower()
        results.append(CheckResult(
            "Forge API (/sdapi/v1/options)",
            "PASS" if ckpt_ok else "WARN",
            f"Online at {base_url} | checkpoint={ckpt} | forge_inference_memory={mem} | xl_GPU_MB={xl_mb}",
            "" if ckpt_ok else "Load Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors in Forge UI or via API."
        ))
    except Exception as e:
        results.append(CheckResult(
            "Forge API (/sdapi/v1/options)",
            "FAIL",
            f"Cannot reach Forge at {base_url} ({e}).",
            "Start Stable Diffusion WebUI Forge (`run.bat` on Windows or `webui.sh` on Linux/Colab) with `--api --xformers` and wait for port 7860."
        ))
        return results

    try:
        scripts = _http_get_json(f"{base_url}/sdapi/v1/script-info", timeout=8)
        script_names = {str(s.get("name", "")).lower(): s for s in scripts if isinstance(s, dict)}
        req_scripts = ["controlnet", "forge couple", "freeu integrated (sd 1.x, sd 2.x, sdxl)", "dynamicthresholding (cfg-fix) integrated"]
        missing_scripts = [r for r in req_scripts if r not in script_names]
        if missing_scripts:
            results.append(CheckResult(
                "Forge Scripts (/sdapi/v1/script-info)",
                "FAIL",
                f"Missing required Forge scripts: {missing_scripts}",
                "Install `sd-forge-couple` (Haoming02/sd-forge-couple) into webui/extensions/ and ensure Forge built-in ControlNet/FreeU/DynamicThresholding are active."
            ))
        else:
            has_reactor = "reactor" in script_names
            results.append(CheckResult(
                "Forge Scripts (/sdapi/v1/script-info)",
                "PASS",
                f"ControlNet, Forge Couple, FreeU, DynamicThresholding present (ReActor API present={has_reactor})."
            ))
    except Exception as e:
        results.append(CheckResult("Forge Scripts (/sdapi/v1/script-info)", "WARN", f"Could not inspect script-info: {e}"))

    try:
        cn_models = _http_get_json(f"{base_url}/controlnet/model_list", timeout=8).get("model_list", [])
        cn_str = " ".join(str(m).lower() for m in cn_models)
        has_openpose = "openpose_xl" in cn_str
        has_ipface = "ip-adapter-plus-face_sdxl_vit-h" in cn_str
        if has_openpose and has_ipface:
            results.append(CheckResult("ControlNet Model List", "PASS", f"{len(cn_models)} models visible (openpose_xl & ip-adapter-plus-face_sdxl_vit-h detected)."))
        else:
            missing_cn = [x for ok, x in [(has_openpose, "openpose_xl"), (has_ipface, "ip-adapter-plus-face_sdxl_vit-h")] if not ok]
            results.append(CheckResult(
                "ControlNet Model List",
                "FAIL",
                f"Missing required ControlNet model(s) in Forge: {missing_cn}",
                "Download missing weights into Forge `webui/models/ControlNet/` and restart/refresh Forge."
            ))
    except Exception as e:
        results.append(CheckResult("ControlNet Model List", "WARN", f"Could not query /controlnet/model_list: {e}"))

    return results


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_weights(
    forge_models_dir: Path,
    project_root: Path,
    verify_sha256: bool = False,
    manifest: Optional[List[Dict[str, Any]]] = None
) -> List[CheckResult]:
    items = manifest or WEIGHT_MANIFEST
    missing_req: List[str] = []
    missing_auto: List[str] = []
    bad_size: List[str] = []
    bad_sha: List[str] = []
    ok_req = 0

    for item in items:
        base = forge_models_dir if item["root"] == "forge_models" else project_root
        candidates = [base / item["rel_path"]] + [base / ap for ap in item.get("alt_paths", [])]
        found = next((p for p in candidates if p.exists() and p.is_file()), None)
        if not found:
            if item["required"] and not item["auto_downloaded"]:
                missing_req.append(f"{item['name']} ({candidates[0]})")
            elif item["auto_downloaded"]:
                missing_auto.append(f"{item['name']} (auto-downloaded on first run)")
            continue

        actual_size = found.stat().st_size
        if abs(actual_size - item["size_bytes"]) > 1024 and item["size_bytes"] > 0:
            bad_size.append(f"{item['name']} (expected {item['size_bytes']} B, got {actual_size} B)")

        if verify_sha256 and item.get("sha256"):
            got_sha = sha256_file(found)
            if got_sha != item["sha256"]:
                bad_sha.append(f"{item['name']} (expected {item['sha256'][:12]}..., got {got_sha[:12]}...)")

        if item["required"]:
            ok_req += 1

    results = []
    if missing_req:
        results.append(CheckResult(
            "Manual Model Weights",
            "FAIL",
            f"Missing {len(missing_req)} required manual weight(s): {'; '.join(missing_req)}",
            "Follow docs/run/04_MODELS_AND_DOWNLOADS.md to download required SDXL/ControlNet/InsightFace weights into Forge models dir."
        ))
    else:
        results.append(CheckResult("Manual Model Weights", "PASS", f"Required manual weights located under {forge_models_dir}."))

    if missing_auto:
        results.append(CheckResult(
            "Auto-Downloaded Weights",
            "WARN",
            f"{len(missing_auto)} runtime/auto-downloaded file(s) not cached yet (will download on first run): {', '.join(missing_auto[:4])}",
            "Ensure internet access on first run so CodeFormer/YOLOv8n/DeepFace/HuggingFace weights cache cleanly."
        ))
    else:
        results.append(CheckResult("Auto-Downloaded Weights", "PASS", "Runtime auto-downloaded weights already cached locally."))

    if bad_size or bad_sha:
        issues = bad_size + bad_sha
        results.append(CheckResult(
            "Weight Integrity",
            "FAIL",
            f"Size/SHA mismatch on {len(issues)} file(s): {'; '.join(issues)}",
            "Re-download corrupted/incomplete files and re-run with --verify-sha256."
        ))
    elif verify_sha256:
        results.append(CheckResult("Weight Integrity (SHA-256)", "PASS", "All present manifest weights matched exact SHA-256 checksums."))
    return results


def check_freeze_and_splits(project_root: Path) -> List[CheckResult]:
    results = []
    # Frozen verifier check
    freeze_script = project_root / "tools" / "check_freeze.py"
    if not freeze_script.exists():
        results.append(CheckResult("Frozen Verifier v1.2", "FAIL", f"Missing {freeze_script}"))
    else:
        cwd_before = os.getcwd()
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            os.chdir(str(project_root))
            import tools.check_freeze as cf
            cf.VERIFY_DIR = project_root / "consistency_v2" / "verify"
            cf.FREEZE_FILE = project_root / "consistency_v2" / "verifier_freeze.json"
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                ok = cf.check_freeze()
            summary = buf_out.getvalue().strip() or "SHA-256 freeze manifest verified."
            if ok:
                results.append(CheckResult("Frozen Verifier v1.2", "PASS", summary))
            else:
                err_txt = buf_err.getvalue().strip()
                results.append(CheckResult("Frozen Verifier v1.2", "FAIL", err_txt or "Verifier freeze hash mismatch!", "Do NOT modify consistency_v2/verify/; run `git checkout -- consistency_v2/verify`."))
        except Exception as e:
            results.append(CheckResult("Frozen Verifier v1.2", "FAIL", f"Freeze check raised {e}"))
        finally:
            os.chdir(cwd_before)

    # Split seal guard
    allow_test2 = os.environ.get("ALLOW_TEST2") == "1"
    allow_res = os.environ.get("ALLOW_RESERVED") == "1"
    if allow_test2 or allow_res:
        results.append(CheckResult(
            "Split Guards (TEST-2 / RESERVED)",
            "FAIL",
            f"Sealed split unlocked in env (ALLOW_TEST2={os.environ.get('ALLOW_TEST2')}, ALLOW_RESERVED={os.environ.get('ALLOW_RESERVED')}).",
            "Unset ALLOW_TEST2 and ALLOW_RESERVED immediately (`Remove-Item Env:ALLOW_TEST2` / `unset ALLOW_TEST2`)."
        ))
    else:
        results.append(CheckResult("Split Guards (TEST-2 / RESERVED)", "PASS", "ALLOW_TEST2 & ALLOW_RESERVED unset (TEST-2 & LoRA reserved splits sealed)."))

    return results


def check_forge_patches(
    forge_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> CheckResult:
    """Checks whether the 3 recommended MMCCP upstream Forge patches are applied cleanly."""
    target_dir = Path(forge_dir) if forge_dir is not None else get_forge_dir()
    patches_dir = (project_root or get_project_root()) / "patches" / "forge"
    try:
        from tools.apply_forge_patches import manage_patches
        rc, report = manage_patches("check", forge_dir=target_dir, patches_dir=patches_dir)
    except Exception as exc:
        return CheckResult("Forge Patches", "WARN", f"Could not inspect Forge patches: {exc}")

    if rc == 0:
        return CheckResult(
            "Forge Patches",
            "PASS",
            f"All {len(report.get('patches', []))} recommended upstream Forge patches applied cleanly.",
        )
    if rc == 1:
        missing = [p["name"] for p in report.get("patches", []) if p.get("state") != "ALREADY_APPLIED"]
        return CheckResult(
            "Forge Patches",
            "WARN",
            f"{len(missing)} recommended upstream Forge patch(es) unapplied ({', '.join(missing)}).",
            f'python tools/apply_forge_patches.py --forge-dir "{target_dir}" --apply',
        )
    return CheckResult(
        "Forge Patches",
        "WARN",
        report.get("error", "Forge git repository mismatch/unavailable for patch check."),
        "See patches/forge/README.md before applying Forge patches manually.",
    )


def run_preflight(
    mode: str = "new-machine",
    skip_forge: bool = False,
    verify_sha256: bool = False,
    project_root: Optional[Path] = None,
    forge_url: Optional[str] = None,
    forge_models_dir: Optional[Path] = None,
) -> List[CheckResult]:
    root = project_root or get_project_root()
    models_dir = forge_models_dir or get_forge_models_dir()
    results: List[CheckResult] = []
    results.append(CheckResult("Platform / Paths", "PASS", f"OS={platform.system()} {platform.release()} | MMCCP_ROOT={root} | FORGE_DIR={get_forge_dir()} | FORGE_PYTHON={get_forge_python()}"))
    results.append(check_python_version(mode))
    results.append(check_git_binary())
    results.extend(check_disk_and_ram(root, mode))
    results.extend(check_gpu_and_torch(mode))
    results.extend(check_packages(root / "requirements-lock.txt", mode))
    results.extend(check_forge_api(forge_url=forge_url, skip_forge=skip_forge))
    results.append(check_forge_patches(project_root=root))
    results.extend(check_weights(models_dir, root, verify_sha256=verify_sha256))
    results.extend(check_freeze_and_splits(root))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MMCCP Setup Preflight Verifier")
    parser.add_argument("--mode", choices=["new-machine", "colab"], default="new-machine", help="Target environment mode")
    parser.add_argument("--fix-hints", action="store_true", help="Print actionable fix commands for WARN/FAIL checks")
    parser.add_argument("--verify-sha256", action="store_true", help="Compute SHA-256 on present model files (slower)")
    parser.add_argument("--skip-forge", action="store_true", help="Skip live Forge API connectivity check (for CPU/offline validation)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args(argv)

    results = run_preflight(
        mode=args.mode,
        skip_forge=args.skip_forge,
        verify_sha256=args.verify_sha256,
    )
    fails = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]

    if args.json:
        print(json.dumps({
            "mode": args.mode,
            "summary": {"pass": len(results) - len(fails) - len(warns), "warn": len(warns), "fail": len(fails)},
            "checks": [asdict(r) for r in results],
        }, indent=2))
    else:
        print("=" * 80)
        print(f"  MMCCP PREFLIGHT CHECK ({args.mode})")
        print("=" * 80)
        for r in results:
            print(f"[{r.status:4s}] {r.name}: {r.detail}")
            if (args.fix_hints or r.status == "FAIL") and r.fix_hint and r.status != "PASS":
                print(f"       FIX: {r.fix_hint}")
        print("-" * 80)
        print(f"SUMMARY: {len(results) - len(fails) - len(warns)} PASS | {len(warns)} WARN | {len(fails)} FAIL")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
