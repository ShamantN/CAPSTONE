"""
Centralized cross-platform path and environment resolution for MMCCP.
Respects explicit environment variables (`MMCCP_ROOT`, `FORGE_URL`, `FORGE_DIR`,
`FORGE_MODELS_DIR`, `FORGE_PYTHON`) while preserving original Windows defaults.
Must remain free of heavy ML imports so preflight/diagnostic scripts load fast.
"""
import os
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Returns repository root, honoring MMCCP_ROOT env override."""
    override = os.environ.get("MMCCP_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def get_forge_url() -> str:
    """Returns WebUI Forge API base URL, honoring FORGE_URL env override."""
    return os.environ.get("FORGE_URL", "http://127.0.0.1:7860").strip().rstrip("/")


def get_forge_dir() -> Path:
    """Returns WebUI Forge root directory, honoring FORGE_DIR env override."""
    override = os.environ.get("FORGE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        return Path(r"C:\Stable-Diffusion")
    return Path.home() / "stable-diffusion-webui-forge"


def get_forge_models_dir() -> Path:
    """Returns Forge `webui/models` directory, honoring FORGE_MODELS_DIR override."""
    override = os.environ.get("FORGE_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    forge_dir = get_forge_dir()
    candidate = forge_dir / "webui" / "models"
    if candidate.exists():
        return candidate
    alt = forge_dir / "models"
    if alt.exists():
        return alt
    return candidate


def get_forge_python() -> Path:
    """
    Resolves Forge's Python interpreter cleanly across Windows, Linux, and Colab.
    Order: FORGE_PYTHON env -> Windows embedded system python -> webui venv -> current sys.executable.
    """
    override = os.environ.get("FORGE_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()

    forge_dir = get_forge_dir()
    if sys.platform == "win32":
        embedded = forge_dir / "system" / "python" / "python.exe"
        if embedded.exists():
            return embedded
        venv_win = forge_dir / "webui" / "venv" / "Scripts" / "python.exe"
        if venv_win.exists():
            return venv_win
    else:
        venv_lin = forge_dir / "webui" / "venv" / "bin" / "python"
        if venv_lin.exists():
            return venv_lin
        alt_lin = forge_dir / "venv" / "bin" / "python"
        if alt_lin.exists():
            return alt_lin

    return Path(sys.executable)


def configure_windows_forge_env() -> None:
    """
    On Windows only, registers Forge's embedded Python DLL paths and site-packages
    when present so shared Torch/Transformers wheels resolve without touching PATH on POSIX.
    """
    if sys.platform != "win32":
        return

    sys_py = get_forge_dir() / "system" / "python"
    if not sys_py.exists():
        return

    dll_paths = [
        sys_py / "Library" / "bin",
        sys_py / "Lib" / "site-packages" / "torch" / "lib",
        sys_py / "Scripts",
        sys_py,
    ]
    current_path = os.environ.get("PATH", "")
    for p in dll_paths:
        if p.exists():
            p_str = str(p)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p_str)
                except Exception:
                    pass
            if p_str not in current_path:
                current_path = p_str + os.pathsep + current_path
    os.environ["PATH"] = current_path

    sd_site = str(sys_py / "Lib" / "site-packages")
    if os.path.exists(sd_site) and sd_site not in sys.path:
        sys.path.append(sd_site)

