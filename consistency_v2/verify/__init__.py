"""
MMCCP v2 Comprehensive Verification Suite
Provides CPU-isolated biometric ensembling, spatial and wardrobe binding verification,
DINOv2/CLIP-I appearance scoring, and automated defect auditing.
"""
import os
import sys

# Ensure torch/transformers from Forge python are discoverable without environment pollution
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

# Enforce CPU execution for all verifier components
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from .biometric_ensemble import BiometricEnsemble
from .binding_checker import BindingChecker
from .appearance_evaluator import AppearanceEvaluator
from .scene_auditor import SceneAuditor

__all__ = [
    "BiometricEnsemble",
    "BindingChecker",
    "AppearanceEvaluator",
    "SceneAuditor",
]
