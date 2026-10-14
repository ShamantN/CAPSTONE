"""
STAGE 22: FORGE API CLIENT
High-reliability HTTP client communicating with local Stable Diffusion Forge WebUI backend.
"""
import requests, json, base64, logging, time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("MMCCP.ForgeClient")


class ForgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip("/")

    def is_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/sdapi/v1/options", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def set_checkpoint(self, checkpoint_name: str) -> bool:
        try:
            payload = {"sd_model_checkpoint": checkpoint_name}
            r = requests.post(f"{self.base_url}/sdapi/v1/options", json=payload, timeout=30)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Failed to set checkpoint: {e}")
            return False

    def txt2img(self, payload: Dict[str, Any], timeout: int = 600) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        """
        Executes txt2img generation. Returns (list of base64 image strings, info metadata dictionary).
        """
        url = f"{self.base_url}/sdapi/v1/txt2img"
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                images = data.get("images", [])
                info = json.loads(data.get("info", "{}"))
                return images, info
            else:
                logger.error(f"txt2img error {r.status_code}: {r.text[:300]}")
                return None, {"error": r.text[:300]}
        except Exception as e:
            logger.error(f"txt2img exception: {e}")
            return None, {"error": str(e)}

    def img2img(self, payload: Dict[str, Any], timeout: int = 600) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        """
        Executes img2img generation. Returns (list of base64 image strings, info metadata dictionary).
        """
        url = f"{self.base_url}/sdapi/v1/img2img"
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                images = data.get("images", [])
                info = json.loads(data.get("info", "{}"))
                return images, info
            else:
                logger.error(f"img2img error {r.status_code}: {r.text[:300]}")
                return None, {"error": r.text[:300]}
        except Exception as e:
            logger.error(f"img2img exception: {e}")
            return None, {"error": str(e)}
