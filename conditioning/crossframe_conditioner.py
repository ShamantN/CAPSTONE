"""
STAGE 18: CROSS-FRAME CONDITIONER
"""
from typing import Dict, Any, List, Optional


class CrossFrameConditioner:
    """
    Maintains cross-frame character visual DNA continuity and state tracking across consecutive scenes.
    """

    def __init__(self):
        self.frame_history: List[Dict[str, Any]] = []

    def record_frame(self, scene_number: int, image_b64: str, metadata: Dict[str, Any]):
        self.frame_history.append({
            "scene_number": scene_number,
            "image_b64": image_b64,
            "metadata": metadata
        })

    def get_previous_frame_anchor(self, character_id: str) -> Optional[str]:
        """Returns the most recent successfully generated frame containing this character."""
        for frame in reversed(self.frame_history):
            chars = frame["metadata"].get("characters", [])
            if character_id in chars:
                return frame["image_b64"]
        return None
