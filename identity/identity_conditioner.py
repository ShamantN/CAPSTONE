"""
STAGE 9: IDENTITY CONDITIONING ABSTRACTION
Abstract Base Class defining the unified interface for all facial/identity conditioning adapters.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from experiments.controlnet_adapter import ControlSpec
from story_engine.scene_planner import SceneExecutionPlan


class IdentityConditioner(ABC):
    """
    Abstract Base Class for identity conditioning engines (IP-Adapter SDXL, InstantID, FaceID Plus v2).
    """

    @abstractmethod
    def get_adapter_name(self) -> str:
        """Returns the human-readable identifier of the identity adapter."""
        pass

    @abstractmethod
    def build_control_specs(self, plan: SceneExecutionPlan) -> List[ControlSpec]:
        """Translates the scene execution plan into Forge ControlNet ControlSpecs."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the required weights and preprocessors exist in Forge."""
        pass
