"""
Consistency v2 Configuration System
Centralizes all toggleable hyper-parameters and flags for M1 through M10.
Includes scope guard preventing evaluation on reserved seeds {606, 707, 808}.
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path


RESERVED_SEEDS = {606, 707, 808}


@dataclass
class ConsistencyV2Config:
    # Mode preset
    mode: str = "default"  # "fast", "default", "quality"

    # M1: Multi-Reference Identity Bank
    enable_identity_bank: bool = True
    bank_max_references: int = 8
    bank_selection_metric: str = "pose_and_scale"

    # M2: Layout Planner (Spatial Binding & Pose Control)
    enable_layout_planner: bool = True
    min_face_size_px_medium: int = 96
    min_face_size_px_wide: int = 64
    openpose_weight: float = 0.70
    openpose_guidance_end: float = 0.75

    # M3: True Regional Conditioning (Forge Couple)
    enable_regional_couple: bool = True
    regional_global_weight: float = 0.40

    # M4: Identity Units (Dual InstantID + IP-Adapter Plus-Face SDXL)
    enable_dual_instantid: bool = True
    instantid_ip_weight: float = 0.65
    instantid_keypoints_weight: float = 0.60
    instantid_guidance_end: float = 0.80
    enable_ip_adapter_plus: bool = True
    ip_adapter_plus_weight: float = 0.55

    # M5: Reliable Identity Transplant
    enable_reliable_face_swap: bool = True
    swap_backend: str = "direct_insightface"  # "direct_insightface" or "reactor_api"
    codeformer_fidelity: float = 0.75

    # M6: Identity-Conditioned Face Detailer
    enable_face_detailer: bool = True
    face_detailer_denoise: float = 0.35
    face_detailer_padding: int = 48
    face_detailer_steps: int = 20

    # M7: Wardrobe and Palette Lock
    enable_wardrobe_lock: bool = True
    wardrobe_delta_e_threshold: float = 38.0
    wardrobe_repair_denoise: float = 0.45

    # M8: Reference-Only Attention Injection
    enable_reference_attention: bool = False
    reference_attention_weight: float = 0.40

    # M9: Best-of-N Candidate Selection & Diagnosis-Driven Healing
    enable_best_of_n: bool = False  # Enabled in quality mode
    best_of_n_candidates: int = 3
    max_healing_retries: int = 2

    # M10: Cross-Scene Identity Memory with Drift Guard
    enable_identity_memory: bool = True
    drift_guard_threshold: float = 0.75
    max_memory_promotions: int = 3

    # Scope Override Extension Point: M11 LoRA training is OUT OF SCOPE
    use_character_lora: bool = False  # Keep disabled; reserved for future tier

    # Generation Base Parameters
    steps_pass1: int = 20
    steps_pass2: int = 15
    cfg_scale: float = 5.5
    sampler_name: str = "DPM++ 2M"
    scheduler: str = "Karras"

    def __post_init__(self):
        if self.mode == "fast":
            self.steps_pass1 = 14
            self.steps_pass2 = 10
            self.enable_face_detailer = False
            self.enable_best_of_n = False
        elif self.mode == "quality":
            self.steps_pass1 = 25
            self.steps_pass2 = 18
            self.enable_face_detailer = True
            self.enable_best_of_n = True
            self.best_of_n_candidates = 3

    def validate_seed(self, seed: int):
        """Scope guard ensuring reserved held-out seeds are never touched."""
        if seed in RESERVED_SEEDS:
            raise ValueError(
                f"FATAL: Seed {seed} is strictly RESERVED for held-out LoRA evaluation. "
                f"Reserved seeds {RESERVED_SEEDS} must NEVER be used for dev, tuning, test, or debug!"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsistencyV2Config":
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def compute_hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
