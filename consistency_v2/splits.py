"""
MMCCP Data Splits Specification & Access Guard
Defines immutable splits (DEV-FAST, DEV-FULL, DEV-2, TEST-2, RESERVED)
and provides strict guards to prevent data leakage and enforce sealing.
"""
import os
import sys
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. Split Definitions ---

# DEV-FAST: screening only (reduced steps: Pass 1 @ 14, Pass 2 @ 10)
DEV_FAST_SCENES = [1, 2, 4, 5]
DEV_FAST_SEEDS = [11, 23, 37]

# DEV-FULL: full steps across all 6 Ebright scenes
DEV_FULL_SCENES = [1, 2, 3, 4, 5, 6]
DEV_FULL_SEEDS = [11, 23, 37]

# DEV-2: old test set, now usable as dev/composition/tuning
DEV_2_EBRIGHT_SCENES = [1, 2, 3, 4, 5, 6]
DEV_2_EBRIGHT_SEEDS = [101, 202, 303, 404, 505]
DEV_2_CHRONO_SEEDS = [101, 202, 303]

# TEST-2 (SEALED): Touched strictly ONCE during final evaluation in Phase R10
TEST_2_EBRIGHT_SCENES = [1, 2, 3, 4, 5, 6]
TEST_2_EBRIGHT_SEEDS = [1101, 1202, 1303, 1404, 1505]
TEST_2_HOLDOUT_SEEDS = [1101, 1202, 1303]
TEST_2_HOLDOUT_SCENES = [1, 2, 3, 4]

# RESERVED: NEVER USED anywhere (not dev, test, tuning, or debug)
RESERVED_SEEDS = [606, 707, 808]

SPLIT_NAMES = ["DEV-FAST", "DEV-FULL", "DEV-2", "TEST-2", "RESERVED"]


def _log_split_access(split: str, caller_info: str):
    """Appends split access event to results/split_access_log.jsonl."""
    log_dir = PROJECT_ROOT / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "split_access_log.jsonl"
    
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split": split,
        "caller": caller_info,
        "process_id": os.getpid(),
        "env_allow_test2": os.environ.get("ALLOW_TEST2", "0"),
        "env_allow_reserved": os.environ.get("ALLOW_RESERVED", "0")
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def assert_split_allowed(split: str, caller_info: str = ""):
    """
    Enforces access guards on data splits:
    - DEV-FAST, DEV-FULL, DEV-2: Always allowed.
    - TEST-2: Requires ALLOW_TEST2=1 environment variable.
    - RESERVED: Requires ALLOW_RESERVED=1 environment variable.
    """
    split_norm = split.upper().replace("_", "-")
    
    if split_norm in ["TEST-2", "TEST2"]:
        if os.environ.get("ALLOW_TEST2") != "1":
            raise PermissionError(
                "Access to sealed split TEST-2 is FORBIDDEN! "
                "TEST-2 is sealed until Phase R10 final evaluation. "
                "To unlock for final evaluation, set environment variable ALLOW_TEST2=1."
            )
        _log_split_access("TEST-2", caller_info)
        
    elif split_norm == "RESERVED":
        if os.environ.get("ALLOW_RESERVED") != "1":
            raise PermissionError(
                "Access to RESERVED split is FORBIDDEN! "
                "Reserved seeds {606, 707, 808} must never be used."
            )
        _log_split_access("RESERVED", caller_info)


def assert_seed_allowed(seed: int, split: Optional[str] = None):
    """
    Validates that a seed is not in the forbidden or sealed pools.
    """
    if seed in RESERVED_SEEDS:
        raise ValueError(
            f"Seed {seed} belongs to the RESERVED seeds {RESERVED_SEEDS} "
            "and must NEVER be used under any circumstances."
        )
        
    all_test2_seeds = TEST_2_EBRIGHT_SEEDS + TEST_2_HOLDOUT_SEEDS
    if seed in all_test2_seeds:
        if os.environ.get("ALLOW_TEST2") != "1":
            raise PermissionError(
                f"Seed {seed} belongs to sealed TEST-2! "
                "It cannot be accessed during development or tuning."
            )
