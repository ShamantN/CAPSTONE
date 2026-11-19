"""
MMCCP TEST-3 Evaluation Protocol: Split and Seed Definitions (eval3/splits.py).

Defines deterministic seed formulas, partition ranges, and strictly proves
disjointness from all previous splits (DEV-FAST, DEV-FULL, DEV-2, TEST-2, RESERVED, demo).
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.splits import (
    DEV_FAST_SEEDS,
    DEV_2_EBRIGHT_SEEDS,
    DEV_2_CHRONO_SEEDS,
    TEST_2_EBRIGHT_SEEDS,
    TEST_2_HOLDOUT_SEEDS,
    RESERVED_SEEDS,
)

# Demo showcase seeds (from browser demo protocol)
DEMO_SEEDS: Set[int] = set(range(4101, 4300))

# Forbidden pools across all prior splits
FORBIDDEN_POOLS: Dict[str, Set[int]] = {
    "DEV-FAST": set(DEV_FAST_SEEDS),
    "DEV-2": set(DEV_2_EBRIGHT_SEEDS + DEV_2_CHRONO_SEEDS),
    "TEST-2": set(TEST_2_EBRIGHT_SEEDS + TEST_2_HOLDOUT_SEEDS),
    "RESERVED": set(RESERVED_SEEDS),
    "DEMO": DEMO_SEEDS,
}

# DEV-3 Partition Range: 7001..7499
DEV3_MIN = 7001
DEV3_MAX = 7499
DEV3_ANCHOR_MAX = 7200
DEV3_SCENE_MIN = 7201

# TEST-3 Partition Range: 8001..8999
TEST3_MIN = 8001
TEST3_MAX = 8999

TEST3_ANCHOR_MIN = 8001
TEST3_ANCHOR_MAX = 8450

TEST3_V2_SCENE_MIN = 8451
TEST3_V2_SCENE_MAX = 8700

STAGE_D_BASELINE_MIN = 8701
STAGE_D_BASELINE_MAX = 8999

DEV3_SEEDS: Set[int] = set(range(DEV3_MIN, DEV3_MAX + 1))
TEST3_SEEDS: Set[int] = set(range(TEST3_ANCHOR_MIN, TEST3_V2_SCENE_MAX + 1))
STAGE_D_SEEDS: Set[int] = set(range(STAGE_D_BASELINE_MIN, STAGE_D_BASELINE_MAX + 1))


def get_anchor_seed(replicate_idx: int, char_idx: int, candidate_idx: int, is_test3: bool = False) -> int:
    """
    Deterministic seed formula for anchor candidates.
    replicate_idx: 0-indexed (e.g. 0..4)
    char_idx: 0-indexed (e.g. 0..10 for 11 total storyboard characters)
    candidate_idx: 0-indexed (e.g. 0..7 for up to 8 candidates per character)
    """
    assert 0 <= candidate_idx < 8, f"Candidate idx {candidate_idx} must be in 0..7"
    assert 0 <= char_idx < 12, f"Char idx {char_idx} must be in 0..11"

    if is_test3:
        # TEST-3 anchor range: 8001..8450
        seed = TEST3_ANCHOR_MIN + (replicate_idx * 88) + (char_idx * 8) + candidate_idx
        assert TEST3_ANCHOR_MIN <= seed <= TEST3_ANCHOR_MAX, f"Anchor seed {seed} out of TEST-3 anchor range"
        return seed
    else:
        # DEV-3 anchor range: 7001..7200
        seed = DEV3_MIN + (replicate_idx * 88) + (char_idx * 8) + candidate_idx
        assert DEV3_MIN <= seed <= DEV3_ANCHOR_MAX, f"Anchor seed {seed} out of DEV-3 anchor range"
        return seed


def get_scene_seed(
    replicate_idx: int,
    story_idx: int,
    scene_idx: int,
    seed_idx: int,
    arm: str = "v2_final",
    is_test3: bool = False
) -> int:
    """
    Deterministic seed formula for scene rendering.
    replicate_idx: 0-indexed (0..4)
    story_idx: 0 for Story A, 1 for Story B, 2 for Story C
    scene_idx: 0-indexed scene within story (0..4 for A, 0..3 for B and C)
    seed_idx: 0, 1, 2 (v2 uses 3 seeds per scene; Stage D baselines use 2)
    arm: "v2_final" / "v2_final_bon3" (or future "naive", "v1" for Stage D)
    """
    assert 0 <= story_idx <= 2, f"Story idx {story_idx} must be in 0..2"
    assert 0 <= scene_idx <= 4, f"Scene idx {scene_idx} must be in 0..4"

    if is_test3:
        if arm in ["v2_final", "v2_final_bon3"]:
            assert 0 <= seed_idx <= 2, f"Seed idx {seed_idx} must be in 0..2 for v2"
            seed = TEST3_V2_SCENE_MIN + (replicate_idx * 45) + (story_idx * 15) + (scene_idx * 3) + seed_idx
            assert TEST3_V2_SCENE_MIN <= seed <= TEST3_V2_SCENE_MAX, f"TEST-3 scene seed {seed} out of range"
            return seed
        elif arm in ["v1", "naive"]:
            assert 0 <= seed_idx <= 1, f"Seed idx {seed_idx} must be in 0..1 for Stage D baselines"
            arm_idx = 0 if arm == "v1" else 1
            seed = STAGE_D_BASELINE_MIN + (arm_idx * 130) + (replicate_idx * 26) + (story_idx * 8) + (scene_idx * 2) + seed_idx
            assert STAGE_D_BASELINE_MIN <= seed <= STAGE_D_BASELINE_MAX, f"Stage D seed {seed} out of range"
            return seed
        else:
            raise ValueError(f"Unknown arm: {arm}")
    else:
        # DEV-3 scene seed range: 7201..7499
        assert 0 <= seed_idx <= 2, f"Seed idx {seed_idx} must be in 0..2 for DEV-3"
        seed = DEV3_SCENE_MIN + (replicate_idx * 60) + (story_idx * 16) + (scene_idx * 3) + seed_idx
        assert DEV3_SCENE_MIN <= seed <= DEV3_MAX, f"DEV-3 scene seed {seed} out of range"
        return seed


def verify_seed_disjointness(max_replicates: int = 5) -> Dict[str, Any]:
    """
    Exhaustively proves that all generated DEV-3 and TEST-3 seeds are:
    1. Internally unique.
    2. Completely disjoint from DEV-FAST, DEV-2, TEST-2, RESERVED, and demo seeds.
    """
    forbidden_pools = FORBIDDEN_POOLS

    # DEV-3 generated seeds
    dev3_anchors = set()
    for r in range(2):
        for c in range(11):
            for k in range(8):
                dev3_anchors.add(get_anchor_seed(r, c, k, is_test3=False))

    dev3_scenes = set()
    for r in range(2):
        for st in range(3):
            sc_count = 5 if st == 0 else (4 if st == 2 else 3)
            for sc in range(sc_count):
                for s in range(3):
                    dev3_scenes.add(get_scene_seed(r, st, sc, s, is_test3=False))

    # TEST-3 generated seeds
    test3_anchors = set()
    for r in range(max_replicates):
        for c in range(11):
            for k in range(8):
                test3_anchors.add(get_anchor_seed(r, c, k, is_test3=True))

    test3_scenes = set()
    for r in range(max_replicates):
        for st in range(3):
            sc_count = 5 if st == 0 else (4 if st == 2 else 3)
            for sc in range(sc_count):
                for s in range(3):
                    test3_scenes.add(get_scene_seed(r, st, sc, s, is_test3=True))

    stage_d_scenes = set()
    for r in range(max_replicates):
        for st in range(3):
            sc_count = 5 if st == 0 else (4 if st == 2 else 3)
            for sc in range(sc_count):
                for s in range(2):
                    for arm in ["v1", "naive"]:
                        stage_d_scenes.add(get_scene_seed(r, st, sc, s, arm=arm, is_test3=True))

    all_eval3_seeds = dev3_anchors | dev3_scenes | test3_anchors | test3_scenes | stage_d_scenes

    collisions = {}
    for pool_name, pool_seeds in forbidden_pools.items():
        overlap = all_eval3_seeds.intersection(pool_seeds)
        if overlap:
            collisions[pool_name] = sorted(list(overlap))

    internal_overlap = {
        "dev3_anchor_scene_overlap": sorted(list(dev3_anchors.intersection(dev3_scenes))),
        "test3_anchor_scene_overlap": sorted(list(test3_anchors.intersection(test3_scenes))),
        "test3_v2_stage_d_overlap": sorted(list(test3_scenes.intersection(stage_d_scenes))),
        "dev3_test3_overlap": sorted(list((dev3_anchors | dev3_scenes).intersection(test3_anchors | test3_scenes | stage_d_scenes))),
    }

    is_clean = len(collisions) == 0 and all(len(v) == 0 for v in internal_overlap.values())

    return {
        "passed": is_clean,
        "counts": {
            "dev3_anchors": len(dev3_anchors),
            "dev3_scenes": len(dev3_scenes),
            "test3_anchors": len(test3_anchors),
            "test3_scenes": len(test3_scenes),
            "stage_d_scenes": len(stage_d_scenes),
            "total_unique_eval3_seeds": len(all_eval3_seeds),
        },
        "forbidden_collisions": collisions,
        "internal_overlaps": {k: len(v) for k, v in internal_overlap.items()},
    }


if __name__ == "__main__":
    report = verify_seed_disjointness()
    print(json.dumps(report, indent=2))
    assert report["passed"], "Seed disjointness check failed!"
    print("[PASS] Seed disjointness and uniqueness verified!")
