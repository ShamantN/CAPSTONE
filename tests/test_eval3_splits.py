"""
Unit tests for eval3.splits seed formulas and isolation guarantees.
"""
import unittest
from eval3.splits import (
    verify_seed_disjointness,
    get_anchor_seed,
    get_scene_seed,
    FORBIDDEN_POOLS,
    DEV3_SEEDS,
    TEST3_SEEDS,
    STAGE_D_SEEDS,
)


class TestEval3Splits(unittest.TestCase):
    def test_seed_disjointness_all_partitions(self):
        """All seeds in eval3 must have zero internal collisions and zero forbidden pool overlaps."""
        report = verify_seed_disjointness()
        self.assertTrue(report["passed"], f"Seed disjointness verification failed: {report}")

    def test_forbidden_pools_completely_isolated(self):
        """None of the eval3 seed pools may intersect with TEST-2, DEV-2, or RESERVED."""
        all_eval3_seeds = DEV3_SEEDS | TEST3_SEEDS | STAGE_D_SEEDS
        for pool_name, pool_seeds in FORBIDDEN_POOLS.items():
            overlap = all_eval3_seeds.intersection(pool_seeds)
            self.assertEqual(len(overlap), 0, f"Collision detected with forbidden pool {pool_name}: {overlap}")

    def test_deterministic_seed_generation(self):
        """Seed generator functions must return deterministic, repeatable integers."""
        s1 = get_anchor_seed(replicate_idx=0, char_idx=1, candidate_idx=2, is_test3=False)
        s2 = get_anchor_seed(replicate_idx=0, char_idx=1, candidate_idx=2, is_test3=False)
        self.assertEqual(s1, s2)
        self.assertTrue(7001 <= s1 <= 7200)

        s3 = get_scene_seed(replicate_idx=1, story_idx=2, scene_idx=3, seed_idx=0, is_test3=True)
        s4 = get_scene_seed(replicate_idx=1, story_idx=2, scene_idx=3, seed_idx=0, is_test3=True)
        self.assertEqual(s3, s4)
        self.assertTrue(8451 <= s3 <= 8700)


if __name__ == "__main__":
    unittest.main()
