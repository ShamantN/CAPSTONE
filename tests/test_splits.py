"""
Unit tests for consistency_v2.splits access guards and split sealing.
"""
import os
import unittest
from pathlib import Path
from consistency_v2.splits import (
    assert_split_allowed,
    assert_seed_allowed,
    DEV_FAST_SEEDS,
    DEV_FULL_SEEDS,
    DEV_2_EBRIGHT_SEEDS,
    TEST_2_EBRIGHT_SEEDS,
    RESERVED_SEEDS
)

class TestSplitsGuards(unittest.TestCase):
    def setUp(self):
        # Ensure clean environment
        if "ALLOW_TEST2" in os.environ:
            del os.environ["ALLOW_TEST2"]
        if "ALLOW_RESERVED" in os.environ:
            del os.environ["ALLOW_RESERVED"]

    def test_dev_splits_allowed(self):
        """DEV-FAST, DEV-FULL, DEV-2 must be permitted unconditionally."""
        assert_split_allowed("DEV-FAST")
        assert_split_allowed("DEV-FULL")
        assert_split_allowed("DEV-2")
        assert_seed_allowed(11)
        assert_seed_allowed(23)
        assert_seed_allowed(101)

    def test_test2_blocked_by_default(self):
        """TEST-2 must be blocked without explicit ALLOW_TEST2=1."""
        with self.assertRaises(PermissionError):
            assert_split_allowed("TEST-2")
        with self.assertRaises(PermissionError):
            assert_seed_allowed(1101)

    def test_test2_unlocked_with_env(self):
        """TEST-2 unlocks and logs access when ALLOW_TEST2=1."""
        os.environ["ALLOW_TEST2"] = "1"
        try:
            assert_split_allowed("TEST-2", caller_info="unit_test")
            assert_seed_allowed(1101)
            
            # Check access log
            log_p = Path("results/split_access_log.jsonl")
            self.assertTrue(log_p.exists())
            with open(log_p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertTrue(any("TEST-2" in l for l in lines))
        finally:
            if "ALLOW_TEST2" in os.environ:
                del os.environ["ALLOW_TEST2"]

    def test_reserved_seeds_strictly_forbidden(self):
        """Reserved seeds 606, 707, 808 must raise ValueError even if ALLOW_TEST2 is set."""
        os.environ["ALLOW_TEST2"] = "1"
        for s in RESERVED_SEEDS:
            with self.assertRaises(ValueError):
                assert_seed_allowed(s)

if __name__ == "__main__":
    unittest.main()
