"""
Unit tests for consistency_v2.splits access guards and split sealing.
"""
import os
import tempfile
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
        # Create isolated temporary log path for testing so real results/split_access_log.jsonl is untouched
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_log_path = Path(self.tmp_dir.name) / "test_split_access.jsonl"
        os.environ["SPLIT_ACCESS_LOG_PATH"] = str(self.test_log_path)

        # Ensure clean environment
        if "ALLOW_TEST2" in os.environ:
            del os.environ["ALLOW_TEST2"]
        if "ALLOW_RESERVED" in os.environ:
            del os.environ["ALLOW_RESERVED"]

    def tearDown(self):
        if "SPLIT_ACCESS_LOG_PATH" in os.environ:
            del os.environ["SPLIT_ACCESS_LOG_PATH"]
        if "ALLOW_TEST2" in os.environ:
            del os.environ["ALLOW_TEST2"]
        if "ALLOW_RESERVED" in os.environ:
            del os.environ["ALLOW_RESERVED"]
        self.tmp_dir.cleanup()

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
        self.assertNotIn("ALLOW_TEST2", os.environ)
        with self.assertRaises(PermissionError):
            assert_split_allowed("TEST-2")
        with self.assertRaises(PermissionError):
            assert_seed_allowed(1101)

    def test_reserved_blocked_by_default(self):
        """RESERVED split and seeds must be blocked when ALLOW_RESERVED is unset."""
        self.assertNotIn("ALLOW_RESERVED", os.environ)
        with self.assertRaises(PermissionError):
            assert_split_allowed("RESERVED")
        for s in RESERVED_SEEDS:
            with self.assertRaises(ValueError):
                assert_seed_allowed(s)

    def test_test2_unlocked_with_env(self):
        """TEST-2 unlocks and logs access to isolated test log when ALLOW_TEST2=1."""
        os.environ["ALLOW_TEST2"] = "1"
        try:
            assert_split_allowed("TEST-2", caller_info="unit_test")
            assert_seed_allowed(1101)
            
            # Check access log in temp directory
            self.assertTrue(self.test_log_path.exists())
            with open(self.test_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertTrue(any("TEST-2" in l for l in lines))
        finally:
            if "ALLOW_TEST2" in os.environ:
                del os.environ["ALLOW_TEST2"]

    def test_reserved_seeds_strictly_forbidden_in_generation(self):
        """Reserved seeds 606, 707, 808 must raise ValueError even if ALLOW_TEST2 is set."""
        os.environ["ALLOW_TEST2"] = "1"
        for s in RESERVED_SEEDS:
            with self.assertRaises(ValueError):
                assert_seed_allowed(s)

if __name__ == "__main__":
    unittest.main()

