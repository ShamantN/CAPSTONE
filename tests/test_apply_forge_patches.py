"""
Unit tests for tools/apply_forge_patches.py using a temporary toy git repository.
Verifies --check, idempotent --apply, idempotent --revert, commit mismatch refusal,
missing .git refusal, and atomic conflict refusal.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.apply_forge_patches import manage_patches


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "--no-pager", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


class TestApplyForgePatches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.forge_root = self.root / "forge"
        self.repo = self.forge_root / "webui"
        self.patches_dir = self.root / "patches"
        self.repo.mkdir(parents=True)
        self.patches_dir.mkdir(parents=True)

        _git(self.repo, "init")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "core.autocrlf", "false")

        self.file_a = self.repo / "alpha.py"
        self.file_b = self.repo / "beta.py"
        self.file_a.write_text("def hello():\n    return 'stock_a'\n", encoding="utf-8", newline="\n")
        self.file_b.write_text("def world():\n    return 'stock_b'\n", encoding="utf-8", newline="\n")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-m", "stock forge")
        self.commit = _git(self.repo, "rev-parse", "HEAD")

        # Create clean patches in self.patches_dir by temporarily modifying and diffing
        self.file_a.write_text("def hello():\n    return 'patched_a'\n", encoding="utf-8", newline="\n")
        (self.patches_dir / "01_a.patch").write_text(_git(self.repo, "diff", "--no-ext-diff", "alpha.py") + "\n", encoding="utf-8", newline="\n")
        _git(self.repo, "checkout", "--", "alpha.py")

        self.file_b.write_text("def world():\n    return 'patched_b'\n", encoding="utf-8", newline="\n")
        (self.patches_dir / "02_b.patch").write_text(_git(self.repo, "diff", "--no-ext-diff", "beta.py") + "\n", encoding="utf-8", newline="\n")
        _git(self.repo, "checkout", "--", "beta.py")

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_apply_idempotent_and_revert(self):
        # 1. Check before apply -> exit 1 (unapplied)
        rc, rep = manage_patches("check", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 1)
        self.assertFalse(rep["all_applied"])

        # 2. Apply -> exit 0
        rc, rep = manage_patches("apply", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 0)
        self.assertEqual(rep["applied"], ["01_a.patch", "02_b.patch"])
        self.assertIn("patched_a", self.file_a.read_text(encoding="utf-8"))
        self.assertIn("patched_b", self.file_b.read_text(encoding="utf-8"))

        # 3. Second apply -> idempotent exit 0, nothing applied
        rc, rep = manage_patches("apply", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 0)
        self.assertEqual(rep["applied"], [])
        self.assertEqual(rep["already_applied"], ["01_a.patch", "02_b.patch"])

        # 4. Check after apply -> exit 0
        rc, rep = manage_patches("check", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 0)
        self.assertTrue(rep["all_applied"])

        # 5. Revert -> exit 0, files back to stock
        rc, rep = manage_patches("revert", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 0)
        self.assertEqual(set(rep["reverted"]), {"01_a.patch", "02_b.patch"})
        self.assertIn("stock_a", self.file_a.read_text(encoding="utf-8"))

        # 6. Second revert -> idempotent exit 0
        rc, rep = manage_patches("revert", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 0)
        self.assertEqual(rep["reverted"], [])

    def test_refuses_commit_mismatch_and_conflict(self):
        rc, rep = manage_patches("apply", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit="deadbeef" * 5)
        self.assertEqual(rc, 2)
        self.assertIn("mismatch", rep["error"].lower())
        self.assertIn("stock_a", self.file_a.read_text(encoding="utf-8"))

        # Modify beta.py directly to create conflict on 02_b.patch, ensure 01_a.patch is NOT touched atomically
        self.file_b.write_text("completely different content\n", encoding="utf-8")
        rc, rep = manage_patches("apply", forge_dir=self.forge_root, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 2)
        self.assertIn("stock_a", self.file_a.read_text(encoding="utf-8"))

    def test_refuses_non_git_dir(self):
        non_git = self.root / "plain_dir"
        non_git.mkdir()
        rc, rep = manage_patches("check", forge_dir=non_git, patches_dir=self.patches_dir, expected_commit=self.commit)
        self.assertEqual(rc, 2)
        self.assertIn(".git", rep["error"])


if __name__ == "__main__":
    unittest.main()

