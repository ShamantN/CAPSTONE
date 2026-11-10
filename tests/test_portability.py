"""
Unit tests for cross-platform path resolution & portability fixes.
Pure CPU / mock tests; zero Forge or GPU required.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from consistency_v2 import runtime_paths
from experiments import memory_utils
from generation.forge_client import ForgeClient
from consistency_v2.face_detailer import FaceDetailer
from consistency_v2.face_swap import DirectFaceSwapper
from story_engine.character_registry import CharacterRegistry


class TestPortability(unittest.TestCase):

    def test_runtime_paths_env_overrides(self):
        with mock.patch.dict(os.environ, {
            "MMCCP_ROOT": tempfile.gettempdir(),
            "FORGE_URL": "http://127.0.0.1:7861/",
            "FORGE_DIR": "/tmp/custom_forge",
            "FORGE_MODELS_DIR": "/tmp/custom_models",
            "FORGE_PYTHON": "/tmp/custom_py",
        }):
            self.assertEqual(runtime_paths.get_project_root(), Path(tempfile.gettempdir()).resolve())
            self.assertEqual(runtime_paths.get_forge_url(), "http://127.0.0.1:7861")
            self.assertEqual(runtime_paths.get_forge_dir(), Path("/tmp/custom_forge"))
            self.assertEqual(runtime_paths.get_forge_models_dir(), Path("/tmp/custom_models"))
            self.assertEqual(runtime_paths.get_forge_python(), Path("/tmp/custom_py"))

    def test_configure_windows_forge_env_noop_on_posix(self):
        with mock.patch.object(sys, "platform", "linux"):
            # Must not raise or touch anything on POSIX
            runtime_paths.configure_windows_forge_env()

    def test_get_host_ram_mb_linux_proc_meminfo(self):
        fake_meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         2048000 kB\n"
            "MemAvailable:    8192000 kB\n"
        )
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch("os.path.exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=fake_meminfo)):
            total, avail, used = memory_utils.get_host_ram_mb()
            self.assertAlmostEqual(total, 16000.0, places=1)
            self.assertAlmostEqual(avail, 8000.0, places=1)
            self.assertAlmostEqual(used, 8000.0, places=1)

    def test_clients_honor_forge_url_env(self):
        with mock.patch.dict(os.environ, {"FORGE_URL": "http://localhost:9999/"}):
            fc = ForgeClient()
            fd = FaceDetailer()
            self.assertEqual(fc.base_url, "http://localhost:9999")
            self.assertEqual(fd.forge_api_url, "http://localhost:9999")

    def test_face_swap_template_has_no_hardcoded_c_drive(self):
        self.assertNotIn(r"C:\Stable-Diffusion", DirectFaceSwapper.WORKER_TEMPLATE)
        self.assertIn("models_root = sys.argv[5]", DirectFaceSwapper.WORKER_TEMPLATE)

    def test_character_registry_reads_local_storyboards_not_author_pc(self):
        reg = CharacterRegistry()
        c = reg.get_character("young_ebright")
        self.assertIsNotNone(c)
        self.assertEqual(c.id, "young_ebright")


if __name__ == "__main__":
    unittest.main()

