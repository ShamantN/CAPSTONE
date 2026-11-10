"""
Unit tests for tools/verify_setup.py.
Uses pure mocks; does NOT import torch/cv2/deepface or download weights so CI stays fast.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.verify_setup as vs


class TestVerifySetup(unittest.TestCase):

    def test_no_heavy_imports_on_module_load(self):
        code = (
            "import sys, tools.verify_setup; "
            "heavy = [m for m in ('torch','cv2','deepface','transformers','ultralytics') if m in sys.modules]; "
            "assert not heavy, f'Unexpected heavy imports: {heavy}'"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_parse_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "requirements-lock.txt"
            p.write_text("# comment\nnumpy==1.26.4\ndeepface==0.0.99\n", encoding="utf-8")
            parsed = vs.parse_lock_file(p)
            self.assertEqual(parsed["numpy"], "1.26.4")
            self.assertEqual(parsed["deepface"], "0.0.99")

    def test_check_python_version_modes(self):
        self.assertEqual(vs.check_python_version("new-machine", (3, 10, 6)).status, "PASS")
        self.assertEqual(vs.check_python_version("new-machine", (3, 11, 9)).status, "WARN")
        self.assertEqual(vs.check_python_version("colab", (3, 11, 11)).status, "PASS")
        self.assertEqual(vs.check_python_version("new-machine", (3, 9, 13)).status, "FAIL")

    def test_check_split_guards(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            res = vs.check_freeze_and_splits(Path("."))
            split_res = [r for r in res if "Split" in r.name][0]
            self.assertEqual(split_res.status, "PASS")

        with mock.patch.dict(os.environ, {"ALLOW_TEST2": "1"}):
            res = vs.check_freeze_and_splits(Path("."))
            split_res = [r for r in res if "Split" in r.name][0]
            self.assertEqual(split_res.status, "FAIL")

    def test_check_weights_manual_vs_auto_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            proj_dir = Path(tmp) / "proj"
            (models_dir / "Stable-diffusion").mkdir(parents=True)
            payload = b"dummy_sdxl_weights"
            fp = models_dir / "Stable-diffusion" / "ckpt.safetensors"
            fp.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()

            manifest = [
                {
                    "name": "Req Manual",
                    "rel_path": "Stable-diffusion/ckpt.safetensors",
                    "root": "forge_models",
                    "required": True,
                    "auto_downloaded": False,
                    "size_bytes": len(payload),
                    "sha256": digest,
                },
                {
                    "name": "Auto Missing",
                    "rel_path": "yolov8n.pt",
                    "root": "project_root",
                    "required": True,
                    "auto_downloaded": True,
                    "size_bytes": 100,
                    "sha256": "0" * 64,
                },
            ]
            res = vs.check_weights(models_dir, proj_dir, verify_sha256=True, manifest=manifest)
            by_name = {r.name: r for r in res}
            self.assertEqual(by_name["Manual Model Weights"].status, "PASS")
            self.assertEqual(by_name["Auto-Downloaded Weights"].status, "WARN")
            self.assertEqual(by_name["Weight Integrity (SHA-256)"].status, "PASS")

            # Missing required manual weight -> FAIL
            fp.unlink()
            res2 = vs.check_weights(models_dir, proj_dir, verify_sha256=False, manifest=manifest)
            by_name2 = {r.name: r for r in res2}
            self.assertEqual(by_name2["Manual Model Weights"].status, "FAIL")

    def test_check_forge_api_mocked(self):
        def fake_get(url, timeout=5):
            if url.endswith("/sdapi/v1/options"):
                return {"sd_model_checkpoint": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", "forge_inference_memory": 3072.0, "xl_GPU_MB": 1600.0}
            if url.endswith("/sdapi/v1/script-info"):
                return [
                    {"name": "controlnet"},
                    {"name": "forge couple"},
                    {"name": "freeu integrated (sd 1.x, sd 2.x, sdxl)"},
                    {"name": "dynamicthresholding (cfg-fix) integrated"},
                ]
            if url.endswith("/controlnet/model_list"):
                return {"model_list": ["openpose_xl [f4251cb4]", "ip-adapter-plus-face_sdxl_vit-h [368cf551]"]}
            raise RuntimeError("Unexpected URL")

        with mock.patch("tools.verify_setup._http_get_json", side_effect=fake_get):
            res = vs.check_forge_api("http://127.0.0.1:7860", skip_forge=False)
            self.assertEqual(len(res), 3)
            self.assertTrue(all(r.status == "PASS" for r in res), res)

    def test_check_forge_api_missing_model_fails(self):
        def fake_get(url, timeout=5):
            if url.endswith("/sdapi/v1/options"):
                return {"sd_model_checkpoint": "Juggernaut-XL"}
            if url.endswith("/sdapi/v1/script-info"):
                return [{"name": "controlnet"}]
            if url.endswith("/controlnet/model_list"):
                return {"model_list": []}
            return {}

        with mock.patch("tools.verify_setup._http_get_json", side_effect=fake_get):
            res = vs.check_forge_api("http://127.0.0.1:7860", skip_forge=False)
            self.assertTrue(any(r.status == "FAIL" for r in res))

    def test_check_forge_patches_pass_and_warn(self):
        with mock.patch("tools.apply_forge_patches.manage_patches", return_value=(0, {"patches": [{"name": "01.patch", "state": "ALREADY_APPLIED"}]})):
            r = vs.check_forge_patches(forge_dir=Path("C:/FakeForge"))
            self.assertEqual(r.status, "PASS")

        with mock.patch("tools.apply_forge_patches.manage_patches", return_value=(1, {"patches": [{"name": "01.patch", "state": "UNAPPLIED"}]})):
            r = vs.check_forge_patches(forge_dir=Path("C:/FakeForge"))
            self.assertEqual(r.status, "WARN")
            self.assertIn("apply_forge_patches.py", r.fix_hint)


if __name__ == "__main__":
    unittest.main()
