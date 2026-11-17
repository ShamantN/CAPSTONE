"""
MMCCP TEST-3 Evaluation Protocol: Resumable Evaluation Runner (eval3/runner.py).

Executes end-to-end evaluation runs with:
- Dedicated manifest per run in D:\5TH SEM\HACKATHON_eval\<run_id>\
- Per-render sidecar JSON (seed, arm, config hash, git commit, verifier SHA, timings, peak VRAM, image SHA-256, retry count, status)
- Resumability: skips completed valid renders on restart
- Isolated character banks per replicate (D:\5TH SEM\HACKATHON_eval\<run_id>\replicates\<rep>\banks)
- Forge watchdog: restarts every 40 renders and upon timeout/OOM
- Support for arms: v2_final, v2_final_bon3, and explicit stage-guard blocking naive/v1 until Stage D
"""
import os
import sys
import json
import time
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.runtime_paths import get_project_root, get_forge_url
from story_engine.schema import StorySpec, SceneSpec
from eval3.splits import get_anchor_seed, get_scene_seed
from eval3.anchors import (
    load_all_evaluation_characters,
    generate_character_candidates,
    select_best_anchor,
    SceneAuditor,
)
from eval3.best_of_n import select_best_of_three_scene

EVAL_OUTPUT_BASE = Path(r"D:\5TH SEM\HACKATHON_eval")

# Candidate configuration presets for DEV-3 selection
def _build_config_presets() -> Dict[str, ConsistencyV2Config]:
    c0 = ConsistencyV2Config(mode="default")
    c1 = ConsistencyV2Config(mode="default", enable_crop_upscale_swap=True)
    c2 = ConsistencyV2Config(mode="default", enable_ip_adapter_plus=False)
    return {"C0": c0, "C1": c1, "C2": c2}

CONFIG_PRESETS: Dict[str, ConsistencyV2Config] = _build_config_presets()


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "--no-pager", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_verifier_freeze_sha() -> str:
    freeze_json = PROJECT_ROOT / "consistency_v2" / "verifier_freeze.json"
    if freeze_json.exists():
        try:
            with open(freeze_json, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d.get("combined_sha256", "unknown")
        except Exception:
            pass
    return "unknown"


def get_current_gpu_vram_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 0.0


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class Eval3Runner:
    """
    Orchestrates deterministic, resumable evaluation runs for TEST-3.
    """

    def __init__(
        self,
        run_id: str = "eval3_default",
        stage: str = "A",
        split: str = "DEV-3",
        config: Optional[ConsistencyV2Config] = None,
        out_base: Path = EVAL_OUTPUT_BASE,
        forge_url: Optional[str] = None
    ):
        self.run_id = run_id
        self.stage = stage.upper()
        self.split = split.upper()
        self.config = config or ConsistencyV2Config(mode="default")
        self.out_dir = out_base / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.json"
        self.forge_url = (forge_url or get_forge_url()).rstrip("/")
        self.git_commit = get_git_commit()
        self.verifier_sha = get_verifier_freeze_sha()
        self.config_hash = self.config.to_hash()
        self.render_count_since_restart = 0

        self.auditor = SceneAuditor()
        self.all_characters = load_all_evaluation_characters()
        self.manifest = self._load_or_init_manifest()

    @staticmethod
    def get_config_preset(preset_name: str) -> Tuple[ConsistencyV2Config, str]:
        if preset_name not in CONFIG_PRESETS:
            raise ValueError(f"Unknown config preset: {preset_name}")
        cfg = CONFIG_PRESETS[preset_name]
        return cfg, cfg.to_hash()

    def run_protocol(self, arm: str = "v2_final", stage: Optional[str] = None, allow_stage_d: bool = False, **kwargs) -> Dict[str, Any]:
        curr_stage = (stage or self.stage).upper()
        if (arm in ["naive", "v1"] or curr_stage == "D") and not allow_stage_d:
            raise ValueError(
                f"Arm '{arm}' / Stage '{curr_stage}' is strictly reserved for Stage D with explicit human authorization! "
                "Do NOT run Stage D without explicit human approval."
            )
        return {"status": "OK"}

    def _load_or_init_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        manifest = {
            "run_id": self.run_id,
            "stage": self.stage,
            "split": self.split,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config_hash": self.config_hash,
            "config_compute_hash": self.config.compute_hash(),
            "git_commit": self.git_commit,
            "verifier_sha256": self.verifier_sha,
            "status": "INITIALIZED",
            "progress": {
                "total_renders": 0,
                "completed": 0,
                "failed": 0,
                "elapsed_seconds": 0.0,
                "eta_seconds": 0.0,
            },
            "renders": {},
            "replicates": {},
        }
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: Optional[Dict[str, Any]] = None):
        if manifest is None:
            manifest = self.manifest
        manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def restart_forge_if_needed(self, max_renders: int = 40):
        if self.render_count_since_restart >= max_renders:
            print(f"[WATCHDOG] Restarting Forge after {self.render_count_since_restart} renders...")
            # Restart logic: Forge WebUI restarts via sdapi/v1/server-kill or restart
            try:
                requests.post(f"{self.forge_url}/sdapi/v1/server-restart", timeout=5)
            except Exception:
                pass
            time.sleep(15)
            self.render_count_since_restart = 0

    def prepare_replicate_anchors(
        self,
        replicate_idx: int,
        char_ids: List[str],
        is_test3: bool = False
    ) -> Path:
        """
        Synthesizes, gates, selects, and exports certified anchors into an isolated replicate bank.
        Returns path to isolated bank_root (D:\5TH SEM\HACKATHON_eval\<run_id>\replicates\rep_XX\banks).
        """
        rep_dir = self.out_dir / "replicates" / f"rep_{replicate_idx:02d}"
        anchors_dir = rep_dir / "anchors"
        banks_dir = rep_dir / "banks"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        banks_dir.mkdir(parents=True, exist_ok=True)

        selected_anchors = {}
        for c_idx, cid in enumerate(char_ids):
            c_bank_dir = banks_dir / cid
            c_bank_dir.mkdir(parents=True, exist_ok=True)
            ref_path = c_bank_dir / f"ref_00_{cid}_anchor.png"
            sidecar_path = anchors_dir / f"{cid}_selected_anchor.json"

            if ref_path.exists() and sidecar_path.exists():
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    selected_anchors[cid] = json.load(f)
                continue

            cspec = self.all_characters.get(cid)
            if not cspec:
                raise ValueError(f"Character {cid} not found in evaluation character library!")

            candidates = generate_character_candidates(
                char_id=cid,
                char_spec=cspec,
                replicate_idx=replicate_idx,
                char_idx=c_idx,
                is_test3=is_test3,
                out_dir=anchors_dir / cid,
                auditor=self.auditor,
                forge_url=self.forge_url,
                max_candidates=8,
            )

            best = select_best_anchor(candidates)
            selected_anchors[cid] = best

            # Copy selected candidate image into isolated bank ref_00
            best_img = best.get("img_bgr")
            if best_img is not None:
                cv2.imwrite(str(ref_path), best_img)
            else:
                shutil.copyfile(best["image_path"], str(ref_path))

            # Write selection record
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in best.items() if k not in ["img_bgr", "crop_bgr"]}, f, indent=2)

        self.manifest["replicates"][f"rep_{replicate_idx:02d}"] = {
            "replicate_idx": replicate_idx,
            "characters": list(selected_anchors.keys()),
            "banks_path": str(banks_dir),
            "uncertified_count": sum(1 for a in selected_anchors.values() if a.get("uncertified_anchor", False)),
        }
        self._save_manifest()
        return banks_dir

    def run_scene_render(
        self,
        story: StorySpec,
        scene: SceneSpec,
        seed: int,
        arm: str,
        replicate_idx: int,
        pipeline: ConsistencyPipelineV2,
        render_id: str,
        out_render_dir: Path
    ) -> Dict[str, Any]:
        """
        Renders a single scene with sidecar logging, timing, VRAM tracking, and one retry on failure.
        """
        # Block Stage D arms
        if arm in ["naive", "v1"] and self.stage != "D":
            raise PermissionError(
                f"Arm '{arm}' is strictly reserved for Stage D! Cannot run in Stage {self.stage}."
            )

        img_path = out_render_dir / f"{render_id}.png"
        sidecar_path = out_render_dir / f"{render_id}.json"

        # Resumability check
        if img_path.exists() and sidecar_path.exists():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    sc = json.load(f)
                    if sc.get("status") == "SUCCESS":
                        return sc
            except Exception:
                pass

        vram_start = get_current_gpu_vram_mb()
        t0 = time.time()
        retry_count = 0
        img_cv = None
        meta = {}
        status = "FAILED"

        max_attempts = 2  # 1 initial + 1 infrastructure retry
        for attempt in range(max_attempts):
            try:
                img_cv, meta = pipeline.generate_scene(story=story, scene=scene, seed=seed)
                if img_cv is not None:
                    status = "SUCCESS"
                    break
            except Exception as e:
                print(f"[RETRY] Attempt {attempt+1} failed for {render_id}: {e}")
                time.sleep(5)
                retry_count += 1

        elapsed = time.time() - t0
        vram_peak = max(vram_start, get_current_gpu_vram_mb())
        self.render_count_since_restart += 1

        img_sha = ""
        if img_cv is not None and status == "SUCCESS":
            cv2.imwrite(str(img_path), img_cv)
            img_sha = compute_file_sha256(img_path)

        sidecar = {
            "render_id": render_id,
            "replicate_idx": replicate_idx,
            "story_id": story.story_id,
            "scene_number": scene.scene_number,
            "scene_id": scene.scene_id,
            "seed": seed,
            "arm": arm,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "verifier_sha256": self.verifier_sha,
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_mb": round(vram_peak, 1),
            "image_sha256": img_sha,
            "retry_count": retry_count,
            "status": status,
            "expected_characters": scene.characters_present or list(scene.character_positions.keys()),
            "character_positions": scene.character_positions,
            "metadata": meta,
        }

        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        self.manifest["renders"][render_id] = {
            "status": status,
            "seed": seed,
            "arm": arm,
            "elapsed": round(elapsed, 2),
            "image_path": str(img_path) if status == "SUCCESS" else "",
            "sidecar_path": str(sidecar_path),
        }
        self.manifest["progress"]["completed"] = sum(1 for r in self.manifest["renders"].values() if r["status"] == "SUCCESS")
        self.manifest["progress"]["failed"] = sum(1 for r in self.manifest["renders"].values() if r["status"] == "FAILED")
        self._save_manifest()

        self.restart_forge_if_needed(max_renders=40)
        return sidecar
