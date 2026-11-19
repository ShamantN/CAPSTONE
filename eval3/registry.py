"""
MMCCP TEST-3 Evaluation Protocol: Run Registry (eval3/registry.py).

Discovers and inspects evaluation runs under D:\5TH SEM\HACKATHON_eval\.
Provides real-time run discovery, status tracking, progress metrics, and ETA computation.
"""
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval3.runner import EVAL_OUTPUT_BASE


def discover_evaluation_runs(base_dir: Path = EVAL_OUTPUT_BASE) -> List[Dict[str, Any]]:
    """Discovers all evaluation runs under base_dir that contain a manifest.json."""
    if not base_dir.exists():
        return []

    runs = []
    for run_dir in sorted(base_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        prog = m.get("progress", {})
                        total = prog.get("total_renders", 0)
                        done = prog.get("completed", 0)
                        failed = prog.get("failed", 0)
                        elapsed = prog.get("elapsed_seconds", 0.0)
                        pct = round((done / total) * 100, 1) if total > 0 else 0.0
                        eta = round(((total - done) / (done / elapsed)), 1) if (done > 0 and elapsed > 0 and total > done) else 0.0

                        runs.append({
                            "run_id": m.get("run_id", run_dir.name),
                            "stage": m.get("stage", "A"),
                            "split": m.get("split", "DEV-3"),
                            "status": m.get("status", "UNKNOWN"),
                            "config_hash": m.get("config_hash", "unknown"),
                            "git_commit": m.get("git_commit", "unknown"),
                            "created_at": m.get("created_at", ""),
                            "updated_at": m.get("updated_at", ""),
                            "completed_renders": done,
                            "failed_renders": failed,
                            "total_renders": total,
                            "progress_pct": pct,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta,
                            "run_dir": str(run_dir),
                        })
                except Exception:
                    pass
    return runs


def get_run_details(run_id: str, base_dir: Path = EVAL_OUTPUT_BASE) -> Optional[Dict[str, Any]]:
    """Loads complete manifest and render summary for a specific run ID."""
    manifest_file = base_dir / run_id / "manifest.json"
    if not manifest_file.exists():
        return None
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
