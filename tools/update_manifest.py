"""
Manifest Registration Utility
Scans results directories and ensures all artifacts (images, sidecars, reports)
are recorded in results/MANIFEST.json with SHA256, size, git commit, split, and timestamps.
"""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "results" / "MANIFEST.json"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(PROJECT_ROOT))
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def update_manifest():
    commit = get_git_commit()

    manifest = {"manifest_version": "2.0", "total_artifacts": 0, "artifacts": []}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    existing = {a["file_path"].replace("/", "\\"): a for a in manifest.get("artifacts", [])}

    # Directories to scan
    scan_dirs = [
        PROJECT_ROOT / "results" / "baseline_v1" / "dev2",
        PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "ebright",
        PROJECT_ROOT / "results" / "baseline_v1" / "test2" / "holdout",
        PROJECT_ROOT / "results" / "rescored_v1.1" / "baseline_v1_dev2",
        PROJECT_ROOT / "results" / "rescored_v1.1" / "baseline_v1_test2_ebright",
        PROJECT_ROOT / "results" / "rescored_v1.1" / "baseline_v1_test2_holdout",
    ]

    added = 0
    updated = 0

    for s_dir in scan_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.iterdir():
            if not p.is_file():
                continue
            rel_path = str(p.relative_to(PROJECT_ROOT)).replace("/", "\\")
            sha = compute_sha256(p)
            size = p.stat().st_size
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()

            # Determine split and mode
            split = "dev2" if "dev2" in rel_path else ("test2" if "test2" in rel_path else "rescored")
            mode = "baseline_v1"

            entry = {
                "file_path": rel_path,
                "sha256": sha,
                "size_bytes": size,
                "config_hash": "baseline_v1",
                "git_commit": commit,
                "split": split,
                "mode": mode,
                "timestamp": mtime
            }

            if rel_path not in existing:
                existing[rel_path] = entry
                added += 1
            else:
                if existing[rel_path]["sha256"] != sha:
                    existing[rel_path] = entry
                    updated += 1

    # Reconstruct sorted list
    sorted_artifacts = sorted(existing.values(), key=lambda x: x["file_path"])
    manifest["artifacts"] = sorted_artifacts
    manifest["total_artifacts"] = len(sorted_artifacts)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest updated: {len(sorted_artifacts)} total artifacts recorded ({added} added, {updated} updated).")


if __name__ == "__main__":
    update_manifest()
