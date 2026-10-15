"""
MMCCP Canonical Deterministic Verifier Freezing Tool
Computes normalized SHA-256 hashes for all verifier source code, helpers, and calibration.json.
Writes consistency_v2/verifier_freeze.json with version 1.1.
"""
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

VERIFY_DIR = Path("consistency_v2/verify")
FREEZE_FILE = Path("consistency_v2/verifier_freeze.json")

def hash_file(file_path: Path) -> str:
    """Computes SHA-256 with newline normalization for cross-platform stability."""
    data = file_path.read_bytes()
    # Normalize CRLF to LF for consistent hashing
    normalized = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()

def freeze_verifier(version: str = "1.1") -> Dict[str, Any]:
    tracked_files = [
        "__init__.py",
        "__main__.py",
        "appearance_evaluator.py",
        "binding_checker.py",
        "biometric_ensemble.py",
        "calibration.py",
        "calibration.json",
        "cli.py",
        "scene_auditor.py",
        "torch_bridge.py",
        "torch_vision_helper.py",
        "visual_review/make_blind_sheet.py",
        "visual_review/make_contact_sheet.py",
        "visual_review/rubric.py"
    ]
    
    file_hashes = {}
    for rel_path in sorted(tracked_files):
        full_p = VERIFY_DIR / rel_path
        if full_p.exists():
            file_hashes[rel_path] = hash_file(full_p)
        else:
            print(f"[WARNING] Tracked verifier file not found: {full_p}", file=sys.stderr)
            
    # Combined hash of sorted file entries
    combined_str = "".join(f"{k}:{file_hashes[k]}\n" for k in sorted(file_hashes.keys()))
    combined_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
    
    freeze_data = {
        "verifier_version": version,
        "combined_sha256": combined_hash,
        "files": file_hashes
    }
    
    FREEZE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FREEZE_FILE, "w", encoding="utf-8") as f:
        json.dump(freeze_data, f, indent=2)
        
    print(f"[FREEZE] Verifier {version} frozen successfully!")
    print(f"  Combined SHA-256: {combined_hash}")
    print(f"  Tracked Files:    {len(file_hashes)}")
    return freeze_data

if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "1.1"
    freeze_verifier(ver)
