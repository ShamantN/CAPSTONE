"""
MMCCP Canonical Verifier Freeze Integrity Check
Verifies that all files in consistency_v2/verify/ match their frozen SHA-256 hashes
recorded in consistency_v2/verifier_freeze.json.
Exits 0 if matched; exits 1 with failure details if tampered/drifted.
"""
import sys
import json
import hashlib
from pathlib import Path

VERIFY_DIR = Path("consistency_v2/verify")
FREEZE_FILE = Path("consistency_v2/verifier_freeze.json")

def hash_file(file_path: Path) -> str:
    """Computes SHA-256 with newline normalization for cross-platform stability."""
    data = file_path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()

def check_freeze() -> bool:
    if not FREEZE_FILE.exists():
        print(f"[ERROR] Verifier freeze file missing: {FREEZE_FILE}", file=sys.stderr)
        return False
        
    with open(FREEZE_FILE, "r", encoding="utf-8") as f:
        freeze_data = json.load(f)
        
    version = freeze_data.get("verifier_version", "unknown")
    recorded_combined = freeze_data.get("combined_sha256", "")
    recorded_files = freeze_data.get("files", {})
    
    mismatches = []
    current_files = {}
    
    for rel_path, expected_hash in sorted(recorded_files.items()):
        full_p = VERIFY_DIR / rel_path
        if not full_p.exists():
            mismatches.append(f"MISSING: {rel_path}")
            continue
        cur_hash = hash_file(full_p)
        current_files[rel_path] = cur_hash
        if cur_hash != expected_hash:
            mismatches.append(f"MODIFIED: {rel_path} (expected {expected_hash[:12]}..., got {cur_hash[:12]}...)")
            
    # Compute current combined hash
    combined_str = "".join(f"{k}:{current_files[k]}\n" for k in sorted(current_files.keys()))
    cur_combined = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
    
    if cur_combined != recorded_combined:
        mismatches.append(f"COMBINED HASH MISMATCH: expected {recorded_combined[:12]}..., got {cur_combined[:12]}...")
        
    if mismatches:
        print(f"[FAIL] Verifier Freeze Integrity Check FAILED for version {version}:", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return False
    else:
        print(f"[PASS] Verifier Freeze Integrity Verified! Version {version} (SHA: {cur_combined[:16]}...)")
        return True

if __name__ == "__main__":
    if check_freeze():
        sys.exit(0)
    else:
        sys.exit(1)
