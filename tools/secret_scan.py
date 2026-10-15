"""
MMCCP Secret Scanner
Scans staged, unstaged, and untracked files for secret patterns:
AIza, sk-, ghp_, github_pat_, hf_, api_key, token, password, secret
"""
import re
import sys
import subprocess
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),
    re.compile(r'sk-[a-zA-Z0-9]{20,48}'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
    re.compile(r'github_pat_[a-zA-Z0-9_]{50,90}'),
    re.compile(r'hf_[a-zA-Z0-9]{34,40}'),
    re.compile(r'(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*["\']([^"\'\s]{8,})["\']')
]

# Patterns that are safe / false positives (like variable names, dummy placeholders)
ALLOW_LIST = [
    "RESERVED_SEEDS",
    "password", # type annotation or generic word
    "token", # nlp token
    "api_key", # parameter name
]

def scan_files(paths):
    findings = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), start=1):
                for pat in SECRET_PATTERNS:
                    m = pat.search(line)
                    if m:
                        matched_str = m.group(0)
                        # Filter out common false positives
                        if any(safe in line for safe in ["#", "//", "test", "dummy", "placeholder", "export", "unique_token"]):
                            continue
                        findings.append((str(path), line_no, matched_str[:20] + "..."))
        except Exception as e:
            pass
    return findings

def main():
    # Get changed/untracked files from git
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    files_to_scan = []
    for line in res.stdout.splitlines():
        if len(line) > 3:
            fpath = line[3:].strip()
            # handle renames
            if " -> " in fpath:
                fpath = fpath.split(" -> ")[1].strip()
            files_to_scan.append(fpath)
            
    print(f"[Secret Scanner] Scanning {len(files_to_scan)} files for credentials...")
    findings = scan_files(files_to_scan)
    if findings:
        print(f"[CRITICAL ALERT] Found {len(findings)} potential secrets:")
        for f in findings:
            print(f"  {f[0]}:{f[1]} -> {f[2]}")
        sys.exit(1)
    else:
        print("[Secret Scanner] CLEAN: No secrets detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
