"""
Idempotent patch manager for upstream Stable Diffusion WebUI Forge customizations.
Validates Forge git commit HEAD == dfdcbab685e57677014f05a3309b48cc87383167,
runs `git apply --check` / `--reverse --check` before touching anything,
and touches nothing outside the resolved Forge webui git directory.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.runtime_paths import get_forge_dir

EXPECTED_FORGE_COMMIT = "dfdcbab685e57677014f05a3309b48cc87383167"
DEFAULT_PATCHES_DIR = PROJECT_ROOT / "patches" / "forge"


def resolve_forge_git_dir(forge_dir: Path) -> Path:
    """Resolves the Forge `webui` git checkout directory under `forge_dir`."""
    forge_dir = Path(forge_dir).expanduser()
    if (forge_dir / "webui" / ".git").exists():
        return forge_dir / "webui"
    if (forge_dir / ".git").exists():
        return forge_dir
    raise RuntimeError(
        f"Could not find a Forge git repository (.git) at '{forge_dir}' or '{forge_dir / 'webui'}'. "
        "Cannot safely verify commit or apply patches."
    )


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--no-pager", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def get_patch_files(patches_dir: Path = DEFAULT_PATCHES_DIR) -> List[Path]:
    files = sorted(Path(patches_dir).glob("*.patch"))
    if not files:
        raise RuntimeError(f"No .patch files found in '{patches_dir}'.")
    return files


def verify_forge_commit(repo_dir: Path, expected_commit: str = EXPECTED_FORGE_COMMIT) -> str:
    res = _git(repo_dir, "rev-parse", "HEAD")
    if res.returncode != 0:
        raise RuntimeError(f"Failed to read git HEAD in '{repo_dir}': {res.stderr.strip()}")
    actual = res.stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"Forge commit mismatch in '{repo_dir}': found '{actual}', "
            f"expected '{expected_commit}'. Refusing to touch repository."
        )
    return actual


def inspect_patch_state(repo_dir: Path, patch_file: Path) -> str:
    """Returns one of: ALREADY_APPLIED, UNAPPLIED, CONFLICT."""
    rev = _git(repo_dir, "apply", "--reverse", "--check", str(patch_file))
    if rev.returncode == 0:
        return "ALREADY_APPLIED"
    fwd = _git(repo_dir, "apply", "--check", str(patch_file))
    if fwd.returncode == 0:
        return "UNAPPLIED"
    return "CONFLICT"


def manage_patches(
    action: str,
    forge_dir: Optional[Path] = None,
    patches_dir: Path = DEFAULT_PATCHES_DIR,
    expected_commit: str = EXPECTED_FORGE_COMMIT,
) -> Tuple[int, Dict[str, object]]:
    """
    Executes `--check`, `--apply`, or `--revert` atomically inside `forge_dir`.
    Returns `(exit_code, report_dict)`.
    """
    target_root = Path(forge_dir) if forge_dir is not None else get_forge_dir()
    try:
        repo_dir = resolve_forge_git_dir(target_root)
        actual_commit = verify_forge_commit(repo_dir, expected_commit=expected_commit)
        patch_files = get_patch_files(patches_dir)
    except RuntimeError as exc:
        return 2, {"action": action, "forge_dir": str(target_root), "error": str(exc), "patches": []}

    states = []
    for pf in patch_files:
        st = inspect_patch_state(repo_dir, pf)
        states.append({"name": pf.name, "path": str(pf), "state": st})

    has_conflict = any(s["state"] == "CONFLICT" for s in states)
    if has_conflict:
        return 2, {
            "action": action,
            "repo_dir": str(repo_dir),
            "commit": actual_commit,
            "error": "One or more patches conflict cleanly with target files; aborting without modifying files.",
            "patches": states,
        }

    if action == "check":
        all_applied = all(s["state"] == "ALREADY_APPLIED" for s in states)
        return (0 if all_applied else 1), {
            "action": action,
            "repo_dir": str(repo_dir),
            "commit": actual_commit,
            "all_applied": all_applied,
            "patches": states,
        }

    if action == "apply":
        applied_now: List[str] = []
        skipped_now: List[str] = []
        for pf in patch_files:
            st = inspect_patch_state(repo_dir, pf)
            if st == "ALREADY_APPLIED":
                skipped_now.append(pf.name)
                continue
            chk = _git(repo_dir, "apply", "--check", str(pf))
            if chk.returncode != 0:
                return 2, {
                    "action": action,
                    "repo_dir": str(repo_dir),
                    "commit": actual_commit,
                    "error": f"Pre-apply check failed on {pf.name}: {chk.stderr.strip()}",
                    "patches": states,
                }
            res = _git(repo_dir, "apply", str(pf))
            if res.returncode != 0:
                return 2, {
                    "action": action,
                    "repo_dir": str(repo_dir),
                    "commit": actual_commit,
                    "error": f"Failed applying {pf.name}: {res.stderr.strip()}",
                    "patches": states,
                }
            applied_now.append(pf.name)
        return 0, {
            "action": action,
            "repo_dir": str(repo_dir),
            "commit": actual_commit,
            "applied": applied_now,
            "already_applied": skipped_now,
        }

    if action == "revert":
        reverted_now: List[str] = []
        skipped_now = []
        for pf in reversed(patch_files):
            st = inspect_patch_state(repo_dir, pf)
            if st == "UNAPPLIED":
                skipped_now.append(pf.name)
                continue
            res = _git(repo_dir, "apply", "--reverse", str(pf))
            if res.returncode != 0:
                return 2, {
                    "action": action,
                    "repo_dir": str(repo_dir),
                    "commit": actual_commit,
                    "error": f"Failed reverting {pf.name}: {res.stderr.strip()}",
                    "patches": states,
                }
            reverted_now.append(pf.name)
        return 0, {
            "action": action,
            "repo_dir": str(repo_dir),
            "commit": actual_commit,
            "reverted": reverted_now,
            "already_stock": skipped_now,
        }

    return 2, {"error": f"Unsupported action: {action}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check, apply, or revert MMCCP patches inside WebUI Forge")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check whether all MMCCP Forge patches are applied (exit 0 if applied, 1 if unapplied, 2 on error/mismatch)")
    group.add_argument("--apply", action="store_true", help="Idempotently apply missing MMCCP patches after commit + git apply --check validation")
    group.add_argument("--revert", action="store_true", help="Idempotently revert MMCCP patches back to stock upstream files")
    parser.add_argument("--forge-dir", default=None, help="Forge root directory (defaults to FORGE_DIR / runtime_paths.get_forge_dir())")
    parser.add_argument("--patches-dir", default=str(DEFAULT_PATCHES_DIR), help="Directory containing *.patch files")
    parser.add_argument("--expected-commit", default=EXPECTED_FORGE_COMMIT, help="Required Forge git commit HEAD")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    action = "check" if args.check else ("apply" if args.apply else "revert")
    code, report = manage_patches(
        action=action,
        forge_dir=Path(args.forge_dir) if args.forge_dir else None,
        patches_dir=Path(args.patches_dir),
        expected_commit=args.expected_commit,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 72)
        print(f"  FORGE PATCH MANAGER ({action.upper()})")
        print("=" * 72)
        if report.get("error"):
            print(f"[ERROR] {report['error']}")
        if "repo_dir" in report:
            print(f"Target Repo: {report['repo_dir']} (HEAD: {report.get('commit', '?')})")
        for p in report.get("patches", []):
            print(f"  - {p['name']}: {p['state']}")
        if "applied" in report:
            print(f"  Applied: {report['applied']} | Already applied: {report['already_applied']}")
        if "reverted" in report:
            print(f"  Reverted: {report['reverted']} | Already stock: {report['already_stock']}")
    return code


if __name__ == "__main__":
    sys.exit(main())

