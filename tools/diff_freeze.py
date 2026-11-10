"""
tools/diff_freeze.py
Computes package drift between initial freeze snapshots and current environments.
Outputs machine-verified markdown to docs_generated/v2/repair/environment_drift.md.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_freeze(file_path: Path):
    packages = {}
    if not file_path.exists():
        return packages
    raw_bytes = file_path.read_bytes()
    # Detect encoding
    if raw_bytes.startswith(b"\xff\xfe") or b"\x00" in raw_bytes[:100]:
        text = raw_bytes.decode("utf-16-le", errors="replace")
    elif raw_bytes.startswith(b"\xfe\xff"):
        text = raw_bytes.decode("utf-16-be", errors="replace")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip().strip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            parts = line.split("==", 1)
            packages[parts[0].lower()] = parts[1]
        elif "@" in line:
            parts = line.split("@", 1)
            packages[parts[0].strip().lower()] = "@" + parts[1].strip()
        else:
            packages[line.lower()] = "installed"
    return packages

def compare_envs(initial_path: Path, current_path: Path, env_name: str) -> list:
    init = parse_freeze(initial_path)
    curr = parse_freeze(current_path)

    added = {k: curr[k] for k in curr if k not in init}
    removed = {k: init[k] for k in init if k not in curr}
    changed = {k: (init[k], curr[k]) for k in curr if k in init and curr[k] != init[k]}

    lines = [
        f"### {env_name}",
        f"- Total packages initially: **{len(init)}**",
        f"- Total packages currently: **{len(curr)}**",
        f"- Newly Added packages: **{len(added)}**",
        f"- Removed packages: **{len(removed)}**",
        f"- Version Changed packages: **{len(changed)}**",
        ""
    ]

    if added:
        lines.append("#### Newly Added Packages:")
        for k, v in sorted(added.items()):
            lines.append(f"- `{k}=={v}`")
        lines.append("")

    if removed:
        lines.append("#### Removed Packages:")
        for k, v in sorted(removed.items()):
            lines.append(f"- `{k}=={v}`")
        lines.append("")

    if changed:
        lines.append("#### Version Changed Packages:")
        for k, (v_old, v_new) in sorted(changed.items()):
            lines.append(f"- `{k}`: `{v_old}` -> `{v_new}`")
        lines.append("")

    if not added and not removed and not changed:
        lines.append("✅ **Zero package drift detected.** Environment is 100% identical to initial snapshot.\n")

    return lines

def main():
    out_file = PROJECT_ROOT / "docs_generated" / "v2" / "repair" / "environment_drift.md"

    lines = [
        "# Environment Drift Audit Report",
        "",
        "> Machine-generated audit comparing current pip freeze against Phase 0 initial snapshots.",
        "> Audits: Rule 8 & Problem B11.",
        ""
    ]

    # 1. my_app_env
    init_app = PROJECT_ROOT / "docs_generated" / "v2" / "my_app_env_freeze.txt"
    curr_app = PROJECT_ROOT / "docs_generated" / "v2" / "repair" / "my_app_env_current.txt"
    lines.extend(compare_envs(init_app, curr_app, "Project Python (`my_app_env`)"))

    # 2. Forge Python
    init_forge = PROJECT_ROOT / "docs_generated" / "v2" / "forge_python_freeze.txt"
    curr_forge = PROJECT_ROOT / "docs_generated" / "v2" / "repair" / "forge_python_current.txt"
    lines.extend(compare_envs(init_forge, curr_forge, "Forge Embedded Python (`C:\\Stable-Diffusion\\system\\python`)"))

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Environment drift audit written to {out_file}")

if __name__ == "__main__":
    main()
