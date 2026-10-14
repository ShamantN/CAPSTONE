# Git Setup & Pre-Flight Forensic Log

**Repository URL:** `https://github.com/ShamantN/CAPSTONE.git`  
**Remote Name:** `origin`  
**GitHub Authenticated User:** `ShamantN` (ID: `146366065`)  
**Commit Email:** `146366065+ShamantN@users.noreply.github.com`  
**Visibility:** `PUBLIC`  
**Timestamp:** October 14, 2026  

---

## 1. Pre-Flight Diagnostic Check

- **Git Version:** `2.53.0.windows.2`
- **Git LFS Version:** `3.7.1` (initialized and hooks updated)
- **GitHub CLI Version:** `gh version 2.80.0`
- **Authentication State:** Logged in to `github.com` account `ShamantN` via keyring (`gho_...` with scopes `gist`, `read:org`, `repo`, `workflow`).
- **Disk Space Available:**
  - `D:\` (Project drive): **204.89 GB free** (out of 399.99 GB)
  - `C:\` (System & Forge drive): **22.53 GB free** (out of 452.60 GB)
- **Windows Power Configuration:**
  - Original Standby AC Timeout: **900 seconds (15 mins)**
  - Original Hibernate AC Timeout: **10,800 seconds (180 mins)**
  - Session Adjustment: Both AC sleep and hibernate set to **0 (disabled)** via `powercfg /change` to prevent sleep during overnight evaluation runs.

---

## 2. Remote Repository State & Case Classification

- Initial remote check:
  - Remote was **NOT empty**: Commit `cf114dc110060baf28dc3edda5e381af7fcbc2d4` on `main` contained an earlier prototype (`cap.py` and initial `README.md`).
- Classification: **CASE C** (Local directory was not yet initialized as a git repo, but remote contained prior history).

### Forensic Actions Taken:
1. Initialized local git repo with default branch `main`.
2. Configured repo-local `user.name` as `ShamantN` and `user.email` as `146366065+ShamantN@users.noreply.github.com`.
3. Added remote `origin https://github.com/ShamantN/CAPSTONE.git`.
4. Fetched `origin`.
5. Executed non-destructive `git reset --mixed origin/main` to preserve all local working files.
6. Created and verified `.gitignore` to strictly exclude virtual environments, heavy models (`*.safetensors`, `*.onnx`), local caches (`edu_outputs/`, `hardcoded_cache.json`), and temporary files.
7. Created `.gitattributes` configuring Git LFS for curated image artifacts (`docs_generated/**/*.png`, `results/curated/**/*.png`).
8. Verified that no single staged file exceeds the 25 MB gate (largest tracked file: 7.27 MB JSON).
9. Performed regex secret scanning across all files to ensure zero API keys or credentials are committed. Created `.env.example`.
10. Added `.github/pull_request_template.md` and `.github/workflows/ci.yml`.
11. Preserved `cap.py` and `README.md` from the remote repository.
12. Created branch `sync/local-baseline` and committed the baseline import:
    `feat(baseline): import full MMCCP multi-module pipeline and documentation (v1)`
13. Tagged annotated tag `baseline-v1` on this baseline commit.
14. Pushed `sync/local-baseline` and tag `baseline-v1` to `origin`.
15. Created Pull Request #1 into `main`:
    `https://github.com/ShamantN/CAPSTONE/pull/1` (left open for user review).
16. Created and checked out integration branch `consistency-v2` from the baseline commit and pushed with upstream tracking:
    `origin/consistency-v2`.

**Current Repository Status:** Clean, synchronized with `origin/consistency-v2`.
