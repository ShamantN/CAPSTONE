# Release Notes — MMCCP `v2.0.0`

- **Tag:** `v2.0.0` (`e5cd94c956c0e407907d09e5bae98158d52d40f3`)
- **Commit on `main`:** `1ecf548f684aa54f6bb8dac8d2d174c358f88809` (PR #37 merge of `finalize/v2.0.0` @ `3cd3895`)
- **Date:** `2026-11-12`
- **Project:** PES University Capstone (`PW26_CGM_02`) — Novel to Video Generation (NVG)
- **Frozen Verifier:** `v1.2` (`SHA-256: 3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6`)

---

## 1. Overview & Active Architecture (`v2.0.0`)

MMCCP `v2.0.0` is a training-free, zero-shot multi-character storyboarding pipeline built over Stable Diffusion XL (`Juggernaut-XL v9`) via WebUI Forge (`sdapi/v1/*`) targeting 8 GB VRAM GPUs (`8192` MiB tested on RTX 3070 Ti Laptop GPU).

In locked `v2.0` (`consistency_v2/config.py` and `consistency_v2/best_config.json`, config hash `df4a0a2648acdfe1`), **8 of 12 numbered methods are ON** (`M1`, `M2`, `M3`, `M4` IP-Adapter Plus-Face `0.55`, `M5`, `M6`, `M7`, `M10`), **4 are OFF** (`M8`, `M9`, `M11`, `M12`), and the **`M4b` Dual InstantID ControlNet sub-option is OFF** (`enable_dual_instantid=False`). `app.py` interactive sidebar defaults match this config hash in `default` mode.

---

## 2. Highlights & Deliverables in `v2.0.0`

- **Documentation, Reproducibility & Limitations Disclosure:**
  - Updated [`README.md`](../README.md), [`README_V2.md`](../README_V2.md), [`CHANGELOG.md`](../CHANGELOG.md), and [`CITATION.cff`](../CITATION.cff) with exact `results/numbers.json` metrics and retracted `v2.0.0-rc2` scorecard framing.
  - Added [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) covering limitations `(a)–(n)` with artifact provenance, [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (CPU-only and DEV GPU reproduction protocols), and sanitized `DEMO ONLY` artifacts ([`docs/demo/METHODS_USED.md`](demo/METHODS_USED.md), [`docs/demo/scorecard.csv`](demo/scorecard.csv)).
  - Added [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`docs/maintainers/REPO_SETTINGS.md`](maintainers/REPO_SETTINGS.md), [`docs/run/LICENSES_AND_MODELS.md`](run/LICENSES_AND_MODELS.md), [`docs/run/MAINTAINER_SETUP.md`](run/MAINTAINER_SETUP.md), `.github/CODEOWNERS`, `.github/pull_request_template.md`, and `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`.
- **Portable Setup, Patches & Colab Package:**
  - Cross-platform path layer (`consistency_v2/runtime_paths.py`), environment preflight checker (`tools/verify_setup.py`), idempotent Forge patch manager (`tools/apply_forge_patches.py` + `patches/forge/*.patch`), runbooks (`docs/run/00_START_HERE.md`–`07_FAQ_AND_COSTS.md`), and `UNTESTED` Colab notebook (`notebooks/MMCCP_Colab.ipynb`).
- **Automated Verification & Claim Governance:**
  - Single-source metric collector (`tools/collect_numbers.py` -> `results/numbers.json`), doc renderer (`tools/render_docs.py`), human blind evaluation validator/scorer (`tools/score_human_blind_test.py` with Clopper-Pearson 95% CIs, exiting `2` while template is unfilled), and quantitative/prose linter (`tools/lint_claims.py`).

---

## 3. Sealed `TEST-2` Scorecard Summary (`results/numbers.json`)

One-shot paired comparison across `42` scenes (`64` face slots) under Frozen Verifier `v1.2` (with strict `0/4` pre-registered hypotheses `P1–P4` met across all splits):

| Split (`v2` vs `v1`) | Renders / Faces | Bio Mean | Rank-1 Accuracy | Spatial Pass | Wardrobe Pass | Defects / Scene |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ebright `TEST-2`** | `30` / `40` | `0.3147` vs `0.3091` (`+0.0056`, `p=0.3754`) | `80.0%` (`32/40`) vs `65.0%` (`26/40`) | `97.5%` (`39/40`) vs `100.0%` | `80.0%` (`32/40`) vs `45.0%` (`p=0.000183`) | `1.50` vs `3.07` (`-51.1%`, `p=0.001453`) |
| **Holdout `TEST-2`** | `12` / `24` | `0.2596` vs `0.2758` (`-0.0162`, `p=0.8213`) | `54.2%` (`13/24`) vs `62.5%` (`15/24`) | `87.5%` (`21/24`) vs `79.2%` | `37.5%` (`9/24`) vs `41.7%` | `1.25` vs `1.92` (`-0.67`, `p=0.3242`) |
| **Combined `TEST-2`** | `42` / `64` | `0.2940` vs `0.2966` (`-0.0026`, `p=0.3233`) | `70.31%` (`45/64`) vs `64.06%` | `93.75%` (`60/64`) vs `92.19%` | `64.06%` (`41/64`) vs `43.75%` (`p=0.004556`) | `1.43` vs `2.74` (`-1.31`, `p=0.000877`) |

- **Peak Whole-System VRAM (`vram_test2_sealed.csv`):** `7984` MiB (`7.797` GiB, `0` OOMs).
- **Combined Binding Gate (`identity_at_position` & `combined_binding`):** `0.0%` for both `v1` and `v2`.

---

## 4. Known Limitations & License Status

- Key limits documented in [`docs/KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md): `(g)` 2-character layout cap; `(h)` static M7 `clothing_colors` overwrite & closed-loop wardrobe metric; `(c)` Holdout parity/lag; `(i)` InsightFace judge overlap; `(j)` `M4` mixed ablation tradeoff (`candidate for a multi-seed re-test in v2.1`) across `DEV-2` and post-hoc exploratory `TEST-2` ablations + unablated `M10` (zero empirical benefit claimed); `(k)` human blind evaluation pending (`docs_generated/v2/visual_review/blind_predictions_human.json`).
- **Licensing:** No root `LICENSE` file is included (`LICENSE_FOR_OWN_CODE=""`; standard default copyright applies and PES University's IP policy has not been checked). Third-party weights/tools used or required by the pipeline (never bundled/redistributed) include non-commercial (`InsightFace`, `CodeFormer`), `AGPL-3.0` (`WebUI Forge`, `YOLOv8n`), and Open RAIL models — see [`docs/run/LICENSES_AND_MODELS.md`](run/LICENSES_AND_MODELS.md).

---

## 5. Verification Summary (`1ecf548` / `v2.0.0`)

- **Fresh Temp Clone Smoke Test (`origin/main` @ `1ecf548`):** `verify_setup.py --mode new-machine --skip-forge --fix-hints` (`10 PASS | 4 WARN | 0 FAIL`), `sync_canonical_banks.py` (`8/8` characters `1.0000` cosine similarity), `unittest discover -s tests -p "test_*.py"` (`83/83 OK` in `32.341s` post-hydration), `check_freeze.py` (`PASS`, `Version 1.2`), `lint_claims.py` (`68/68` docs, `0` issues), `audit_test2_isolation.py` (`PASSED`). (`apply_forge_patches.py --check` reported CRLF/whitespace `CONFLICT` on `C:\Stable-Diffusion\webui` while `git apply --ignore-whitespace --reverse --check` succeeded `0` on all 3 patches; `tests.verifier_controls.test_controls` skipped/errored on fresh clone because local canary image is gitignored.)
- **Main Workspace (`D:\5TH SEM\HACKATHON` @ `1ecf548`):** Canonical banks synced (`8/8` `SHA-256` exact match, `1.0000` cosine), `83/83` unit tests `OK` (`33.183s`), `15/15` verifier controls `OK` (`202.975s`), `68/68` claim-lint docs `0` issues, `audit_test2_isolation.py` `PASSED`, `secret_scan.py` `CLEAN`, `check_freeze.py` `PASS`.

