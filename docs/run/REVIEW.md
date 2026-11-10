# MMCCP v2.0 — Run Package Pre-Merge Self-Review (`REVIEW.md`)

- **Review Status:** `PASSED AUTOMATED + MANUAL AUDIT` (2026-11-10)
- **Target Branch:** `docs/run-guides` (based on `main` `625b365` / `v2.0.0-rc3`)

---

## 1. Automated Gate Checks

| Check | Scope | Result | Notes |
| :--- | :--- | :---: | :--- |
| **File length cap (`< 600` lines each)** | All `11` markdown files under `docs/run/` | **PASS** | Range `47–338` lines per file (`01_NEW_MACHINE_WINDOWS.md` is `338` lines) |
| **Required status + baseline headers** | All files under `docs/run/` | **PASS** | Top lines explicitly declare `TESTED` / `VERIFIED` / `UNTESTED` + branch/commit |
| **Stale marker grep** | Unresolved draft tags across `docs/run/` | **PASS (`0` hits)** | Verified by regex scan |
| **Secret scan (`tools/secret_scan.py`)** | All repo modified + untracked files | **PASS (`CLEAN`)** | Also verified `0` API key/token literals in `docs/run/` & `notebooks/` |
| **Relative markdown links** | All relative markdown targets in `docs/run/*` | **PASS (`0` broken)** | Verified across all `docs/run/*.md` files |
| **Personal path leak audit** | `01_NEW_MACHINE_WINDOWS.md` & newbie guides | **PASS** | Uses generic `<MMCCP_ROOT>` / `C:\Projects\MMCCP` & `env` vars |
| **Notebook structural validation** | `notebooks/MMCCP_Colab.ipynb` (`34` cells: `18` md, `16` code) | **PASS** | `json.loads`, nbformat v4 keys, Python `ast.parse`, `bash -n` on `30` shell cmds, `7/7` URL/size/SHA parity with `WEIGHT_MANIFEST` |
| **Forge patch CLI + clean copy test** | `patches/forge/*.patch`, `tools/apply_forge_patches.py` | **PASS (`recommended`)** | Toy-repo unit tests `3/3 OK`; clean copy at `D:\5TH SEM\_forge_clean_test\` tested unpatched & patched (`RC=0` each, no M4 crash reproduced on Plus-Face) |
| **Fresh `venv` Frozen Verifier parity** | DEV-2 (`30` renders, `40` faces) + `15` controls | **PASS (`0.00000000` diff)** | `max_abs_diff = 0.00000000` vs `report_v1_2.json`; `15/15` `test_controls.py` `OK` |
| **Frozen Verifier (`v1.2`) integrity** | `tools/check_freeze.py` | **PASS** | `Version 1.2 (SHA: 3fa69071f888643b...)` untouched |
| **Split hygiene (`tools/audit_test2_isolation.py`)** | Whole repo Python tree + notebook | **PASS** | `TEST-2` & `RESERVED` splits untouched; zero unauthorized seed references |
| **Canonical docs consistency (`tests.test_docs_numbers`)** | `README_V2.md` & `docs_generated/v2/*` | **PASS (`4/4 OK`)** | Re-rendered cleanly via `tools/render_docs.py` |

---

## 2. `NEEDS SOURCE` Register (Every Unverified Checksum or Missing Upstream Spec)

Per zero-fabrication rules, items without an exact verified SHA-256 checksum on local disk / official host are explicitly tagged `NEEDS SOURCE`:

| Item | What Is Verified | What Still Needs Source | Document Location |
| :--- | :--- | :--- | :--- |
| **`webui_forge_cu121_torch231.7z`** | Live GitHub Release HTTP `200`, byte size `1,870,203,670` (`1.74 GiB`), license (`GPL-3.0`) | Exact SHA-256 (upstream uses rolling `latest` tag; extract commit on disk is `dfdcbab6...`) | [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) |
| **`mattmdjaga/segformer_b2_clothes`** | HuggingFace URL (`HTTP 200`), `109,493,236` B, HF license `other` | Standalone file SHA-256 (`NEEDS SOURCE` — stored inside local HF blob cache, hash not extracted separately) | [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) |
| **`facebook/dinov2-base`** | HuggingFace URL (`HTTP 200`), `346,345,912` B, `apache-2.0` | Standalone file SHA-256 (`NEEDS SOURCE` — HF blob cache) | [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) |
| **`openai/clip-vit-base-patch32`** | HuggingFace URL (`HTTP 200`), `605,247,071` B, MIT | Standalone file SHA-256 (`NEEDS SOURCE` — HF blob cache) | [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) |
| **`Salesforce/blip-image-captioning-base`** | HuggingFace URL (`HTTP 200`), `989,820,849` B, `bsd-3-clause` | Standalone file SHA-256 (`NEEDS SOURCE` — HF blob cache) | [04_MODELS_AND_DOWNLOADS.md](04_MODELS_AND_DOWNLOADS.md) |

---

## 3. `TESTED` vs `UNTESTED` Register

| Procedure / Surface | Classification | Evidence / Exact Scope |
| :--- | :---: | :--- |
| **Windows fresh `git clone` -> `py -3.10` `venv` -> `torch 2.3.1+cu121` -> `requirements.txt`** | **TESTED** | Verified end-to-end in `D:\5TH SEM\_fresh_machine_test` (`TEST_LOG.md`), `torch.cuda.is_available() == True` |
| **Preflight (`tools/verify_setup.py`) pre-Forge SHA-256 & live Forge API** | **TESTED** | `11 PASS / 3 WARN / 0 FAIL` pre-Forge; `12 PASS / 3 WARN / 0 FAIL` live Forge (`tests/test_verify_setup.py` `7/7`) |
| **Canonical bank JSON->PNG hydration (`tools/sync_canonical_banks.py`) & `72` CPU unit tests** | **TESTED** | `8/8` anchors decoded with byte-exact SHA-256, cosine `1.0000`, `72/72` `unittest` pass (`21.35s`) |
| **1-scene zero-cache DEV smoke test (`scene 1, seed 11, --mode fast`) + VRAM + visual review** | **TESTED** | Render `174.86s`, total w/ anchors+eval `312.71s`, peak system VRAM `7887.0 MiB` (`286` samples), output inspected via `read_file` |
| **Streamlit `app.py` headless boot + zero-cache reset (`Remove-Item hardcoded_cache.json ...`)** | **TESTED** | HTTP `200 ok` on `/_stcore/health`, clean cleanup preserving original `D:\5TH SEM\HACKATHON` caches |
| **Full 6-scene zero-cache `v2 default` demo & DEV-2 rescore** | **TESTED** *(on main repo)* | `18.72 min` 6-scene demo (`2026-11-10` archive) & DEV-2 `report_v1_2.json` — not repeated inside `_fresh_machine_test` |
| **Empty-PC WebUI Forge `.7z` extraction & manual `~14–22 GiB` weight download** | **UNTESTED** | Tested against pre-existing `C:\Stable-Diffusion` read-only; live HTTP `HEAD` & SHA-256 verified |
| **Unpatched upstream Forge `IPAdapterPlus.py` / `controlnet.py` / `utils.py` (`v2 default`)** | **TESTED** *(no crash reproduced on `Plus-Face`)* | Clean stock copy at `D:\5TH SEM\_forge_clean_test` completed `scene 1, seed 11` (`RC=0`); 3 patches shipped as recommended hardening |
| **Fresh `venv` DEV-2 rescore + `15` Verifier controls** | **TESTED** | `max_abs_diff = 0.00000000` vs `results/bundle_dev2/report_v1_2.json`; `15/15` `test_controls.py` OK |
| **Linux setup (`02_LINUX_NOTES.md`)** | **UNTESTED** | Portability helpers & `/proc/meminfo` unit-tested with mocks (`tests/test_portability.py` `6/6`), no live Linux GPU run |
| **Google Colab (`03_GOOGLE_COLAB.md` + `notebooks/MMCCP_Colab.ipynb`)** | **UNTESTED by choice** | Statically validated JSON/AST/`bash -n`/SHA parity; no live Colab T4 execution |
| **`< 8 GB` VRAM GPUs (e.g., 6 GB `--lowvram`)** | **UNTESTED / UNSUPPORTED** | `8 GB` GPU already reaches `7887–7984 MiB` system peak |

---

## 4. 10-Minute Skim Review Check

- **Newcomer flow (`00_START_HERE.md` -> `01_NEW_MACHINE_WINDOWS.md` -> `04_MODELS_AND_DOWNLOADS.md`):** Path decision table sits on screen 1; Windows steps are copy-pasteable with exact expected outputs and common traps (`py -3.10`, OpenPose rename, `sync_canonical_banks.py`, Forge boot warm-up, pre-existing Forge IP-Adapter patch note).
- **Zero fabrication check:** Every package version, timing, byte count, SHA-256, URL, VRAM sample, and limitation in `docs/run/` comes verbatim from local tool outputs, `step0_raw_facts.json`, `results/numbers.json`, or `TEST_LOG.md`.
