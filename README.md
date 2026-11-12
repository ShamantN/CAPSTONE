# Multi-Modal Character Consistency Pipeline (MMCCP) v2.0

**PES University Capstone Project (`PW26_CGM_02`) — Parent Initiative: Novel to Video Generation (NVG)**  
**Department of Computer Science and Engineering | Faculty Guide:** Prof. Chitra G M  
**Student Team:** Samiya Naaz, Shamant Nagabhushana, Shihab K A, Shivu K C  
**Current Release Tag (Pre-Release Finalization):** `v2.0.0-rc3` (`main` PR pending; final `v2.0.0` release tag blocked until PR merge & Phase 6–7 release checks)

> [!IMPORTANT]
> **Quick Navigation:**
> - **Run From Scratch (Windows / Linux / Google Colab):** [`docs/run/00_START_HERE.md`](docs/run/00_START_HERE.md)
> - **Preflight Environment Check (with `my_app_env` active or explicit Windows path):** `my_app_env\Scripts\python.exe tools\verify_setup.py`
> - **Colab Notebook (`UNTESTED` live):** [`notebooks/MMCCP_Colab.ipynb`](notebooks/MMCCP_Colab.ipynb) *(checklist in [`docs/run/03_GOOGLE_COLAB.md`](docs/run/03_GOOGLE_COLAB.md))*
> - **Full Measured Scorecard & Statistical Audit:** [`README_V2.md`](README_V2.md) & [`docs_generated/v2/07_measured_results.md`](docs_generated/v2/07_measured_results.md)
> - **Honest Limitations `(a)–(n)`:** [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)
> - **Reproducibility Commands:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
> - **Contributor Rules & Governance:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 1. Overview & Hardware Target

MMCCP is a training-free, zero-shot multi-character storyboarding pipeline built over Stable Diffusion XL (`Juggernaut-XL v9`) via WebUI Forge (`sdapi/v1/*`) and evaluated under a frozen CPU deterministic verifier (`v1.2`, SHA `3fa69071f888643b...`).

- **Tested Target Hardware:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (`8192` MiB VRAM), Intel Core i7-12650H, `32` GB DDR4 RAM, Windows 11 Home, Python `3.10`, PyTorch `2.3.1+cu121`.
- **No Weights in Git:** Zero binary model weights or generated PNG datasets are committed (`C:\Stable-Diffusion\` and local caches supply weights via [`docs/run/04_MODELS_AND_DOWNLOADS.md`](docs/run/04_MODELS_AND_DOWNLOADS.md)).

---

## 2. Architecture Modules (`M1–M12` Status in Frozen `v2.0`)

In locked `v2.0` (`consistency_v2/config.py` + `consistency_v2/best_config.json`), **8 of 12 numbered methods are ON** (`M1`, `M2`, `M3`, `M4` IP-Adapter Plus-Face, `M5`, `M6`, `M7`, `M10`), **4 of 12 are OFF** (`M8`, `M9`, `M11`, `M12`), and the **`M4b` Dual InstantID sub-option is OFF**:

| Module | Component | Default `v2.0` Config | Implementation / Empirical Note |
| :--- | :--- | :---: | :--- |
| **M1** | Multi-Reference Identity Bank (`identity_bank.py`) | **ON** | Stores certified canonical portrait embeddings & torso CIELAB vectors (`single_anchor` mode used in demo) |
| **M2** | Programmatic Layout Planner (`layout_planner.py`) | **ON** | Non-overlapping boxes + OpenPose COCO-18 guidance; **capped at 2 characters max** |
| **M3** | True Regional Conditioning (`regional.py`) | **ON** | `sd-forge-couple` 17-arg API (`regional_global_weight=0.40`); 2-column partition |
| **M4b** | Dual InstantID sub-option (`identity_units.py`) | **OFF** | `enable_dual_instantid=False` to avoid OpenPose spatial contention & VRAM overflow |
| **M4** | IP-Adapter Plus-Face SDXL (`identity_units.py`) | **ON** (`0.55`) | Active in `v2.0`; removing `M4` edged `full_v2` on 4-judge Bio inside `0.03` noise (`DEV-2` & post-hoc exploratory `TEST-2` ablation) while losing `Facenet512` & `DINOv2` -> **candidate for a multi-seed re-test in `v2.1`** |
| **M5** | Direct InsightFace Face Transplant (`face_swap.py`) | **ON** | CPU `inswapper_128.onnx` + CodeFormer (`0.85` fidelity); `enable_crop_upscale_swap=False` in config |
| **M6** | Adaptive Face Detailer (`face_detailer.py`) | **ON** | Padded face inpaint refinement (`denoise=0.28`, `steps=20`, `padding=48`) |
| **M7** | Wardrobe & Palette Lock (`wardrobe.py`) | **ON** | SegFormer clothes mask + CIELAB transfer toward canonical `clothing_colors` (`Delta E <= 38.0`, `blend_strength=0.70` code-locked; closed-loop with verifier check; static studio attire only) |
| **M8** | Reference-Only Attention (`reference_unit.py`) | **OFF** | `enable_reference_attention=False` (avoids studio background bleed & VRAM growth) |
| **M9** | Best-of-N & Healing (`candidates.py`, `healing_v2.py`) | **OFF** | `enable_best_of_n=False` in default mode and **not executed inside `generate_scene()`** |
| **M10** | Cross-Scene Identity Memory (`identity_memory.py`) | **ON** | ON in locked eval config (`0.75` drift guard) and has **no LOO ablation arm**, so **no empirical benefit is claimed** (`OFF` during browser demo) |
| **M11** | Character LoRA Training (`use_character_lora`) | **OFF** | Explicitly **OUT OF SCOPE** (`{606, 707, 808}` permanently quarantined) |
| **M12** | Collage-Init Composition (`collage_init.py`) | **OFF** | Implemented & unit-tested; `enable_collage_init=False` in locked `v2.0` default config |

---

## 3. Sealed `TEST-2` Headline Results (From `results/numbers.json`)

Evaluated **one-shot** under Frozen Verifier `v1.2` across `42` paired scenes (`64` face slots) comparing MMCCP `v2.0` against Baseline `v1` (`v2.0.0-rc2` post-hoc scorecard shifts retracted; see [`README_V2.md`](README_V2.md) and [`docs_generated/v2/PREREG_RECONCILIATION.md`](docs_generated/v2/PREREG_RECONCILIATION.md)):

| Split (`v2` vs `v1`) | Renders / Faces | Bio Mean | Rank-1 Accuracy | Spatial Pass | Wardrobe Pass | Defects / Scene |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ebright `TEST-2`** | `30` / `40` | `0.3147` vs `0.3091` (`+0.0056`, `p=0.3754`) | **`80.0%`** (`32/40`) vs `65.0%` (`26/40`) | `97.5%` (`39/40`) vs `100.0%` | **`80.0%`** (`32/40`) vs `45.0%` (`p=0.000183`) | **`1.50`** vs `3.07` (`-51.1%`, `p=0.001453`) |
| **Holdout `TEST-2`** | `12` / `24` | `0.2596` vs `0.2758` (`-0.0162`, `p=0.8213`) | `54.2%` (`13/24`) vs `62.5%` (`15/24`) | `87.5%` (`21/24`) vs `79.2%` | `37.5%` (`9/24`) vs `41.7%` | `1.25` vs `1.92` (`-0.67`, `p=0.3242`) |
| **Combined `TEST-2`** | `42` / `64` | `0.2940` vs `0.2966` (`-0.0026`, `p=0.3233`) | `70.31%` (`45/64`) vs `64.06%` | `93.75%` (`60/64`) vs `92.19%` | `64.06%` (`41/64`) vs `43.75%` (`p=0.004556`) | `1.43` vs `2.74` (`-1.31`, `p=0.000877`) |

- **Strict Pre-Registered Hypotheses (`P1–P4`):** **`0/4` all-split rules satisfied** (`P1` CI crosses zero; `P2` Spatial & `P3` Wardrobe meet targets on Ebright/Combined while failing Holdout; `P4` fails both `<= 7.4 GB` whole-system VRAM ceiling and `0` extra YOLO persons).
- **VRAM & Stability (`vram_test2_sealed.csv`, `8,112` samples):** Whole-system peak **`7984` MiB** (`7.797` GiB, `52`s above `7400` MiB, `0` OOMs; `7011` MiB process delta above `973` MiB idle baseline).
- **Combined Binding Gate (`identity_at_position` & `combined_binding`):** `0.0%` for **both** `v1` and `v2` under strict `3/4` judge absolute cosine calibration (`verifier_v1.3_proposal.md`).

---

## 4. Summary of Key Limitations & Demo Assets

- **Architectural & Evaluation Gaps:** Holdout 3-character Scene 4 drops its 3rd character (`(g)`), static studio `clothing_colors` overwrite graduation attire & `M7` wardrobe scoring is closed-loop on that target (`(h)`), Holdout unseen characters show parity/lag behind `v1` (`(c)`), InsightFace `buffalo_l` judge alignment introduces partial circularity (`(i)`), `no_m4_ip_adapter` slightly edged `full_v2` on 4-judge Bio inside `0.03` noise while losing `Facenet512`/`DINOv2` in `DEV-2` & post-hoc exploratory `TEST-2` ablations (`(j)`), `M10` has no ablation arm (`(j)`), and human blind evaluation (`HUMAN_BLIND_TEST.md`) remains pending (`(k)`). Read all `14` items in **[`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)**.
- **From-Scratch Browser Showcase (`DEMO ONLY`):** Sanitized method matrix and verifier scorecard from the local Ebright Scenes `1–5` browser demo live in [`docs/demo/METHODS_USED.md`](docs/demo/METHODS_USED.md) and [`docs/demo/scorecard.csv`](docs/demo/scorecard.csv).

---

## 5. Contributing, Licenses & Citation

- **Contributing:** Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening any branch or PR (`check_freeze.py`, `audit_test2_isolation.py`, `secret_scan.py`, and `unittest` gates apply).
- **Licenses:** No weights are redistributed here, and **no root `LICENSE` file exists for MMCCP's original code** (`LICENSE_FOR_OWN_CODE=""`; standard default copyright applies and the university's IP policy has not been checked). Upstream dependencies/weights **used or required by the pipeline (not redistributed)** include non-commercial (`InsightFace`, `CodeFormer`), `AGPL-3.0` (`WebUI Forge`, `YOLOv8n`), and Open RAIL terms — see [`docs/run/LICENSES_AND_MODELS.md`](docs/run/LICENSES_AND_MODELS.md).
- **Citation:** See [`CITATION.cff`](CITATION.cff).
