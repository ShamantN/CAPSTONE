# Known Limitations & Honest Disclosure (`MMCCP v2.0`)

This document consolidates every verified technical, statistical, hardware, governance, and documentation limitation identified through frozen `TEST-2` evaluation, exploratory diagnostics, upstream license checks, and live runs (`results/numbers.json`, [`docs_generated/v2/07_measured_results.md`](../docs_generated/v2/07_measured_results.md), [`docs_generated/v2/PREREG_RECONCILIATION.md`](../docs_generated/v2/PREREG_RECONCILIATION.md)).

> [!NOTE]
> Open engineering improvements `(g)` and `(h)` below are scoped as beginner-friendly `v2.1` exploration tasks on **DEV splits only** in [`CONTRIBUTING.md`](../CONTRIBUTING.md#4-starter-tasks-for-collaborators-v21-roadmap--not-started-in-frozen-v20).

---

### (a) Child / Young-Character Biometric Sensitivity & Low Similarity
- **Observation:** On both from-scratch generation and certain `TEST-2` contrasts, young/child character slots exhibit high variance or degraded face recognition scores under the frozen 4-judge biometric ensemble (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`).
- **Evidence:** On combined `TEST-2` (`n=19` child face slots), paired `v2 - v1` biometric delta is `-0.0404` (cluster 95% CI `[-0.1665, +0.0404]`; [`results/numbers.json`](../results/numbers.json)), and on Holdout `TEST-2` (`n=9` child faces) `v2` scored `0.3132` vs `v1` `0.3854`. In the from-scratch `DEMO ONLY` browser showcase (`docs/demo/scorecard.csv`), `young_ebright` scored `0.0572` (Scene 1) and `0.1183` (Scene 2) with `0%` Rank-1 against other family characters.

### (b) Unmeasurable Combined Binding Gate (`0.0%` for Both `v1` and `v2`)
- **Observation:** Under Frozen Verifier `v1.2` ([`consistency_v2/verify/cli.py`](../consistency_v2/verify/cli.py)), both `identity_at_position_pass_rate` and `combined_binding_pass_rate` are `0.0%` (`0/40` Ebright `TEST-2`, `0/24` Holdout `TEST-2`) for **both** Baseline `v1` and MMCCP `v2`.
- **Root Cause & Evidence:** `ensemble_passed` requires at least `3` of `4` absolute judge cosine thresholds to pass simultaneously, whereas diffusion crops produce low raw cosines on `SFace` (`0.1365`), `GhostFaceNet` (`0.0294`), and `ArcFace` (`0.0313`) on Ebright `TEST-2` ([`results/numbers.json`](../results/numbers.json); [`docs_generated/v2/verifier_v1.3_proposal.md`](../docs_generated/v2/verifier_v1.3_proposal.md)). Because the gate saturates at zero across all conditions, combined binding is unmeasurable under frozen `v1.2`.

### (c) Holdout Generalization Parity / Lag vs Baseline `v1`
- **Observation:** On unseen Holdout characters (`n=12` scenes, `24` face slots), MMCCP `v2` was **not better than Baseline `v1`**.
- **Evidence:** Holdout `TEST-2` `v2` vs `v1`: Bio Mean `0.2596` vs `0.2758` (`Delta = -0.0162`, cluster CI `[-0.0920, +0.0661]`), Rank-1 `54.2%` (`13/24`) vs `62.5%` (`15/24`, `Delta = -8.3%`), Wardrobe `37.5%` (`9/24`) vs `41.7%` (`10/24`), Spatial `87.5%` (`21/24`, failing the `90.0%` target) vs `79.2%`, Defects/Scene `1.25` vs `1.92` (`Delta = -0.67`, cluster CI `[-2.33, +0.83]`; [`results/numbers.json`](../results/numbers.json)).

### (d) Pre-Registered Hypothesis Scorecard: `0/4` Strict All-Split Rules Met
- **Observation:** Evaluated strictly against the R10 pre-registration rules defined prior to `TEST-2` unsealing, MMCCP `v2.0` satisfied **`0/4`** all-split rules (`P1`, `P2`, `P3`, `P4` all `NOT MET` globally; `P2` and `P3` passed on Ebright and Combined only while failing Holdout).
- **Evidence:** [`docs_generated/v2/PREREG_RECONCILIATION.md`](../docs_generated/v2/PREREG_RECONCILIATION.md) & [`results/test2_evaluation_summary.json`](../results/test2_evaluation_summary.json). Post-hoc redefined criteria in `v2.0.0-rc2` (`7904ad5`) are formally retracted.

### (e) Whole-System Peak VRAM Exceeded `7.4 GB` Budget (`0` OOMs)
- **Observation:** Although zero CUDA out-of-memory (`OOM`) crashes occurred, continuous 1 Hz whole-system `nvidia-smi` telemetry exceeded the `7.4 GB` (`7400` MB / `7577.6` MiB) target ceiling.
- **Evidence:** In [`results/telemetry/vram_test2_sealed.csv`](../results/telemetry/vram_test2_sealed.csv) (`8,112` samples), whole-system peak VRAM reached **`7984` MiB** (`7.797` GiB / `7.984` GB), spending `52`s above `7400` MiB (`29`s above `7577.6` MiB across `36` episodes). Subtracting the measured `973` MiB pre-render desktop baseline yields `7011` MiB (`6.847` GiB) process-attributable delta (`results/numbers.json`, [`docs_generated/v2/07_measured_results.md`](../docs_generated/v2/07_measured_results.md)). Across all `9` telemetry CSVs (`157,357` samples), max reached `8005` MiB.

### (f) Extra-Person Hallucinations & YOLO Divergence
- **Observation:** `P4` required `0` extra YOLO person detections; `1/42` `v2` `TEST-2` renders was flagged by [`tools/extra_person_detector.py`](../tools/extra_person_detector.py) (`conf=0.40`).
- **Evidence:** Visual review and detection logs confirm `v2` Ebright `scene_5_seed_1202.png` rendered `3` figures when `2` were expected (lower-right duplicate bust). Separately, Verifier `v1.2` YOLO also flagged `scene_2_seed_1303.png` (`2` YOLO vs `1` face/expected, whereas standalone detector logged `1`; [`results/test2_extra_person_audit.json`](../results/test2_extra_person_audit.json)).

### (g) Hard 2-Character Maximum Scene Cap (`M2 / M3 / M5` Architectural Limit)
- **Observation:** Scenes requiring `>= 3` simultaneous characters drop the 3rd character cleanly because layout planning, regional attention prompts, and face-swap slot builders only handle `2` characters.
- **Root Cause & Evidence:** Inspecting all `3` Holdout Scene 4 (`clara_thorn`, `elena_child`, `dr_marcus_vance`) renders showed only `2` people generated (`YOLO=2`), dropping `elena_child` from M5 swap specs and collapsing Spatial Pass Rate to `87.5%` (`21/24`; [`docs_generated/v2/07_measured_results.md`](../docs_generated/v2/07_measured_results.md)). Tracked for `v2.1` in [`CONTRIBUTING.md`](../CONTRIBUTING.md#4-starter-tasks-for-collaborators-v21-roadmap--not-started-in-frozen-v20).

### (h) Static Studio-Anchor Wardrobe Lock Overwriting Scene-Specific Attire & Closed-Loop Scoring (`M7`)
- **Observation:** `WardrobeLock` (`M7`) locks every character to one static studio `clothing_colors` CIELAB vector instead of scene-level costume overrides, causing genuine color stains and state-blind verifier penalties. Also `blend_strength=0.70` is hardcoded in [`consistency_v2/pipeline_v2.py`](../consistency_v2/pipeline_v2.py). In addition, `M7` recolours segmented garment regions toward the **exact same canonical `clothing_colors` CIELAB target** that Frozen Verifier `v1.2`'s `wardrobe_pass_rate` measures (`Delta E <= 38.0`), so the `+35.0%` Ebright `wardrobe_pass_rate` gain is a **closed-loop target-adherence check** rather than independent perceptual evidence of outfit fidelity.
- **Evidence:** In Ebright Scene 6 (`age_state="graduation"`, black graduation gown), `ebright.json` specifies `college_ebright.clothing_colors = ["mustard yellow"]`. M7 stained the chest area of black robes mustard-olive across all `5` Scene 6 renders, and Verifier `v1.2` penalised both `v2` (`Delta E = 38.77..48.94 > 38.0`) and clean `v1` black gowns (`53.75..76.54`). Tracked for `v2.1` in [`CONTRIBUTING.md`](../CONTRIBUTING.md#4-starter-tasks-for-collaborators-v21-roadmap--not-started-in-frozen-v20).

### (i) Partial Scoring Circularity (`InsightFace buffalo_l` Source vs Judge Space)
- **Observation:** `M5` face swap (`inswapper_128.onnx`) operates in the `InsightFace buffalo_l` embedding space using the exact same canonical anchor portraits graded by Verifier `v1.2`.
- **Evidence:** [`results/test2_ebright_circularity_coherence.json`](../results/test2_ebright_circularity_coherence.json) shows `InsightFace_buffalo_l` cosine jumped `+0.0989` and correlates strongly (`r = 0.7439`) with the 4-judge ensemble (`Facenet512` `+0.0665`), whereas `ArcFace` (`-0.0020`, `r = -0.1431`) and `GhostFaceNet` (`+0.0059`, `r = -0.3293`) show minimal movement or negative correlation.

### (j) `M4` IP-Adapter Plus Mixed Tradeoff, Unablated `M10`, & Unexecuted `enable_best_of_n`
- **Observation (`M4` Balanced Tradeoff — Candidate for Multi-Seed Re-Test in `v2.1`):** Removing `M4` (`no_m4_ip_adapter`) edged out `full_v2` on 4-judge biometric ensemble on **both** `DEV-2` (`0.3423` vs `0.3323`, `+0.0100`) and `TEST-2` (`0.3373` vs `0.3269`, `+0.0104` — *post-hoc exploratory ablation on `TEST-2` after unsealing; not used to tune `v2.0`*, run inside `tools/run_sealed_test2.py` PID `1608` at `2026-11-06T05:45:11Z`, scored `13:38:16` before `13:38:50` score inspection; `consistency_v2/best_config.json` unchanged since `2026-11-03`). However, both ensemble gains lie inside the `0.03` single-seed (`n=8` faces) diffusion noise band, while **removing `M4` sharply degraded `Facenet512` (`0.7087 -> 0.6218` DEV-2; `0.7000 -> 0.6060` TEST-2), `DINOv2` structural similarity (`0.5878 -> 0.4935` DEV-2; `0.6069 -> 0.4943` TEST-2), and DEV-2 defects (`1.17 -> 1.50`)**. Thus `M4` is a **candidate for a multi-seed re-test in `v2.1`**, not an unconditional disable.
  - **`M10` Unablated Status:** `M10` (`enable_identity_memory=True`, `drift_guard_threshold=0.75`) is **ON** in the locked evaluation config (`consistency_v2/best_config.json`) and has **no LOO ablation arm** (`DEV-2` and `TEST-2` only ablated `M2–M7`), so **no empirical benefit is claimed for `M10`**.
  - **Unexecuted `M9` Flag:** `enable_best_of_n` in [`consistency_v2/config.py`](../consistency_v2/config.py) is only logged in `methods_active` and not executed inside [`consistency_v2/pipeline_v2.py`](../consistency_v2/pipeline_v2.py) (`generate_scene()`).

| Metric (`n=6` Scenes, `8` Faces) | `DEV-2` `full_v2` | `DEV-2` `no_m4` | `DEV-2` Delta | `TEST-2` `full_v2` *(post-hoc)* | `TEST-2` `no_m4` *(post-hoc)* | `TEST-2` Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-Judge Bio Ensemble** | `0.3323` | `0.3423` | `+0.0100` *(noise)* | `0.3269` | `0.3373` | `+0.0104` *(noise)* |
| `Facenet512` *(drop w/o M4)* | `0.7087` | `0.6218` | **`-0.0869`** | `0.7000` | `0.6060` | **`-0.0940`** |
| `SFace` | `0.1588` | `0.1536` | `-0.0052` | `0.1391` | `0.1556` | `+0.0165` |
| `GhostFaceNet` | `0.0343` | `0.0548` | `+0.0205` | `0.0342` | `0.0452` | `+0.0110` |
| `ArcFace` | `0.0312` | `0.0441` | `+0.0129` | `0.0400` | `0.0469` | `+0.0069` |
| **`DINOv2` Sim** *(drop w/o M4)* | `0.5878` | `0.4935` | **`-0.0943`** | `0.6069` | `0.4943` | **`-0.1126`** |
| **Rank-1 Accuracy** | `87.5%` | `87.5%` | `0.0%` | `87.5%` | `87.5%` | `0.0%` |
| **Spatial Pass** | `100.0%` | `100.0%` | `0.0%` | `100.0%` | `100.0%` | `0.0%` |
| **Wardrobe Pass** | `87.5%` | `87.5%` | `0.0%` | `87.5%` | `87.5%` | `0.0%` |
| **Defects / Scene** | `1.17` | `1.50` | **`+0.33`** *(worse)* | `1.50` | `1.50` | `0.00` |

- **Evidence:** [`results/numbers.json`](../results/numbers.json) (`test2_ablation.m4_comparison`, `dev2_ablation.m4_comparison`) and [`consistency_v2/pipeline_v2.py`](../consistency_v2/pipeline_v2.py).

### (k) Human Blind Evaluation Incomplete (Prior Visual Checks Were Automated Proxies)
- **Observation:** Prior DEV-2 blind A/B preference (`10/12` = `83.33%`, `p=0.0386`), blind Re-ID (`20/20`), canary catches (`3/3`), and VLM agreement (`kappa`) were **automated scripts, not independent human ratings**.
- **Evidence:** [`docs_generated/v2/visual_review/blind_predictions_human.json`](../docs_generated/v2/visual_review/blind_predictions_human.json) remains unfilled (`human_blind_test_status = "PENDING_HUMAN_INPUT"` in [`results/numbers.json`](../results/numbers.json); protocol in [`HUMAN_BLIND_TEST.md`](../docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md)).

### (l) Zero-Shot CLIP Prop & Attribute False Positives in Frozen Verifier `v1.2`
- **Observation:** Open-vocabulary CLIP probes in Frozen Verifier `v1.2` emit recurring false positives on clean period imagery and graduation gowns (e.g., `"beard on child"` from chin shadows; `"casual clothes, shorts, t-shirt"` on black gowns; `colour_inversion` spurious triggers).
- **Evidence:** [`docs_generated/v2/verifier_v1.3_proposal.md`](../docs_generated/v2/verifier_v1.3_proposal.md) reports a `63.3%` baseline FP rate on 30 period portraits and proposes dual-threshold contrastive softmax prompts for future `v1.3`.

### (m) Licensing Restrictions Block Commercial Use & No Root `LICENSE` Exists
- **Observation:** Upstream components used or required by the pipeline (fetched externally, not redistributed) prohibit commercial deployment without separate agreements (`InsightFace`: non-commercial research; `CodeFormer`: `S-Lab License 1.0` non-commercial; `WebUI Forge` & `YOLOv8n`: `AGPL-3.0`; `Juggernaut-XL` / `SDXL-Lightning`: Open RAIL use-based restrictions), and MMCCP ships **without** an OSI root `LICENSE` file (`LICENSE_FOR_OWN_CODE=""`; standard default copyright applies and the university's IP policy has not been checked). Several HF model cards also lack standardized license fields (`thibaud/controlnet-openpose-sdxl-1.0`, `mattmdjaga/segformer_b2_clothes`, `openai/clip-vit-base-patch32`).
- **Evidence:** [`docs/run/LICENSES_AND_MODELS.md`](run/LICENSES_AND_MODELS.md).

### (n) Sealed `TEST-2` Permanence, Untested Platforms & Legacy v1 Doc Links
- **Observation:** `TEST-2` (`1101..1505`, `holdout_v2.json`) is permanently burned after one-shot unsealing (`results/split_access_log.jsonl`). Furthermore, while Windows fresh-clone CPU & local GPU steps were tested ([`docs/run/TEST_LOG.md`](run/TEST_LOG.md)), the Linux guide ([`02_LINUX_NOTES.md`](run/02_LINUX_NOTES.md)), Google Colab notebook ([`MMCCP_Colab.ipynb`](../notebooks/MMCCP_Colab.ipynb)), and clean-Windows Forge `.7z` extraction remain `UNTESTED`. Legacy index [`docs/00_MASTER_INDEX.md`](00_MASTER_INDEX.md) still references `v1` draft explanation files (`docs/files/*`) that were never populated.

---

## Appendix: Streamlit UI (`app.py`) Defaults vs Locked Evaluation Config (`best_config.json`)

| Setting / Aspect | `consistency_v2/best_config.json` (Locked `v2.0`) | `app.py` Interactive Default (Post-Fix) | Impact / Note |
| :--- | :---: | :---: | :--- |
| `enable_dual_instantid` (`M4b`) | `False` | `False` *(fixed from `True`)* | `v2_m4` toggle now defaults `False`, matching locked hash `df4a0a2648acdfe1` (`df4a0a2648ac`) |
| `enable_ip_adapter_plus` (`M4`) | `True` (`0.55`) | `True` (`0.55`, inherited default) | Not exposed as a sidebar toggle; uses `ConsistencyV2Config` default |
| `M1, M2, M3, M5, M6, M7, M10` | `True` (`default` mode) | `True` (`default` mode) | `8` of `12` numbered methods ON (`M4b`, `M8`, `M9`, `M11`, `M12` OFF) |
| Non-toggled hyperparameters | `steps 20/15`, `cfg 5.5`, `FreeU/DynThresh=True`, `crop_upscale_swap=False` | Identical via `ConsistencyV2Config` defaults | No UI sliders exposed; identical resolved config dictionary |
| Render scope & Tab 4 metrics | `42` `TEST-2` renders (`results/numbers.json`) | `ebright.json` with fixed `seed=11`; Tab 4 shows legacy static Phase 3 cards (`app.py:1004-1007`) | Interactive studio showcase (`seed=11` DEV-FAST); not a `TEST-2` benchmark |
