# MMCCP v2.0: Measured Results, Statistical Analysis & Exploratory Diagnostics

**Capstone Project ID:** `PW26_CGM_02` | **Institution:** PES University  
**Evaluation Verifier:** Frozen Verifier `v1.2` (`SHA: 3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6`)

---

## 1. Canonical TEST-2 Results & Dual Bootstrap CIs (`10000` Resamples, Seed `42`)

Defect unit definitions:
- **Defects per Scene (Primary):** Mean defects across unique rendered images (`n=30` Ebright, `n=12` Holdout, `n=42` Combined).
- **Defects per Face Instance (Secondary / Instance-Weighted):** Scene defect count replicated across each face slot (`n=40` Ebright, `n=24` Holdout, `n=64` Combined); reported separately so multi-character scenes are not conflated with scene averages.

### 1.1 Ebright TEST-2 (`30` Renders, `40` Face Slots, `40` Paired Slots)
- **Biometric Ensemble Mean:** `0.3147` (vs v1 `0.3091`, `Delta = +0.0056`, scene-cluster 95% CI `[-0.0149, +0.0317]`, image-level 95% CI `[-0.0124, +0.0244]`, Wilcoxon `W=343.0`, `p=0.3754` -> parity, CI includes 0)
  - Adult Bio (`n=20`): `0.3280` (v1 `0.3290`)
  - Teen Bio (`n=10`): `0.3139` (v1 `0.2778`)
  - Child Bio (`n=10`): `0.2889` (v1 `0.3006`)
- **Rank-1 Discrimination Accuracy:** `80.0%` (`32/40`, v1 `65.0%` `26/40`, `Delta = +15.0%`, image 95% CI `[+2.4%, +28.6%]` excludes 0, cluster 95% CI `[+0.0%, +33.3%]` touches 0, Wilcoxon `p=0.0578`)
- **Spatial Position Pass Rate:** `97.5%` (`39/40` vs v1 `100.0%` `40/40`, `Delta = -2.5%`)
- **Wardrobe Palette Pass Rate:** `80.0%` (`32/40`, v1 `45.0%` `18/40`, `Delta = +35.0%`, cluster 95% CI `[+20.0%, +48.6%]`, image 95% CI `[+22.5%, +47.6%]`, Wilcoxon `W=0.0`, `p=0.000183`)
- **Mean Defects per Scene (Primary, `n=30`):** `1.50` (v1 `3.07`, `Delta = -1.57` / `-51.1%`, cluster 95% CI `[-3.20, -0.13]`, image 95% CI `[-2.37, -0.77]`, Wilcoxon `W=39.5`, `p=0.001453`)
- **Mean Defects per Face Instance (Secondary, `n=40`):** `1.525` (v1 `2.575`, `Delta = -1.050`)
- **Combined Binding & Identity-at-Position Pass Rates:** `0.0%` (`0/40`) and `0.0%` (`0/40`) for **both** v1 and v2.

### 1.2 Holdout TEST-2 (`12` Renders, `24` Face Slots)
- **Biometric Ensemble Mean:** `0.2596` (vs v1 `0.2758`, `Delta = -0.0162`, cluster CI `[-0.0920, +0.0661]`, image CI `[-0.0633, +0.0412]`, Wilcoxon `p=0.8213`)
  - Adult Bio (`n=15`): `0.2274` (v1 `0.2100`) | Child Bio (`n=9`): `0.3132` (v1 `0.3854`)
- **Rank-1 Accuracy:** `54.2%` (`13/24` vs v1 `62.5%` `15/24`, `Delta = -8.3%`, cluster CI `[-27.3%, +12.5%]`, image CI `[-20.8%, +8.7%]`, `p=0.3173`)
- **Spatial Pass Rate:** `87.5%` (`21/24` vs v1 `79.2%` `19/24`, `Delta = +8.3%`, image CI `[+0.0%, +19.2%]`) — below `90.0%` target.
- **Wardrobe Pass Rate:** `37.5%` (`9/24` vs v1 `41.7%` `10/24`, `Delta = -4.2%`, `p=0.7055`) — below `60.0%` target.
- **Mean Defects per Scene (Primary, `n=12`):** `1.25` (v1 `1.92`, `Delta = -0.67`, cluster CI `[-2.33, +0.83]`, image CI `[-1.67, +0.33]`, `p=0.3242`)
- **Mean Defects per Face Instance (Secondary, `n=24`):** `1.417` (v1 `1.792`, `Delta = -0.375`)
- **Combined Binding & Identity-at-Position:** `0.0%` (`0/24`) for both v1 and v2.

### 1.3 Combined TEST-2 (`42` Renders, `64` Face Slots)
- **Bio Mean:** `0.2940` vs v1 `0.2966` (`Delta = -0.0026`, cluster CI `[-0.0390, +0.0330]`, image CI `[-0.0252, +0.0211]`, `p=0.3233`)
- **Rank-1:** `70.31%` (`45/64` vs v1 `64.06%`, `Delta = +6.25%`, image CI `[-4.6%, +17.7%]`)
- **Spatial / Wardrobe:** `93.75%` (`60/64`) / `64.06%` (`41/64`)
- **Defects/Scene (Primary):** `1.43` vs v1 `2.74` (`Delta = -1.31`, cluster CI `[-2.55, -0.16]`, image CI `[-1.95, -0.64]`, `p=0.000877`)

---

## 2. Original Pre-Registered Scorecard (`P1`–`P4`)

See full table and post-hoc reconciliation in [`PREREG_RECONCILIATION.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/PREREG_RECONCILIATION.md):
- **`P1 (Original): Delta Re-ID (Biometric Ensemble & Rank-1) > 0 with 95% CI lower bound > 0` — [`NOT MET`]**: Ebright Biometric Ensemble Delta = +0.0056 (cluster 95% CI [-0.0149, 0.0317], image 95% CI [-0.0124, 0.0244], Wilcoxon p=0.3754); Holdout Delta = -0.0162 (cluster CI [-0.092, 0.0661]); Combined Delta = -0.0026 (cluster CI [-0.039, 0.033]). Lower bound <= 0 across splits.
- **`P2 (Original): Spatial assignment pass rate >= 90.0%` — [`NOT MET`]**: Ebright Spatial Pass = 97.5% (39/40, >=90% MET) and Combined = 93.75% (60/64, >=90%), but Holdout Spatial Pass = 87.5% (21/24, <90% NOT MET due to missing 3rd character on all 3 Scene 4 renders).
- **`P3 (Original): Wardrobe Delta-E pass rate >= 60.0%` — [`NOT MET`]**: Ebright Wardrobe Pass = 80.0% (32/40, >=60% MET) and Combined = 64.06% (41/64, >=60%), but Holdout Wardrobe Pass = 37.5% (9/24, <60% NOT MET vs v1 41.7%).
- **`P4 (Original): Engineering (Peak VRAM <= 7.4 GB, 0 OOMs, 0 extra YOLO persons)` — [`NOT MET`]**: Whole-system Peak VRAM = 7984 MiB (7.797 GiB > 7400 MB / 7577.6 MiB target; 52s > 7400 MiB; 29s > 7577.6 MiB), while process-attributable delta (max - 973 MiB idle) = 7011 MiB (6.847 GiB), 0 OOMs, and extra_person_detector.py (conf=0.40) flagged 1/42 extra YOLO persons (v2 Ebright scene_5_seed_1202.png; Verifier YOLO also flagged scene_2_seed_1303.png).

---

## 3. Leave-One-Out Ablation Tables (TEST-2 Seed `1101` & DEV-2 Seed `101`)

Single-seed (`n=8` faces) biometric deltas smaller than `~0.03` are within diffusion seed noise. On **both** DEV-2 and TEST-2, `no_m4_ip_adapter` exceeded `full_v2`, motivating disabling M4 IP-Adapter Plus in `v2.1` exploration while keeping frozen `v2.0` intact.

| Arm | TEST-2 Bio (`n=8`) | TEST-2 `Delta Bio` | TEST-2 Pixel Diff | DEV-2 Bio (`n=8`) | DEV-2 `Delta Bio` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `full_v2` | 0.3269 | +0.0000 | 0.00 px | 0.3323 | +0.0000 |
| `no_m2_layout` | 0.3167 | -0.0102 | 66.48 px | 0.2198 | -0.1125 |
| `no_m3_couple` | 0.3064 | -0.0205 | 19.85 px | 0.3169 | -0.0154 |
| `no_m4_ip_adapter` | 0.3373 | +0.0104 | 65.33 px | 0.3423 | +0.0100 |
| `no_m5_swap` | 0.2954 | -0.0315 | 0.79 px | 0.2818 | -0.0505 |
| `no_m6_detailer` | 0.3181 | -0.0088 | 0.44 px | 0.3149 | -0.0174 |
| `no_m7_wardrobe` | 0.3227 | -0.0042 | 2.59 px | 0.3108 | -0.0215 |

---

## 4. Exploratory Defect Root-Cause Findings (`EXPLORATORY`)

All images below were inspected directly with the image viewer before reading their scores:
1. **Holdout Scene 4 Missing 3rd Character (`clara_thorn` left, `elena_child` center, `dr_marcus_vance` right — CONFIRMED ARCHITECTURAL LIMIT):**
   - Visual inspection of all `3` Holdout Scene 4 renders confirmed strictly `2` rendered figures around a table with a blank middle wall gap, and the right-hand person exhibited a mustached male face over long feminine hair.
   - Root cause confirmed in code + sidecars: `layout_planner.py` constructs only a 2-slot pair (`left_char`, `right_char`), `regional.py` sends 2 Basic columns, `face_swap.py`/M5 receives only `2` swap specs (`clara_thorn` and `dr_marcus_vance`, dropping `elena_child`), and `prompt_builder.py` collapses non-first characters into the right column while banning `"3persons"` for all scenes with `>= 2` characters. Verifier then matched the right face to `elena_child` by spatial zone, scoring `dr_marcus_vance` `0.0000` (missing).

| Holdout Scene 4 Render | Faces/Expected (YOLO) | Swapped Specs | Per-Character Verifier Outcomes |
| :--- | :---: | :--- | :--- |
| `scene_4_seed_1101.png` | 2/3 (YOLO=2) | clara_thorn, dr_marcus_vance | clara_thorn=0.1751(sp=True,w_dE=41.04); elena_child=0.0349(sp=True,w_dE=70.34); dr_marcus_vance=0.0000(sp=False,w_dE=99.0) |
| `scene_4_seed_1202.png` | 2/3 (YOLO=2) | clara_thorn, dr_marcus_vance | clara_thorn=0.1557(sp=True,w_dE=46.0); elena_child=0.0882(sp=True,w_dE=52.47); dr_marcus_vance=0.0000(sp=False,w_dE=99.0) |
| `scene_4_seed_1303.png` | 2/3 (YOLO=2) | clara_thorn, dr_marcus_vance | clara_thorn=0.1501(sp=True,w_dE=47.72); elena_child=0.0881(sp=True,w_dE=64.23); dr_marcus_vance=0.0000(sp=False,w_dE=99.0) |

2. **Ebright Scene 6 Graduation Gown Color Corruption (`EXPLORATORY` — BOTH REAL DEFECT AND VERIFIER ARTIFACT CONFIRMED):**
   - Visual inspection across all `5` Scene 6 renders confirmed a black graduation gown and mortarboard with a rectangular olive/mustard chest stain bleeding onto the top edge of the diploma folder.
   - Root cause: `ebright.json` defines `college_ebright.clothing_colors = ["mustard yellow"]` while Scene 6 specifies `age_state = "graduation"` (`black graduation gown`). M7 (`pipeline_v2.py` -> `wardrobe.py`) reads `clothing_colors` rather than scene-state clothing, staining the torso box toward mustard yellow (**real visual corruption** + CLIP flags for `casual clothes`, `shorts`, `t-shirt`), **and** Frozen Verifier `v1.2` (`cli.py`) also scores against `clothing_colors`, causing both v2 (`Delta E = 38.77..48.94 > 38.0`) and unstained v1 black gowns (`Delta E = 53.75..76.54`) to fail (**verifier state-blindness artifact**).

| Scene 6 Render | v2 Bio Mean | v2 Wardrobe `Delta E` | v1 Wardrobe `Delta E` | Verifier Defects |
| :--- | :---: | :---: | :---: | :--- |
| `scene_6_seed_1101.png` | 0.4121 | 38.77 (pass=False) | 57.52 | forbidden_prop_present: casual clothes, forbidden_prop_present: shorts, forbidden_prop_present: t-shirt |
| `scene_6_seed_1202.png` | 0.3518 | 48.94 (pass=False) | 67.31 | forbidden_prop_present: casual clothes, forbidden_prop_present: shorts, forbidden_prop_present: t-shirt |
| `scene_6_seed_1303.png` | 0.3640 | 44.72 (pass=False) | 53.75 | forbidden_prop_present: casual clothes, forbidden_prop_present: shorts, forbidden_prop_present: t-shirt |
| `scene_6_seed_1404.png` | 0.3740 | 40.00 (pass=False) | 70.37 | forbidden_prop_present: casual clothes, forbidden_prop_present: shorts, forbidden_prop_present: t-shirt |
| `scene_6_seed_1505.png` | 0.4137 | 40.92 (pass=False) | 76.54 | forbidden_prop_present: casual clothes, forbidden_prop_present: shorts, forbidden_prop_present: t-shirt |

3. **Extra-Person / YOLO Divergence (`scene_2_seed_1303` vs `scene_5_seed_1202` — `EXPLORATORY`):**
   - Visual inspection of `scene_2_seed_1303.png` showed `1` boy behind a butterfly table (Verifier `report_v1_2.json` logged `YOLO=2, faces=1, defects=[]`, while standalone `tools/extra_person_detector.py` logged `YOLO=1`).
   - Visual inspection of `scene_5_seed_1202.png` showed **`3` distinct figures** (left `college_ebright`, upper-right `james_wong`, and a giant duplicated `james_wong` bust overlapping in the lower-right foreground), matching both Verifier (`YOLO=3, faces=3`) and `extra_person_detector.py` (`1/30` on v2 TEST-2 Ebright).

---

## 5. Hardware Telemetry & Excursion Attribution
- **Sealed TEST-2 (`vram_test2_sealed.csv`, `8112` 1 Hz samples):** Whole-system peak **`7984` MiB** (`7.797` GiB / `7.984` GB), median `6247.0` MiB, mean `5243.1` MiB, p95 `7265.0` MiB, p99 `7328.0` MiB. Pre-render desktop idle baseline = `973` MiB (`32` samples); process-attributable delta peak = `7011` MiB (`6.847` GiB). Whole-system excursions exceeded `7400` MiB for **`52`s** (`29`s > `7577.6` MiB) across **`36` episodes**; all episodes aligned cleanly with early Pass 1 txt2img model/ControlNet/IP-Adapter allocation (`6–31s` into renders, including the `7984` MiB peak in `scene_5_seed_1303`) or Pass 1->Pass 2 img2img handoff (`49–131s`), and **zero** occurred during frozen scoring.
- **All `9` Telemetry CSVs Combined (`157357` samples):** Max `8005` MiB, p99 `7442.0` MiB, `1838`s > `7400` MiB (`1020`s > `7577.6` MiB).

---

## 6. Circularity & Cross-Scene Coherence Audit (TEST-2 Ebright, Read-Only)

### 6.1 Final Face Similarity to Source Anchor: InsightFace (`buffalo_l w600k_r50`) vs 4 Judges

M5 (`inswapper_128`) operates inside the InsightFace `buffalo_l` embedding space and receives the same canonical anchor used by Verifier v1.2, introducing partial scoring circularity.

| Judge / Embedder | v1 Mean | v2 Mean | Paired Delta (`v2 - v1`) | Image 95% CI | v2 Pearson r vs InsightFace |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `InsightFace_buffalo_l` | 0.7408 | 0.8397 | +0.0989 | [+0.0551, +0.1396] | 1.0000 |
| `Ensemble_4Judge` | 0.2013 | 0.2202 | +0.0189 | [+0.0044, +0.0321] | 0.7439 |
| `Facenet512` | 0.6171 | 0.6836 | +0.0665 | [+0.0316, +0.0969] | 0.8881 |
| `SFace` | 0.1311 | 0.1365 | +0.0053 | [-0.0133, +0.0225] | 0.3939 |
| `GhostFaceNet` | 0.0235 | 0.0294 | +0.0059 | [-0.0203, +0.0320] | -0.3293 |
| `ArcFace` | 0.0333 | 0.0313 | -0.0020 | [-0.0138, +0.0107] | -0.1431 |

### 6.2 Anchor-Free Cross-Scene Coherence (Same-Character / Impostor Ratio, TEST-2 Ebright)

| Embedder | v1 Genuine | v1 Impostor | v1 Ratio | v2 Genuine | v2 Impostor | v2 Ratio | v1 Diff-Scene Gen | v2 Diff-Scene Gen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-Judge Ensemble** | 0.6161 | 0.1801 | 3.4207 | 0.7631 | 0.1998 | 3.8192 | 0.5971 | 0.7468 |
| `InsightFace_buffalo_l` | 0.7105 | 0.2359 | 3.0114 | 0.8399 | 0.2151 | 3.9041 | 0.6896 | 0.8263 |
| `Facenet512` | 0.7384 | 0.1898 | 3.8904 | 0.8290 | 0.1954 | 4.2426 | 0.7003 | 0.8155 |
| `SFace` | 0.5676 | 0.1966 | 2.8869 | 0.7472 | 0.2169 | 3.4449 | 0.5545 | 0.7330 |
| `GhostFaceNet` | 0.5265 | 0.1693 | 3.1108 | 0.7378 | 0.1917 | 3.8479 | 0.5248 | 0.7218 |
| `ArcFace` | 0.6318 | 0.1647 | 3.8349 | 0.7383 | 0.1952 | 3.7827 | 0.6087 | 0.7170 |


---

## 7. Visual Review Status (`automated, not human`)
- Prior Phase R8 evaluations on DEV-2 (`10/12` blind A/B preference = `83.33%`, `p=0.0386`; `20/20` blind crop Re-ID; `3/3` planted canaries; VLM `kappa` proxy) are **automated, not human** evaluations (`blind_review_summary.json` / `blind_evaluation_report.md` marked `AUTOMATED-SUPERSEDED`).
- Human TEST-2 Ebright blind test is staged awaiting user scoring in [`HUMAN_BLIND_TEST.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md) and [`blind_predictions_human.json`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/visual_review/blind_predictions_human.json).
