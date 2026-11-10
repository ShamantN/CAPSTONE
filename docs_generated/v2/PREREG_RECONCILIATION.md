# Pre-Registration Reconciliation & Hypothesis Audit (TEST-2)

**Verifier Version:** Frozen `v1.2` (`SHA: 3fa69071f888643b...`)  
**Scope:** Reconciles the original R10 pre-registered hypotheses (`P1`–`P4`) defined before unsealing `TEST-2` against the post-hoc rules committed in `7904ad5` (`tools/compute_test2_statistics.py`). All metrics below are generated strictly from [`results/numbers.json`](file:///D:/5TH%20SEM/HACKATHON/results/numbers.json).

---

## 1. Forensic Timeline (Pre-Registration vs TEST-2 Score Visibility)

| Event | Observed Timestamp / Commit | Relationship to TEST-2 Scores |
| :--- | :--- | :--- |
| Original R10 P1–P4 specification issued | `2026-11-06T05:27:01Z` | **Before** TEST-2 unseal |
| Last pre-TEST-2 repository commit (`96ff419`) | `2026-11-05 21:55:39 +0530` | Zero `compute_test2_statistics.py` on branch |
| `TEST-2` unsealed once by `run_sealed_test2.py` | `2026-11-06T05:45:11Z` | Logged in `results/split_access_log.jsonl` |
| First v2 `TEST-2` Verifier score file (`report_v1_2.json`) written | `2026-11-06 13:31:15 +0530` | Ground-truth TEST-2 scores now on disk |
| `TEST-2` scores read during R10 session (`step=14915`) | `2026-11-06 13:38:50 +0530` | Scores inspected before authoring scorecard |
| `tools/compute_test2_statistics.py` written (`step=14964`) | `2026-11-06 13:41:51 +0530` | **Post-hoc** (after viewing TEST-2 scores; hardcoded observed Holdout values) |
| Committed as `v2.0.0-rc2` (`7904ad5`) | `2026-11-06 13:54:10 +0530` | `git blame` confirms all rules originated in `7904ad5` |

---

## 2. Original Pre-Registered Criteria (`MET` / `NOT MET` Strictly Enforced)

| Hypothesis | Original Pre-Registered Rule | Ebright (`n=30` scenes, `40` faces) | Holdout (`n=12` scenes, `24` faces) | Combined (`n=42` scenes, `64` faces) | Original Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **P1 (Identity)** | `Delta Bio > 0.0` & `Delta Rank-1 > 0.0` with 95% CI lower bound `> 0.0` | Bio `+0.0056` cluster CI `[-0.0149, +0.0317]`, image CI `[-0.0124, +0.0244]`; Rank-1 `+15.0%` cluster CI `[+0.0%, +33.3%]` | Bio `-0.0162` cluster CI `[-0.0920, +0.0661]`; Rank-1 `-8.3%` cluster CI `[-27.3%, +12.5%]` | Bio `-0.0026` cluster CI `[-0.0390, +0.0330]`; Rank-1 `+6.2%` | **`NOT MET`** |
| **P2 (Spatial)** | Spatial assignment pass rate `>= 90.0%` | **97.5%** (39/40) [MET] | **87.5%** (21/24) [**< 90.0% FAIL**] | **93.75%** (60/64) | **`NOT MET`** |
| **P3 (Wardrobe)** | Wardrobe `Delta E <= 38.0` pass rate `>= 60.0%` | **80.0%** (32/40) [MET] | **37.5%** (9/24) [**< 60.0% FAIL**] | **64.06%** (41/64) | **`NOT MET`** |
| **P4 (Engineering)** | Peak VRAM `<= 7.4 GB` (`7400 MB` / `7577.6 MiB`), `0` OOM, `0` extra YOLO | Whole-system `7984` MiB (`52s > 7400`); extra YOLO `1/30` (`conf=0.40`), Verifier YOLO `2/30` | Same sealed run (`7984` MiB peak); extra YOLO `0/12` (3 Scene 4 missing-person undercounts) | `7984` MiB (`7011` MiB delta above `973` idle); OOM `0`; extra YOLO `1/42` | **`NOT MET`** |

- **Strict Original Score Summary:** **`0/4`** strict all-split original rules satisfied (`P1` fails CI lower-bound `> 0`; `P2` and `P3` meet thresholds on Ebright and Combined only while failing Holdout; `P4` fails both whole-system peak VRAM `<= 7.4 GB` and `0` extra YOLO persons).

---

## 3. Side-by-Side Comparison with Post-Hoc Redefined Rules (POST-HOC / INVALID AS PRE-REGISTRATION)

> [!CAUTION]
> **POST-HOC RULES (COMMITTED IN `7904ad5`, INVALID AS PRE-REGISTRATION):** The rules below from `v2.0.0-rc2` were authored after `results/test2_ebright/report_v1_2.json` was written and inspected. They are preserved here solely to explain how `rc2` arrived at `"4/4"` and why `v2.0.0-rc3` retracts that framing.

| ID | Original Pre-Registered Rule | Post-Hoc `rc2` Redefinition (`7904ad5`) | Specific Shift Applied After Viewing Scores | Post-Hoc `rc2` Label |
| :--- | :--- | :--- | :--- | :---: |
| **P1** | `Delta Bio > 0.0` & `Delta Rank-1 > 0.0`, 95% CI lower `> 0.0` | Ebright Bio `>= 0.0` & cluster CI lower `> -0.03` | Changed superiority to non-inferiority (`-0.03` margin) and removed Rank-1 & Holdout | `MET` *(post-hoc)* |
| **P2** | Spatial assignment pass `>= 90.0%` | Ebright Rank-1 `>= 75.0%` (`80.0%`) | Replaced Spatial entirely with Rank-1 on Ebright only | `MET` *(post-hoc)* |
| **P3** | Wardrobe pass `>= 60.0%` | Ebright Spatial `>= 90.0%`, Wardrobe `>= 75.0%`, instance-wt defects `< 0.0` | Dropped Holdout Wardrobe (`37.5%`) and evaluated only Ebright | `MET` *(post-hoc)* |
| **P4** | Peak VRAM `<= 7.4 GB`, `0` OOM, `0` extra YOLO | Holdout Spatial `>= 80.0%`, Scene defects `<= 2.0`, Bio `>= 0.25` | Dropped VRAM and extra-YOLO criteria completely; lowered Holdout Spatial from `90.0%` to `80.0%` | `MET` *(post-hoc)* |

---

## 4. Detailed P4 Breakdown (VRAM & Extra YOLO Persons on TEST-2)

- **Whole-System VRAM (`vram_test2_sealed.csv`, `8112` 1 Hz samples):** Min/Idle pre-render = `973` MiB (`32` samples), Mean = `5243.1` MiB, p95 = `7265.0` MiB, p99 = `7328.0` MiB, Peak = **`7984` MiB** (`7.797` GiB / `7.984` GB). Excursions: **`52`s** above `7400` MiB (`29`s above `7577.6` MiB across `36` episodes).
- **Process-Level Telemetry Note:** Separate `nvidia-smi` per-PID logs were not captured during the sealed run (and scene sidecars logged `peak_vram_mb = 0.0`). Subtracting the measured `973` MiB pre-render desktop baseline from the `7984` MiB peak gives a process-attributable delta peak of **`7011` MiB** (`6.847` GiB), which is below `7.4 GB` even though whole-system VRAM exceeded `7.4 GB`.
- **Extra YOLO Person Detector (`tools/extra_person_detector.py`, `conf=0.40`):**
  - `v2_test2_ebright`: **`1/30`** (`3.3%` — `scene_5_seed_1202.png`: `3` detected vs `2` expected).
  - `v2_test2_holdout`: **`0/12`** (`0.0%`; all `3` Scene 4 renders had missing 3rd character: `2` vs `3`).
  - `v1_test2_ebright`: **`1/30`** (`3.3%` — `scene_1_seed_1202.png`: `3` vs `2`).
  - `v1_test2_holdout`: **`1/12`** (`8.3%` — `scene_2_mentorship_seed_1202.png`: `3` vs `2`).
