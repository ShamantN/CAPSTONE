## CORRECTION NOTICE: Reconciled Finalization Audit (`v2.0.0-rc3` Candidate)

> [!WARNING]
> **RETRACTING OVERSTATED `v2.0.0-rc2` SCORECARD CLAIMS:** The earlier `v2.0.0-rc2` (`7904ad5`) comment reported `4/4 MET` hypotheses using post-hoc redefined criteria written on `2026-11-06 13:41:51 +0530` after `TEST-2` scores were inspected (`2026-11-06 13:31:15 +0530`), conflated per-scene (`1.50`) with instance-weighted (`-1.05`) defect metrics, and omitted whole-system VRAM excursions (`7984` MiB). Below are the reconciled, script-verified numbers from [`results/numbers.json`](file:///D:/5TH%20SEM/HACKATHON/results/numbers.json).

### 1. Reconciled Sealed TEST-2 Scorecard (`42` Renders: `30` Ebright + `12` Holdout)

- **Ebright TEST-2 (`30` scenes, `40` face slots):**
  - **Biometric Ensemble:** `0.3147` vs v1 `0.3091` (`Delta = +0.0056`, cluster CI `[-0.0149, +0.0317]`, image CI `[-0.0124, +0.0244]`, Wilcoxon `W=343.0`, `p=0.3754`) -> **Identity parity with Baseline v1**.
  - **Rank-1 Disambiguation:** `80.0%` (`32/40`) vs v1 `65.0%` (`26/40`, `Delta = +15.0%`, image CI `[+2.4%, +28.6%]`, cluster CI `[+0.0%, +33.3%]`).
  - **Wardrobe Palette (`Delta E <= 38.0`):** `80.0%` (`32/40`) vs v1 `45.0%` (`18/40`, `Delta = +35.0%`, cluster CI `[+20.0%, +48.6%]`, image CI `[+22.5%, +47.6%]`, `p=0.000183`).
  - **Mean Defects / Scene (Primary):** `1.50` vs v1 `3.07` (`Delta = -1.57/scene`, `-51.1%`, cluster CI `[-3.20, -0.13]`, image CI `[-2.37, -0.77]`, `p=0.001453`).
- **Holdout TEST-2 (`12` scenes, `24` face slots — NOT better than v1):**
  - **Biometric Ensemble:** `0.2596` vs v1 `0.2758` (`Delta = -0.0162`, cluster CI `[-0.0920, +0.0661]`, `p=0.8213`).
  - **Rank-1 / Spatial / Wardrobe / Scene Defects:** `54.2%` (`13/24` vs v1 `62.5%`), Spatial `87.5%` (`21/24` vs v1 `79.2%`), Wardrobe `37.5%` (`9/24` vs v1 `41.7%`), Scene Defects `1.25` vs v1 `1.92` (`Delta = -0.67`, `p=0.3242`).
- **Combined Binding & Identity-at-Position:** `0.0%` (`0/40` Ebright, `0/24` Holdout) for both v1 and v2.

### 2. Original Pre-Registered Hypotheses (`P1`–`P4` — Strictly `0/4` MET)

1. **`P1 (Original): Delta Re-ID (Biometric Ensemble & Rank-1) > 0 with 95% CI lower bound > 0` — [`NOT MET`]**: Ebright Biometric Ensemble Delta = +0.0056 (cluster 95% CI [-0.0149, 0.0317], image 95% CI [-0.0124, 0.0244], Wilcoxon p=0.3754); Holdout Delta = -0.0162 (cluster CI [-0.092, 0.0661]); Combined Delta = -0.0026 (cluster CI [-0.039, 0.033]). Lower bound <= 0 across splits.
2. **`P2 (Original): Spatial assignment pass rate >= 90.0%` — [`NOT MET`]**: Ebright Spatial Pass = 97.5% (39/40, >=90% MET) and Combined = 93.75% (60/64, >=90%), but Holdout Spatial Pass = 87.5% (21/24, <90% NOT MET due to missing 3rd character on all 3 Scene 4 renders).
3. **`P3 (Original): Wardrobe Delta-E pass rate >= 60.0%` — [`NOT MET`]**: Ebright Wardrobe Pass = 80.0% (32/40, >=60% MET) and Combined = 64.06% (41/64, >=60%), but Holdout Wardrobe Pass = 37.5% (9/24, <60% NOT MET vs v1 41.7%).
4. **`P4 (Original): Engineering (Peak VRAM <= 7.4 GB, 0 OOMs, 0 extra YOLO persons)` — [`NOT MET`]**: Whole-system Peak VRAM = 7984 MiB (7.797 GiB > 7400 MB / 7577.6 MiB target; 52s > 7400 MiB; 29s > 7577.6 MiB), while process-attributable delta (max - 973 MiB idle) = 7011 MiB (6.847 GiB), 0 OOMs, and extra_person_detector.py (conf=0.40) flagged 1/42 extra YOLO persons (v2 Ebright scene_5_seed_1202.png; Verifier YOLO also flagged scene_2_seed_1303.png).

### 3. Telemetry, Ablation & Methodological Notes
- **VRAM (`vram_test2_sealed.csv`):** Whole-system peak `7984` MiB (`52s > 7400` MiB), `973` MiB idle, `7011` MiB process delta, `0` OOMs.
- **Ablation (`no_m4_ip_adapter`):** Outperformed `full_v2` on DEV-2 (`0.3423` vs `0.3323`) and TEST-2 (`0.3373` vs `0.3269`); M4 should be tested OFF in future `v2.1` exploration (`|Delta| < 0.03` on 1 seed is within noise).
- **Visual Review:** Prior blind/canary/VLM artifacts were **automated, not human**; human blind evaluation (`HUMAN_BLIND_TEST.md`) is pending. Verifier freeze (`v1.2`) remains intact (`check_freeze.py` PASS); PR #19 stays **OPEN**.
