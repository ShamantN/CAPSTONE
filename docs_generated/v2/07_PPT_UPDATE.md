# Capstone Presentation Slide Update: Reconciled Verified State

**Project:** MMCCP v2.0 (`PW26_CGM_02`) | **Verifier:** `v1.2`

## Key Reconciled Slide Numbers (`TEST-2`, `n=30` Ebright + `12` Holdout)
1. **Ebright TEST-2 (In-Domain Story):**
   - **Biometric Parity:** `0.3091 -> 0.3147` (`Delta = +0.0056`, cluster CI `[-0.0149, +0.0317]`, image CI `[-0.0124, +0.0244]`, `p=0.3754` — includes 0).
   - **Rank-1 Re-ID:** `65.0%` (`26/40`) -> **`80.0%`** (`32/40`, `+15.0%`, image CI `[+2.4%, +28.6%]`, 6-cluster CI `[+0.0%, +33.3%]`).
   - **Wardrobe Fidelity:** `45.0%` (`18/40`) -> **`80.0%`** (`32/40`, `+35.0%`, `p=0.000183`).
   - **Defects per Scene (Primary):** `3.07` -> **`1.50`** (`-1.57/scene`, `-51.1%`, `p=0.001453`). *(Secondary instance-weighted delta: `-1.050`).*
2. **Holdout TEST-2 (Unseen Characters — NOT Better Than v1):**
   - Bio `0.2758 -> 0.2596` (`-0.0162`), Rank-1 `62.5% -> 54.2%`, Wardrobe `41.7% -> 37.5%`, Spatial `87.5%` (`21/24` — 3-char Scene 4 dropped `1` character), Defects/Scene `1.92 -> 1.25`.
3. **Engineering Telemetry & Extra Persons:**
   - Whole-system peak VRAM **`7984` MiB** (`7.797` GiB; `52s > 7400` MiB), process-attributable delta above `973` MiB idle = `7011` MiB, `0` OOMs; extra YOLO person (`conf=0.40`): `1/42` (`scene_5_seed_1202`).
4. **Original Pre-Registered Hypotheses (`P1`–`P4`):**
   - Strict all-split verdict: **`0/4` MET** (`P1` NOT MET; `P2`/`P3` pass Ebright `97.5%`/`80.0%` and Combined `93.75%`/`64.06%` only, failing Holdout; `P4` NOT MET).
5. **Known Limits & Future Work:**
   - `identity_at_position` & `combined_binding` = `0.0%` for both v1 & v2; post-hoc exploratory ablations (not used to tune `v2.0`) show `no_m4_ip_adapter` slightly exceeded `full_v2` on 4-judge Bio inside `0.03` noise while dropping `Facenet512`/`DINOv2` (candidate for a multi-seed re-test in `v2.1`); `M10` is ON with no ablation arm; prior visual review checks were **automated, not human** (human TEST-2 blind test pending).
