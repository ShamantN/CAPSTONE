# MMCCP v2.0 Spark Notes for Defense / Review (Reconciled)

- **What MMCCP v2.0 Does:** Training-free conditioning & refinement pipeline (`M1` Bank, `M2` OpenPose Layout, `M3` Regional Couple, `M4` IP-Adapter Plus, `M5` Direct InsightFace Swap, `M6` Adaptive Face Detailer, `M7` CIELAB Wardrobe Lock).
- **What Actually Improved on Sealed TEST-2 Ebright (`n=30` scenes):**
  - Biometric similarity was at parity with Baseline v1 (`0.3091 -> 0.3147`, `Delta = +0.0056`, CI crosses 0), while **Rank-1 identity disambiguation** rose `65.0% -> 80.0%` (`32/40` vs `26/40`, image CI `[+2.4%, +28.6%]`), **Wardrobe pass rate** rose `45.0% -> 80.0%` (`p=0.000183`), and **scene-level defects** dropped `3.07 -> 1.50` (`-51.1%`, `p=0.001453`).
- **Where v2.0 Fell Short (Honest Defense Takeaways):**
  - **Holdout (`n=12` scenes):** v2 was not better than v1 (`Bio 0.2596 vs 0.2758`, `Rank-1 54.2% vs 62.5%`, `Wardrobe 37.5% vs 41.7%`), and `M2`/`M3`/`M5` dropped the 3rd character on all `3` Holdout Scene 4 renders (`Spatial = 87.5% < 90.0%`).
  - **Pre-Registration (`0/4` strict):** Only `P2` Spatial & `P3` Wardrobe passed on Ebright alone; original `P1`, Holdout `P2`/`P3`, and `P4` (`7984` MiB whole-system peak > `7.4 GB`, `1/42` extra YOLO person) were `NOT MET`.
  - **Methodology Caveats:** `combined_binding = 0.0%` in both v1 and v2; `no_m4_ip_adapter` outperformed `full_v2` on DEV-2 & TEST-2 (`recommend M4 OFF in v2.1`); prior blind checks were **automated, not human**; scoring shares InsightFace anchor family with `M5`.
