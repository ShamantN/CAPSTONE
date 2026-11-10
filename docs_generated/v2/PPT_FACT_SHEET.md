# PPT Fact Sheet: Reconciled Ground-Truth Numbers

- **Ebright TEST-2 (`n=30` scenes, `40` faces):**
  - Bio Mean: `0.3147` (v1 `0.3091`, `Delta = +0.0056`, cluster CI `[-0.0149, +0.0317]`)
  - Rank-1 Accuracy: `80.0%` (`32/40`, v1 `65.0%`)
  - Wardrobe Pass: `80.0%` (`32/40`, v1 `45.0%`, `p=0.000183`)
  - Spatial Pass: `97.5%` (`39/40`)
  - Defects/Scene (Primary): `1.50` (v1 `3.07`, `-51.1%`, `p=0.001453`)
- **Holdout TEST-2 (`n=12` scenes, `24` faces — NOT better than v1):**
  - Bio Mean: `0.2596` (v1 `0.2758`, `Delta = -0.0162`)
  - Rank-1 Accuracy: `54.2%` (`13/24`, v1 `62.5%`)
  - Spatial / Wardrobe Pass: `87.5%` (`21/24`) / `37.5%` (`9/24`)
  - Defects/Scene (Primary): `1.25` (v1 `1.92`)
- **Combined Binding Gate (v1 & v2):** `0.0%` (unmeasurable under Verifier `v1.2`)
- **VRAM Telemetry (`vram_test2_sealed.csv`):** Whole-system peak `7984 MiB` (`52s > 7400` MiB), `973 MiB` idle, `7011 MiB` process delta, `0` OOMs
- **Extra YOLO (`conf=0.40`):** `1/42` v2 TEST-2 renders
- **Original Pre-Registered Hypotheses:** `0/4` strict all-split MET (`P1` NOT MET, `P2` NOT MET [Ebright 97.5%, Holdout 87.5%], `P3` NOT MET [Ebright 80.0%, Holdout 37.5%], `P4` NOT MET)
- **Blind Review:** Automated proxy (`10/12`), **not human**; human blind test pending
- **Frozen Verifier:** `v1.2 (SHA: 3fa69071f888643b...)`
