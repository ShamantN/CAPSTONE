# MMCCP v2.0 — Project Handoff Checklist (`v2.0.0-rc3` + PR #35)

---

## 1. Merge & Verify PR #35 (`docs/run-guides` -> `main`)
1. Open **PR #35**: **https://github.com/ShamantN/CAPSTONE/pull/35** (`docs/run-guides` -> `main`, base `625b365`).
2. Confirm CI passes cleanly and skim these two ground-truth audit artifacts in PR #35 before clicking **Squash and merge** (or Merge):
   - Execution transcript & timings: [`docs/run/TEST_LOG.md`](TEST_LOG.md)
   - Self-review, `UNTESTED` & `NEEDS SOURCE` registers: [`docs/run/REVIEW.md`](REVIEW.md)
   - Recommended Forge patches & `--apply` CLI: [`patches/forge/README.md`](../../patches/forge/README.md) & [`tools/apply_forge_patches.py`](../../tools/apply_forge_patches.py)
3. After merging PR #35 into `main`, verify live on local `main`:
   ```powershell
   git --no-pager checkout main && git --no-pager pull --ff-only origin main
   .\my_app_env\Scripts\python.exe tools\check_freeze.py
   ```
   *(Once merged to `main`, `https://colab.research.google.com/github/ShamantN/CAPSTONE/blob/main/notebooks/MMCCP_Colab.ipynb` becomes live.)*

---

## 2. Optional Future Work Left for the Human (`NOT Started`)

### A. Human Blind Visual Evaluation (`TEST-2` Ebright — Awaiting Human Input)
- Automated R8 A/B (`10/12`) and Re-ID checks were proxy runs, **not** human evaluations (`docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md`).
- To complete human scoring: inspect `docs_generated/v2/visual_review/human_blind_ab/pair_01..30_blind.png` and `human_blind_reid_crops/reid_board.png`, record selections in [`docs_generated/v2/visual_review/blind_predictions_human.json`](../../docs_generated/v2/visual_review/blind_predictions_human.json), and run [`tools/score_blind_test.py`](../../tools/score_blind_test.py).

### B. Optional Google Colab Live Verification (`UNTESTED`)
- Open the Colab link above after merging PR #35 and walk [`03_GOOGLE_COLAB.md`](03_GOOGLE_COLAB.md) Cells `0–15` cleanly on a T4 GPU (no Colab run has been executed yet).

### C. Optional `v2.1` Architectural Bugs *(Deliberately Not Started in Frozen `v2.0`)*
1. **3+ Character Layout Cap:** `layout_planner.py`, `regional.py`, `prompt_builder.py`, and M5 swap enforce a hard 2-character per-scene limit — causing Holdout Scene 4 (`3` expected characters) to drop 1 character on all `3` `TEST-2` seeds (`Spatial 87.5%`).
2. **Scene-State Wardrobe Shifts:** M7 `WardrobeLock` enforces a single static canonical anchor wardrobe CIELAB vector per character (`<= 38.0`), lacking per-scene outfit changes when a story script specifies a costume change mid-story (contributing to Holdout `Wardrobe 37.5%`).

---

## 3. What to Tell the Mentor (10 Lines — Strictly From `CLAIMS_AND_LIMITS.md`)

> 1. Under frozen Verifier `v1.2` (`SHA: 3fa69071...`) on sealed `TEST-2` (`42` renders: `30` Ebright, `12` Holdout), MMCCP v2 delivered strong **Ebright wardrobe accuracy (`45.0% -> 80.0%`, `+35.0%`, `p=0.000183`)**.
> 2. **Per-scene defects dropped `51.1%` on Ebright** (`3.07 -> 1.50/scene`, `p=0.001453`) and **`2.74 -> 1.43/scene`** across Combined `TEST-2` (`p=0.000877`).
> 3. **Ebright Rank-1 identity disambiguation rose `65.0% -> 80.0%` (`+15.0%`)** with image-level 95% CI `[+2.4%, +28.6%]` (while 6-scene cluster CI `[+0.0%, +33.3%]` touches zero, `p=0.0578`).
> 4. **Ebright Biometric Ensemble identity stayed at parity** with Baseline v1 (`0.3091 -> 0.3147`, `+0.0056`, `p=0.3754`) and hit **`97.5%` (`39/40`) Ebright spatial accuracy**.
> 5. **Strictly speaking, `0/4` original all-split pre-registered rules passed** because **Holdout (`12` scenes)** did not beat v1 (`Bio 0.2596 vs 0.2758`, `Rank-1 54.2% vs 62.5%`, `Wardrobe 37.5% vs 41.7%`, `Spatial 87.5%`).
> 6. Holdout Scene 4 failed all `3` seeds because M2/M3/M5 currently hard-cap at **2 characters per scene** (dropping the 3rd person).
> 7. Combined `spatial + identity + wardrobe` binding read **`0.0%` for both v1 and v2** under frozen `v1.2` calibration, and scoring is partly circular (`InsightFace 0.8397 vs 0.7408`).
> 8. Zero OOMs occurred (`7011 MiB` / `6.847 GiB` process delta), though whole-system VRAM reached **`7984 MiB` (`7.797 GiB`)**, exceeding the `7.4 GB` ceiling for `52s`.
> 9. In single-seed ablations (`TEST-2` arm is post-hoc exploratory after unsealing; not used to tune `v2.0`), `no_m4_ip_adapter` slightly exceeded `full_v2` on 4-judge Bio inside `0.03` noise (`0.3373 vs 0.3269` on `TEST-2`, `0.3423 vs 0.3323` on `DEV-2`) while losing `Facenet512`/`DINOv2` — making M4 a **candidate for a multi-seed re-test in `v2.1`**, and `M10` is ON with no ablation arm.
> 10. All code/verifier/split audits pass cleanly (`PR #35` adds reproducible fresh-clone guides, preflight tooling, and recommended Forge patches; only human blind review & optional Colab validation remain unexecuted).

