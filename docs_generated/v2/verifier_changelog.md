# Verifier v1.2 Changelog

> **Verifier Version:** `1.2`  
> **Status:** FROZEN  
> **Combined SHA-256:** `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6`  
> **Date:** 2026-10-18  
> **Supercedes:** Verifier v1.1 (`verifier-v1.1-frozen` tag, `ee3b29d5...`)

---

## 1. Summary of Changes

| Component | Verifier v1.1 Flaw | Verifier v1.2 Fix | Evidence / Verification |
|---|---|---|---|
| **Calibrated Thresholds & Default Loading** | `load_calibration()` looked for `data["thresholds"]`, but `calibration.json` had `thresholds_adult`, `thresholds_child`, etc. Furthermore `BiometricEnsemble.__init__()` had `calibration_path=None` by default. As a result, the ensemble remained at uncalibrated `DEFAULT_THRESHOLDS` (0.40, 0.45, 0.42, 0.48). With real face similarities ~0.20-0.25, `pass_count >= 3` was impossible, pinning `ensemble_passed` and `identity_at_position_pass` to exact 0.0% for 100% of benchmark images (including baseline v1). | `__init__` now defaults to `calibration.json`. Added top-level `"thresholds"` and `"thresholds_teen"` to `calibration.json`. Updated `DEFAULT_THRESHOLDS` to calibrated values (`Facenet512: 0.22, SFace: 0.24, GhostFaceNet: 0.18, ArcFace: 0.25`). Calibrated thresholds allow binding metrics to move realistically. | Control Test 11; genuine bank reference pass rate jumps from 47-80% to 92-100%. |
| **Multi-Face Scene Matching (Crowd Bug)** | `cli.py` matched characters by arbitrary detection index (`detected_faces[i]`). In Scene 6 Graduation (1 expected face, 4-14 detected faces in crowd), `college_ebright` (`i=0`) was matched to a tiny 30px audience member on the left edge rather than the 137px hero subject on stage in the center. | Implemented `assign_faces_to_characters()` using spatial zone and bounding box prominence (area $w \times h$). Filters out background crowd faces (< 20% max area or < 56px) when assigning hero subjects. | Control Test 12; synthetic 3-face crowd correctly matches center hero (140px), not spectator (35px). |
| **Small-Crop ArcFace Collapse Guard** | When fed small blurry crops (< 56px), ArcFace produces degenerate normalized embeddings yielding artificial pseudo-similarities > 0.95 to any reference face, while Facenet512 correctly rejects them (< 0.15). | Added small-crop guard in `score_against_bank`: if face width < 56px and ArcFace similarity exceeds Facenet512 by > 0.25, ArcFace is discounted relative to Facenet512. | Control Test 13; small 40px crop pseudo-similarity clamped. |
| **CLIP Prop Zero-Shot Calibration** | Binary forced-choice `["a scene showing {p}", "a scene without {p}"]` caused severe confirmation bias due to CLIP text negation weakness. Clean period scenes (Scene 1, Scene 2) were flagged as containing `laptop`, `smartphone`, and `computer` at 75-100% false-alarm rates. | Calibrated contrastive prompt sets with domain-appropriate negative/vintage room descriptors (`"a photo of an empty desk or vintage furniture"`, `"a historical period room with wooden furniture and paper books"`). Requires $\ge 0.70$ probability and strict margin over null scenes. | Control Test 14; 0 false positive props detected on clean period storyboard images. |
| **3-Way Age Disaggregation** | `teen_ebright` was previously lumped into the adult category (`is_child = bool("young" in cid)`). | Disaggregated into 3 distinct biological age stages: `child` (`young_ebright`), `teen` (`teen_ebright`), and `adult` (`college_ebright`, `james_wong`, `mother`). Summary reports metrics for all three separately. | Control Test 15; explicit 3-way split in `age_split_metrics`. |
| **Persistent Vision Worker IPC** | Previous one-shot subprocess bridge incurred ~35s cold-start overhead per image loading PyTorch, Ultralytics, and Transformers. | Added `--worker` mode to `torch_vision_helper.py` and persistent bidirectional JSON-lines IPC in `torch_bridge.py`, reducing per-image vision latency to 0.60s while maintaining 100% bit-exact mathematical outputs. | Control Tests 1-15 verified with sub-second per-call turnaround. |

---

## 2. Frozen File Manifest

| File | SHA-256 (LF-normalized) |
|---|---|
| `__init__.py` | `3b59657dfe90bcdc72e4d3d39ee16cfb0c45dac8f4aeb00fefa138aea8d94334` |
| `__main__.py` | `d8a58bc354041bbe9e53518a1a612938cc26e2aecc06adf52f7825bb9385cf14` |
| `appearance_evaluator.py` | `42d1ad69fdfe213a6b68f6e9bfd4898b2756a0aaebea74836874fb12c85773ae` |
| `binding_checker.py` | `4cf2eda80cce7dcf7828a649e46f827b5974bc7d9fce03e9b71f229970c27748` |
| `biometric_ensemble.py` | `13f1fd4f97f2d5720a0e3b50f86dcfc1737ff6a5a593e17155bf4f41dfadbd60` |
| `calibration.json` | `e2927e53e40b14e275151f60abaf01498478f3cb5b465dcb84d6776ba62b7bef` |
| `calibration.py` | `33a8454a1c4e8cb50d07289154483435e95891d2ff30b60c5b92b33dc5a46491` |
| `cli.py` | `11ac342f7ded00832cfb32f35bb0dd1981727dbe5a6a0cde41251b93d17c5b9b` |
| `scene_auditor.py` | `73f605bb6a5919f3191caf4a2ec6d1387f0c978c05251675f2f6b288cdcfb81f` |
| `torch_bridge.py` | `be7aee9591746b6fe17f2999a52f6cfa8854ea62f1a680210efa145830985be0` |
| `torch_vision_helper.py` | `b15f878eec2d5923f20f484001e58c44d233113eac8cc5a072d6fb4f8cc186ef` |
| `visual_review/make_blind_sheet.py` | `1a0c4e12cfe0822d72061eafcfc5e5f982e4cf92c26ec9c492b4ca30f792f546` |
| `visual_review/make_contact_sheet.py` | `c3d6daebc2e504306c108128301b53bd59329d1ebcabd365757ab002889d1455` |
| `visual_review/rubric.py` | `c432a8a73dafc3bc6044b50e287ebf96d7207039eaa1e1ea5bd6be2a70df1a10` |

**Canonical Combined SHA-256:** `3fa69071f888643bccb1c21ee732415ac81996a9f3cb0b71dedd68086fecb4b6`

---

## 3. Control Tests Suite

The verifier behavior is guarded by 15 deterministic unit tests in `tests/verifier_controls/test_controls.py`:
1. `test_01_identical_image_gives_max_similarity`: Cosine similarity == 1.0 (within 0.005).
2. `test_02_horizontal_flip_fails_spatial_binding`: Inversion of left/right fails spatial binding.
3. `test_03_color_inversion_flagged`: Inversion detector catches color-inverted canary.
4. `test_04_hue_rotated_shirt_fails_wardrobe`: Hue-rotated shirt fails $\Delta E \le 10.0$ threshold.
5. `test_05_pasted_wrong_face_breaks_rank1`: Mother's face queried against Ebright fails Rank-1.
6. `test_06_duplicated_face_triggers_clone_alarm`: Cloned face triggers intra-scene clone alarm.
7. `test_07_blank_image_reports_missing_people`: Blank image reports missing expected characters.
8. `test_08_jpeg_resave_within_small_tolerance`: JPEG compression alters similarity by $< 0.002$.
9. `test_09_clean_image_has_zero_defects`: Clean portrait has 0 defect taxonomy violations.
10. `test_10_detector_health_tracked`: All vision detectors verify `ran=True`.
11. `test_11_calibrated_thresholds_loaded`: BiometricEnsemble automatically loads calibrated v1.2 thresholds.
12. `test_12_spatial_prominence_assigns_hero`: Multi-face scene matching assigns foreground hero face, not background crowd.
13. `test_13_small_crop_arcface_guard`: Small crops (<56px) with divergent pseudo-similarity are discounted.
14. `test_14_calibrated_props_no_false_positive_on_clean_period`: Calibrated CLIP prop detection triggers 0 false alarms on clean period scenes.
15. `test_15_three_way_age_disaggregation`: Child, teen, and adult age categories are tracked as separate non-overlapping groups.
