# Verifier v1.1 Changelog

> **Verifier Version:** `1.1`  
> **Status:** FROZEN  
> **Combined SHA-256:** `ee3b29d5a3511f1bad0efed114b079d4c4478beeeafd6c13b9053ebb99c9b8ac`  
> **Date:** 2026-10-15  
> **Supercedes:** Verifier v1.0 (`verifier-frozen` tag)

---

## 1. Summary of Changes

| Component | Verifier v1.0 Flaw | Verifier v1.1 Fix | Evidence / Verification |
|---|---|---|---|
| **SegFormer Bridge** | In `my_app_env`, missing `torch` caused fallback to central-torso crop (`int(th*0.2):int(th*0.8)`). In Scene 1, sampled mahogany dining table, reporting 25% wardrobe accuracy. | Executed in Forge Python via `torch_bridge.py`. Central-torso fallback deleted. If $< 50$ px clothing found, reports `seg_failed`. Expanded clothing classes to `(4, 5, 6, 7, 17)`. | Control Test 4; zero silent fallbacks. |
| **Face Crop Embedding** | `DeepFace.represent()` re-ran OpenCV Haar cascade on cropped faces. If Haar failed on tightly-cropped face, whole crop or fallback was used, dropping similarity on JPEG compression to 0.71. | Added `detector_backend="skip"`. DeepFace skips redundant face detection and normalizes the audited face crop directly. | Control Test 8; JPEG compression similarity $> 0.998$ (diff $< 0.0013$). |
| **Calibration & EER** | Trivial calibration on 2 images per character gave 7 genuine and 32 impostor pairs, resulting in artificial EER = 0.0000. No child/adult splits. | Calibrated on 63 DEV scenes: 859 genuine pairs (561 child, 298 adult), 1,850 impostor pairs (1,408 hard cross-age/adult pairs). Realistic EER (0.4597 - 0.4773) with bootstrap 95% CIs. | `consistency_v2/verify/calibration.json` |
| **Wardrobe $\Delta E$ Calibration** | $\Delta E$ threshold was arbitrarily set to 25.0 without labeled ground truth. | Labeled 60 clothing crops (30 positive, 30 negative) across 8 target colors. Evaluated on 30 holdout crops: optimal $T^* = 10.0$ achieved 95.0% balanced accuracy (90% sensitivity, 100% specificity). | `consistency_v2/verify/calibration.json` |
| **Binding Decomposition** | Single binary binding metric obscured whether position, wardrobe, or identity failed. | Decomposed into: `spatial_position_pass_rate`, `wardrobe_palette_pass_rate`, `identity_at_position_pass_rate`, and `combined_binding_pass_rate`. | Reported in `report.json` and `summary.csv`. |
| **Age Subgroup Split** | Face recognizers have higher variance on children under 10; Ebright story has 3 child scenes and 3 adult scenes. Not analyzed previously. | Verifier automatically tracks child vs adult metrics across identity, Rank-1, and binding. | Separate child and adult summary dictionaries in `report.json`. |
| **Detector Health Tracking** | Verifier previously didn't track whether detectors actually ran or silently failed. | Explicit 9-point detector health monitoring (`retinaface`, `yolov8n`, `segformer`, `clip`, `dinov2`, `facenet512`, `sface`, `ghostfacenet`, `arcface`). | Reported in `detector_health` block. |
| **Deterministic Freeze** | Divergent hash tools used raw byte hashing sensitive to CRLF vs LF on Windows. | Canonical `tools/freeze_verifier.py` and `tools/check_freeze.py` with newline normalization. | CI gate passes consistently. |

---

## 2. Frozen File Manifest

| File | SHA-256 (LF-normalized) |
|---|---|
| `__init__.py` | `e45736073679d2b584f11f9b840674ce57e086f841fe95a036f89189eba4b373` |
| `__main__.py` | `90c802660ba74e43d83035a4ba6ec9ff593f21b6258bec20c94b1779702b7b63` |
| `appearance_evaluator.py` | `56069935c0d6fad6922f34f29dfc037281576dd00e1c9fba49372cd96d9b59f4` |
| `binding_checker.py` | `efb10c9d300062aeb28dd6615b3cba14dcf73c9451121fa25d97bcfae8387063` |
| `biometric_ensemble.py` | `1bc72e8111626f86cae7284ce40212001a15555462fc14f9611b85db8c946e3a` |
| `calibration.json` | `888f4e2f3d5ee1ba08154e1dfb5faeb925501861036d0b57e79cff3c749a9fb1` |
| `calibration.py` | `65849b6f3c9574aecb9adc6a82fdb7e4778e9ac1cd5d10d5bd27b5dca5648c69` |
| `cli.py` | `a906231efb7ee0c13bb101d24835260197a9f77b78913d33efea43734e5653db` |
| `scene_auditor.py` | `bf731af6b398d798d019c40493a280633a50ff5bf2653001399eec7f8f802c77` |
| `torch_bridge.py` | `42d82178984eea50dc8fa51643182f7e24d71f616903b08987daeda4c40e40fb` |
| `torch_vision_helper.py` | `9b3626e257ebbebb6cf4b46c6fcbf0c33ee3fe98971034444983e20ec42ef39b` |
| `visual_review/make_blind_sheet.py` | `83384d0fe56a11b7d869c2a14fa6e772f3e34c385dd4b5772d3ba6060bf751bc` |
| `visual_review/make_contact_sheet.py` | `66ebbe2aae654712f304dfbf389589d4993b9400e8aa3d0f74034a12ee19004c` |
| `visual_review/rubric.py` | `9cc9ba80421067eba0f33a913402f65d758e8bcf161c50a4eab6832ee1ad3871` |

**Canonical Combined SHA-256:** `ee3b29d5a3511f1bad0efed114b079d4c4478beeeafd6c13b9053ebb99c9b8ac`

---

## 3. Control Tests Suite

The verifier behavior is guarded by 10 deterministic unit tests in `tests/verifier_controls/test_controls.py`:
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
