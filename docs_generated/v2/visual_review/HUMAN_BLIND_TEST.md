# Human Blind Evaluation Protocol (TEST-2 Ebright)

> [!IMPORTANT]
> **STATUS: AWAITING HUMAN COMPLETION (DO NOT AUTO-FILL)**  
> All previous Phase R8 blind, canary, and VLM checks (`blind_study_report.json`, `canary_audit_report.json`, `vlm_second_opinion_report.md`, and superseded `blind_review_summary.json`) were **automated proxy checks, not human evaluations**.  
> This human protocol tests genuine perceptual preference and character re-identification on **TEST-2 Ebright** (`v1` Baseline vs `v2.0.0-rc2/rc3`, 30 shuffled pairs + 20 anonymous face crops).

---

## Part 1: Blind A/B Pairwise Preference (30 Shuffled Pairs)

Inspect each side-by-side composite image in `docs_generated/v2/visual_review/human_blind_ab/<pair_id>_blind.png`.
For each pair, judge BOTH **character identity consistency** and **overall visual quality**, and record `"A"`, `"B"`, or `"tie"` under `ab_preference.<pair_id>.selected_option` in:
- [`docs_generated/v2/visual_review/blind_predictions_human.json`](blind_predictions_human.json)

| Pair ID | Blind Composite Path |
| :--- | :--- |
| `pair_01` | `docs_generated/v2/visual_review/human_blind_ab/pair_01_blind.png` |
| `pair_02` | `docs_generated/v2/visual_review/human_blind_ab/pair_02_blind.png` |
| `pair_03` | `docs_generated/v2/visual_review/human_blind_ab/pair_03_blind.png` |
| `pair_04` | `docs_generated/v2/visual_review/human_blind_ab/pair_04_blind.png` |
| `pair_05` | `docs_generated/v2/visual_review/human_blind_ab/pair_05_blind.png` |
| `pair_06` | `docs_generated/v2/visual_review/human_blind_ab/pair_06_blind.png` |
| `pair_07` | `docs_generated/v2/visual_review/human_blind_ab/pair_07_blind.png` |
| `pair_08` | `docs_generated/v2/visual_review/human_blind_ab/pair_08_blind.png` |
| `pair_09` | `docs_generated/v2/visual_review/human_blind_ab/pair_09_blind.png` |
| `pair_10` | `docs_generated/v2/visual_review/human_blind_ab/pair_10_blind.png` |
| `pair_11` | `docs_generated/v2/visual_review/human_blind_ab/pair_11_blind.png` |
| `pair_12` | `docs_generated/v2/visual_review/human_blind_ab/pair_12_blind.png` |
| `pair_13` | `docs_generated/v2/visual_review/human_blind_ab/pair_13_blind.png` |
| `pair_14` | `docs_generated/v2/visual_review/human_blind_ab/pair_14_blind.png` |
| `pair_15` | `docs_generated/v2/visual_review/human_blind_ab/pair_15_blind.png` |
| `pair_16` | `docs_generated/v2/visual_review/human_blind_ab/pair_16_blind.png` |
| `pair_17` | `docs_generated/v2/visual_review/human_blind_ab/pair_17_blind.png` |
| `pair_18` | `docs_generated/v2/visual_review/human_blind_ab/pair_18_blind.png` |
| `pair_19` | `docs_generated/v2/visual_review/human_blind_ab/pair_19_blind.png` |
| `pair_20` | `docs_generated/v2/visual_review/human_blind_ab/pair_20_blind.png` |
| `pair_21` | `docs_generated/v2/visual_review/human_blind_ab/pair_21_blind.png` |
| `pair_22` | `docs_generated/v2/visual_review/human_blind_ab/pair_22_blind.png` |
| `pair_23` | `docs_generated/v2/visual_review/human_blind_ab/pair_23_blind.png` |
| `pair_24` | `docs_generated/v2/visual_review/human_blind_ab/pair_24_blind.png` |
| `pair_25` | `docs_generated/v2/visual_review/human_blind_ab/pair_25_blind.png` |
| `pair_26` | `docs_generated/v2/visual_review/human_blind_ab/pair_26_blind.png` |
| `pair_27` | `docs_generated/v2/visual_review/human_blind_ab/pair_27_blind.png` |
| `pair_28` | `docs_generated/v2/visual_review/human_blind_ab/pair_28_blind.png` |
| `pair_29` | `docs_generated/v2/visual_review/human_blind_ab/pair_29_blind.png` |
| `pair_30` | `docs_generated/v2/visual_review/human_blind_ab/pair_30_blind.png` |

---

## Part 2: Anonymous Face Re-Identification (20 Crops)

Inspect `docs_generated/v2/visual_review/human_blind_reid_crops/reid_board.png` (or individual files `docs_generated/v2/visual_review/human_blind_reid_crops/crop_01.png` .. `docs_generated/v2/visual_review/human_blind_reid_crops/crop_20.png`).
Reference anchors (for character identity visual lookup):
- `young_ebright`: `experiments/clean_run_ground_truth/anchors/young_ebright_ground_truth.png`
- `teen_ebright`: `experiments/clean_run_ground_truth/anchors/teen_ebright_ground_truth.png`
- `college_ebright`: `experiments/clean_run_ground_truth/anchors/college_ebright_ground_truth.png`
- `mother`: `experiments/clean_run_ground_truth/anchors/mother_ground_truth.png`
- `james_wong`: `experiments/clean_run_ground_truth/anchors/james_wong_ground_truth.png`

For each `crop_01` .. `crop_20`, fill `blind_reid.<crop_id>.predicted_character_id` with one of: `young_ebright`, `teen_ebright`, `college_ebright`, `mother`, `james_wong`.

| Crop ID | Image Path |
| :--- | :--- |
| `crop_01` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_01.png` |
| `crop_02` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_02.png` |
| `crop_03` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_03.png` |
| `crop_04` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_04.png` |
| `crop_05` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_05.png` |
| `crop_06` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_06.png` |
| `crop_07` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_07.png` |
| `crop_08` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_08.png` |
| `crop_09` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_09.png` |
| `crop_10` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_10.png` |
| `crop_11` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_11.png` |
| `crop_12` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_12.png` |
| `crop_13` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_13.png` |
| `crop_14` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_14.png` |
| `crop_15` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_15.png` |
| `crop_16` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_16.png` |
| `crop_17` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_17.png` |
| `crop_18` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_18.png` |
| `crop_19` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_19.png` |
| `crop_20` | `docs_generated/v2/visual_review/human_blind_reid_crops/crop_20.png` |

---

## Next Step (Scoring Only After Human Return)

Once `blind_predictions_human.json` is filled by a human evaluator, score it against the sealed key in `scratch/sealed_keys/human_test2_ebright/`.
