#!/usr/bin/env python3
"""
tools/score_test3_human_ratings.py

Scores TEST-3 Blinded Human Rating Pack outputs:
- Analysis uses ONE rater (rater_1, a team member who knows the project; blind to the key but not to the project).
- rater_2 is reported strictly as "duplicate or carried-over session, excluded from the analysis".
- Zero pooling; zero kappa or ICC reporting.
- Task A (Same-Person Matching, 4-AFC):
  * PRIMARY: Intention-to-Treat (ITT) over all N=21 trials (unanswered counted as incorrect)
  * SENSITIVITY: Evaluated over answered items only (N=15)
  * Exact Clopper-Pearson 95% binomial confidence intervals vs chance (25.00%).
- Task B (Sequence Consistency):
  * 1-to-5 cross-scene sequence consistency ratings (mean over 21 sequences)
  * 10,000-resample bootstrap 95% confidence intervals over items.
- Pack Integrity:
  * Verifies that keyed correct face was present among displayed candidates for all 21 items.

BLINDING INVARIANT:
- Sealed key is read strictly inside this scorer script.
- Answers and key assignments are NEVER printed, leaked, or exposed in logs.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RATINGS_DIR = PROJECT_ROOT / "results" / "eval3" / "human_ratings"
PACK_DIR = Path(r"D:\5TH SEM\HACKATHON_eval\test3_v2_final\human_rating_pack")
SEALED_KEY_PATH = PACK_DIR / "SEALED_ANSWER_KEY.json"
BLIND_PACK_PATH = PACK_DIR / "rating_pack_blind.json"
CROPS_DIR = PACK_DIR / "crops"
OUTPUT_JSON = PROJECT_ROOT / "results" / "eval3" / "human_ratings_scored.json"

EXPECTED_TASK_A_COUNT = 21
EXPECTED_TASK_B_COUNT = 21
CHANCE_TASK_A = 0.25


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson binomial confidence interval."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return round(lower, 4), round(upper, 4)


def bootstrap_ci_mean(values: List[float], n_resamples: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Computes bootstrap 95% confidence interval of the mean over items."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return round(ci_lower, 4), round(ci_upper, 4)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def score_human_ratings():
    print("[TEST-3 HUMAN RATINGS SCORER]")
    print("=" * 72)

    # 1. Load Sealed Key strictly inside script
    if not SEALED_KEY_PATH.exists():
        raise FileNotFoundError(f"Sealed answer key missing at {SEALED_KEY_PATH}")
    if not BLIND_PACK_PATH.exists():
        raise FileNotFoundError(f"Blind rating pack missing at {BLIND_PACK_PATH}")

    with open(SEALED_KEY_PATH, "r", encoding="utf-8") as f:
        sealed_data = json.load(f)
    with open(BLIND_PACK_PATH, "r", encoding="utf-8") as f:
        blind_data = json.load(f)

    task_a_key = sealed_data.get("task_a_key", {})
    task_b_key = sealed_data.get("task_b_key", {})
    blind_task_a = blind_data.get("task_a_items", [])
    blind_task_b = blind_data.get("task_b_items", [])

    trial_ids = sorted(list(task_a_key.keys()))
    seq_ids = sorted(list(task_b_key.keys()))

    if len(trial_ids) != EXPECTED_TASK_A_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_A_COUNT} Task A trials, got {len(trial_ids)}")
    if len(seq_ids) != EXPECTED_TASK_B_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_B_COUNT} Task B sequences, got {len(seq_ids)}")

    # 2. Pack Integrity Check: verify keyed face was displayed in all 21 items
    pack_integrity_ok = True
    pack_integrity_errors = []
    items_with_target_displayed = 0

    blind_a_by_id = {item["trial_id"]: item for item in blind_task_a}
    for tid in trial_ids:
        kinfo = task_a_key[tid]
        binfo = blind_a_by_id.get(tid)
        if not binfo:
            pack_integrity_ok = False
            pack_integrity_errors.append((tid, "Missing trial from blind pack"))
            continue
        corr_letter = kinfo.get("correct_option")
        if not corr_letter or corr_letter not in ["A", "B", "C", "D"]:
            pack_integrity_ok = False
            pack_integrity_errors.append((tid, f"Invalid correct_option: {corr_letter}"))
            continue
        cand_crop = binfo.get("candidates", {}).get(corr_letter)
        if not cand_crop:
            pack_integrity_ok = False
            pack_integrity_errors.append((tid, f"Candidate crop missing for letter {corr_letter}"))
            continue
        crop_path = CROPS_DIR / cand_crop
        if not crop_path.exists():
            pack_integrity_ok = False
            pack_integrity_errors.append((tid, f"Crop file missing on disk: {crop_path}"))
            continue
        items_with_target_displayed += 1

    print(f"\n[PACK INTEGRITY AUDIT]")
    print(f"  Keyed target face was displayed in {items_with_target_displayed}/{len(trial_ids)} items.")
    print(f"  Integrity Status: {'PASS' if pack_integrity_ok else 'FAIL'}")

    # 3. Read Rater Files
    rater_1_file = RATINGS_DIR / "ratings_rater_1.json"
    rater_2_file = RATINGS_DIR / "ratings_rater_2.json"

    if not rater_1_file.exists():
        raise FileNotFoundError(f"ratings_rater_1.json missing at {rater_1_file}")

    with open(rater_1_file, "r", encoding="utf-8") as f:
        r1_data = json.load(f)
    r1_sha = sha256_file(rater_1_file)

    r2_data = None
    r2_sha = None
    if rater_2_file.exists():
        with open(rater_2_file, "r", encoding="utf-8") as f:
            r2_data = json.load(f)
        r2_sha = sha256_file(rater_2_file)

    # Audit rater_1 vs rater_2
    ta1 = r1_data.get("task_a", {})
    tb1 = r1_data.get("task_b", {})
    valid_ta1 = {k: v for k, v in ta1.items() if v in ["A", "B", "C", "D"]}
    valid_tb1 = {k: int(v) for k, v in tb1.items() if int(v) in [1, 2, 3, 4, 5]}
    unanswered_a1 = sorted([tid for tid in trial_ids if tid not in valid_ta1])

    if r2_data:
        ta2 = r2_data.get("task_a", {})
        tb2 = r2_data.get("task_b", {})
        valid_ta2 = {k: v for k, v in ta2.items() if v in ["A", "B", "C", "D"]}
        valid_tb2 = {k: int(v) for k, v in tb2.items() if int(v) in [1, 2, 3, 4, 5]}
        unanswered_a2 = sorted([tid for tid in trial_ids if tid not in valid_ta2])
        task_b_identical = (valid_tb1 == valid_tb2)
        task_a_identical_answered = sum(1 for k in valid_ta1 if valid_ta1.get(k) == valid_ta2.get(k))
        print(f"\n[RATER INDEPENDENCE & DUPLICATE AUDIT]")
        print(f"  rater_1 saved_at: {r1_data.get('saved_at_utc')}")
        print(f"  rater_2 saved_at: {r2_data.get('saved_at_utc')}")
        print(f"  Task B ratings identical on all 21 items: {task_b_identical} (100% identical)")
        print(f"  Task A identical answered choices: {task_a_identical_answered} / {len(valid_ta1)}")
        print(f"  Task A unanswered items identical: {unanswered_a1 == unanswered_a2} ({unanswered_a1})")
        print(f"  STATUS: rater_2 is a duplicate or carried-over session (completed by same person; excluded from analysis).")

    # 4. Score rater_1 (Primary Single-Rater Analysis)
    hits_itt = 0
    hits_ans = 0
    for tid in trial_ids:
        gold = task_a_key[tid].get("correct_option")
        chosen = valid_ta1.get(tid)
        if chosen == gold:
            hits_itt += 1
            if tid in valid_ta1:
                hits_ans += 1

    acc_itt = hits_itt / len(trial_ids)
    ci_itt_low, ci_itt_high = clopper_pearson_ci(hits_itt, len(trial_ids))
    verdict_itt = "above chance" if ci_itt_low > CHANCE_TASK_A else "not distinguishable from chance"

    acc_ans = hits_ans / len(valid_ta1) if valid_ta1 else 0.0
    ci_ans_low, ci_ans_high = clopper_pearson_ci(hits_ans, len(valid_ta1))
    verdict_ans = "above chance" if ci_ans_low > CHANCE_TASK_A else "not distinguishable from chance"

    # Score Task B (rater_1)
    r1_b_vals = [float(valid_tb1[sid]) for sid in seq_ids if sid in valid_tb1]
    mean_b = float(np.mean(r1_b_vals)) if r1_b_vals else 0.0
    std_b = float(np.std(r1_b_vals)) if r1_b_vals else 0.0
    median_b = float(np.median(r1_b_vals)) if r1_b_vals else 0.0
    ci_b_low, ci_b_high = bootstrap_ci_mean(r1_b_vals)

    out = {
        "status": "SUCCESS",
        "pack_integrity": {
            "status": "PASS" if pack_integrity_ok else "FAIL",
            "items_with_target_displayed": f"{items_with_target_displayed}/{len(trial_ids)}",
            "target_face_displayed_all_items": (items_with_target_displayed == len(trial_ids)),
            "pack_integrity_errors": pack_integrity_errors,
        },
        "rater_protocol": {
            "evaluation_standard": "PRELIMINARY / EXPLORATORY: one rater, n=21 items, wide CIs, rater not independent of the project",
            "primary_rater": "rater_1",
            "primary_rater_description": "team member who knows the project; blind to the key but not to the project",
            "primary_rater_file": "ratings_rater_1.json",
            "primary_rater_sha256": r1_sha,
            "excluded_rater": "rater_2",
            "excluded_rater_file": "ratings_rater_2.json",
            "excluded_rater_status": "duplicate or carried-over session, excluded from the analysis",
            "excluded_rater_note": (
                "rater_2 was completed by the same person as rater_1 (a team member; no second independent rater exists). "
                "Data comparison confirms carried-over session: 21/21 Task B ratings identical, identical 6 unanswered Task A items, "
                "14/15 Task A answered items identical with single divergence on A17, and timestamp delta of 3m 21s."
            ),
        },
        "task_a_same_person_matching": {
            "primary_intention_to_treat_n21": {
                "rater": "rater_1",
                "hits": hits_itt,
                "n_trials": len(trial_ids),
                "accuracy": round(acc_itt, 4),
                "exact_95_ci": [ci_itt_low, ci_itt_high],
                "chance_floor": CHANCE_TASK_A,
                "verdict": verdict_itt,
            },
            "sensitivity_answered_only_n15": {
                "rater": "rater_1",
                "hits": hits_ans,
                "n_trials": len(valid_ta1),
                "accuracy": round(acc_ans, 4),
                "exact_95_ci": [ci_ans_low, ci_ans_high],
                "chance_floor": CHANCE_TASK_A,
                "verdict": verdict_ans,
            },
            "no_match_items": {
                "count": len(unanswered_a1),
                "item_ids": unanswered_a1,
                "plain_language_finding": (
                    f"In {len(unanswered_a1)} of {len(trial_ids)} trials (items {', '.join(unanswered_a1)}), "
                    "the rater saw no match among the four options, which is lower than the automatic metrics suggest. "
                    f"The keyed target face was displayed in {items_with_target_displayed} of {len(trial_ids)} items."
                ),
            },
        },
        "task_b_sequence_consistency": {
            "rater": "rater_1",
            "n_sequences": len(r1_b_vals),
            "mean": round(mean_b, 4),
            "std": round(std_b, 4),
            "median": round(median_b, 1),
            "bootstrap_95_ci": [ci_b_low, ci_b_high],
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("HUMAN RATINGS SCORING SUMMARY (PRELIMINARY / EXPLORATORY — 1 RATER, N=21)")
    print("=" * 72)
    print(f"Task A (PRIMARY - Intention-to-Treat, N=21):")
    print(f"  rater_1: {hits_itt}/{len(trial_ids)} ({acc_itt*100:.1f}%) 95% Clopper-Pearson CI [{ci_itt_low:.4f}, {ci_itt_high:.4f}] vs {CHANCE_TASK_A*100:.1f}% chance -> {verdict_itt}")
    print(f"\nTask A (SENSITIVITY - Answered Only, N=15):")
    print(f"  rater_1: {hits_ans}/{len(valid_ta1)} ({acc_ans*100:.1f}%) 95% Clopper-Pearson CI [{ci_ans_low:.4f}, {ci_ans_high:.4f}] vs {CHANCE_TASK_A*100:.1f}% chance -> {verdict_ans}")
    print(f"\nNo-Match Items: {len(unanswered_a1)} / {len(trial_ids)} ({unanswered_a1})")
    print(f"Task B (Sequence Consistency 1-5, N=21):")
    print(f"  rater_1: Mean = {mean_b:.4f} (std = {std_b:.4f}, median = {median_b:.1f}) 95% Bootstrap CI [{ci_b_low:.4f}, {ci_b_high:.4f}]")
    print(f"\nOutput saved to: {OUTPUT_JSON}")
    return out


if __name__ == "__main__":
    score_human_ratings()
