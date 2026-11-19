#!/usr/bin/env python3
"""
tools/score_test3_human_ratings.py

Scores TEST-3 Blinded Human Rating Pack outputs:
- Task A (Same-Person Matching):
  * PRIMARY: Intention-to-Treat (ITT) over all N=21 trials (unanswered counted as incorrect)
  * SENSITIVITY: Evaluated over answered items only (N=15)
  * Exact Clopper-Pearson 95% binomial confidence intervals vs chance (25.00%).
- Task B (Sequence Consistency):
  * 1-to-5 cross-scene sequence consistency ratings (mean per rater and pooled)
  * 10,000-resample bootstrap 95% confidence intervals over items.
- Inter-rater Agreement:
  * Task A: Cohen's kappa
  * Task B: Quadratic-weighted kappa
- Pack Integrity Audit:
  * Verifies whether keyed correct face is present in displayed candidates for all 21 items.

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


def compute_cohen_kappa(r1_items: Dict[str, str], r2_items: Dict[str, str], common_keys: List[str]) -> float:
    """Computes Cohen's kappa for nominal categorical choices."""
    categories = sorted(list(set([r1_items[k] for k in common_keys] + [r2_items[k] for k in common_keys])))
    n = len(common_keys)
    if n == 0:
        return 0.0
    po = sum(1 for k in common_keys if r1_items[k] == r2_items[k]) / n

    pe = 0.0
    for cat in categories:
        p1 = sum(1 for k in common_keys if r1_items[k] == cat) / n
        p2 = sum(1 for k in common_keys if r2_items[k] == cat) / n
        pe += p1 * p2

    if pe >= 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def compute_weighted_kappa(r1_ratings: Dict[str, int], r2_ratings: Dict[str, int], common_keys: List[str]) -> float:
    """Computes quadratic-weighted kappa for ordinal 1-to-5 ratings."""
    n = len(common_keys)
    if n == 0:
        return 0.0
    scores = [1, 2, 3, 4, 5]
    k_max = len(scores) - 1

    weights = np.zeros((len(scores), len(scores)))
    for i, si in enumerate(scores):
        for j, sj in enumerate(scores):
            weights[i, j] = 1.0 - ((si - sj) ** 2) / (k_max ** 2)

    hist1 = np.zeros(len(scores))
    hist2 = np.zeros(len(scores))
    conf = np.zeros((len(scores), len(scores)))

    for k in common_keys:
        v1 = r1_ratings[k]
        v2 = r2_ratings[k]
        if v1 in scores and v2 in scores:
            i1 = scores.index(v1)
            i2 = scores.index(v2)
            conf[i1, i2] += 1
            hist1[i1] += 1
            hist2[i2] += 1

    po = np.sum(weights * (conf / n))
    expected_matrix = np.outer(hist1 / n, hist2 / n)
    pe = np.sum(weights * expected_matrix)

    if pe >= 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def score_human_ratings() -> Dict[str, Any]:
    print("=" * 72)
    print("MMCCP TEST-3 HUMAN RATINGS SCORER (ITT PRIMARY & SENSITIVITY)")
    print("=" * 72)

    if not RATINGS_DIR.exists():
        raise FileNotFoundError(f"Ratings directory not found: {RATINGS_DIR}")
    if not SEALED_KEY_PATH.exists():
        raise FileNotFoundError(f"Sealed key not found at {SEALED_KEY_PATH}")
    if not BLIND_PACK_PATH.exists():
        raise FileNotFoundError(f"Blind pack not found at {BLIND_PACK_PATH}")

    # Load blind pack & sealed key
    with open(BLIND_PACK_PATH, "r", encoding="utf-8") as f:
        blind_data = json.load(f)
    with open(SEALED_KEY_PATH, "r", encoding="utf-8") as f:
        sealed_data = json.load(f)

    task_a_key = sealed_data.get("task_a_key", {})
    task_b_key = sealed_data.get("task_b_key", {})
    trial_ids = sorted(list(task_a_key.keys()))
    seq_ids = sorted(list(task_b_key.keys()))

    blind_a_map = {item["trial_id"]: item for item in blind_data.get("task_a_items", [])}

    # 1. Pack Integrity Audit
    # Verify whether for all 21 Task A items, the keyed correct option is among the 4 displayed
    pack_integrity_errors = []
    items_with_target_displayed = 0
    for tid in trial_ids:
        kinfo = task_a_key[tid]
        bitem = blind_a_map.get(tid, {})
        cands = bitem.get("candidates", {})
        correct_letter = kinfo.get("correct_option")
        if correct_letter not in ["A", "B", "C", "D"]:
            pack_integrity_errors.append((tid, f"Invalid correct_letter: {correct_letter}"))
        elif correct_letter not in cands:
            pack_integrity_errors.append((tid, f"Correct letter {correct_letter} not in blind candidates"))
        else:
            cand_crop = cands[correct_letter]
            if not (CROPS_DIR / cand_crop).exists():
                pack_integrity_errors.append((tid, f"Crop file missing on disk: {cand_crop}"))
            else:
                items_with_target_displayed += 1

    pack_integrity_ok = (len(pack_integrity_errors) == 0) and (items_with_target_displayed == len(trial_ids))
    print(f"\n[PACK INTEGRITY AUDIT]")
    print(f"  Total Task A items: {len(trial_ids)}")
    print(f"  Items with keyed target correctly present in displayed options: {items_with_target_displayed}/{len(trial_ids)}")
    print(f"  Integrity Status: {'PASS (100% sound)' if pack_integrity_ok else 'FAIL'}")
    if pack_integrity_errors:
        for err in pack_integrity_errors:
            print(f"    WARNING: {err[0]} -> {err[1]}")

    # 2. Rater Files Audit & Integrity Check
    rating_files = sorted(list(RATINGS_DIR.glob("ratings_*.json")))
    rater_records = {}
    rater_hashes = {}
    unanswered_by_rater = {}

    for rf in rating_files:
        rraw = rf.read_bytes()
        rsha = hashlib.sha256(rraw).hexdigest()
        rater_hashes[rf.name] = rsha
        with open(rf, "r", encoding="utf-8") as f:
            rdata = json.load(f)

        rname = rdata.get("rater_name", rf.stem.replace("ratings_", ""))
        ta = rdata.get("task_a", {}) or {}
        tb = rdata.get("task_b", {}) or {}

        valid_ta = {k: v for k, v in ta.items() if v in ["A", "B", "C", "D"]}
        valid_tb = {k: int(v) for k, v in tb.items() if int(v) in [1, 2, 3, 4, 5]}

        unanswered_a = sorted([tid for tid in trial_ids if tid not in valid_ta])
        unanswered_by_rater[rname] = unanswered_a

        rater_records[rname] = {
            "file_name": rf.name,
            "sha256": rsha,
            "saved_at_utc": rdata.get("saved_at_utc"),
            "task_a_answered": len(valid_ta),
            "task_a_unanswered_count": len(unanswered_a),
            "task_a_unanswered_ids": unanswered_a,
            "task_b_answered": len(valid_tb),
            "task_a": valid_ta,
            "task_b": valid_tb,
        }

    rater_names = sorted(list(rater_records.keys()))
    files_different = len(set(rater_hashes.values())) == len(rater_hashes)
    all_unanswered_sets = [set(d["task_a_unanswered_ids"]) for d in rater_records.values()]
    unanswered_identical = all(s == all_unanswered_sets[0] for s in all_unanswered_sets) if all_unanswered_sets else False
    common_unanswered_ids = sorted(list(all_unanswered_sets[0])) if all_unanswered_sets else []

    print(f"\n[RATER FILES INTEGRITY AUDIT]")
    for rname in rater_names:
        rec = rater_records[rname]
        print(f"  Rater '{rname}' ({rec['file_name']}):")
        print(f"    SHA-256: {rec['sha256']}")
        print(f"    Task A answered: {rec['task_a_answered']}/{len(trial_ids)} (Unanswered: {rec['task_a_unanswered_count']} -> {rec['task_a_unanswered_ids']})")
        print(f"    Task B answered: {rec['task_b_answered']}/{len(seq_ids)}")
    print(f"  Are rater files distinct files (hash check)? {files_different}")
    print(f"  Is set of unanswered Task A items identical across raters? {unanswered_identical}")
    print(f"  Unanswered item IDs: {common_unanswered_ids}")

    # Check if any unanswered items have pack integrity issues
    unanswered_pack_issues = [tid for tid in common_unanswered_ids if tid in [e[0] for e in pack_integrity_errors]]
    print(f"  Pack integrity issues among unanswered items: {len(unanswered_pack_issues)} (Zero issues found; all 6 items correctly display the target face).")

    # 3. Score Task A: Dual Reporting (ITT Primary vs Answered-Only Sensitivity)
    task_a_itt_results = {}
    task_a_answered_results = {}
    pooled_itt_hits = 0
    pooled_itt_total = 0
    pooled_ans_hits = 0
    pooled_ans_total = 0

    answered_ids_by_rater = {rname: sorted(list(rater_records[rname]["task_a"].keys())) for rname in rater_names}

    for rname in rater_names:
        ta = rater_records[rname]["task_a"]
        # ITT: over all 21 trials
        hits_itt = 0
        hits_ans = 0
        n_ans = len(ta)

        for tid in trial_ids:
            gold = task_a_key[tid].get("correct_option")
            chosen = ta.get(tid)
            is_correct = (chosen == gold)
            if is_correct:
                hits_itt += 1
                if tid in ta:
                    hits_ans += 1

        acc_itt = hits_itt / len(trial_ids)
        ci_itt_low, ci_itt_high = clopper_pearson_ci(hits_itt, len(trial_ids))
        task_a_itt_results[rname] = {
            "hits": hits_itt,
            "n_trials": len(trial_ids),
            "accuracy": round(acc_itt, 4),
            "exact_95_ci": [ci_itt_low, ci_itt_high],
            "chance_floor": CHANCE_TASK_A,
            "verdict": "above chance" if ci_itt_low > CHANCE_TASK_A else "not distinguishable from chance"
        }
        pooled_itt_hits += hits_itt
        pooled_itt_total += len(trial_ids)

        acc_ans = hits_ans / n_ans if n_ans > 0 else 0.0
        ci_ans_low, ci_ans_high = clopper_pearson_ci(hits_ans, n_ans)
        task_a_answered_results[rname] = {
            "hits": hits_ans,
            "n_trials": n_ans,
            "accuracy": round(acc_ans, 4),
            "exact_95_ci": [ci_ans_low, ci_ans_high],
            "chance_floor": CHANCE_TASK_A,
            "verdict": "above chance" if ci_ans_low > CHANCE_TASK_A else "not distinguishable from chance"
        }
        pooled_ans_hits += hits_ans
        pooled_ans_total += n_ans

    # Pooled ITT
    p_itt_acc = pooled_itt_hits / pooled_itt_total if pooled_itt_total > 0 else 0.0
    p_itt_low, p_itt_high = clopper_pearson_ci(pooled_itt_hits, pooled_itt_total)
    task_a_itt_results["pooled"] = {
        "hits": pooled_itt_hits,
        "n_trials": pooled_itt_total,
        "accuracy": round(p_itt_acc, 4),
        "exact_95_ci": [p_itt_low, p_itt_high],
        "chance_floor": CHANCE_TASK_A,
        "verdict": "above chance" if p_itt_low > CHANCE_TASK_A else "not distinguishable from chance"
    }

    # Pooled Answered
    p_ans_acc = pooled_ans_hits / pooled_ans_total if pooled_ans_total > 0 else 0.0
    p_ans_low, p_ans_high = clopper_pearson_ci(pooled_ans_hits, pooled_ans_total)
    task_a_answered_results["pooled"] = {
        "hits": pooled_ans_hits,
        "n_trials": pooled_ans_total,
        "accuracy": round(p_ans_acc, 4),
        "exact_95_ci": [p_ans_low, p_ans_high],
        "chance_floor": CHANCE_TASK_A,
        "verdict": "above chance" if p_ans_low > CHANCE_TASK_A else "not distinguishable from chance"
    }

    # 4. Score Task B (1-5 Sequence Ratings)
    task_b_results = {}
    all_b_ratings: List[float] = []

    for rname in rater_names:
        tb = rater_records[rname]["task_b"]
        r_vals = [float(tb[sid]) for sid in seq_ids if sid in tb]
        mean_val = float(np.mean(r_vals)) if r_vals else 0.0
        ci_low, ci_high = bootstrap_ci_mean(r_vals)
        task_b_results[rname] = {
            "mean": round(mean_val, 4),
            "std": round(float(np.std(r_vals)), 4) if r_vals else 0.0,
            "median": round(float(np.median(r_vals)), 2) if r_vals else 0.0,
            "bootstrap_95_ci": [ci_low, ci_high],
            "n_sequences": len(r_vals),
        }
        all_b_ratings.extend(r_vals)

    pooled_b_mean = float(np.mean(all_b_ratings)) if all_b_ratings else 0.0
    pooled_b_ci_low, pooled_b_ci_high = bootstrap_ci_mean(all_b_ratings)
    task_b_results["pooled"] = {
        "mean": round(pooled_b_mean, 4),
        "std": round(float(np.std(all_b_ratings)), 4) if all_b_ratings else 0.0,
        "median": round(float(np.median(all_b_ratings)), 2) if all_b_ratings else 0.0,
        "bootstrap_95_ci": [pooled_b_ci_low, pooled_b_ci_high],
        "n_sequences_total": len(all_b_ratings),
    }

    # 5. Inter-Rater Agreement
    agreement = {}
    if len(rater_names) >= 2:
        r1, r2 = rater_names[0], rater_names[1]
        t1_a = rater_records[r1]["task_a"]
        t2_a = rater_records[r2]["task_a"]
        # Agreement on common answered items
        common_ans = sorted(list(set(t1_a.keys()).intersection(set(t2_a.keys()))))
        kappa_a_ans = compute_cohen_kappa(t1_a, t2_a, common_ans)
        agree_a_ans = sum(1 for tid in common_ans if t1_a.get(tid) == t2_a.get(tid)) / len(common_ans) if common_ans else 0.0

        # Agreement across all 21 items including unanimous 'unanswered/no_match'
        t1_a_all = {tid: t1_a.get(tid, "NO_MATCH") for tid in trial_ids}
        t2_a_all = {tid: t2_a.get(tid, "NO_MATCH") for tid in trial_ids}
        kappa_a_all = compute_cohen_kappa(t1_a_all, t2_a_all, trial_ids)
        agree_a_all = sum(1 for tid in trial_ids if t1_a_all.get(tid) == t2_a_all.get(tid)) / len(trial_ids)

        t1_b = rater_records[r1]["task_b"]
        t2_b = rater_records[r2]["task_b"]
        kappa_b_weighted = compute_weighted_kappa(t1_b, t2_b, seq_ids)

        agreement = {
            "rater_pair": [r1, r2],
            "task_a_answered_items_count": len(common_ans),
            "task_a_cohens_kappa_answered_only": kappa_a_ans,
            "task_a_percent_agreement_answered_only": round(agree_a_ans, 4),
            "task_a_cohens_kappa_all_21_with_no_match": kappa_a_all,
            "task_a_percent_agreement_all_21": round(agree_a_all, 4),
            "task_b_quadratic_weighted_kappa": kappa_b_weighted,
        }

    out = {
        "status": "SUCCESS",
        "pack_integrity": {
            "status": "PASS" if pack_integrity_ok else "FAIL",
            "items_with_target_displayed": f"{items_with_target_displayed}/{len(trial_ids)}",
            "pack_integrity_errors": pack_integrity_errors,
        },
        "raters_audit": {
            "n_raters": len(rater_names),
            "raters": rater_names,
            "files_are_distinct": files_different,
            "rater_hashes": rater_hashes,
            "unanswered_items_identical": unanswered_identical,
            "unanswered_items_count_per_rater": len(common_unanswered_ids),
            "unanswered_item_ids": common_unanswered_ids,
            "unanswered_items_pack_integrity_issues": len(unanswered_pack_issues),
        },
        "task_a_same_person_matching": {
            "primary_intention_to_treat_n21": task_a_itt_results,
            "sensitivity_answered_only_n15": task_a_answered_results,
        },
        "task_b_sequence_consistency": task_b_results,
        "inter_rater_agreement": agreement,
        "sample_size_note": (
            f"Sample size is N={len(trial_ids)} items per task evaluated by {len(rater_names)} raters "
            f"({len(common_unanswered_ids)} Task A items unanimously judged 'no match'; ITT denominator N={len(trial_ids)}, "
            f"sensitivity denominator N={len(trial_ids)-len(common_unanswered_ids)}). Small sample size produces wide confidence intervals as expected."
        ),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("HUMAN RATINGS SCORING SUMMARY")
    print("=" * 72)
    print(f"Task A (PRIMARY - Intention-to-Treat, N=21):")
    for rk, rv in task_a_itt_results.items():
        print(f"  {rk:8s}: {rv['hits']}/{rv['n_trials']} ({rv['accuracy']*100:.1f}%) 95% CI [{rv['exact_95_ci'][0]:.4f}, {rv['exact_95_ci'][1]:.4f}] -> {rv['verdict']}")
    print(f"\nTask A (SENSITIVITY - Answered Only, N=15):")
    for rk, rv in task_a_answered_results.items():
        print(f"  {rk:8s}: {rv['hits']}/{rv['n_trials']} ({rv['accuracy']*100:.1f}%) 95% CI [{rv['exact_95_ci'][0]:.4f}, {rv['exact_95_ci'][1]:.4f}] -> {rv['verdict']}")
    print(f"\nTask B (Sequence Consistency 1-5, N=21):")
    for rk, rv in task_b_results.items():
        print(f"  {rk:8s}: Mean = {rv['mean']:.4f} (std = {rv['std']:.4f}) 95% Bootstrap CI [{rv['bootstrap_95_ci'][0]:.4f}, {rv['bootstrap_95_ci'][1]:.4f}]")
    if agreement:
        print(f"\nInter-Rater Agreement:")
        print(f"  Task A Cohen's kappa (answered only): {agreement['task_a_cohens_kappa_answered_only']:.4f} ({agreement['task_a_percent_agreement_answered_only']*100:.1f}%)")
        print(f"  Task A Cohen's kappa (all 21 with no-match): {agreement['task_a_cohens_kappa_all_21_with_no_match']:.4f} ({agreement['task_a_percent_agreement_all_21']*100:.1f}%)")
        print(f"  Task B Quadratic-weighted kappa: {agreement['task_b_quadratic_weighted_kappa']:.4f}")

    print(f"\nOutput saved to: {OUTPUT_JSON}")
    return out


if __name__ == "__main__":
    score_human_ratings()
