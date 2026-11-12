#!/usr/bin/env python3
"""Score human blind A/B preference and 5-character Re-ID predictions.

Reads human predictions from ``docs_generated/v2/visual_review/blind_predictions_human.json``
and scores them against the sealed keys in ``scratch/sealed_keys/human_test2_ebright/``.

Refuses to score if any prediction is blank/incomplete (exits with code 2 and
``PENDING_HUMAN_INPUT``), or if any entry contains an invalid value (exits with
code 1). Uses exact ``scipy.stats.binomtest`` Clopper-Pearson 95% CIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_predictions_human.json"
)
DEFAULT_SEALED_DIR = PROJECT_ROOT / "scratch" / "sealed_keys" / "human_test2_ebright"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results" / "human_blind_study_report.json"
DEFAULT_OUT_MD = (
    PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "HUMAN_BLIND_STUDY_REPORT.md"
)

VALID_AB_OPTIONS = {"A", "B", "TIE"}
VALID_CHARACTER_IDS = (
    "young_ebright",
    "teen_ebright",
    "college_ebright",
    "mother",
    "james_wong",
)
REID_CHANCE_BASELINE = 1.0 / len(VALID_CHARACTER_IDS)  # 0.20 (5 classes)
AB_CHANCE_BASELINE = 0.50


class PendingHumanInputError(ValueError):
    """Raised when one or more human prediction fields remain unfilled."""


class PredictionValidationError(ValueError):
    """Raised when prediction schema or values are invalid."""


def _normalize_ab_choice(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    upper = text.upper()
    return "tie" if upper == "TIE" else upper


def _normalize_char_id(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip().lower()


def validate_predictions(
    pred_data: Dict[str, Any],
    expected_ab_ids: Optional[Iterable[str]] = None,
    expected_reid_ids: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Validate structure, completeness, and allowed values of human predictions."""
    if not isinstance(pred_data, dict):
        raise PredictionValidationError("Predictions root must be a JSON object.")

    ab_preds = pred_data.get("ab_preference")
    reid_preds = pred_data.get("blind_reid")
    if not isinstance(ab_preds, dict) or not ab_preds:
        raise PredictionValidationError("Missing or empty 'ab_preference' object.")
    if not isinstance(reid_preds, dict) or not reid_preds:
        raise PredictionValidationError("Missing or empty 'blind_reid' object.")

    blank_ab: List[str] = []
    invalid_ab: List[str] = []
    for pair_id in sorted(ab_preds.keys()):
        entry = ab_preds[pair_id]
        if not isinstance(entry, dict):
            invalid_ab.append(f"{pair_id} (not an object)")
            continue
        choice = _normalize_ab_choice(entry.get("selected_option", ""))
        if not choice:
            blank_ab.append(pair_id)
        elif choice.upper() not in VALID_AB_OPTIONS:
            invalid_ab.append(f"{pair_id}={choice!r}")

    blank_reid: List[str] = []
    invalid_reid: List[str] = []
    valid_char_set = set(VALID_CHARACTER_IDS)
    for crop_id in sorted(reid_preds.keys()):
        entry = reid_preds[crop_id]
        if not isinstance(entry, dict):
            invalid_reid.append(f"{crop_id} (not an object)")
            continue
        char_id = _normalize_char_id(entry.get("predicted_character_id", ""))
        if not char_id:
            blank_reid.append(crop_id)
        elif char_id not in valid_char_set:
            invalid_reid.append(f"{crop_id}={char_id!r}")

    if invalid_ab or invalid_reid:
        parts = []
        if invalid_ab:
            parts.append(f"invalid ab_preference values: {', '.join(invalid_ab)}")
        if invalid_reid:
            parts.append(f"invalid blind_reid values: {', '.join(invalid_reid)}")
        raise PredictionValidationError("; ".join(parts))

    if blank_ab or blank_reid:
        raise PendingHumanInputError(
            "PENDING_HUMAN_INPUT: Human blind evaluation is incomplete "
            f"({len(blank_ab)}/{len(ab_preds)} A/B pairs blank, "
            f"{len(blank_reid)}/{len(reid_preds)} Re-ID crops blank). "
            "Fill docs_generated/v2/visual_review/blind_predictions_human.json before scoring."
        )

    if expected_ab_ids is not None:
        exp_ab = set(expected_ab_ids)
        got_ab = set(ab_preds.keys())
        if exp_ab != got_ab:
            raise PredictionValidationError(
                f"A/B key mismatch: missing={sorted(exp_ab - got_ab)}, "
                f"unexpected={sorted(got_ab - exp_ab)}"
            )

    if expected_reid_ids is not None:
        exp_reid = set(expected_reid_ids)
        got_reid = set(reid_preds.keys())
        if exp_reid != got_reid:
            raise PredictionValidationError(
                f"Re-ID key mismatch: missing={sorted(exp_reid - got_reid)}, "
                f"unexpected={sorted(got_reid - exp_reid)}"
            )

    return ab_preds, reid_preds


def _resolve_v2_choice(key_entry: Dict[str, Any]) -> str:
    if key_entry.get("v2_choice") in {"A", "B"}:
        return str(key_entry["v2_choice"])
    if key_entry.get("option_a") == "v2":
        return "A"
    if key_entry.get("option_b") == "v2":
        return "B"
    raise PredictionValidationError(f"Cannot resolve v2_choice from sealed A/B key entry: {key_entry}")


def score_human_blind_test(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    sealed_dir: Path = DEFAULT_SEALED_DIR,
    out_json: Optional[Path] = DEFAULT_OUT_JSON,
    out_md: Optional[Path] = DEFAULT_OUT_MD,
) -> Dict[str, Any]:
    """Validate predictions first, unblind with sealed keys, compute exact CIs, and optionally write reports."""
    pred_data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    # Validate completeness BEFORE reading any sealed key files.
    validate_predictions(pred_data)

    sealed_ab_path = Path(sealed_dir) / "sealed_ab_key.json"
    sealed_reid_path = Path(sealed_dir) / "sealed_reid_key.json"
    sealed_ab = json.loads(sealed_ab_path.read_text(encoding="utf-8"))
    sealed_reid = json.loads(sealed_reid_path.read_text(encoding="utf-8"))

    ab_preds, reid_preds = validate_predictions(
        pred_data,
        expected_ab_ids=sealed_ab.keys(),
        expected_reid_ids=sealed_reid.keys(),
    )

    ab_details: List[Dict[str, Any]] = []
    v2_wins = 0
    v1_wins = 0
    ties = 0

    for pair_id in sorted(sealed_ab.keys()):
        k = sealed_ab[pair_id]
        v2_opt = _resolve_v2_choice(k)
        pred_choice = _normalize_ab_choice(ab_preds[pair_id]["selected_option"])
        if pred_choice == "tie":
            outcome = "tie"
            ties += 1
            v2_won: Optional[bool] = None
        elif pred_choice == v2_opt:
            outcome = "v2_win"
            v2_wins += 1
            v2_won = True
        else:
            outcome = "v1_win"
            v1_wins += 1
            v2_won = False

        ab_details.append(
            {
                "pair_id": pair_id,
                "scene": k.get("scene_number"),
                "seed": k.get("seed"),
                "v2_option": v2_opt,
                "selected_option": pred_choice,
                "outcome": outcome,
                "v2_won": v2_won,
                "notes": str(ab_preds[pair_id].get("notes", "") or ""),
            }
        )

    total_ab = len(sealed_ab)
    decisive_ab = v2_wins + v1_wins
    if decisive_ab > 0:
        ab_win_rate_decisive = v2_wins / decisive_ab
        ab_test = binomtest(v2_wins, decisive_ab, p=AB_CHANCE_BASELINE)
        ab_ci = ab_test.proportion_ci(confidence_level=0.95, method="exact")
        ab_ci_low, ab_ci_high = float(ab_ci.low), float(ab_ci.high)
        ab_pvalue = float(ab_test.pvalue)
    else:
        ab_win_rate_decisive = 0.0
        ab_ci_low, ab_ci_high = 0.0, 1.0
        ab_pvalue = 1.0

    reid_details: List[Dict[str, Any]] = []
    reid_correct = 0
    per_char: Dict[str, Dict[str, Any]] = {
        cid: {"total": 0, "correct": 0, "accuracy": 0.0} for cid in VALID_CHARACTER_IDS
    }

    for crop_id in sorted(sealed_reid.keys()):
        k = sealed_reid[crop_id]
        true_id = _normalize_char_id(k.get("true_character_id", ""))
        pred_id = _normalize_char_id(reid_preds[crop_id]["predicted_character_id"])
        is_correct = pred_id == true_id
        if is_correct:
            reid_correct += 1
        if true_id not in per_char:
            per_char[true_id] = {"total": 0, "correct": 0, "accuracy": 0.0}
        per_char[true_id]["total"] += 1
        if is_correct:
            per_char[true_id]["correct"] += 1

        reid_details.append(
            {
                "crop_id": crop_id,
                "source": k.get("source", ""),
                "true_character_id": true_id,
                "predicted_character_id": pred_id,
                "is_correct": is_correct,
            }
        )

    for cid, stats in per_char.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else None

    total_reid = len(sealed_reid)
    reid_acc = reid_correct / total_reid if total_reid > 0 else 0.0
    reid_test = binomtest(reid_correct, total_reid, p=REID_CHANCE_BASELINE)
    reid_ci = reid_test.proportion_ci(confidence_level=0.95, method="exact")
    reid_ci_low, reid_ci_high = float(reid_ci.low), float(reid_ci.high)

    report_data: Dict[str, Any] = {
        "evaluation_type": str(pred_data.get("evaluation_type", "human_blind_test2_ebright")),
        "evaluator_name": str(pred_data.get("evaluator_name", "") or ""),
        "date": str(pred_data.get("date", "") or ""),
        "blind_ab_study": {
            "total_pairs": total_ab,
            "decisive_pairs": decisive_ab,
            "v2_wins": v2_wins,
            "v1_wins": v1_wins,
            "ties": ties,
            "v2_win_rate_decisive": round(ab_win_rate_decisive, 4),
            "v2_win_rate_all_pairs": round(v2_wins / total_ab, 4) if total_ab > 0 else 0.0,
            "chance_baseline": AB_CHANCE_BASELINE,
            "clopper_pearson_95_ci_decisive": [round(ab_ci_low, 4), round(ab_ci_high, 4)],
            "binomial_p_value_vs_chance": round(ab_pvalue, 6),
            "pairs": ab_details,
        },
        "blind_reid_study": {
            "total_crops": total_reid,
            "correct_reid": reid_correct,
            "reid_accuracy": round(reid_acc, 4),
            "chance_baseline": REID_CHANCE_BASELINE,
            "num_classes": len(VALID_CHARACTER_IDS),
            "clopper_pearson_95_ci": [round(reid_ci_low, 4), round(reid_ci_high, 4)],
            "binomial_p_value_vs_chance": round(float(reid_test.pvalue), 6),
            "per_character": per_char,
            "crops": reid_details,
        },
    }

    if out_json is not None:
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")

    if out_md is not None:
        out_md_path = Path(out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# Human Blind Evaluation Report (TEST-2 Ebright)",
            "",
            f"- **Evaluator:** `{report_data['evaluator_name'] or 'unspecified'}`",
            f"- **Date:** `{report_data['date'] or 'unspecified'}`",
            "",
            "## Summary",
            "",
            "| Study | N | Outcome | Rate | Clopper-Pearson 95% CI | Chance | Two-sided Binomial p |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            (
                f"| Blind A/B (decisive non-tie) | {decisive_ab} ({total_ab} total, {ties} ties) | "
                f"{v2_wins} v2 / {v1_wins} v1 | {ab_win_rate_decisive*100:.1f}% | "
                f"[{ab_ci_low*100:.1f}%, {ab_ci_high*100:.1f}%] | 50.0% | {ab_pvalue:.6f} |"
            ),
            (
                f"| Blind Re-ID (5 classes) | {total_reid} | {reid_correct} correct | "
                f"{reid_acc*100:.1f}% | [{reid_ci_low*100:.1f}%, {reid_ci_high*100:.1f}%] | "
                f"20.0% | {float(reid_test.pvalue):.6f} |"
            ),
            "",
            "## Per-Character Re-ID Breakdown",
            "",
            "| Character ID | Correct / Total | Accuracy |",
            "| :--- | :--- | :--- |",
        ]
        for cid, stats in per_char.items():
            acc_str = f"{stats['accuracy']*100:.1f}%" if stats["accuracy"] is not None else "n/a"
            md_lines.append(f"| `{cid}` | {stats['correct']}/{stats['total']} | {acc_str} |")
        out_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report_data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score filled human blind-test predictions against sealed keys.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate predictions schema/completeness without reading sealed keys or writing output.",
    )
    args = parser.parse_args(argv)

    try:
        if args.check_only:
            pred_data = json.loads(args.predictions.read_text(encoding="utf-8"))
            ab_preds, reid_preds = validate_predictions(pred_data)
            print(f"[OK] Validated {len(ab_preds)} A/B predictions and {len(reid_preds)} Re-ID predictions.")
            return 0
        report = score_human_blind_test(
            predictions_path=args.predictions,
            sealed_dir=args.sealed_dir,
            out_json=args.out_json,
            out_md=args.out_md,
        )
        ab = report["blind_ab_study"]
        reid = report["blind_reid_study"]
        print(
            f"[AB] v2 wins: {ab['v2_wins']}/{ab['decisive_pairs']} decisive "
            f"({ab['ties']} ties; total={ab['total_pairs']}), "
            f"rate={ab['v2_win_rate_decisive']*100:.1f}%, "
            f"95% CI={ab['clopper_pearson_95_ci_decisive']}, p={ab['binomial_p_value_vs_chance']}"
        )
        print(
            f"[RE-ID] correct: {reid['correct_reid']}/{reid['total_crops']} "
            f"({reid['reid_accuracy']*100:.1f}% vs 20.0% chance), "
            f"95% CI={reid['clopper_pearson_95_ci']}, p={reid['binomial_p_value_vs_chance']}"
        )
        return 0
    except PendingHumanInputError as exc:
        print(f"[PENDING] {exc}", file=sys.stderr)
        return 2
    except (PredictionValidationError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
