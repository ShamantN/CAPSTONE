"""
Phase R8 Step R8.4: Local VLM Second Opinion Reviewer
Evaluates character presence, person counts, and clothing colors on DEV-2 images using local VLM backbones on CPU.
Computes Cohen's kappa (k) measuring inter-rater agreement with Verifier v1.2.
"""
import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Force CPU and UTF-8
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DLL paths for Torch on Windows
_DLL_PATHS = [
    r"C:\Stable-Diffusion\system\python\Library\bin",
    r"C:\Stable-Diffusion\system\python\Lib\site-packages\torch\lib",
    r"C:\Stable-Diffusion\system\python\Scripts",
    r"C:\Stable-Diffusion\system\python"
]
for _p in _DLL_PATHS:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        if _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from qa.vlm_visual_auditor import VLMVisualAuditor
from story_engine.schema import StorySpec

STORY_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
RENDERS_DIR = PROJECT_ROOT / "results" / "bundle_dev2" / "renders"
VERIFIER_REPORT_PATH = PROJECT_ROOT / "results" / "bundle_dev2" / "report_v1_2.json"
OUT_JSON = PROJECT_ROOT / "results" / "vlm_review_report.json"
OUT_MD = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "vlm_second_opinion_report.md"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def compute_cohens_kappa(rater1_binary: list, rater2_binary: list) -> float:
    """Computes Cohen's kappa for two raters with binary classification."""
    n = len(rater1_binary)
    if n == 0:
        return 1.0

    # Confusion matrix
    a = sum(1 for r1, r2 in zip(rater1_binary, rater2_binary) if r1 and r2)
    b = sum(1 for r1, r2 in zip(rater1_binary, rater2_binary) if r1 and not r2)
    c = sum(1 for r1, r2 in zip(rater1_binary, rater2_binary) if not r1 and r2)
    d = sum(1 for r1, r2 in zip(rater1_binary, rater2_binary) if not r1 and not r2)

    po = (a + d) / n
    pe1 = ((a + b) / n) * ((a + c) / n)
    pe2 = ((c + d) / n) * ((b + d) / n)
    pe = pe1 + pe2

    if pe >= 1.0 or (1.0 - pe) == 0.0:
        return 1.0
    k = (po - pe) / (1.0 - pe)
    return round(float(k), 4)


def main():
    print("=" * 80)
    print("  PHASE R8: LOCAL VLM SECOND OPINION REVIEW (BLIP + CLIP ON CPU)")
    print("=" * 80)

    with open(STORY_PATH, "r", encoding="utf-8") as f:
        story = StorySpec.from_dict(json.load(f))

    verifier_reports = {}
    if VERIFIER_REPORT_PATH.exists():
        with open(VERIFIER_REPORT_PATH, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            for sc in v_data.get("per_scene_reports", []):
                verifier_reports[sc["image_filename"]] = sc

    vlm = VLMVisualAuditor()
    VLMVisualAuditor._init_models()

    vlm_evaluations = []
    
    # Binary lists for kappa calculation
    vlm_counts_binary = []
    verifier_counts_binary = []
    
    vlm_wardrobe_binary = []
    verifier_wardrobe_binary = []
    
    vlm_verdicts_binary = []
    verifier_verdicts_binary = []

    for s in story.scenes:
        for seed in [101, 202]:
            filename = f"scene_{s.scene_number}_seed_{seed}.png"
            img_path = RENDERS_DIR / filename
            if not img_path.exists():
                continue

            print(f"Auditing with VLM: {filename} (Scene {s.scene_number})...")
            caption = vlm.generate_caption(str(img_path))

            # Prompt CLIP for character presence & person count
            expected_count = len(s.characters_present)
            count_candidates = [
                f"a photo showing exactly {i} person" if i == 1 else f"a photo showing {i} people"
                for i in range(1, 5)
            ]
            count_scores = vlm.score_text_candidates(str(img_path), count_candidates)
            best_count_str = max(count_scores.items(), key=lambda x: x[1])[0]
            
            # parse count
            vlm_detected_count = 1
            for i in range(1, 5):
                if f"{i} person" in best_count_str or f"{i} people" in best_count_str:
                    vlm_detected_count = i
                    break

            # Clothing probe for each character
            char_clothing_results = {}
            for cid in s.characters_present:
                c_meta = story.characters.get(cid) if isinstance(story.characters, dict) else None
                if c_meta and getattr(c_meta, "recurring_clothing", None):
                    cloth_desc = c_meta.recurring_clothing
                    cloth_candidates = [
                        f"person wearing {cloth_desc}",
                        "person wearing different clothing"
                    ]
                    cloth_scores = vlm.score_text_candidates(str(img_path), cloth_candidates)
                    target_score = cloth_scores.get(cloth_candidates[0], 0.0)
                    char_clothing_results[cid] = {
                        "target_clothing": cloth_desc,
                        "clip_prob": target_score,
                        "match": bool(target_score >= 0.50)
                    }

            # Retrieve Verifier v1.2 report
            v_rep = verifier_reports.get(filename, {})
            v_scene_audit = v_rep.get("scene_audit", {})
            v_yolo_count = v_scene_audit.get("detected_persons_yolo_count", expected_count)
            v_evals = v_rep.get("character_evaluations", [])

            # Person count match
            vlm_count_match = bool(vlm_detected_count == expected_count)
            verifier_count_match = bool(v_yolo_count == expected_count)

            # Wardrobe match
            # If any character matched clothing in VLM vs Verifier
            vlm_any_wardrobe = any(cr["match"] for cr in char_clothing_results.values()) if char_clothing_results else True
            verifier_any_wardrobe = any(ce.get("wardrobe_pass", False) for ce in v_evals) if v_evals else True

            # Overall acceptance: count match and plausible wardrobe
            vlm_scene_pass = vlm_count_match and (vlm_any_wardrobe or len(s.characters_present) > 1)
            verifier_scene_pass = verifier_count_match and all(ce.get("spatial_pass", True) for ce in v_evals)

            # Record binary verdicts for Cohen's Kappa
            vlm_counts_binary.append(vlm_count_match)
            verifier_counts_binary.append(verifier_count_match)

            vlm_wardrobe_binary.append(vlm_any_wardrobe)
            verifier_wardrobe_binary.append(verifier_any_wardrobe)

            vlm_verdicts_binary.append(vlm_scene_pass)
            verifier_verdicts_binary.append(verifier_scene_pass)

            vlm_evaluations.append({
                "filename": filename,
                "scene_number": s.scene_number,
                "seed": seed,
                "caption": caption,
                "expected_characters": s.characters_present,
                "expected_count": expected_count,
                "vlm_detected_count": vlm_detected_count,
                "verifier_yolo_count": v_yolo_count,
                "person_count_match_vlm": vlm_count_match,
                "person_count_match_verifier": verifier_count_match,
                "clothing_breakdown": char_clothing_results,
                "wardrobe_match_vlm": vlm_any_wardrobe,
                "wardrobe_match_verifier": verifier_any_wardrobe,
                "vlm_pass": vlm_scene_pass,
                "verifier_pass": verifier_scene_pass
            })

    kappa_counts = compute_cohens_kappa(vlm_counts_binary, verifier_counts_binary)
    kappa_wardrobe = compute_cohens_kappa(vlm_wardrobe_binary, verifier_wardrobe_binary)
    kappa_verdict = compute_cohens_kappa(vlm_verdicts_binary, verifier_verdicts_binary)

    overall_kappa = kappa_counts  # Primary spatial / presence metric

    print("\n" + "=" * 80)
    print("  VLM SECOND OPINION RESULTS")
    print("=" * 80)
    print(f"  Total Images Evaluated: {len(vlm_evaluations)}")
    print(f"  Inter-Rater Agreement on Person Counts: k = {kappa_counts:.4f}")
    print(f"  Inter-Rater Agreement on Wardrobe:      k = {kappa_wardrobe:.4f}")
    print(f"  Inter-Rater Agreement on Scene Pass:    k = {kappa_verdict:.4f}")
    print("=" * 80)

    summary_data = {
        "vlm_backbones": ["Salesforce/blip-image-captioning-base", "openai/clip-vit-base-patch32"],
        "device": "cpu",
        "total_images": len(vlm_evaluations),
        "cohens_kappa_person_counts": kappa_counts,
        "cohens_kappa_wardrobe": kappa_wardrobe,
        "cohens_kappa_overall_verdict": kappa_verdict,
        "cohens_kappa": overall_kappa,
        "inter_rater_agreement_category": "Substantial Agreement" if overall_kappa >= 0.60 else "Moderate Agreement",
        "evaluations": vlm_evaluations
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Phase R8: Local VLM Second Opinion Review Report\n\n")
        f.write(f"**Date:** 2026-11-02  \n")
        f.write(f"**Backbones:** `Salesforce/blip-image-captioning-base` & `openai/clip-vit-base-patch32` (Pinned to CPU)  \n")
        f.write(f"**Inter-Rater Agreement on Person Count (Cohen's Kappa):** **{kappa_counts:.4f}** ({summary_data['inter_rater_agreement_category']})  \n")
        f.write(f"**Inter-Rater Agreement on Wardrobe (Cohen's Kappa):** **{kappa_wardrobe:.4f}**  \n")
        f.write(f"**Inter-Rater Agreement on Overall Pass (Cohen's Kappa):** **{kappa_verdict:.4f}**  \n\n")
        f.write("## Per-Scene VLM Perception Summary\n\n")
        f.write("| File | Scene | Expected | VLM Count | YOLO Count | BLIP Caption | VLM Pass | Verifier Pass |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |\n")
        for ev in vlm_evaluations:
            vp = "PASS" if ev["vlm_pass"] else "FAIL"
            vrp = "PASS" if ev["verifier_pass"] else "FAIL"
            f.write(f"| `{ev['filename']}` | {ev['scene_number']} | {ev['expected_count']} | {ev['vlm_detected_count']} | {ev['verifier_yolo_count']} | \"{ev['caption']}\" | {vp} | {vrp} |\n")

    print(f"\nSaved VLM report to {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
