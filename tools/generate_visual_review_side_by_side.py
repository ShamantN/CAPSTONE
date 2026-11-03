"""
Phase R8 Visual Review Contact Sheet Generator: Side-by-Side Comparison (v1 vs v2).
Constructs side-by-side full-resolution comparison sheets for Scenes 1-6 on DEV-2.
"""
import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V1_DIR = PROJECT_ROOT / "results" / "baseline_v1" / "dev2"
V2_DIR = PROJECT_ROOT / "results" / "bundle_dev2" / "renders"
OUT_DIR = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "side_by_side"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load V1 and V2 evaluation reports for annotations
V1_REPORT_PATH = PROJECT_ROOT / "results" / "rescored_v1.2" / "baseline_v1_dev2" / "report.json"
V2_REPORT_PATH = PROJECT_ROOT / "results" / "bundle_dev2" / "report_v1_2.json"

v1_scores = {}
if V1_REPORT_PATH.exists():
    with open(V1_REPORT_PATH, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
        for sc in v1_data.get("per_scene_reports", v1_data.get("scene_reports", [])):
            v1_scores[sc["image_filename"]] = sc

v2_scores = {}
if V2_REPORT_PATH.exists():
    with open(V2_REPORT_PATH, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
        for sc in v2_data.get("per_scene_reports", v2_data.get("scene_reports", [])):
            v2_scores[sc["image_filename"]] = sc


def build_comparison_sheet(scene_num: int, seed: int):
    filename = f"scene_{scene_num}_seed_{seed}.png"
    v1_img_p = V1_DIR / filename
    v2_img_p = V2_DIR / filename

    if not v1_img_p.exists() or not v2_img_p.exists():
        print(f"Skipping {filename}: not found in both directories")
        return None

    img_v1 = cv2.imread(str(v1_img_p))
    img_v2 = cv2.imread(str(v2_img_p))

    # Target resolution for display: 960x640 per image
    target_w, target_h = 960, 640
    img_v1_r = cv2.resize(img_v1, (target_w, target_h))
    img_v2_r = cv2.resize(img_v2, (target_w, target_h))

    # Canvas dimensions
    header_h = 90
    footer_h = 130
    canvas_w = target_w * 2 + 40
    canvas_h = target_h + header_h + footer_h

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (20, 22, 26)  # Dark charcoal slate

    # 1. Header
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (30, 34, 42), -1)
    cv2.line(canvas, (0, header_h), (canvas_w, header_h), (50, 56, 70), 2)
    header_text = f"SCENE {scene_num} (Seed {seed}) - VISUAL AUDIT: BASELINE v1 (Left) vs LOCKED v2 BUNDLE (Right)"
    cv2.putText(canvas, header_text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 245), 2, cv2.LINE_AA)

    # 2. Place Images
    x_v1 = 15
    x_v2 = target_w + 25
    y_img = header_h + 15

    canvas[y_img:y_img+target_h, x_v1:x_v1+target_w] = img_v1_r
    canvas[y_img:y_img+target_h, x_v2:x_v2+target_w] = img_v2_r

    cv2.rectangle(canvas, (x_v1, y_img), (x_v1+target_w, y_img+target_h), (120, 50, 50), 2)
    cv2.rectangle(canvas, (x_v2, y_img), (x_v2+target_w, y_img+target_h), (50, 140, 50), 2)

    # Image Labels
    cv2.putText(canvas, "BASELINE v1 (No Pipeline Fixes)", (x_v1 + 20, y_img + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LOCKED v2 BUNDLE (OpenPose+Couple+IP+Swap+Wardrobe)", (x_v2 + 20, y_img + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 200), 2, cv2.LINE_AA)

    # 3. Footer Metrics
    v1_info = v1_scores.get(filename, {})
    v2_info = v2_scores.get(filename, {})

    y_footer = y_img + target_h + 25
    cv2.line(canvas, (0, y_footer - 10), (canvas_w, y_footer - 10), (45, 50, 62), 1)

    # v1 metrics
    v1_evals = v1_info.get("character_evaluations", [])
    v1_text_lines = []
    for ce in v1_evals:
        cid = ce.get("character_id", "")
        bio = ce.get("biometric_ensemble_score", 0.0)
        sp = "PASS" if ce.get("spatial_pass") else "FAIL"
        wd = "PASS" if ce.get("wardrobe_pass") else "FAIL"
        v1_text_lines.append(f"{cid}: Bio={bio:.4f} | Spatial={sp} | Wardrobe={wd}")
    if not v1_text_lines:
        v1_text_lines.append("No evaluation record found for v1")

    # v2 metrics
    v2_evals = v2_info.get("character_evaluations", [])
    v2_text_lines = []
    for ce in v2_evals:
        cid = ce.get("character_id", "")
        bio = ce.get("biometric_ensemble_score", 0.0)
        sp = "PASS" if ce.get("spatial_pass") else "FAIL"
        wd = "PASS" if ce.get("wardrobe_pass") else "FAIL"
        v2_text_lines.append(f"{cid}: Bio={bio:.4f} | Spatial={sp} | Wardrobe={wd}")
    if not v2_text_lines:
        v2_text_lines.append("No evaluation record found for v2")

    for i, line in enumerate(v1_text_lines):
        cv2.putText(canvas, line, (x_v1 + 10, y_footer + 25 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 210), 1, cv2.LINE_AA)

    for i, line in enumerate(v2_text_lines):
        cv2.putText(canvas, line, (x_v2 + 10, y_footer + 25 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 240, 220), 1, cv2.LINE_AA)

    out_path = OUT_DIR / f"comparison_scene_{scene_num}_seed_{seed}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Generated comparison sheet: {out_path}")
    return out_path


def main():
    print("=" * 80)
    print("  PHASE R8: GENERATING SIDE-BY-SIDE CONTACT SHEETS FOR DEV-2")
    print("=" * 80)

    generated = []
    # Generate for all 6 scenes, seed 101 (canonical DEV-2)
    for s_num in range(1, 7):
        for seed in [101, 202]:
            p = build_comparison_sheet(s_num, seed)
            if p:
                generated.append(p)

    print(f"\nGenerated {len(generated)} side-by-side comparison sheets in {OUT_DIR}")


if __name__ == "__main__":
    main()

