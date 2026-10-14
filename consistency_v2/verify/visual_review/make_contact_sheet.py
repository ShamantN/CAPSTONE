"""
Contact Sheet Generator (VR-1)
Builds unified visual contact sheets showing full render, bank references, detected face/body crops,
and overlay metrics for visual auditing.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text_with_bg(img_np: np.ndarray, text: str, pos: tuple, font_scale: float = 0.5, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img_np, (x, y - th - 4), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.putText(img_np, text, (x + 2, y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)


def make_contact_sheet(
    scene_img_bgr: np.ndarray,
    scene_meta: Dict[str, Any],
    character_evals: List[Dict[str, Any]],
    output_path: Path
) -> np.ndarray:
    """
    Constructs a 1600x1200 contact sheet combining:
    - Header banner with scene title, seed, and overall verdict
    - Main scene canvas with localized bounding boxes
    - Character comparison cards (Bank Reference vs Scene Face/Body Crops)
    """
    sheet_w = 1600
    sheet_h = 1350
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    canvas[:] = (24, 26, 30)  # Dark slate background

    # 1. Header Banner
    s_num = scene_meta.get("scene_number", 1)
    s_title = scene_meta.get("title", "Storyboard Scene")
    s_seed = scene_meta.get("seed", -1)
    overall_pass = all(c.get("overall_pass", False) for c in character_evals) if character_evals else False

    banner_color = (34, 139, 34) if overall_pass else (30, 30, 180)  # Green or Red in BGR
    cv2.rectangle(canvas, (0, 0), (sheet_w, 70), (35, 38, 44), -1)
    cv2.line(canvas, (0, 70), (sheet_w, 70), (55, 60, 70), 2)

    title_text = f"SCENE {s_num}: {s_title} | Seed: {s_seed} | MMCCP v2 Verifier"
    cv2.putText(canvas, title_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 245), 2, cv2.LINE_AA)

    status_tag = "VERIFIED PASS" if overall_pass else "DEFECT / FAIL"
    draw_text_with_bg(canvas, f"[{status_tag}]", (sheet_w - 220, 48), font_scale=0.8, text_color=(255, 255, 255), bg_color=banner_color)

    # 2. Main Scene Render (Scale to approx 900x600 centered on left)
    sh, sw = scene_img_bgr.shape[:2]
    target_scene_w = 960
    target_scene_h = int(target_scene_w * (sh / sw))
    scene_resized = cv2.resize(scene_img_bgr.copy(), (target_scene_w, target_scene_h))

    # Draw boxes on scene copy
    for ceval in character_evals:
        bbox = ceval.get("face_bbox")
        if bbox:
            scale_x = target_scene_w / sw
            scale_y = target_scene_h / sh
            bx = int(bbox["x"] * scale_x)
            by = int(bbox["y"] * scale_y)
            bw = int(bbox["w"] * scale_x)
            bh = int(bbox["h"] * scale_y)
            box_col = (0, 220, 0) if ceval.get("overall_pass", False) else (0, 0, 240)
            cv2.rectangle(scene_resized, (bx, by), (bx + bw, by + bh), box_col, 2)
            cv2.putText(scene_resized, ceval.get("character_id", ""), (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1, cv2.LINE_AA)

    canvas[90:90 + target_scene_h, 30:30 + target_scene_w] = scene_resized

    # 3. Sidebar: Scene & Defect Diagnostics
    diag_x = 1020
    diag_y = 100
    cv2.putText(canvas, "Scene Verification Diagnostics", (diag_x, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 230), 2, cv2.LINE_AA)
    cv2.line(canvas, (diag_x, diag_y + 10), (sheet_w - 30, diag_y + 10), (70, 75, 85), 1)

    info_lines = [
        f"Characters Evaluated: {len(character_evals)}",
        f"Prompt: {scene_meta.get('positive_prompt', '')[:55]}...",
        f"Sampler: DPM++ 2M Karras (Pass1: 25, Pass2: 15)",
        f"VRAM Peak: {scene_meta.get('vram_peak_mb', 0.0)} MB",
        f"Render Latency: {scene_meta.get('runtime_seconds', 0.0)}s"
    ]
    cur_y = diag_y + 35
    for il in info_lines:
        cv2.putText(canvas, il, (diag_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 175, 185), 1, cv2.LINE_AA)
        cur_y += 24

    # Defect checklist summary
    defects = scene_meta.get("defects", [])
    cur_y += 10
    cv2.putText(canvas, f"Taxonomy Defects Flagged: {len(defects)}", (diag_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 100, 100) if defects else (100, 220, 100), 1, cv2.LINE_AA)
    cur_y += 22
    if not defects:
        cv2.putText(canvas, "  - Zero visual hallucinations detected", (diag_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 200, 140), 1, cv2.LINE_AA)
        cur_y += 20
    else:
        for d in defects[:4]:
            cv2.putText(canvas, f"  ! {d[:45]}", (diag_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 80, 80), 1, cv2.LINE_AA)
            cur_y += 18

    # 4. Bottom Cards: Character Reference vs Crops Comparison
    card_y = 90 + target_scene_h + 30
    cv2.line(canvas, (20, card_y - 15), (sheet_w - 20, card_y - 15), (55, 60, 70), 2)
    cv2.putText(canvas, "Character Biometric & Wardrobe Binding Evidence", (30, card_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 245), 2, cv2.LINE_AA)

    card_start_y = card_y + 20
    card_w = (sheet_w - 60) // max(1, len(character_evals))

    for idx, ceval in enumerate(character_evals):
        cx = 30 + (idx * card_w)
        cy = card_start_y
        cid = ceval.get("character_id", f"char_{idx}")
        c_pass = ceval.get("overall_pass", False)

        # Draw card container
        cv2.rectangle(canvas, (cx, cy), (cx + card_w - 20, sheet_h - 25), (35, 38, 44), -1)
        border_col = (50, 180, 50) if c_pass else (50, 50, 200)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w - 20, sheet_h - 25), border_col, 2)

        # Card header
        cv2.putText(canvas, f"Character: {cid}", (cx + 12, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 240), 2, cv2.LINE_AA)

        # Images: Reference Anchor on Left, Detected Face on Right
        img_size = 130
        ref_bgr = ceval.get("reference_crop_bgr")
        face_bgr = ceval.get("detected_face_crop_bgr")

        if ref_bgr is not None and ref_bgr.size > 0:
            ref_sq = cv2.resize(ref_bgr, (img_size, img_size))
            canvas[cy + 40:cy + 40 + img_size, cx + 12:cx + 12 + img_size] = ref_sq
            cv2.putText(canvas, "Bank Ref", (cx + 12, cy + 40 + img_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 165, 175), 1, cv2.LINE_AA)

        if face_bgr is not None and face_bgr.size > 0:
            face_sq = cv2.resize(face_bgr, (img_size, img_size))
            canvas[cy + 40:cy + 40 + img_size, cx + 24 + img_size:cx + 24 + (2 * img_size)] = face_sq
            cv2.putText(canvas, "Scene Face", (cx + 24 + img_size, cy + 40 + img_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 165, 175), 1, cv2.LINE_AA)

        # Quantitative Metrics List
        metrics_y = cy + 40 + img_size + 40
        m_lines = [
            f"Ensemble Biometric: {ceval.get('biometric_ensemble_score', 0.0):.4f} ({'PASS' if ceval.get('biometric_pass', False) else 'FAIL'})",
            f"Rank-1 Discrimination: {'YES' if ceval.get('rank1_correct', False) else 'NO'} (Margin: {ceval.get('discrimination_margin', 0.0):+.3f})",
            f"Spatial Position: {ceval.get('observed_position', '')} ({'PASS' if ceval.get('spatial_pass', False) else 'FAIL'})",
            f"Wardrobe Delta E: {ceval.get('wardrobe_delta_e', 0.0):.1f} ({'PASS' if ceval.get('wardrobe_pass', False) else 'FAIL'})",
            f"Face Sharpness: {ceval.get('face_sharpness', 0.0):.1f} (Width: {ceval.get('face_width_px', 0)}px)"
        ]
        for ml in m_lines:
            cv2.putText(canvas, ml, (cx + 12, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
            metrics_y += 20

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return canvas
