"""
PyTorch Vision Helper for MMCCP v2.1 Verification Suite
Executes YOLOv8n, SegFormer, DINOv2, and CLIP in Forge's isolated Python environment on CPU.
Communicates via JSON config files.
Includes full detector health tracking and strictly eliminates central-torso fallbacks.
"""
import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image

# Force CPU execution to prevent GPU VRAM contention
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
    CLIPProcessor,
    CLIPModel
)
from ultralytics import YOLO

# Global model caches
_DINO_PROC = None
_DINO_MODEL = None
_SEG_PROC = None
_SEG_MODEL = None
_CLIP_PROC = None
_CLIP_MODEL = None
_YOLO_MODEL = None


def get_dinov2():
    global _DINO_PROC, _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL.eval()
    return _DINO_PROC, _DINO_MODEL


def get_segformer():
    global _SEG_PROC, _SEG_MODEL
    if _SEG_MODEL is None:
        _SEG_PROC = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _SEG_MODEL = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _SEG_MODEL.eval()
    return _SEG_PROC, _SEG_MODEL


def get_clip():
    global _CLIP_PROC, _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL.eval()
    return _CLIP_PROC, _CLIP_MODEL


def get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL


def process_image(
    image_path: Path,
    face_bboxes: List[Dict[str, int]],
    reference_image_paths: List[Path],
    required_props: List[str],
    forbidden_props: List[str]
) -> Dict[str, Any]:
    detector_health = {
        "yolov8n": False,
        "segformer": False,
        "clip": False,
        "dinov2": False
    }

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "error": f"Failed to load image at {image_path}",
            "detector_health": detector_health
        }

    h_img, w_img = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(img_rgb)

    # 1. YOLOv8 Person Count
    persons = []
    try:
        yolo = get_yolo()
        y_res = yolo(img_bgr, device="cpu", verbose=False)
        if y_res and len(y_res) > 0 and y_res[0].boxes:
            for b in y_res[0].boxes:
                if int(b.cls[0]) == 0 and float(b.conf[0]) >= 0.40:
                    xyxy = b.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    persons.append({
                        "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                        "confidence": round(float(b.conf[0]), 4)
                    })
        detector_health["yolov8n"] = True
    except Exception as e:
        print(f"[ERROR] YOLOv8 execution failed: {e}", file=sys.stderr)

    # 2. CLIP Solarization, Inversion Probe, and Props Check
    inv_prob = 0.0
    req_missing = []
    forb_present = []
    try:
        clip_proc, clip_model = get_clip()
        inv_prompts = [
            "a standard realistic color photograph",
            "a color-inverted photographic negative with glowing eyes, cyan skin, and inverted lighting"
        ]
        inputs = clip_proc(text=inv_prompts, images=pil_full, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = clip_model(**inputs)
        inv_prob = float(out.logits_per_image.softmax(dim=1).cpu().numpy()[0][1])

        for p in required_props:
            p_prompts = [f"a scene showing {p}", f"a scene without {p}"]
            c_in = clip_proc(text=p_prompts, images=pil_full, return_tensors="pt", padding=True)
            with torch.no_grad():
                c_out = clip_model(**c_in)
            prob = float(c_out.logits_per_image.softmax(dim=1).cpu().numpy()[0][0])
            if prob < 0.35:
                req_missing.append({"prop": p, "score": round(prob, 4)})

        for p in forbidden_props:
            p_prompts = [f"a scene showing {p}", f"a scene without {p}"]
            c_in = clip_proc(text=p_prompts, images=pil_full, return_tensors="pt", padding=True)
            with torch.no_grad():
                c_out = clip_model(**c_in)
            prob = float(c_out.logits_per_image.softmax(dim=1).cpu().numpy()[0][0])
            if prob >= 0.65:
                forb_present.append({"prop": p, "score": round(prob, 4)})

        detector_health["clip"] = True
    except Exception as e:
        print(f"[ERROR] CLIP execution failed: {e}", file=sys.stderr)

    # 3. SegFormer Wardrobe Segmentation for each face bbox
    clothing_labs = []
    clothing_seg_statuses = []
    clothing_seg_pixels = []

    try:
        seg_proc, seg_model = get_segformer()
        detector_health["segformer"] = True

        for fbox in face_bboxes:
            fx, fy, fw, fh = fbox.get("x", 0), fbox.get("y", 0), fbox.get("w", 0), fbox.get("h", 0)
            y1 = min(h_img, fy + int(0.7 * fh))
            y2 = min(h_img, fy + int(3.5 * fh))
            x1 = max(0, fx - int(0.6 * fw))
            x2 = min(w_img, fx + int(1.6 * fw))

            if (y2 - y1) < 20 or (x2 - x1) < 20:
                clothing_labs.append(None)
                clothing_seg_statuses.append("invalid_bbox_dimensions")
                clothing_seg_pixels.append(0)
                continue

            torso_bgr = img_bgr[y1:y2, x1:x2]
            torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
            pil_torso = Image.fromarray(torso_rgb)

            s_in = seg_proc(images=pil_torso, return_tensors="pt")
            with torch.no_grad():
                s_out = seg_model(**s_in)

            upsampled = torch.nn.functional.interpolate(
                s_out.logits,
                size=(torso_rgb.shape[0], torso_rgb.shape[1]),
                mode="bilinear",
                align_corners=False
            )
            mask = upsampled.argmax(dim=1)[0].cpu().numpy()

            # SegFormer id2label: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 17: Scarf
            cloth_mask = (mask == 4) | (mask == 5) | (mask == 6) | (mask == 7) | (mask == 17)
            seg_count = int(np.sum(cloth_mask))

            # STRICT RULE: No central-torso fallback!
            # If fewer than 50 clothing pixels detected, mark seg_failed and return None
            if seg_count < 50:
                clothing_labs.append(None)
                clothing_seg_statuses.append("seg_failed")
                clothing_seg_pixels.append(seg_count)
                continue

            lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            l_std = lab[:, :, 0] * (100.0 / 255.0)
            a_std = lab[:, :, 1] - 128.0
            b_std = lab[:, :, 2] - 128.0

            sel_l = l_std[cloth_mask]
            sel_a = a_std[cloth_mask]
            sel_b = b_std[cloth_mask]

            if len(sel_l) > 0:
                clothing_labs.append([
                    float(np.median(sel_l)),
                    float(np.median(sel_a)),
                    float(np.median(sel_b))
                ])
                clothing_seg_statuses.append("ok")
                clothing_seg_pixels.append(seg_count)
            else:
                clothing_labs.append(None)
                clothing_seg_statuses.append("seg_failed")
                clothing_seg_pixels.append(seg_count)

    except Exception as e:
        print(f"[ERROR] SegFormer execution failed: {e}", file=sys.stderr)
        traceback.print_exc()
        detector_health["segformer"] = False
        while len(clothing_labs) < len(face_bboxes):
            clothing_labs.append(None)
            clothing_seg_statuses.append("seg_failed")
            clothing_seg_pixels.append(0)

    # 4. DINOv2 Appearance Similarity
    target_dino_sims = []
    try:
        dino_proc, dino_model = get_dinov2()
        ref_dino_embs = []
        for rp in reference_image_paths:
            if rp.exists():
                r_img = Image.open(str(rp)).convert("RGB")
                d_in = dino_proc(images=r_img, return_tensors="pt")
                with torch.no_grad():
                    d_out = dino_model(**d_in)
                cls_t = d_out.last_hidden_state[:, 0, :].cpu().numpy()[0]
                norm = np.linalg.norm(cls_t)
                ref_dino_embs.append(cls_t / norm if norm > 0 else cls_t)

        for fbox in face_bboxes:
            fx, fy, fw, fh = fbox.get("x", 0), fbox.get("y", 0), fbox.get("w", 0), fbox.get("h", 0)
            crop_bgr = img_bgr[max(0, fy):min(h_img, fy + fh), max(0, fx):min(w_img, fx + fw)]
            if crop_bgr.size > 0 and ref_dino_embs:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                d_in = dino_proc(images=pil_crop, return_tensors="pt")
                with torch.no_grad():
                    d_out = dino_model(**d_in)
                c_cls = d_out.last_hidden_state[:, 0, :].cpu().numpy()[0]
                c_norm = np.linalg.norm(c_cls)
                if c_norm > 0:
                    c_cls = c_cls / c_norm
                sims = [float(np.dot(c_cls, r_emb)) for r_emb in ref_dino_embs]
                target_dino_sims.append(round(max(sims), 4))
            else:
                target_dino_sims.append(0.0)

        detector_health["dinov2"] = True
    except Exception as e:
        print(f"[ERROR] DINOv2 execution failed: {e}", file=sys.stderr)
        while len(target_dino_sims) < len(face_bboxes):
            target_dino_sims.append(0.0)

    return {
        "person_count_yolo": len(persons),
        "person_boxes": persons,
        "is_inverted": bool(inv_prob >= 0.50),
        "inversion_probability": round(inv_prob, 4),
        "missing_required_props": req_missing,
        "detected_forbidden_props": forb_present,
        "clothing_labs": clothing_labs,
        "clothing_seg_statuses": clothing_seg_statuses,
        "clothing_seg_pixels": clothing_seg_pixels,
        "dinov2_similarities": target_dino_sims,
        "detector_health": detector_health
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    res = process_image(
        image_path=Path(cfg["image_path"]),
        face_bboxes=cfg.get("face_bboxes", []),
        reference_image_paths=[Path(p) for p in cfg.get("reference_image_paths", [])],
        required_props=cfg.get("required_props", []),
        forbidden_props=cfg.get("forbidden_props", [])
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
