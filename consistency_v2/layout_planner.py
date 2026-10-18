"""
M2: Programmatic Layout Planner & OpenPose Skeleton Synthesizer
Generates geometric region masks and standard OpenPose COCO-18 skeleton control images
from storyboard scene specifications (character_positions, shot_type, camera_angle).
Guarantees face size >= 96px for medium shots and >= 64px for wide shots on 1024x1024 canvas.
"""
import math
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np


# OpenPose COCO-18 Body Connections and Standard Colors (RGB)
COCO_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

COCO_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170]
]


@dataclass
class CharacterLayoutBox:
    character_id: str
    target_zone: str  # "left", "center", "right"
    face_bbox: Dict[str, int]  # {x, y, w, h}
    body_bbox: Dict[str, int]  # {x, y, w, h}
    normalized_region: List[float]  # [y1, y2, x1, x2] in 0.0-1.0 range
    scale_category: str  # "close", "medium", "wide"


class LayoutPlanner:
    """
    Translates storyboard scene parameters into deterministic spatial regions and OpenPose control images.
    """

    CANVAS_SIZE = 1024

    # Normalized canonical pose templates: centered at (0, 0) with height 1.0
    CANONICAL_POSES = {
        "standing_front": {
            0: (0.0, -0.42),    # Nose
            1: (0.0, -0.32),    # Neck
            2: (-0.14, -0.30),  # R-Shoulder
            3: (-0.17, -0.10),  # R-Elbow
            4: (-0.17, 0.08),   # R-Wrist
            5: (0.14, -0.30),   # L-Shoulder
            6: (0.17, -0.10),   # L-Elbow
            7: (0.17, 0.08),    # L-Wrist
            8: (-0.08, -0.02),  # R-Hip
            9: (-0.09, 0.22),   # R-Knee
            10: (-0.09, 0.46),  # R-Ankle
            11: (0.08, -0.02),  # L-Hip
            12: (0.09, 0.22),   # L-Knee
            13: (0.09, 0.46),   # L-Ankle
            14: (-0.03, -0.45), # R-Eye
            15: (0.03, -0.45),  # L-Eye
            16: (-0.07, -0.43), # R-Ear
            17: (0.07, -0.43)   # L-Ear
        },
        "sitting_bench": {
            0: (0.0, -0.38),
            1: (0.0, -0.28),
            2: (-0.13, -0.26),
            3: (-0.15, -0.08),
            4: (-0.10, 0.05),
            5: (0.13, -0.26),
            6: (0.15, -0.08),
            7: (0.10, 0.05),
            8: (-0.08, -0.02),
            9: (-0.14, 0.18),
            10: (-0.14, 0.40),
            11: (0.08, -0.02),
            12: (0.14, 0.18),
            13: (0.14, 0.40),
            14: (-0.03, -0.41),
            15: (0.03, -0.41),
            16: (-0.07, -0.39),
            17: (0.07, -0.39)
        }
    }

    @classmethod
    def resolve_position_ratio(cls, position_str: str, default_ratio: float = 0.50) -> Tuple[float, str]:
        """Maps natural language position description to normalized horizontal ratio and zone."""
        p = position_str.lower()
        if "left" in p:
            return 0.28, "left"
        elif "right" in p:
            return 0.72, "right"
        elif "center" in p or "middle" in p:
            return 0.50, "center"
        return default_ratio, "center"

    @classmethod
    def compute_scale_params(cls, shot_type: str) -> Tuple[float, float, str]:
        """
        Computes total figure height (px), face width (px), and scale category.
        Ensures strict face size minimums: >=96px for medium, >=64px for wide.
        """
        st = shot_type.lower()
        if "extreme close-up" in st or "macro" in st:
            fig_height = 1400.0
            face_width = 300.0
            cat = "close"
        elif "close-up" in st or "portrait" in st or "hero" in st or "medium close-up" in st:
            fig_height = 1100.0
            face_width = 180.0
            cat = "close"
        elif "medium shot" in st or "two-shot" in st:
            fig_height = 780.0
            face_width = 120.0  # Well above 96px minimum
            cat = "medium"
        elif "wide shot" in st or "full shot" in st or "long shot" in st:
            fig_height = 540.0
            face_width = 75.0   # Well above 64px minimum
            cat = "wide"
        else:
            fig_height = 720.0
            face_width = 110.0
            cat = "medium"

        return fig_height, face_width, cat

    @classmethod
    def plan_scene_layout(
        cls,
        character_positions: Dict[str, str],
        shot_type: str = "medium shot",
        canvas_w: int = 1024,
        canvas_h: int = 1024
    ) -> List[CharacterLayoutBox]:
        """
        Computes non-overlapping spatial layout boxes and region masks for all characters in scene.
        """
        chars = list(character_positions.keys())
        num_chars = len(chars)
        fig_h, face_w, scale_cat = cls.compute_scale_params(shot_type)

        boxes: List[CharacterLayoutBox] = []

        if num_chars == 1:
            cid = chars[0]
            pos_str = character_positions[cid]
            r_x, zone = cls.resolve_position_ratio(pos_str, 0.50)

            cx = int(canvas_w * r_x)
            cy = int(canvas_h * 0.52)
            face_cy = int(cy - (fig_h * 0.38))
            face_h = int(face_w * 1.2)

            face_box = {
                "x": max(0, int(cx - face_w / 2)),
                "y": max(0, int(face_cy - face_h / 2)),
                "w": int(face_w),
                "h": int(face_h)
            }
            body_box = {
                "x": max(0, int(cx - (fig_h * 0.25))),
                "y": max(0, int(cy - (fig_h * 0.5))),
                "w": int(fig_h * 0.5),
                "h": int(fig_h)
            }
            norm_region = [0.0, 1.0, 0.0, 1.0]

            boxes.append(CharacterLayoutBox(
                character_id=cid,
                target_zone=zone,
                face_bbox=face_box,
                body_bbox=body_box,
                normalized_region=norm_region,
                scale_category=scale_cat
            ))

        elif num_chars >= 2:
            # Multi-character horizontal partitioning
            left_char = chars[0]
            right_char = chars[1]

            # Assign left and right characters based on spec
            for cid in chars:
                p = character_positions.get(cid, "").lower()
                if "left" in p:
                    left_char = cid
                elif "right" in p:
                    right_char = cid

            ordered_pair = [left_char, right_char]
            r_offsets = [0.28, 0.72]
            zones = ["left", "right"]
            norm_regions = [
                [0.0, 1.0, 0.0, 0.52],  # Left region
                [0.0, 1.0, 0.48, 1.0]   # Right region
            ]

            for i, cid in enumerate(ordered_pair):
                r_x = r_offsets[i]
                zone = zones[i]
                cx = int(canvas_w * r_x)
                cy = int(canvas_h * 0.55)
                face_cy = int(cy - (fig_h * 0.38))
                face_h = int(face_w * 1.2)

                face_box = {
                    "x": max(0, int(cx - face_w / 2)),
                    "y": max(0, int(face_cy - face_h / 2)),
                    "w": int(face_w),
                    "h": int(face_h)
                }
                body_box = {
                    "x": max(0, int(cx - (fig_h * 0.22))),
                    "y": max(0, int(cy - (fig_h * 0.5))),
                    "w": int(fig_h * 0.44),
                    "h": int(fig_h)
                }

                boxes.append(CharacterLayoutBox(
                    character_id=cid,
                    target_zone=zone,
                    face_bbox=face_box,
                    body_bbox=body_box,
                    normalized_region=norm_regions[i],
                    scale_category=scale_cat
                ))

        return boxes

    @classmethod
    def render_openpose_canvas(
        cls,
        layout_boxes: List[CharacterLayoutBox],
        pose_type: str = "standing_front",
        canvas_w: int = 1024,
        canvas_h: int = 1024
    ) -> np.ndarray:
        """
        Renders a standard black ControlNet OpenPose COCO-18 skeleton control image.
        Uses OpenCV lines and circles drawn in exact OpenPose color order.
        """
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        template = cls.CANONICAL_POSES.get(pose_type, cls.CANONICAL_POSES["standing_front"])

        for box in layout_boxes:
            b = box.body_bbox
            cx = b["x"] + b["w"] / 2.0
            cy = b["y"] + b["h"] / 2.0
            scale_h = float(b["h"])

            # Map keypoints to canvas coordinates
            kpts = {}
            for kid, (nx, ny) in template.items():
                px = int(cx + (nx * scale_h))
                py = int(cy + (ny * scale_h))
                # Clamp within canvas bounds
                px = max(0, min(canvas_w - 1, px))
                py = max(0, min(canvas_h - 1, py))
                kpts[kid] = (px, py)

            # Draw limbs (lines)
            for idx, (k1, k2) in enumerate(COCO_LIMBS):
                if k1 in kpts and k2 in kpts:
                    pt1 = kpts[k1]
                    pt2 = kpts[k2]
                    rgb_color = COCO_COLORS[idx % len(COCO_COLORS)]
                    bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
                    thickness = max(3, int(scale_h * 0.007))
                    cv2.line(canvas, pt1, pt2, bgr_color, thickness, lineType=cv2.LINE_AA)

            # Draw keypoints (circles)
            joint_radius = max(4, int(scale_h * 0.008))
            for kid, (px, py) in kpts.items():
                if kid in [0, 14, 15, 16, 17]:
                    # Head keypoints in pink/magenta
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.circle(canvas, (px, py), joint_radius, color, -1, lineType=cv2.LINE_AA)

        return canvas

    @classmethod
    def render_depth_layout_hint(
        cls,
        layout_boxes: List[CharacterLayoutBox],
        canvas_w: int = 1024,
        canvas_h: int = 1024
    ) -> np.ndarray:
        """
        Renders a coarse grayscale depth map proxy (white = foreground figures, dark = background).
        """
        depth_canvas = np.full((canvas_h, canvas_w), 40, dtype=np.uint8)  # Dark gray background

        # Add floor gradient
        horizon = int(canvas_h * 0.60)
        for y in range(horizon, canvas_h):
            depth_canvas[y, :] = int(40 + (y - horizon) / (canvas_h - horizon) * 60)

        # Draw figures with soft gradient
        for box in layout_boxes:
            b = box.body_bbox
            bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]

            # Figure ellipse
            center = (int(bx + bw / 2), int(by + bh / 2))
            axes = (int(bw * 0.45), int(bh * 0.48))
            cv2.ellipse(depth_canvas, center, axes, 0, 0, 360, 210, -1, lineType=cv2.LINE_AA)

            # Face circle (closer to camera)
            fb = box.face_bbox
            fc = (int(fb["x"] + fb["w"] / 2), int(fb["y"] + fb["h"] / 2))
            fr = int(fb["w"] * 0.52)
            cv2.circle(depth_canvas, fc, fr, 240, -1, lineType=cv2.LINE_AA)

        # Smooth depth map
        depth_canvas = cv2.GaussianBlur(depth_canvas, (31, 31), 0)
        return cv2.cvtColor(depth_canvas, cv2.COLOR_GRAY2BGR)

    @classmethod
    def image_to_base64(cls, img_bgr: np.ndarray) -> str:
        """Encodes an OpenCV image to base64 JPEG format."""
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
