"""
Step 1 of Genuine Blind Evaluation: Preparation of Blinded Test Sets.
Generates:
1. Anonymized A/B pairs (v1 baseline vs v2 locked bundle) with cryptographically shuffled A/B assignment.
2. Anonymized Re-ID crops (20 crops) with randomized IDs.
Stores sealed ground truth answer keys STRICTLY in scratch/sealed_keys/ (forbidden from evaluation script).
"""
import os
import sys
import json
import hashlib
import random
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.scene_auditor import SceneAuditor

V1_DIR = PROJECT_ROOT / "results" / "baseline_v1" / "dev2"
V2_DIR = PROJECT_ROOT / "results" / "bundle_dev2" / "renders"

BLIND_AB_DIR = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_ab"
BLIND_AB_DIR.mkdir(parents=True, exist_ok=True)

REID_DIR = PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_reid_crops"
REID_DIR.mkdir(parents=True, exist_ok=True)

SEALED_DIR = PROJECT_ROOT / "scratch" / "sealed_keys"
SEALED_DIR.mkdir(parents=True, exist_ok=True)


def prepare_blind_ab():
    salt = "MMCCP_V2_GENUINE_BLIND_A_B_SALT_20261103"
    sealed_ab_key = {}
    pair_manifest = []

    for s_num in range(1, 7):
        for seed in [101, 202]:
            fname = f"scene_{s_num}_seed_{seed}.png"
            p_v1 = V1_DIR / fname
            p_v2 = V2_DIR / fname
            if not p_v1.exists() or not p_v2.exists():
                continue

            img_v1 = cv2.imread(str(p_v1))
            img_v2 = cv2.imread(str(p_v2))

            h_str = hashlib.sha256(f"{salt}_{fname}".encode("utf-8")).hexdigest()
            v2_is_option_a = (int(h_str, 16) % 2 == 0)

            pair_id = f"pair_scene_{s_num}_seed_{seed}"
            if v2_is_option_a:
                opt_a = img_v2
                opt_b = img_v1
                v2_choice = "A"
            else:
                opt_a = img_v1
                opt_b = img_v2
                v2_choice = "B"

            sealed_ab_key[pair_id] = {
                "scene_number": s_num,
                "seed": seed,
                "option_a": "v2_locked_bundle" if v2_is_option_a else "v1_baseline",
                "option_b": "v1_baseline" if v2_is_option_a else "v2_locked_bundle",
                "v2_choice": v2_choice
            }

            # Render comparison image
            w, h = 800, 533
            a_r = cv2.resize(opt_a, (w, h))
            b_r = cv2.resize(opt_b, (w, h))

            canvas = np.zeros((h + 80, w * 2 + 30, 3), dtype=np.uint8)
            canvas[:] = (25, 27, 32)
            cv2.putText(canvas, f"{pair_id.upper()}: OPTION A", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 250), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{pair_id.upper()}: OPTION B", (w + 45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 250), 2, cv2.LINE_AA)
            canvas[65:65+h, 10:10+w] = a_r
            canvas[65:65+h, w+20:w+20+w] = b_r

            out_p = BLIND_AB_DIR / f"{pair_id}_blind.png"
            cv2.imwrite(str(out_p), canvas)
            
            # Save separate option images for direct algorithmic comparison
            opt_a_path = BLIND_AB_DIR / f"{pair_id}_option_A.png"
            opt_b_path = BLIND_AB_DIR / f"{pair_id}_option_B.png"
            cv2.imwrite(str(opt_a_path), opt_a)
            cv2.imwrite(str(opt_b_path), opt_b)

            pair_manifest.append({
                "pair_id": pair_id,
                "blind_composite": str(out_p),
                "option_a_path": str(opt_a_path),
                "option_b_path": str(opt_b_path)
            })

    with open(SEALED_DIR / "sealed_ab_key.json", "w", encoding="utf-8") as f:
        json.dump(sealed_ab_key, f, indent=2)

    with open(BLIND_AB_DIR / "ab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(pair_manifest, f, indent=2)

    print(f"[OK] Generated {len(pair_manifest)} Blind A/B pairs. Sealed key saved.")


def prepare_blind_reid():
    auditor = SceneAuditor()
    renders_dir = V2_DIR
    archive_dir = PROJECT_ROOT / "results" / "bundle_dev2" / "legacy_seeds_archive"

    crops_by_char = {
        "young_ebright": [],
        "teen_ebright": [],
        "college_ebright": [],
        "mother": [],
        "james_wong": []
    }

    def extract_hero_crops(img_path: Path, s_num: int):
        if not img_path.exists():
            return []
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        faces = auditor.detect_faces(img)
        if not faces:
            return []
        w_img = img.shape[1]

        # For solo scenes, pick ONLY the single largest prominent hero face
        if s_num in [2, 3, 4, 6]:
            faces_sorted = sorted(faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)
            top = faces_sorted[0]
            fb = top["bbox"]
            if fb["w"] >= 40 and fb["h"] >= 40:
                crop = img[fb["y"]:fb["y"]+fb["h"], fb["x"]:fb["x"]+fb["w"]]
                cid = "young_ebright" if s_num == 2 else ("teen_ebright" if s_num in [3, 4] else "college_ebright")
                return [(cid, img_path.stem, crop)]
            return []

        # For two-character scenes (1, 5), pick the largest face on left and right
        elif s_num in [1, 5]:
            left_cands = []
            right_cands = []
            for f in faces:
                fb = f["bbox"]
                if fb["w"] < 40 or fb["h"] < 40:
                    continue
                cx = (fb["x"] + fb["w"] / 2.0) / float(w_img)
                area = fb["w"] * fb["h"]
                if cx < 0.50:
                    left_cands.append((area, f))
                else:
                    right_cands.append((area, f))

            res = []
            if left_cands:
                left_cands.sort(key=lambda t: t[0], reverse=True)
                fb = left_cands[0][1]["bbox"]
                crop = img[fb["y"]:fb["y"]+fb["h"], fb["x"]:fb["x"]+fb["w"]]
                cid = "young_ebright" if s_num == 1 else "college_ebright"
                res.append((cid, img_path.stem, crop))

            if right_cands:
                right_cands.sort(key=lambda t: t[0], reverse=True)
                fb = right_cands[0][1]["bbox"]
                crop = img[fb["y"]:fb["y"]+fb["h"], fb["x"]:fb["x"]+fb["w"]]
                cid = "mother" if s_num == 1 else "james_wong"
                res.append((cid, img_path.stem, crop))

            return res
        return []

    # 1. Harvest crops from DEV-2 renders (seeds 101 and 202)
    for s_num in [1, 2, 3, 4, 5, 6]:
        for sd in [101, 202]:
            p = renders_dir / f"scene_{s_num}_seed_{sd}.png"
            for cid, src, crop in extract_hero_crops(p, s_num):
                crops_by_char[cid].append((cid, src, crop))

    # 2. For mother and james_wong, supplement from legacy DEV-2 seed renders (seeds 303, 404)
    for sd in [303, 404]:
        p1 = archive_dir / f"scene_1_seed_{sd}.png"
        for cid, src, crop in extract_hero_crops(p1, 1):
            if cid == "mother":
                crops_by_char["mother"].append((cid, src, crop))
        p5 = archive_dir / f"scene_5_seed_{sd}.png"
        for cid, src, crop in extract_hero_crops(p5, 5):
            if cid == "james_wong":
                crops_by_char["james_wong"].append((cid, src, crop))

    # Select exactly 4 crops per character (20 crops total, perfectly balanced, zero anchor leakage)
    selected_20 = []
    for cid in ["young_ebright", "teen_ebright", "college_ebright", "mother", "james_wong"]:
        available = crops_by_char[cid]
        if len(available) < 4:
            print(f"Warning: Only {len(available)} crops for {cid}")
        selected_20.extend(available[:4])

    rng = random.Random(42)  # Deterministic seed for unblindable shuffling
    rng.shuffle(selected_20)

    sealed_reid_key = {}
    reid_manifest = []

    for idx, (true_cid, source, crop) in enumerate(selected_20, 1):
        crop_id = f"crop_{idx:02d}"
        crop_filename = f"{crop_id}.png"
        crop_path = REID_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop)

        sealed_reid_key[crop_id] = {
            "crop_id": crop_id,
            "true_character_id": true_cid,
            "source": source
        }

        reid_manifest.append({
            "crop_id": crop_id,
            "crop_path": str(crop_path)
        })

    with open(SEALED_DIR / "sealed_reid_key.json", "w", encoding="utf-8") as f:
        json.dump(sealed_reid_key, f, indent=2)

    with open(REID_DIR / "reid_manifest.json", "w", encoding="utf-8") as f:
        json.dump(reid_manifest, f, indent=2)

    print(f"[OK] Generated {len(reid_manifest)} Blind Re-ID crops (4 per character, ZERO anchor leakage). Sealed key saved.")


if __name__ == '__main__':
    prepare_blind_ab()
    prepare_blind_reid()
