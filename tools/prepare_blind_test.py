"""
Step 1 of Blind Evaluation: Preparation of Shuffled Blinded Test Sets (Automated or Human).
Generates:
1. Anonymized A/B pairs (v1 baseline vs v2) with cryptographically shuffled A/B assignment
   and optional full-dataset shuffled pair IDs (pair_01..pair_N).
2. Anonymized Re-ID crops (20 crops, 4 per Ebright character) with randomized IDs + composite board.
3. Optional human instruction markdown + blank predictions JSON template (never pre-filled!).

Stores sealed ground-truth answer keys STRICTLY in --sealed-dir (default scratch/sealed_keys/),
and never prints answer keys to stdout.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.scene_auditor import SceneAuditor


def _discover_common_pairs(v1_dir: Path, v2_dir: Path) -> List[Tuple[int, int, str]]:
    v1_files = {p.name: p for p in v1_dir.glob("scene_*_seed_*.png")}
    v2_files = {p.name: p for p in v2_dir.glob("scene_*_seed_*.png")}
    common = []
    for fname in sorted(set(v1_files.keys()) & set(v2_files.keys())):
        m = re.search(r"scene_(\d+)_seed_(\d+)", fname)
        if m:
            common.append((int(m.group(1)), int(m.group(2)), fname))
    common.sort()
    return common


def prepare_blind_ab(
    v1_dir: Path,
    v2_dir: Path,
    blind_ab_dir: Path,
    sealed_dir: Path,
    salt: str,
    shuffle_seed: int,
    auto_all: bool,
) -> List[str]:
    blind_ab_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    sealed_ab_key = {}
    pair_manifest = []
    pair_ids = []

    if auto_all:
        items = _discover_common_pairs(v1_dir, v2_dir)
        rng = random.Random(shuffle_seed)
        rng.shuffle(items)
        use_anonymous_ids = True
    else:
        items = [(s_num, seed, f"scene_{s_num}_seed_{seed}.png") for s_num in range(1, 7) for seed in [101, 202]]
        use_anonymous_ids = False

    for idx, (s_num, seed, fname) in enumerate(items, 1):
        p_v1 = v1_dir / fname
        p_v2 = v2_dir / fname
        if not p_v1.exists() or not p_v2.exists():
            continue

        img_v1 = cv2.imread(str(p_v1))
        img_v2 = cv2.imread(str(p_v2))
        if img_v1 is None or img_v2 is None:
            continue

        h_str = hashlib.sha256(f"{salt}_{fname}".encode("utf-8")).hexdigest()
        v2_is_option_a = (int(h_str, 16) % 2 == 0)

        pair_id = f"pair_{idx:02d}" if use_anonymous_ids else f"pair_scene_{s_num}_seed_{seed}"
        if v2_is_option_a:
            opt_a, opt_b, v2_choice = img_v2, img_v1, "A"
        else:
            opt_a, opt_b, v2_choice = img_v1, img_v2, "B"

        sealed_ab_key[pair_id] = {
            "scene_number": s_num,
            "seed": seed,
            "source_filename": fname,
            "option_a": "v2" if v2_is_option_a else "v1_baseline",
            "option_b": "v1_baseline" if v2_is_option_a else "v2",
            "v2_choice": v2_choice,
        }

        w, h = 768, int(round(768 * opt_a.shape[0] / max(1, opt_a.shape[1])))
        h_b = int(round(768 * opt_b.shape[0] / max(1, opt_b.shape[1])))
        ch = max(h, h_b)
        a_r = cv2.resize(opt_a, (w, ch))
        b_r = cv2.resize(opt_b, (w, ch))

        canvas = np.zeros((ch + 75, w * 2 + 30, 3), dtype=np.uint8)
        canvas[:] = (25, 27, 32)
        cv2.putText(canvas, f"{pair_id.upper()}: OPTION A", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 250), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{pair_id.upper()}: OPTION B", (w + 45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 250), 2, cv2.LINE_AA)
        canvas[60:60 + ch, 10:10 + w] = a_r
        canvas[60:60 + ch, w + 20:w + 20 + w] = b_r

        out_p = blind_ab_dir / f"{pair_id}_blind.png"
        cv2.imwrite(str(out_p), canvas)

        opt_a_path = blind_ab_dir / f"{pair_id}_option_A.png"
        opt_b_path = blind_ab_dir / f"{pair_id}_option_B.png"
        cv2.imwrite(str(opt_a_path), opt_a)
        cv2.imwrite(str(opt_b_path), opt_b)

        pair_manifest.append({
            "pair_id": pair_id,
            "blind_composite": str(out_p.relative_to(PROJECT_ROOT)),
            "option_a_path": str(opt_a_path.relative_to(PROJECT_ROOT)),
            "option_b_path": str(opt_b_path.relative_to(PROJECT_ROOT)),
        })
        pair_ids.append(pair_id)

    (sealed_dir / "sealed_ab_key.json").write_text(json.dumps(sealed_ab_key, indent=2), encoding="utf-8")
    (blind_ab_dir / "ab_manifest.json").write_text(json.dumps(pair_manifest, indent=2), encoding="utf-8")
    print(f"[OK] Generated {len(pair_manifest)} Blind A/B pairs in {blind_ab_dir}. Sealed key stored privately.")
    return pair_ids


def _extract_hero_crops(auditor: SceneAuditor, img_path: Path, s_num: int):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    faces = auditor.detect_faces(img)
    if not faces:
        return []
    h_img, w_img = img.shape[:2]

    def padded_crop(fb):
        bw, bh = fb["w"], fb["h"]
        px, py = int(bw * 0.2), int(bh * 0.2)
        x1 = max(0, fb["x"] - px)
        y1 = max(0, fb["y"] - py)
        x2 = min(w_img, fb["x"] + bw + px)
        y2 = min(h_img, fb["y"] + bh + py)
        return img[y1:y2, x1:x2]

    if s_num in [2, 3, 4, 6]:
        top = sorted(faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)[0]
        fb = top["bbox"]
        if fb["w"] >= 40 and fb["h"] >= 40:
            cid = "young_ebright" if s_num == 2 else ("teen_ebright" if s_num in [3, 4] else "college_ebright")
            return [(cid, img_path.stem, padded_crop(fb))]
        return []
    elif s_num in [1, 5]:
        left_cands, right_cands = [], []
        for f in faces:
            fb = f["bbox"]
            if fb["w"] < 40 or fb["h"] < 40:
                continue
            cx = (fb["x"] + fb["w"] / 2.0) / float(w_img)
            area = fb["w"] * fb["h"]
            (left_cands if cx < 0.50 else right_cands).append((area, f))
        res = []
        if left_cands:
            fb = sorted(left_cands, key=lambda t: t[0], reverse=True)[0][1]["bbox"]
            cid = "young_ebright" if s_num == 1 else "college_ebright"
            res.append((cid, img_path.stem, padded_crop(fb)))
        if right_cands:
            fb = sorted(right_cands, key=lambda t: t[0], reverse=True)[0][1]["bbox"]
            cid = "mother" if s_num == 1 else "james_wong"
            res.append((cid, img_path.stem, padded_crop(fb)))
        return res
    return []


def prepare_blind_reid(
    v2_dir: Path,
    reid_dir: Path,
    sealed_dir: Path,
    shuffle_seed: int,
    auto_all: bool,
) -> List[str]:
    reid_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    auditor = SceneAuditor()
    chars = ["young_ebright", "teen_ebright", "college_ebright", "mother", "james_wong"]
    crops_by_char = {c: [] for c in chars}

    if auto_all:
        for p in sorted(v2_dir.glob("scene_*_seed_*.png")):
            m = re.search(r"scene_(\d+)_seed_(\d+)", p.stem)
            if not m:
                continue
            s_num = int(m.group(1))
            for cid, src, crop in _extract_hero_crops(auditor, p, s_num):
                crops_by_char[cid].append((cid, src, crop))
    else:
        archive_dir = PROJECT_ROOT / "results" / "bundle_dev2" / "legacy_seeds_archive"
        for s_num in range(1, 7):
            for sd in [101, 202]:
                p = v2_dir / f"scene_{s_num}_seed_{sd}.png"
                if p.exists():
                    for cid, src, crop in _extract_hero_crops(auditor, p, s_num):
                        crops_by_char[cid].append((cid, src, crop))
        for sd in [303, 404]:
            p1 = archive_dir / f"scene_1_seed_{sd}.png"
            if p1.exists():
                for cid, src, crop in _extract_hero_crops(auditor, p1, 1):
                    if cid == "mother":
                        crops_by_char["mother"].append((cid, src, crop))
            p5 = archive_dir / f"scene_5_seed_{sd}.png"
            if p5.exists():
                for cid, src, crop in _extract_hero_crops(auditor, p5, 5):
                    if cid == "james_wong":
                        crops_by_char["james_wong"].append((cid, src, crop))

    rng = random.Random(shuffle_seed)
    selected_20 = []
    for cid in chars:
        avail = list(crops_by_char[cid])
        if auto_all:
            rng.shuffle(avail)
        if len(avail) < 4:
            print(f"Warning: only {len(avail)} crops found for {cid}")
        selected_20.extend(avail[:4])

    rng.shuffle(selected_20)

    sealed_reid_key = {}
    reid_manifest = []
    crop_ids = []

    tile_sz = 256
    cols, rows = 5, 4
    board = np.zeros((rows * (tile_sz + 40) + 40, cols * (tile_sz + 20) + 20, 3), dtype=np.uint8)
    board[:] = (24, 26, 30)

    for idx, (true_cid, source, crop) in enumerate(selected_20, 1):
        crop_id = f"crop_{idx:02d}"
        crop_path = reid_dir / f"{crop_id}.png"
        cv2.imwrite(str(crop_path), crop)

        sealed_reid_key[crop_id] = {
            "crop_id": crop_id,
            "true_character_id": true_cid,
            "source": source,
        }
        reid_manifest.append({
            "crop_id": crop_id,
            "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
        })
        crop_ids.append(crop_id)

        r_idx, c_idx = (idx - 1) // cols, (idx - 1) % cols
        x0 = 20 + c_idx * (tile_sz + 20)
        y0 = 30 + r_idx * (tile_sz + 40)
        resized = cv2.resize(crop, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
        board[y0:y0 + tile_sz, x0:x0 + tile_sz] = resized
        cv2.putText(board, crop_id, (x0 + 6, y0 + tile_sz + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 245), 2, cv2.LINE_AA)

    board_path = reid_dir / "reid_board.png"
    cv2.imwrite(str(board_path), board)

    (sealed_dir / "sealed_reid_key.json").write_text(json.dumps(sealed_reid_key, indent=2), encoding="utf-8")
    (reid_dir / "reid_manifest.json").write_text(json.dumps(reid_manifest, indent=2), encoding="utf-8")
    print(f"[OK] Generated {len(reid_manifest)} Blind Re-ID crops + board in {reid_dir}. Sealed key stored privately.")
    return crop_ids


def write_human_blind_assets(
    md_out: Path,
    pred_out: Path,
    ab_dir: Path,
    reid_dir: Path,
    pair_ids: List[str],
    crop_ids: List[str],
) -> None:
    pred_template = {
        "evaluation_type": "human_blind_evaluation_pending",
        "instructions": "Fill 'selected_option' with 'A', 'B', or 'tie' for each pair; fill 'predicted_character_id' with one of ['young_ebright', 'teen_ebright', 'college_ebright', 'mother', 'james_wong'] for each crop.",
        "evaluator_name": "",
        "date": "",
        "ab_preference": {
            pid: {
                "selected_option": "",
                "notes": "",
            }
            for pid in pair_ids
        },
        "blind_reid": {
            cid: {
                "predicted_character_id": "",
            }
            for cid in crop_ids
        },
    }
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.write_text(json.dumps(pred_template, indent=2), encoding="utf-8")

    ab_rel = ab_dir.relative_to(PROJECT_ROOT).as_posix()
    reid_rel = reid_dir.relative_to(PROJECT_ROOT).as_posix()

    lines = [
        "# Human Blind Evaluation Protocol (TEST-2 Ebright)",
        "",
        "> [!IMPORTANT]",
        "> **STATUS: AWAITING HUMAN COMPLETION (DO NOT AUTO-FILL)**  ",
        "> All previous Phase R8 blind, canary, and VLM checks (`blind_study_report.json`, `canary_audit_report.json`, `vlm_second_opinion_report.md`, and superseded `blind_review_summary.json`) were **automated proxy checks, not human evaluations**.  ",
        "> This human protocol tests genuine perceptual preference and character re-identification on **TEST-2 Ebright** (`v1` Baseline vs `v2.0.0-rc2/rc3`, 30 shuffled pairs + 20 anonymous face crops).",
        "",
        "---",
        "",
        "## Part 1: Blind A/B Pairwise Preference (30 Shuffled Pairs)",
        "",
        f"Inspect each side-by-side composite image in `{ab_rel}/<pair_id>_blind.png`.",
        "For each pair, judge BOTH **character identity consistency** and **overall visual quality**, and record `\"A\"`, `\"B\"`, or `\"tie\"` under `ab_preference.<pair_id>.selected_option` in:",
        f"- [`{pred_out.relative_to(PROJECT_ROOT).as_posix()}`](file:///{pred_out.resolve().as_posix()})",
        "",
        "| Pair ID | Blind Composite Path |",
        "| :--- | :--- |",
    ]
    for pid in pair_ids:
        lines.append(f"| `{pid}` | `{ab_rel}/{pid}_blind.png` |")

    lines.extend([
        "",
        "---",
        "",
        "## Part 2: Anonymous Face Re-Identification (20 Crops)",
        "",
        f"Inspect `{reid_rel}/reid_board.png` (or individual files `{reid_rel}/crop_01.png` .. `{reid_rel}/crop_20.png`).",
        "Reference anchors (for character identity visual lookup):",
        "- `young_ebright`: `experiments/clean_run_ground_truth/anchors/young_ebright_ground_truth.png`",
        "- `teen_ebright`: `experiments/clean_run_ground_truth/anchors/teen_ebright_ground_truth.png`",
        "- `college_ebright`: `experiments/clean_run_ground_truth/anchors/college_ebright_ground_truth.png`",
        "- `mother`: `experiments/clean_run_ground_truth/anchors/mother_ground_truth.png`",
        "- `james_wong`: `experiments/clean_run_ground_truth/anchors/james_wong_ground_truth.png`",
        "",
        "For each `crop_01` .. `crop_20`, fill `blind_reid.<crop_id>.predicted_character_id` with one of: `young_ebright`, `teen_ebright`, `college_ebright`, `mother`, `james_wong`.",
        "",
        "| Crop ID | Image Path |",
        "| :--- | :--- |",
    ])
    for cid in crop_ids:
        lines.append(f"| `{cid}` | `{reid_rel}/{cid}.png` |")

    lines.extend([
        "",
        "---",
        "",
        "## Next Step (Scoring Only After Human Return)",
        "",
        "Once `blind_predictions_human.json` is filled by a human evaluator, score it against the sealed key in `scratch/sealed_keys/human_test2_ebright/`.",
    ])

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote human blind test guide: {md_out}")
    print(f"[OK] Wrote BLANK human predictions template: {pred_out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Blind A/B and Re-ID evaluation assets")
    parser.add_argument("--v1-dir", type=str, default=str(PROJECT_ROOT / "results" / "baseline_v1" / "dev2"))
    parser.add_argument("--v2-dir", type=str, default=str(PROJECT_ROOT / "results" / "bundle_dev2" / "renders"))
    parser.add_argument("--ab-dir", type=str, default=str(PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_ab"))
    parser.add_argument("--reid-dir", type=str, default=str(PROJECT_ROOT / "docs_generated" / "v2" / "visual_review" / "blind_reid_crops"))
    parser.add_argument("--sealed-dir", type=str, default=str(PROJECT_ROOT / "scratch" / "sealed_keys"))
    parser.add_argument("--salt", type=str, default="MMCCP_V2_GENUINE_BLIND_A_B_SALT_20261103")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-all", action="store_true", help="Discover all common scene/seed pairs and generate anonymous pair IDs")
    parser.add_argument("--md-out", type=str, default="")
    parser.add_argument("--pred-out", type=str, default="")
    args = parser.parse_args()

    def _res(p_str: str) -> Path:
        p = Path(p_str)
        return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()

    v1_dir = _res(args.v1_dir)
    v2_dir = _res(args.v2_dir)
    ab_dir = _res(args.ab_dir)
    reid_dir = _res(args.reid_dir)
    sealed_dir = _res(args.sealed_dir)

    pair_ids = prepare_blind_ab(v1_dir, v2_dir, ab_dir, sealed_dir, args.salt, args.seed, args.auto_all)
    crop_ids = prepare_blind_reid(v2_dir, reid_dir, sealed_dir, args.seed, args.auto_all)

    if args.md_out and args.pred_out:
        write_human_blind_assets(_res(args.md_out), _res(args.pred_out), ab_dir, reid_dir, pair_ids, crop_ids)


if __name__ == "__main__":
    main()
