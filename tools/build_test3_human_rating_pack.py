"""
MMCCP TEST-3 Evaluation Protocol: Human-Rating Pack & Face Strips Builder (B4).

Builds from the completed TEST-3 replicates (rep_00, rep_01, rep_02):
1. Blinded Human-Rating Pack (fixed RNG seed=20261118):
   - Task (a): Same-person matching — 21 trials (7 eligible multi-scene characters x 3 replicates):
     1 probe face crop + 4 candidate face crops from OTHER scenes of the SAME replicate
     (exactly 1 genuine same-character match + 3 different-character distractors, shuffled A/B/C/D).
   - Task (b): 1-to-5 sequence-consistency rating — 21 character face sequences
     (7 eligible multi-scene characters x 3 replicates; distinct-scene face crops).
   - Opaque crop filenames (crop_0001.png, ...) with zero identity/scene leakage.
   - Public blind file (UI reads): D:\\5TH SEM\\HACKATHON_eval\\test3_v2_final\\human_rating_pack\\rating_pack_blind.json
   - Sealed answer key (UI NEVER reads): D:\\5TH SEM\\HACKATHON_eval\\test3_v2_final\\human_rating_pack\\SEALED_ANSWER_KEY.json
   - Small metadata receipt (in repo): results/eval3/human_rating_pack_manifest.json
2. Cross-Scene Face Strips for Tab 5 Viewer browsing:
   - D:\\5TH SEM\\HACKATHON_eval\\test3_v2_final\\face_strips\\<rep_key>\\<char_id>_strip.png
"""
import os
import sys
import json
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval3.runner import EVAL_OUTPUT_BASE

RUN_ID = "test3_v2_final"
PACK_SEED = 20261118

ELIGIBLE_CHARS = [
    "young_ebright",
    "teen_ebright",
    "elena_child",
    "dr_marcus_vance",
    "leo_child",
    "maya_adult",
    "arthur_elder",
]

ALL_CHARS = [
    "young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong",
    "elena_child", "dr_marcus_vance", "clara_thorn",
    "leo_child", "maya_adult", "arthur_elder",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def crop_face_with_margin(img_bgr: np.ndarray, bbox: Dict[str, Any], out_size: int = 256, margin: float = 0.25) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = max(1, int(bbox.get("w", 1)))
    h = max(1, int(bbox.get("h", 1)))
    # Make square with margin
    side = int(max(w, h) * (1.0 + 2.0 * margin))
    cx = x + w // 2
    cy = y + h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w_img, x1 + side)
    y2 = min(h_img, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def main():
    run_dir = EVAL_OUTPUT_BASE / RUN_ID
    pack_dir = run_dir / "human_rating_pack"
    crops_dir = pack_dir / "crops"
    strips_dir = run_dir / "face_strips"
    crops_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    completed_reps = manifest.get("completed_replicates", ["rep_00", "rep_01", "rep_02"])

    df_faces = pd.read_csv(PROJECT_ROOT / "results" / "eval3" / "test3_per_face.csv")
    df_scenes = pd.read_csv(PROJECT_ROOT / "results" / "eval3" / "test3_per_scene.csv")
    bon3_ids = set(df_scenes[df_scenes["bon3_selected"] == True]["render_id"].tolist())

    rng = np.random.RandomState(PACK_SEED)

    # Index detected faces per (rep_key, char_id) across distinct scenes (preferring bon3 renders first)
    faces_by_rep_char: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rep_key in completed_reps:
        r_idx = int(rep_key.split("_")[1])
        scenes_dir = run_dir / "replicates" / rep_key / "scenes"
        sub = df_faces[(df_faces["replicate_idx"] == r_idx) & (df_faces["character_id"] != "unassigned")]
        for _, row in sub.iterrows():
            cid = str(row["character_id"])
            rid = str(row["render_id"])
            img_p = scenes_dir / f"{rid}.png"
            if not img_p.exists():
                continue
            rec = {
                "rep_key": rep_key,
                "replicate_idx": r_idx,
                "story_label": str(row["story_label"]),
                "scene_number": int(row["scene_number"]),
                "global_scene_key": f"{row['story_label']}_sc{int(row['scene_number'])}",
                "render_id": rid,
                "bon3_selected": rid in bon3_ids,
                "image_path": str(img_p),
                "bbox": {"x": int(row["bbox_x"]), "y": int(row["bbox_y"]), "w": int(row["bbox_w"]), "h": int(row["bbox_h"])},
                "sharpness": float(row["sharpness"]),
            }
            faces_by_rep_char.setdefault((rep_key, cid), []).append(rec)

    # Helper to pick one canonical face per distinct scene for (rep_key, cid), preferring bon3
    def get_scene_distinct_faces(rep_key: str, cid: str) -> List[Dict[str, Any]]:
        recs = faces_by_rep_char.get((rep_key, cid), [])
        by_scene: Dict[str, List[Dict[str, Any]]] = {}
        for r in recs:
            by_scene.setdefault(r["global_scene_key"], []).append(r)
        picked = []
        for sk in sorted(by_scene.keys()):
            cands = by_scene[sk]
            # Prefer bon3_selected, then highest sharpness
            cands.sort(key=lambda x: (1 if x["bon3_selected"] else 0, x["sharpness"]), reverse=True)
            picked.append(cands[0])
        return picked

    # --- 1. Build Cross-Scene Face Strips for Tab 5 Viewer ---
    strips_built = 0
    for rep_key in completed_reps:
        rep_strip_dir = strips_dir / rep_key
        rep_strip_dir.mkdir(parents=True, exist_ok=True)
        anchors_dir = run_dir / "replicates" / rep_key / "anchors"
        for cid in ALL_CHARS:
            tiles = []
            # Tile 0: Certified Anchor
            anc_json = anchors_dir / f"{cid}_selected_anchor.json"
            if anc_json.exists():
                am = json.load(open(anc_json, "r", encoding="utf-8"))
                a_img = cv2.imread(am["image_path"]) if am.get("image_path") else None
                if a_img is not None:
                    tile = cv2.resize(a_img, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.putText(tile, "ANCHOR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    tiles.append(tile)
            # Subsequent tiles: Distinct scene bon3 face crops
            sc_faces = get_scene_distinct_faces(rep_key, cid)
            for sf in sc_faces:
                im = cv2.imread(sf["image_path"])
                if im is None:
                    continue
                crop = crop_face_with_margin(im, sf["bbox"], out_size=224, margin=0.25)
                lbl = f"{sf['story_label'].replace('Story_', 'S')}-sc{sf['scene_number']}"
                cv2.putText(crop, lbl, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                tiles.append(crop)
            if tiles:
                strip = np.hstack(tiles)
                cv2.imwrite(str(rep_strip_dir / f"{cid}_strip.png"), strip)
                strips_built += 1

    # --- 2. Build Blinded Human-Rating Pack (Task A + Task B) ---
    crop_counter = 0
    crop_cache: Dict[Tuple[str, int, int, int, int], str] = {}

    def save_blind_crop(rec: Dict[str, Any]) -> str:
        nonlocal crop_counter
        b = rec["bbox"]
        key = (rec["image_path"], b["x"], b["y"], b["w"], b["h"])
        if key in crop_cache:
            return crop_cache[key]
        crop_counter += 1
        fname = f"crop_{crop_counter:04d}.png"
        im = cv2.imread(rec["image_path"])
        crop = crop_face_with_margin(im, b, out_size=256, margin=0.25)
        cv2.imwrite(str(crops_dir / fname), crop)
        crop_cache[key] = fname
        return fname

    # Task (a): 21 Same-Person Matching trials (7 eligible chars x 3 reps)
    task_a_raw: List[Dict[str, Any]] = []
    for rep_key in completed_reps:
        for cid in ELIGIBLE_CHARS:
            sc_faces = get_scene_distinct_faces(rep_key, cid)
            if len(sc_faces) < 2:
                continue
            # Pick probe scene and genuine target scene (different scenes, same replicate)
            idx_order = rng.permutation(len(sc_faces))
            probe_rec = sc_faces[int(idx_order[0])]
            target_rec = sc_faces[int(idx_order[1])]

            # Pick 3 distinct distractor characters from OTHER scenes of the SAME replicate
            other_chars = [c for c in ALL_CHARS if c != cid]
            rng.shuffle(other_chars)
            distractors = []
            for oc in other_chars:
                oc_faces = [f for f in get_scene_distinct_faces(rep_key, oc) if f["global_scene_key"] != probe_rec["global_scene_key"]]
                if oc_faces:
                    pick = oc_faces[int(rng.randint(0, len(oc_faces)))]
                    distractors.append((oc, pick))
                if len(distractors) == 3:
                    break

            if len(distractors) < 3:
                continue

            probe_file = save_blind_crop(probe_rec)
            cand_entries = [
                {"is_target": True, "char_id": cid, "rec": target_rec, "crop_file": save_blind_crop(target_rec)}
            ]
            for oc, drec in distractors:
                cand_entries.append({"is_target": False, "char_id": oc, "rec": drec, "crop_file": save_blind_crop(drec)})

            perm = rng.permutation(4)
            letters = ["A", "B", "C", "D"]
            shuffled_cands = {}
            correct_letter = ""
            cand_key_detail = {}
            for pos_i, p_idx in enumerate(perm):
                L = letters[pos_i]
                ce = cand_entries[int(p_idx)]
                shuffled_cands[L] = ce["crop_file"]
                if ce["is_target"]:
                    correct_letter = L
                cand_key_detail[L] = {
                    "char_id": ce["char_id"],
                    "render_id": ce["rec"]["render_id"],
                    "scene": ce["rec"]["global_scene_key"],
                    "is_target": ce["is_target"],
                }

            task_a_raw.append({
                "probe_crop": probe_file,
                "candidates": shuffled_cands,
                "_key": {
                    "replicate": rep_key,
                    "character_id": cid,
                    "probe_render_id": probe_rec["render_id"],
                    "probe_scene": probe_rec["global_scene_key"],
                    "correct_option": correct_letter,
                    "candidates_detail": cand_key_detail,
                },
            })

    # Shuffle Task (a) order globally
    order_a = rng.permutation(len(task_a_raw))
    task_a_blind = []
    task_a_key = {}
    for new_i, old_i in enumerate(order_a, 1):
        tid = f"A{new_i:02d}"
        item = task_a_raw[int(old_i)]
        task_a_blind.append({
            "trial_id": tid,
            "probe_crop": item["probe_crop"],
            "candidates": item["candidates"],
        })
        task_a_key[tid] = item["_key"]

    # Task (b): 21 Sequence-Consistency rating items (7 eligible chars x 3 reps)
    task_b_raw: List[Dict[str, Any]] = []
    for rep_key in completed_reps:
        for cid in ELIGIBLE_CHARS:
            sc_faces = get_scene_distinct_faces(rep_key, cid)
            if len(sc_faces) < 2:
                continue
            crop_files = [save_blind_crop(sf) for sf in sc_faces]
            task_b_raw.append({
                "face_crops": crop_files,
                "_key": {
                    "replicate": rep_key,
                    "character_id": cid,
                    "n_scenes": len(sc_faces),
                    "render_ids": [sf["render_id"] for sf in sc_faces],
                    "scenes": [sf["global_scene_key"] for sf in sc_faces],
                },
            })

    order_b = rng.permutation(len(task_b_raw))
    task_b_blind = []
    task_b_key = {}
    for new_i, old_i in enumerate(order_b, 1):
        sid = f"B{new_i:02d}"
        item = task_b_raw[int(old_i)]
        task_b_blind.append({
            "sequence_id": sid,
            "face_crops": item["face_crops"],
        })
        task_b_key[sid] = item["_key"]

    blind_pack = {
        "pack_id": f"{RUN_ID}_human_pack_v1",
        "run_id": RUN_ID,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "randomization_seed": PACK_SEED,
        "recommended_raters": 2,
        "crops_dir": str(crops_dir),
        "task_a_instructions": (
            "Task A — Same-Person Matching (4-AFC, chance = 25%): Look at the PROBE face on the left, "
            "then select which ONE of the four candidate faces (A, B, C, or D) from other scenes is the SAME person."
        ),
        "task_b_instructions": (
            "Task B — Cross-Scene Identity Consistency Rating (1 to 5): Look at the sequence of faces shown "
            "across different scenes and rate how consistently they look like the SAME person "
            "(1 = Completely different people, 2 = Mostly different, 3 = Ambiguous/somewhat similar, "
            "4 = Same person with minor drift, 5 = Clearly the exact same person)."
        ),
        "task_a_items": task_a_blind,
        "task_b_items": task_b_blind,
    }

    sealed_key = {
        "pack_id": f"{RUN_ID}_human_pack_v1",
        "sealed_warning": "SEALED ANSWER KEY — MUST NEVER BE READ BY THE STREAMLIT UI.",
        "randomization_seed": PACK_SEED,
        "task_a_key": task_a_key,
        "task_b_key": task_b_key,
    }

    blind_path = pack_dir / "rating_pack_blind.json"
    key_path = pack_dir / "SEALED_ANSWER_KEY.json"
    with open(blind_path, "w", encoding="utf-8") as f:
        json.dump(blind_pack, f, indent=2)
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(sealed_key, f, indent=2)

    # Write small non-secret manifest into results/eval3/
    pack_manifest = {
        "pack_id": blind_pack["pack_id"],
        "run_id": RUN_ID,
        "created_at_utc": blind_pack["created_at_utc"],
        "randomization_seed": PACK_SEED,
        "recommended_raters": 2,
        "n_task_a_trials": len(task_a_blind),
        "task_a_chance_level": 0.25,
        "n_task_b_sequences": len(task_b_blind),
        "n_blind_crops": crop_counter,
        "n_face_strips_built": strips_built,
        "blind_pack_path": str(blind_path),
        "blind_pack_sha256": sha256_file(blind_path),
        "sealed_key_path": str(key_path),
        "sealed_key_sha256": sha256_file(key_path),
        "ratings_output_dir": "results/eval3/human_ratings",
    }
    manifest_out = PROJECT_ROOT / "results" / "eval3" / "human_rating_pack_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(pack_manifest, f, indent=2)

    print(f"[B4 PACK BUILT] Task A: {len(task_a_blind)} trials (4-AFC, chance=0.25) | Task B: {len(task_b_blind)} sequences (1-5)")
    print(f"  Blind crops saved: {crop_counter} | Face strips built: {strips_built}")
    print(f"  Blind pack: {blind_path} (SHA256: {pack_manifest['blind_pack_sha256'][:16]}...)")
    print(f"  Sealed key: {key_path} (SHA256: {pack_manifest['sealed_key_sha256'][:16]}...)")
    print(f"  Repo manifest: {manifest_out}")


if __name__ == "__main__":
    main()
