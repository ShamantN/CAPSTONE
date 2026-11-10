"""
MMCCP Showcase Scene Runner, Auditor, and Montage Builder
Executes Best-of-3 candidate generation for Ebright storyboard scenes 1-6,
applies the Selection Hierarchy, audits with Verifier v1.2,
generates per-scene contact sheets, final 2x3 montage, and scorecard CSV.
"""
import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Ensure DLL paths for torch/deepface on Windows
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
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec, SceneSpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from ultralytics import YOLO

YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = Path("yolov8n.pt")


def detect_persons_yolo(model: YOLO, img_bgr: np.ndarray, conf_thresh: float = 0.40) -> List[Dict[str, Any]]:
    """Runs YOLOv8 person detection (class 0)."""
    results = model(img_bgr, conf=conf_thresh, verbose=False)
    persons = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                persons.append({
                    "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "confidence": round(conf, 3)
                })
    return persons


def initialize_demo_bank(anchors_dir: Path, bank_dir: Path) -> IdentityBankManager:
    """Populates an IdentityBankManager from the generated demo anchors."""
    bank_dir.mkdir(parents=True, exist_ok=True)
    mgr = IdentityBankManager(bank_root=bank_dir)

    characters = ["young_ebright", "teen_ebright", "college_ebright", "mother", "james_wong"]
    for cid in characters:
        anchor_file = anchors_dir / f"{cid}.png"
        if anchor_file.exists():
            print(f"  Registering anchor into demo bank: {cid} from {anchor_file.name}")
            mgr.initialize_character_bank(cid, [anchor_file])
        else:
            print(f"  [WARNING] Anchor file missing for {cid}: {anchor_file}")

    mgr.load_all_banks()
    return mgr


def build_contact_sheet(
    candidates_info: List[Dict[str, Any]],
    scene_num: int,
    out_path: Path
):
    """Creates a high-resolution horizontal contact sheet for candidate review."""
    panels = []
    panel_w = 480
    panel_h = 270  # 16:9 aspect ratio

    for cand in candidates_info:
        img_p = Path(cand["image_path"])
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        thumb = cv2.resize(img, (panel_w, panel_h))
        # Add header banner with metadata
        banner_h = 70
        canvas = np.zeros((panel_h + banner_h, panel_w, 3), dtype=np.uint8)
        canvas[banner_h:, :] = thumb

        # Banner background
        is_winner = cand.get("is_winner", False)
        bg_color = (0, 100, 0) if is_winner else (40, 40, 40)
        cv2.rectangle(canvas, (0, 0), (panel_w, banner_h), bg_color, -1)

        # Status text
        k = cand["candidate_index"]
        seed = cand["seed"]
        faces = cand.get("face_count", 0)
        yolo_p = cand.get("yolo_person_count", 0)
        ens_score = cand.get("ensemble_score", 0.0)
        status = "WINNER (SELECTED)" if is_winner else cand.get("status", "CANDIDATE")

        cv2.putText(canvas, f"Cand #{k} | Seed {seed} | {status}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, f"Faces: {faces} | YOLO: {yolo_p} | Ensemble: {ens_score:.4f}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        reasons = cand.get("notes", "")
        if reasons:
            cv2.putText(canvas, f"Notes: {reasons[:45]}", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

        panels.append(canvas)

    if panels:
        sheet = np.hstack(panels)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), sheet)
        print(f"  [SAVED] Contact sheet -> {out_path.name}")


def build_final_montage(
    final_image_paths: List[Path],
    out_path: Path
):
    """Assembles the 2x3 final montage of the 6 selected storyboard scenes."""
    panels = []
    panel_w = 640
    panel_h = 360  # 16:9

    for s_idx, p in enumerate(final_image_paths, start=1):
        if p.exists():
            img = cv2.imread(str(p))
            thumb = cv2.resize(img, (panel_w, panel_h))
        else:
            thumb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(thumb, f"Scene {s_idx} Missing", (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Label scene number in bottom corner
        cv2.putText(thumb, f"Scene {s_idx}", (15, panel_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(thumb)

    while len(panels) < 6:
        blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panels.append(blank)

    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    montage = np.vstack([row1, row2])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage)
    print(f"  [SAVED] Final 2x3 montage -> {out_path.name} ({montage.shape[1]}x{montage.shape[0]})")


def run_showcase_for_set(
    set_name: str,
    out_dir: Path,
    anchors_dir: Path,
    settings_json_path: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cands_dir = out_dir / "candidates"
    final_dir = out_dir / "final"
    sheets_dir = out_dir / "contact_sheets"
    bank_dir = out_dir / "banks"

    cands_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load config
    cfg_data = {}
    if settings_json_path.exists():
        with open(settings_json_path, "r", encoding="utf-8") as f:
            raw_s = json.load(f)
            cfg_data = raw_s.get("config", raw_s)
    config = ConsistencyV2Config.from_dict(cfg_data)
    # Ensure crop-upscale swap is active as locked
    config.enable_crop_upscale_swap = True

    # 2. Build demo bank
    print(f"\n[PHASE 1] Initializing Demo Bank from Anchors...")
    bank_mgr = initialize_demo_bank(anchors_dir, bank_dir)

    # 3. Instantiate Pipeline
    print(f"[PHASE 2] Initializing ConsistencyPipelineV2 (Config Hash: {config.compute_hash()})...")
    pipeline = ConsistencyPipelineV2(config=config)
    pipeline.bank_manager = bank_mgr
    pipeline.memory_manager.bank_manager = bank_mgr

    # Enforce Forge memory parameters
    try:
        pipeline.client.set_options({"forge_inference_memory": 1024.0, "xl_GPU_MB": 3200.0})
    except Exception as e:
        print(f"  Warning setting Forge options: {e}")

    # 4. Load Storyboard
    story_path = PROJECT_ROOT / "storyboards" / "ebright.json"
    with open(story_path, "r", encoding="utf-8") as f:
        story_dict = json.load(f)
    story = StorySpec.from_dict(story_dict)

    # 5. Define Seeds
    base_seed = 4200 if set_name == "A" else 5200
    scene_seed_map = {
        s: [base_seed + s * 10 + 1, base_seed + s * 10 + 2, base_seed + s * 10 + 3]
        for s in range(1, 7)
    }

    print(f"Seed Map for Set {set_name}: {scene_seed_map}")

    # Auditor & Models
    auditor = SceneAuditor()
    ensemble = BiometricEnsemble()
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    final_picks: List[Path] = []
    scorecard_rows: List[Dict[str, Any]] = []
    all_observations: List[str] = [
        f"# Visual Observations & Audit Log: Set {set_name}\n\n",
        f"- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- **Base Config Hash:** {config.compute_hash()}\n",
        f"- **Pipeline:** MMCCP v2.0 locked bundle (`crop_upscale_swap=True`)\n\n"
    ]

    # 6. Render & Audit Loop
    for scene in story.scenes:
        s_num = scene.scene_number
        s_id = scene.scene_id
        expected_chars = scene.characters_present or []
        expected_face_count = len(expected_chars)

        print(f"\n=======================================================")
        print(f"--- SCENE {s_num}: '{scene.title}' ({scene.shot_type}) ---")
        print(f"Expected characters ({expected_face_count}): {expected_chars}")
        print(f"=======================================================")

        scene_cands_info: List[Dict[str, Any]] = []
        seeds = scene_seed_map[s_num]

        for k_idx, seed in enumerate(seeds, start=1):
            cand_img_path = cands_dir / f"scene_{s_num}_cand{k_idx}.png"
            cand_meta_path = cands_dir / f"scene_{s_num}_cand{k_idx}.json"

            t0 = time.time()
            print(f"\nRendering Candidate {k_idx}/3 (Seed {seed})...")
            img_bgr, meta = pipeline.generate_scene(story, scene, seed=seed)
            render_dt = time.time() - t0

            if img_bgr is None:
                print(f"  [ERROR] Render failed for seed {seed}")
                continue

            cv2.imwrite(str(cand_img_path), img_bgr)

            # Audit
            faces = auditor.detect_faces(img_bgr)
            face_count = len(faces)

            # Extra person YOLO audit
            yolo_persons = detect_persons_yolo(yolo_model, img_bgr, conf_thresh=0.40)
            yolo_person_count = len(yolo_persons)
            extra_persons = (yolo_person_count > expected_face_count)

            # Biometric scoring against newly minted anchors
            sims = []
            char_scores = {}
            for cid in expected_chars:
                ref = bank_mgr.select_best_reference(cid, strategy="single_anchor")
                if ref and "path" in ref and Path(ref["path"]).exists():
                    ref_bgr = cv2.imread(ref["path"])
                    if ref_bgr is not None:
                        embs_ref = ensemble.extract_face_embeddings(ref_bgr)
                        embs_scene = ensemble.extract_face_embeddings(img_bgr)
                        # Cosine similarity for matching models
                        mod_sims = []
                        for m in BiometricEnsemble.JUDGE_MODELS:
                            if m in embs_ref and m in embs_scene:
                                mod_sims.append(float(np.dot(embs_ref[m], embs_scene[m])))
                        if mod_sims:
                            c_score = float(np.mean(mod_sims))
                            char_scores[cid] = round(c_score, 4)
                            sims.append(c_score)

            mean_ens_score = round(float(np.mean(sims)), 4) if sims else 0.0

            # Validity evaluation
            has_correct_face_count = (face_count == expected_face_count)
            no_extra_yolo = not extra_persons
            is_valid = bool(has_correct_face_count and no_extra_yolo)

            status = "VALID" if is_valid else "DEFECTIVE"
            notes = []
            if not has_correct_face_count:
                notes.append(f"Faces {face_count}!={expected_face_count}")
            if extra_persons:
                notes.append(f"YOLO extra person ({yolo_person_count})")

            cand_record = {
                "set": set_name,
                "scene_number": s_num,
                "scene_title": scene.title,
                "candidate_index": k_idx,
                "seed": seed,
                "image_path": str(cand_img_path),
                "render_time_sec": round(render_dt, 2),
                "expected_faces": expected_face_count,
                "face_count": face_count,
                "yolo_person_count": yolo_person_count,
                "extra_person_detected": extra_persons,
                "ensemble_score": mean_ens_score,
                "character_scores": char_scores,
                "is_valid": is_valid,
                "status": status,
                "notes": "; ".join(notes) if notes else "Clean",
                "is_winner": False
            }

            with open(cand_meta_path, "w", encoding="utf-8") as f:
                json.dump(cand_record, f, indent=2)

            scene_cands_info.append(cand_record)
            scorecard_rows.append(cand_record)
            print(f"  Result: valid={is_valid} faces={face_count} yolo={yolo_person_count} ens={mean_ens_score:.4f} notes={cand_record['notes']} (time: {render_dt:.1f}s)")

        # 7. Apply Selection Hierarchy
        valid_cands = [c for c in scene_cands_info if c["is_valid"]]
        if valid_cands:
            winner = max(valid_cands, key=lambda c: c["ensemble_score"])
        else:
            print(f"  [NOTICE] No strictly valid candidates for Scene {s_num}. Selecting best available.")
            winner = max(scene_cands_info, key=lambda c: (c["face_count"] == expected_face_count, c["ensemble_score"]))

        winner["is_winner"] = True
        winner_path = final_dir / f"scene_{s_num}.png"
        shutil.copy(winner["image_path"], str(winner_path))
        final_picks.append(winner_path)

        print(f"\n--> WINNER Scene {s_num}: Candidate #{winner['candidate_index']} (Seed {winner['seed']}, Score {winner['ensemble_score']}) -> {winner_path.name}")

        # Record visual observation template
        all_observations.append(f"### Scene {s_num}: {scene.title} ({scene.shot_type})\n")
        all_observations.append(f"- **Selected Candidate:** #{winner['candidate_index']} (Seed `{winner['seed']}`)\n")
        all_observations.append(f"- **Biometric Ensemble Score:** `{winner['ensemble_score']:.4f}`\n")
        all_observations.append(f"- **Detected Faces:** {winner['face_count']} (Expected: {expected_face_count})\n")
        all_observations.append(f"- **YOLO Persons:** {winner['yolo_person_count']}\n")
        all_observations.append(f"- **Rejection Log:**\n")
        for c in scene_cands_info:
            if not c["is_winner"]:
                all_observations.append(f"  - Cand #{c['candidate_index']} (Seed {c['seed']}): {c['status']} — {c['notes']} (Score: {c['ensemble_score']:.4f})\n")
        all_observations.append("\n")

        # 8. Build Scene Contact Sheet
        sheet_path = sheets_dir / f"scene_{s_num}_contact_sheet.png"
        build_contact_sheet(scene_cands_info, s_num, sheet_path)

    # 9. Build 2x3 Final Montage
    print("\n[PHASE 3] Building 2x3 Final Montage...")
    montage_path = out_dir / "montage_final.png"
    build_final_montage(final_picks, montage_path)

    # 10. Write Scorecard CSV
    csv_path = out_dir / "scorecard.csv"
    import csv
    fieldnames = [
        "set", "scene_number", "scene_title", "candidate_index", "seed",
        "expected_faces", "face_count", "yolo_person_count", "extra_person_detected",
        "ensemble_score", "is_valid", "is_winner", "render_time_sec", "notes", "image_path"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in scorecard_rows:
            writer.writerow(r)
    print(f"  [SAVED] Scorecard CSV -> {csv_path.name} ({len(scorecard_rows)} records)")

    # 11. Write Observations MD
    obs_path = out_dir / "observations.md"
    with open(obs_path, "w", encoding="utf-8") as f:
        f.writelines(all_observations)
    print(f"  [SAVED] Observations MD -> {obs_path.name}")

    print(f"\n=======================================================")
    print(f"[COMPLETE] Set {set_name} Execution Finished Successfully!")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run showcase scenes and candidates.")
    parser.add_argument("--set-name", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--anchors-dir", type=str, required=True)
    parser.add_argument("--settings", type=str, default=r"D:\5TH SEM\HACKATHON_demo_outputs\20261027_1123\settings.json")
    args = parser.parse_args()

    run_showcase_for_set(
        set_name=args.set_name,
        out_dir=Path(args.out_dir),
        anchors_dir=Path(args.anchors_dir),
        settings_json_path=Path(args.settings)
    )
