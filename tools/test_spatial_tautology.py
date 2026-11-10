"""
MMCCP Phase R7: Spatial Check Tautology Test (Tools Only)
Strictly independent test (freeze untouched).

Objective:
Determines whether `spatial_pass` can ever fail given zone-based assignment.
Performs an empirical swap experiment:
1. Loads Scene 1 Seed 11 (two-shot: young_ebright expected left, mother expected right).
2. Runs frozen Verifier v1.2 on original image: records face assignment and spatial_pass.
3. Deliberately swaps the two character faces:
   - Slices face crop A (left) and face crop B (right)
   - Pastes face crop B into A's position (left), and face crop A into B's position (right).
   - Saves `results/swapped_spatial_test_scene_1.png`.
4. Runs frozen Verifier v1.2 on swapped image: records face assignment, spatial_pass, and biometrics.
5. Evaluates whether spatial_pass detected the position swap, or tautologically passed.
"""
import os
import sys
import io
import json
from pathlib import Path
import cv2
import numpy as np

# Force UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

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
        try: os.add_dll_directory(_p)
        except: pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from consistency_v2.verify.cli import assign_faces_to_characters
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.binding_checker import BindingChecker
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble

def run_spatial_tautology_test():
    print("=" * 80)
    print("MMCCP Phase R7: Spatial Position Tautology Experiment")
    print("=" * 80)

    # 1. Source image: Scene 1 Seed 11
    # Look in bundle iter3 renders or baseline_v1 dev_full
    orig_path = PROJECT_ROOT / "results" / "iterations" / "loop-W3W4-iter3-closeup-scale-fidelity" / "renders" / "scene_1_seed_11.png"
    if not orig_path.exists():
        orig_path = PROJECT_ROOT / "results" / "baseline_v1" / "dev_full" / "scene_1_seed_11.png"

    print(f"Loading source image: {orig_path}")
    orig_img = cv2.imread(str(orig_path))
    assert orig_img is not None, f"Failed to load {orig_path}"
    h_img, w_img = orig_img.shape[:2]

    auditor = SceneAuditor()
    binder = BindingChecker()
    verifier = BiometricEnsemble()

    expected_chars = ["young_ebright", "mother"]
    character_positions = {
        "young_ebright": "sitting on the left",
        "mother": "sitting on the right"
    }

    # 2. Evaluate original image
    print("\n--- [EVALUATING ORIGINAL IMAGE] ---")
    orig_faces = auditor.detect_faces(orig_img)
    print(f"Detected {len(orig_faces)} faces in original image.")
    for i, f in enumerate(orig_faces):
        print(f"  Face {i}: bbox={f['bbox']}, center_x={f['bbox']['x'] + f['bbox']['w']/2:.1f}")

    orig_assigned = assign_faces_to_characters(
        detected_faces=orig_faces,
        expected_chars=expected_chars,
        character_positions=character_positions,
        image_width=w_img
    )

    orig_results = {}
    for cid in expected_chars:
        tup = orig_assigned.get(cid)
        if tup:
            idx, face_item = tup
            sp_pass, obs_zone, r_x = binder.verify_spatial_position(
                face_bbox=face_item["bbox"],
                image_width=w_img,
                expected_position_str=character_positions[cid]
            )
            orig_results[cid] = {
                "face_idx": idx,
                "bbox": face_item["bbox"],
                "center_ratio_rx": r_x,
                "observed_zone": obs_zone,
                "spatial_pass": sp_pass
            }
            print(f"  Character '{cid}': assigned Face {idx} at rx={r_x} (zone={obs_zone}) -> spatial_pass={sp_pass}")

    # 3. Create Deliberately Swapped Image
    # Swap face crops between the two detected faces
    print("\n--- [CREATING DELIBERATELY SWAPPED IMAGE] ---")
    # Identify left face and right face
    faces_sorted = sorted(orig_faces, key=lambda f: f["bbox"]["x"])
    left_face = faces_sorted[0]
    right_face = faces_sorted[1]

    lx, ly, lw, lh = left_face["bbox"]["x"], left_face["bbox"]["y"], left_face["bbox"]["w"], left_face["bbox"]["h"]
    rx, ry, rw, rh = right_face["bbox"]["x"], right_face["bbox"]["y"], right_face["bbox"]["w"], right_face["bbox"]["h"]

    left_crop = orig_img[ly:ly+lh, lx:lx+lw].copy()
    right_crop = orig_img[ry:ry+rh, rx:rx+rw].copy()

    swapped_img = orig_img.copy()
    # Resize right crop into left bbox, and left crop into right bbox
    swapped_left = cv2.resize(right_crop, (lw, lh), interpolation=cv2.INTER_LANCZOS4)
    swapped_right = cv2.resize(left_crop, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    swapped_img[ly:ly+lh, lx:lx+lw] = swapped_left
    swapped_img[ry:ry+rh, rx:rx+rw] = swapped_right

    out_swapped_path = PROJECT_ROOT / "results" / "swapped_spatial_test_scene_1.png"
    cv2.imwrite(str(out_swapped_path), swapped_img)
    print(f"Saved swapped image: {out_swapped_path}")
    print("  Left position now contains: MOTHER'S FACE")
    print("  Right position now contains: EBRIGHT'S FACE")

    # 4. Evaluate Swapped Image with Verifier v1.2
    print("\n--- [EVALUATING SWAPPED IMAGE WITH VERIFIER v1.2] ---")
    swapped_faces = auditor.detect_faces(swapped_img)
    print(f"Detected {len(swapped_faces)} faces in swapped image.")

    swapped_assigned = assign_faces_to_characters(
        detected_faces=swapped_faces,
        expected_chars=expected_chars,
        character_positions=character_positions,
        image_width=w_img
    )

    swapped_results = {}
    for cid in expected_chars:
        tup = swapped_assigned.get(cid)
        if tup:
            idx, face_item = tup
            sp_pass, obs_zone, r_x = binder.verify_spatial_position(
                face_bbox=face_item["bbox"],
                image_width=w_img,
                expected_position_str=character_positions[cid]
            )
            swapped_results[cid] = {
                "face_idx": idx,
                "bbox": face_item["bbox"],
                "center_ratio_rx": r_x,
                "observed_zone": obs_zone,
                "spatial_pass": sp_pass
            }
            print(f"  Character '{cid}': assigned Face {idx} at rx={r_x} (zone={obs_zone}) -> spatial_pass={sp_pass}")

    # 5. Scientific Conclusion
    print("\n" + "=" * 80)
    print("EMPIRICAL FINDING & DIAGNOSIS:")
    both_passed_orig = all(r["spatial_pass"] for r in orig_results.values())
    both_passed_swapped = all(r["spatial_pass"] for r in swapped_results.values())
    print(f"Original image spatial_pass: {both_passed_orig}")
    print(f"Swapped image spatial_pass:  {both_passed_swapped}")

    is_tautological = bool(both_passed_swapped and both_passed_orig)
    if is_tautological:
        conclusion = (
            "PROVEN TAUTOLOGICAL: Even when characters' faces were deliberately swapped between "
            "left and right positions, spatial_pass still reported 100% PASS for both characters! "
            "Reason: assign_faces_to_characters assigns faces purely based on horizontal zone score (rx), "
            "assigning whatever face is on the left to the left character, and whatever face is on the right "
            "to the right character. verify_spatial_position then checks if the assigned face is in that zone, "
            "which is guaranteed by definition. Therefore, spatial_pass CANNOT fail based on identity."
        )
    else:
        conclusion = "Spatial pass detected the swapped positions."

    print(f"\n{conclusion}")

    report = {
        "original_results": orig_results,
        "swapped_results": swapped_results,
        "tautology_confirmed": is_tautological,
        "conclusion": conclusion
    }
    out_json = PROJECT_ROOT / "results" / "spatial_tautology_experiment.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved experiment results to: {out_json}")

if __name__ == "__main__":
    run_spatial_tautology_test()
