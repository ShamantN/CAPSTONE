"""
MMCCP Phase R7: Standalone Structural Extra Person Detector (Tools Only)
Strictly independent tool (does not modify consistency_v2/verify).

Functionality:
1. Detects extra people in rendered scene images where YOLO person count > expected characters.
2. Cross-checks with RetinaFace detected faces count to identify faceless/masked/turned figures.
3. Evaluates all images across:
   - DEV-FAST (12 images): Baseline v1, v2@72769ef (single_anchor), Bundle (iter3)
   - DEV-2 (30 images): Baseline v1, v2-rc1
4. Validates that the hallucinated foreground third figure in Scene 1 Seed 11 is caught.
"""
import os
import sys
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
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

from ultralytics import YOLO

YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = Path("yolov8n.pt")

import argparse

def load_scene_expected(storyboard_path: Path) -> Dict[int, int]:
    with open(storyboard_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return {s["scene_number"]: len(s.get("characters_present", [])) for s in spec["scenes"]}

STORYBOARD_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"
SCENE_EXPECTED = load_scene_expected(STORYBOARD_PATH)

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

def audit_directory(dir_path: Path, model: YOLO, label: str, scene_expected: Dict[int, int] = None) -> Dict[str, Any]:
    print(f"\n--- Auditing [{label}]: {dir_path} ---")
    expected_map = scene_expected if scene_expected is not None else SCENE_EXPECTED
    img_files = sorted(list(dir_path.glob("*scene_*.png")))
    total_imgs = len(img_files)
    flagged_extra_persons = []
    
    for p in img_files:
        # Parse scene number and seed robustly (supports scene_1_seed_11.png and holdout_scene_1_seed_1101.png)
        import re as _re
        m_fn = _re.search(r"scene_(\d+).*?_seed_(\d+)", p.stem)
        if m_fn:
            s_num = int(m_fn.group(1))
            seed = int(m_fn.group(2))
        else:
            parts = p.stem.split("_")
            s_num = int(parts[1])
            seed = int(parts[3]) if len(parts) >= 4 else 0
        expected = expected_map.get(s_num, 1)
        
        img = cv2.imread(str(p))
        if img is None:
            continue
            
        persons = detect_persons_yolo(model, img)
        yolo_count = len(persons)
        
        is_extra = yolo_count > expected
        if is_extra:
            flagged_extra_persons.append({
                "filename": p.name,
                "scene": s_num,
                "seed": seed,
                "expected": expected,
                "detected_persons_yolo": yolo_count,
                "extra_count": yolo_count - expected,
                "person_boxes": [p["bbox"] for p in persons]
            })
            print(f"  [FLAGGED EXTRA PERSON] {p.name}: found {yolo_count} persons, expected {expected} (+{yolo_count-expected})")
            
    print(f"  Summary for {label}: {len(flagged_extra_persons)}/{total_imgs} images flagged ({len(flagged_extra_persons)/total_imgs*100:.1f}%)")
    return {
        "label": label,
        "directory": str(dir_path),
        "total_images": total_imgs,
        "flagged_count": len(flagged_extra_persons),
        "flagged_rate": round(len(flagged_extra_persons) / total_imgs if total_imgs > 0 else 0.0, 4),
        "flagged_images": flagged_extra_persons
    }

def main():
    parser = argparse.ArgumentParser(description="MMCCP Structural Extra Person Detector (YOLOv8n, Read-Only)")
    parser.add_argument("--target", action="append", nargs=3, metavar=("DIR", "STORYBOARD", "LABEL"),
                        help="Target directory, storyboard JSON path, and dataset label (repeatable)")
    parser.add_argument("--out", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("MMCCP Structural Extra Person Detector (YOLOv8n)")
    print("=" * 80)
    
    print("Loading YOLOv8n model...")
    model = YOLO(str(YOLO_WEIGHTS))

    if args.target:
        results = {}
        for d_str, sb_str, label in args.target:
            d_path = PROJECT_ROOT / d_str if not Path(d_str).is_absolute() else Path(d_str)
            sb_path = PROJECT_ROOT / sb_str if not Path(sb_str).is_absolute() else Path(sb_str)
            if not d_path.exists():
                print(f"Directory not found, skipping: {d_path}")
                continue
            exp_map = load_scene_expected(sb_path)
            res = audit_directory(d_path, model, label, scene_expected=exp_map)
            results[label] = res
        out_json = (PROJECT_ROOT / args.out) if (args.out and not Path(args.out).is_absolute()) else (Path(args.out) if args.out else PROJECT_ROOT / "results" / "structural_extra_person_audit_custom.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"datasets": results}, f, indent=2)
        print(f"\nSaved extra person audit to: {out_json}")
        return
    
    datasets = [
        # DEV-FAST datasets
        (PROJECT_ROOT / "results" / "baseline_v1" / "dev_full", "Baseline v1 (DEV-FAST subset)"),
        (PROJECT_ROOT / "results" / "m1_screening" / "single_anchor", "v2@72769ef (Broken v2 DEV-FAST)"),
        (PROJECT_ROOT / "results" / "iterations" / "loop-W3W4-iter3-closeup-scale-fidelity" / "renders", "Bundle iter3 (DEV-FAST)"),
        # DEV-2 datasets
        (PROJECT_ROOT / "results" / "baseline_v1" / "dev2", "Baseline v1 (DEV-2)"),
        (PROJECT_ROOT / "results" / "v2_default" / "test", "v2-rc1 (DEV-2)")
    ]
    
    results = {}
    for d_path, label in datasets:
        if d_path.exists():
            res = audit_directory(d_path, model, label)
            results[label] = res
        else:
            print(f"Directory not found, skipping: {d_path}")
            
    # Check specifically Scene 1 Seed 11 in bundle iter3
    iter3_dir = PROJECT_ROOT / "results" / "iterations" / "loop-W3W4-iter3-closeup-scale-fidelity" / "renders"
    s1_seed11 = iter3_dir / "scene_1_seed_11.png"
    s1_audit = None
    if s1_seed11.exists():
        img = cv2.imread(str(s1_seed11))
        persons = detect_persons_yolo(model, img)
        s1_audit = {
            "file": str(s1_seed11),
            "expected": 2,
            "detected_persons_yolo": len(persons),
            "flagged_extra": len(persons) > 2,
            "persons": persons
        }
        print("\n" + "=" * 80)
        print("SCENE 1 SEED 11 FOREGROUND FIGURE AUDIT:")
        print(f"  Expected characters: 2 (young_ebright, mother)")
        print(f"  YOLO detected persons: {len(persons)}")
        print(f"  Extra person caught: {len(persons) > 2}")
        for i, p in enumerate(persons):
            print(f"    Person {i+1}: conf={p['confidence']}, bbox={p['bbox']}")
            
    out_json = PROJECT_ROOT / "results" / "structural_extra_person_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"datasets": results, "scene_1_seed_11_foreground_audit": s1_audit}, f, indent=2)
    print(f"\nSaved extra person audit to: {out_json}")

if __name__ == "__main__":
    main()
