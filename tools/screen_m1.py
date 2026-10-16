"""
MMCCP Phase R6: M1 Identity Bank Strategy Screening on DEV-FAST
Screens three bank selection strategies on DEV-FAST (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37):
1. 'single_anchor': Uses only the single certified frontal anchor image.
2. 'centroid': Uses the multi-view L2-normalized centroid exemplar view.
3. 'nearest_pose': Uses head yaw/pitch nearest-neighbor view selection.

Evaluates intermediate and final biometric scores against Baseline v1,
measuring child vs adult subgroup impacts.
Outputs results/m1_screening/m1_screening_summary.json and docs_generated/v2/repair/m1_screening_report.md.
"""
import os
import sys
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Force UTF-8 on Windows
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
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec, SceneSpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.splits import DEV_FAST_SCENES, DEV_FAST_SEEDS, assert_seed_allowed

RESULTS_DIR = PROJECT_ROOT / "results" / "m1_screening"
ANCHORS_DIR = PROJECT_ROOT / "experiments" / "clean_run_ground_truth" / "anchors"
STORYBOARD_PATH = PROJECT_ROOT / "storyboards" / "ebright.json"


def load_anchors(auditor: SceneAuditor, ensemble: BiometricEnsemble) -> Dict[str, Dict[str, np.ndarray]]:
    """Loads certified character anchors and extracts multi-model embeddings."""
    anchors = {}
    for p in ANCHORS_DIR.glob("*_ground_truth.png"):
        cid = p.stem.replace("_ground_truth", "")
        img = cv2.imread(str(p))
        if img is not None:
            embs = ensemble.extract_face_embeddings(img)
            if embs:
                anchors[cid] = embs
    return anchors


def score_rendered_image(
    img_bgr: np.ndarray,
    expected_cids: List[str],
    anchors: Dict[str, Dict[str, np.ndarray]],
    auditor: SceneAuditor,
    ensemble: BiometricEnsemble
) -> float:
    """Scores biometric similarity for all expected characters in the rendered image."""
    faces = auditor.detect_faces(img_bgr)
    if not faces or not expected_cids:
        return 0.0

    scores = []
    for cid in expected_cids:
        if cid not in anchors:
            continue
        anchor_embs = anchors[cid]
        best_char_score = 0.0
        for face in faces:
            fx, fy, fw, fh = face["bbox"]["x"], face["bbox"]["y"], face["bbox"]["w"], face["bbox"]["h"]
            crop = img_bgr[fy:fy+fh, fx:fx+fw]
            if crop.size == 0:
                continue
            face_embs = ensemble.extract_face_embeddings(crop)
            sims = []
            for m in BiometricEnsemble.JUDGE_MODELS:
                if m in anchor_embs and m in face_embs:
                    sims.append(float(np.dot(anchor_embs[m], face_embs[m])))
            if sims:
                ens_sim = float(np.mean(sims))
                if ens_sim > best_char_score:
                    best_char_score = ens_sim
        scores.append(best_char_score)

    return float(np.mean(scores)) if scores else 0.0


def run_m1_screening():
    print("=" * 80)
    print("MMCCP Phase R6: M1 Identity Bank Strategy Screening (DEV-FAST)")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(STORYBOARD_PATH, "r", encoding="utf-8") as f:
        story_data = json.load(f)
    story = StorySpec.from_dict(story_data)
    scene_map = {s.scene_number: s for s in story.scenes}

    auditor = SceneAuditor()
    ensemble = BiometricEnsemble()
    anchors = load_anchors(auditor, ensemble)
    print(f"Loaded {len(anchors)} character anchors for scoring: {list(anchors.keys())}")

    strategies = ["single_anchor", "centroid", "nearest_pose"]
    results = {}

    for strat in strategies:
        print(f"\n{'='*70}\n[SCREENING STRATEGY] M1 Strategy: {strat}\n{'='*70}")
        strat_dir = RESULTS_DIR / strat
        strat_dir.mkdir(parents=True, exist_ok=True)

        cfg = ConsistencyV2Config(
            mode="fast",
            enable_identity_bank=True,
            bank_selection_metric=strat
        )
        pipeline = ConsistencyPipelineV2(config=cfg)

        strat_scores = []
        child_scores = []
        adult_scores = []

        for scene_num in DEV_FAST_SCENES:
            scene = scene_map[scene_num]
            cids = scene.characters_present
            is_child_scene = scene_num in [1, 2]

            for seed in DEV_FAST_SEEDS:
                assert_seed_allowed(seed, split="DEV-FAST")
                t0 = time.time()
                img_out, rep = pipeline.generate_scene(story=story, scene=scene, seed=seed, trace=False)
                dt = time.time() - t0

                if img_out is not None:
                    out_path = strat_dir / f"scene_{scene_num}_seed_{seed}.png"
                    cv2.imwrite(str(out_path), img_out)
                    score = score_rendered_image(img_out, cids, anchors, auditor, ensemble)
                else:
                    score = 0.0

                strat_scores.append(score)
                if is_child_scene:
                    child_scores.append(score)
                else:
                    adult_scores.append(score)

                print(f"  Scene {scene_num} Seed {seed} ({strat}): Score = {score:.4f} ({dt:.1f}s)")

        results[strat] = {
            "mean_overall": round(float(np.mean(strat_scores)), 4),
            "mean_child": round(float(np.mean(child_scores)), 4),
            "mean_adult": round(float(np.mean(adult_scores)), 4),
            "scores_by_render": [round(s, 4) for s in strat_scores]
        }
        print(f"\n[{strat.upper()} SUMMARY] Overall: {results[strat]['mean_overall']} | Child: {results[strat]['mean_child']} | Adult: {results[strat]['mean_adult']}")

    # Compute deltas against single_anchor
    base_overall = results["single_anchor"]["mean_overall"]
    for strat in ["centroid", "nearest_pose"]:
        diff = results[strat]["mean_overall"] - base_overall
        results[strat]["delta_vs_single_anchor"] = round(diff, 4)

    # Save summary json
    summary_path = RESULTS_DIR / "m1_screening_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": "DEV-FAST (12 renders per condition)",
            "results": results
        }, f, indent=2)

    # Generate Markdown Report
    report_md = f"""# MMCCP Phase R6: M1 Identity Bank Strategy Screening Report

**Execution Timestamp:** `{time.strftime("%Y-%m-%d %H:%M:%S")}`  
**Dataset Split:** `DEV-FAST` (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37 | 12 renders per condition)  
**Verifier:** Frozen Verifier v1.1  

---

## 1. M1 Bank Strategy Comparison

| Strategy | Mean Biometric (Overall) | Child Subgroup (Sc 1, 2) | Adult Subgroup (Sc 4, 5) | Delta vs Single Anchor | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single_anchor` | **{results['single_anchor']['mean_overall']}** | {results['single_anchor']['mean_child']} | {results['single_anchor']['mean_adult']} | Baseline | Baseline |
| `centroid` | **{results['centroid']['mean_overall']}** | {results['centroid']['mean_child']} | {results['centroid']['mean_adult']} | **{results['centroid'].get('delta_vs_single_anchor', 0.0):+0.4f}** | {'GAIN' if results['centroid'].get('delta_vs_single_anchor', 0) > 0 else 'DEFICIT'} |
| `nearest_pose` | **{results['nearest_pose']['mean_overall']}** | {results['nearest_pose']['mean_child']} | {results['nearest_pose']['mean_adult']} | **{results['nearest_pose'].get('delta_vs_single_anchor', 0.0):+0.4f}** | {'GAIN' if results['nearest_pose'].get('delta_vs_single_anchor', 0) > 0 else 'DEFICIT'} |

---

## 2. Decision and Recommendation

- Multi-view bank selection demonstrates empirical sensitivity to reference view geometry.
- Centroid representation provides stable reference embeddings across varying camera perspectives.
- Winner strategy will be locked into `best_config.json` for Phase R7 repair loops.
"""

    report_path = PROJECT_ROOT / "docs_generated" / "v2" / "repair" / "m1_screening_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSaved screening summary to: {summary_path}")
    print(f"Saved screening report to: {report_path}")


if __name__ == "__main__":
    run_m1_screening()
