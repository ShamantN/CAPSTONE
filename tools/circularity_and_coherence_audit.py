"""
MMCCP Tools: Circularity & Cross-Scene Coherence Audit (Read-Only Analysis)

Computes on an evaluation render pair (v1 vs v2) without modifying frozen verifier:
1. Circularity Check: Final assigned face similarity to the canonical source anchor
   measured by InsightFace (buffalo_l w600k_r50) vs the 4 DeepFace Verifier v1.2 judges
   (Facenet512, SFace, GhostFaceNet, ArcFace), including paired deltas and correlations.
2. Cross-Scene Coherence (Anchor-Free): Same-character pairwise similarity across scenes/seeds
   divided by impostor (different-character) similarity across the 4 judges, Ensemble, and InsightFace.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consistency_v2.verify.biometric_ensemble import BiometricEnsemble
from consistency_v2.verify.scene_auditor import SceneAuditor
from consistency_v2.verify.cli import assign_faces_to_characters

FORGE_PYTHON = Path(r"C:\Stable-Diffusion\system\python\python.exe")

INSIGHTFACE_WORKER_CODE = r'''
import json, sys, cv2, numpy as np
from insightface.app import FaceAnalysis

tasks = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def best_match(faces, bbox=None):
    if not faces:
        return None
    if bbox is None:
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    tx = bbox["x"] + bbox["w"] / 2.0
    ty = bbox["y"] + bbox["h"] / 2.0
    best_f = None
    best_dist = 1e18
    for f in faces:
        fx = (f.bbox[0] + f.bbox[2]) / 2.0
        fy = (f.bbox[1] + f.bbox[3]) / 2.0
        d = (fx - tx)**2 + (fy - ty)**2
        if d < best_dist:
            best_dist = d
            best_f = f
    # Accept match only if center is reasonably near (within max(w, h) * 1.25)
    max_r = max(bbox["w"], bbox["h"], 40) * 1.25
    if best_dist ** 0.5 > max_r:
        return None
    return best_f

anchor_embs = {}
for cid, apath in tasks["anchors"].items():
    img = cv2.imread(apath)
    if img is None:
        continue
    faces = app.get(img)
    f = best_match(faces, None)
    if f is not None and f.normed_embedding is not None:
        anchor_embs[cid] = f.normed_embedding.tolist()

img_faces_cache = {}
instance_embs = {}
for item in tasks["instances"]:
    key = item["key"]
    ipath = item["image_path"]
    img = cv2.imread(ipath)
    if img is None:
        continue
    if ipath not in img_faces_cache:
        img_faces_cache[ipath] = app.get(img)
    faces = img_faces_cache[ipath]
    f = best_match(faces, item["bbox"])
    if f is None:
        # Fallback: crop padded region and upscale 2x for small faces
        h_img, w_img = img.shape[:2]
        b = item["bbox"]
        pad_w, pad_h = int(b["w"] * 0.6), int(b["h"] * 0.6)
        x1, y1 = max(0, b["x"] - pad_w), max(0, b["y"] - pad_h)
        x2, y2 = min(w_img, b["x"] + b["w"] + pad_w), min(h_img, b["y"] + b["h"] + pad_h)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            up = cv2.resize(crop, (max(128, crop.shape[1]*2), max(128, crop.shape[0]*2)), interpolation=cv2.INTER_LANCZOS4)
            cf = app.get(up)
            f = best_match(cf, None)
    if f is not None and f.normed_embedding is not None:
        instance_embs[key] = f.normed_embedding.tolist()

with open(sys.argv[2], "w", encoding="utf-8") as out_f:
    json.dump({"anchors": anchor_embs, "instances": instance_embs}, out_f)
'''


def _find_render_files(render_dir: Path) -> Dict[Tuple[int, int], Path]:
    mapping = {}
    for p in sorted(render_dir.glob("*scene_*.png")):
        m = re.search(r"scene_(\d+).*?_seed_(\d+)", p.stem)
        if m:
            mapping[(int(m.group(1)), int(m.group(2)))] = p
    return mapping


def _extract_version_instances(
    version_label: str,
    render_dir: Path,
    story_spec: Dict[str, Any],
    auditor: SceneAuditor,
    ensemble: BiometricEnsemble,
    anchor_deepface_embs: Dict[str, Dict[str, np.ndarray]],
    report_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    files = _find_render_files(render_dir)
    scene_map = {s["scene_number"]: s for s in story_spec["scenes"]}
    characters = story_spec.get("characters", {})
    cached_faces_by_name: Dict[str, List[Dict[str, Any]]] = {}
    if report_path is not None and report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        for sr in rep.get("per_scene_reports", []):
            fname = sr.get("image_filename", "")
            dfaces = sr.get("scene_audit", {}).get("detected_faces", [])
            if fname and dfaces is not None:
                cached_faces_by_name[fname] = dfaces
    instances = []

    for idx_f, ((s_num, seed), img_path) in enumerate(sorted(files.items()), 1):
        scene = scene_map.get(s_num)
        if scene is None:
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]
        if img_path.name in cached_faces_by_name:
            detected = []
            for df in cached_faces_by_name[img_path.name]:
                b = df.get("bbox", {})
                x, y, bw, bh = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
                crop = img_bgr[max(0, y):min(h_img, y + bh), max(0, x):min(w_img, x + bw)]
                detected.append({
                    "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                    "confidence": float(df.get("confidence", 1.0)),
                    "crop_bgr": crop,
                    "landmarks": df.get("landmarks", {}),
                })
        else:
            detected = auditor.detect_faces(img_bgr)
        detected = sorted(detected, key=lambda f: f["bbox"]["x"])
        if idx_f % 5 == 0 or idx_f == len(files):
            print(f"  [{version_label}] {idx_f}/{len(files)} images...", flush=True)
        expected_chars = scene.get("characters_present", [])
        char_positions = scene.get("character_positions", {})
        assigned = assign_faces_to_characters(
            detected_faces=detected,
            expected_chars=expected_chars,
            character_positions=char_positions,
            image_width=w_img,
        )
        for cid in expected_chars:
            key = f"{version_label}|s{s_num}|seed{seed}|{cid}"
            age_group = characters.get(cid, {}).get("age_group", "adult")
            tup = assigned.get(cid)
            if tup is None:
                instances.append({
                    "key": key,
                    "version": version_label,
                    "scene": s_num,
                    "seed": seed,
                    "character_id": cid,
                    "age_group": age_group,
                    "image_path": str(img_path),
                    "bbox": None,
                    "detected": False,
                    "judge_anchor_sims": {m: 0.0 for m in BiometricEnsemble.JUDGE_MODELS},
                    "ensemble_anchor_sim": 0.0,
                    "deepface_embs": None,
                })
                continue
            _, f_item = tup
            crop = f_item.get("crop_bgr")
            embs = ensemble.extract_face_embeddings(crop) if crop is not None and crop.size > 0 else {}
            a_embs = anchor_deepface_embs.get(cid, {})
            j_sims = {}
            for m in BiometricEnsemble.JUDGE_MODELS:
                if m in embs and m in a_embs:
                    j_sims[m] = float(np.dot(embs[m], a_embs[m]))
                else:
                    j_sims[m] = 0.0
            ens = float(np.mean(list(j_sims.values())))
            instances.append({
                "key": key,
                "version": version_label,
                "scene": s_num,
                "seed": seed,
                "character_id": cid,
                "age_group": age_group,
                "image_path": str(img_path),
                "bbox": f_item["bbox"],
                "detected": True,
                "judge_anchor_sims": j_sims,
                "ensemble_anchor_sim": ens,
                "deepface_embs": {m: embs[m].tolist() for m in embs if m in BiometricEnsemble.JUDGE_MODELS},
            })
    return instances


def _compute_coherence(instances: List[Dict[str, Any]], models: List[str]) -> Dict[str, Any]:
    by_char: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        if inst["detected"]:
            by_char.setdefault(inst["character_id"], []).append(inst)

    chars = sorted(by_char.keys())
    genuine_all: Dict[str, List[float]] = {m: [] for m in models}
    genuine_diff_scene: Dict[str, List[float]] = {m: [] for m in models}
    impostor_all: Dict[str, List[float]] = {m: [] for m in models}

    for cid in chars:
        lst = by_char[cid]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                e1 = lst[i].get("all_embs", {})
                e2 = lst[j].get("all_embs", {})
                diff_scene = (lst[i]["scene"] != lst[j]["scene"])
                for m in models:
                    if m in e1 and m in e2:
                        sim = float(np.dot(e1[m], e2[m]))
                        genuine_all[m].append(sim)
                        if diff_scene:
                            genuine_diff_scene[m].append(sim)

    for i_c in range(len(chars)):
        for j_c in range(i_c + 1, len(chars)):
            for inst1 in by_char[chars[i_c]]:
                for inst2 in by_char[chars[j_c]]:
                    e1 = inst1.get("all_embs", {})
                    e2 = inst2.get("all_embs", {})
                    for m in models:
                        if m in e1 and m in e2:
                            sim = float(np.dot(e1[m], e2[m]))
                            impostor_all[m].append(sim)

    per_model = {}
    for m in models:
        g_vals = genuine_all[m]
        gd_vals = genuine_diff_scene[m]
        i_vals = impostor_all[m]
        g_mean = float(np.mean(g_vals)) if g_vals else 0.0
        gd_mean = float(np.mean(gd_vals)) if gd_vals else 0.0
        i_mean = float(np.mean(i_vals)) if i_vals else 0.0
        ratio = g_mean / i_mean if i_mean > 0 else 0.0
        diff_ratio = gd_mean / i_mean if i_mean > 0 else 0.0
        per_model[m] = {
            "genuine_mean": round(g_mean, 4),
            "genuine_diff_scene_mean": round(gd_mean, 4),
            "impostor_mean": round(i_mean, 4),
            "coherence_ratio": round(ratio, 4),
            "diff_scene_coherence_ratio": round(diff_ratio, 4),
            "separation_gap": round(g_mean - i_mean, 4),
            "num_genuine_pairs": len(g_vals),
            "num_genuine_diff_scene_pairs": len(gd_vals),
            "num_impostor_pairs": len(i_vals),
        }

    judge_models = BiometricEnsemble.JUDGE_MODELS
    ens_gen = float(np.mean([per_model[m]["genuine_mean"] for m in judge_models]))
    ens_gd = float(np.mean([per_model[m]["genuine_diff_scene_mean"] for m in judge_models]))
    ens_imp = float(np.mean([per_model[m]["impostor_mean"] for m in judge_models]))
    return {
        "four_judge_ensemble": {
            "genuine_mean": round(ens_gen, 4),
            "genuine_diff_scene_mean": round(ens_gd, 4),
            "impostor_mean": round(ens_imp, 4),
            "coherence_ratio": round(ens_gen / ens_imp, 4) if ens_imp > 0 else 0.0,
            "diff_scene_coherence_ratio": round(ens_gd / ens_imp, 4) if ens_imp > 0 else 0.0,
            "separation_gap": round(ens_gen - ens_imp, 4),
        },
        "per_model": per_model,
        "detected_instances_per_character": {cid: len(by_char[cid]) for cid in chars},
    }


def main():
    parser = argparse.ArgumentParser(description="Circularity & Cross-Scene Coherence Audit (Read-Only)")
    parser.add_argument("--v1-dir", required=True)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--anchors-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--v1-report", default="")
    parser.add_argument("--v2-report", default="")
    parser.add_argument("--cache", default=str(PROJECT_ROOT / "scratch" / "circularity_deepface_cache.json"))
    args = parser.parse_args()

    v1_dir = Path(args.v1_dir)
    v2_dir = Path(args.v2_dir)
    anchors_dir = Path(args.anchors_dir)
    with open(args.storyboard, "r", encoding="utf-8") as f:
        story_spec = json.load(f)

    auditor = SceneAuditor()
    ensemble = BiometricEnsemble()

    anchor_paths = {}
    for cid in story_spec.get("characters", {}).keys():
        ap = anchors_dir / f"{cid}_ground_truth.png"
        if ap.exists():
            anchor_paths[cid] = str(ap)
    cache_p = Path(args.cache)
    if cache_p.exists():
        print(f"[1/3] Loading cached DeepFace 4-judge embeddings from {cache_p}...", flush=True)
        cached = json.loads(cache_p.read_text(encoding="utf-8"))
        v1_insts = cached["v1_insts"]
        v2_insts = cached["v2_insts"]
    else:
        anchor_deepface_embs = {}
        for cid, ap_str in anchor_paths.items():
            img = cv2.imread(ap_str)
            embs = ensemble.extract_face_embeddings(img)
            anchor_deepface_embs[cid] = {m: embs[m] for m in BiometricEnsemble.JUDGE_MODELS if m in embs}

        v1_rep_p = Path(args.v1_report) if args.v1_report else None
        v2_rep_p = Path(args.v2_report) if args.v2_report else None
        print("[1/3] Extracting DeepFace 4-judge embeddings for v1 and v2...", flush=True)
        v1_insts = _extract_version_instances("v1", v1_dir, story_spec, auditor, ensemble, anchor_deepface_embs, report_path=v1_rep_p)
        print(f"  [v1 done: {len(v1_insts)} slots]", flush=True)
        v2_insts = _extract_version_instances("v2", v2_dir, story_spec, auditor, ensemble, anchor_deepface_embs, report_path=v2_rep_p)
        print(f"  [v2 done: {len(v2_insts)} slots]", flush=True)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps({"v1_insts": v1_insts, "v2_insts": v2_insts}), encoding="utf-8")

    print("[2/3] Extracting InsightFace (buffalo_l w600k_r50) embeddings via Forge Python...", flush=True)
    worker_tasks = {
        "anchors": anchor_paths,
        "instances": [
            {"key": inst["key"], "image_path": inst["image_path"], "bbox": inst["bbox"]}
            for inst in (v1_insts + v2_insts) if inst["detected"] and inst["bbox"] is not None
        ],
    }

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd)
        worker_py = tmp_path / "if_worker.py"
        tasks_json = tmp_path / "tasks.json"
        out_json = tmp_path / "out.json"
        worker_py.write_text(INSIGHTFACE_WORKER_CODE, encoding="utf-8")
        tasks_json.write_text(json.dumps(worker_tasks), encoding="utf-8")
        res = subprocess.run([str(FORGE_PYTHON), str(worker_py), str(tasks_json), str(out_json)], capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"InsightFace worker failed: {res.stderr}\n{res.stdout}")
        if_data = json.loads(out_json.read_text(encoding="utf-8"))

    if_anchors = {cid: np.array(vec, dtype=np.float32) for cid, vec in if_data["anchors"].items()}
    if_insts = {k: np.array(vec, dtype=np.float32) for k, vec in if_data["instances"].items()}

    for inst in (v1_insts + v2_insts):
        k = inst["key"]
        cid = inst["character_id"]
        all_embs = {}
        if inst["deepface_embs"]:
            for m, vec in inst["deepface_embs"].items():
                all_embs[m] = np.array(vec, dtype=np.float32)
        if k in if_insts and cid in if_anchors:
            if_vec = if_insts[k]
            all_embs["InsightFace_buffalo_l"] = if_vec
            inst["insightface_anchor_sim"] = float(np.dot(if_vec, if_anchors[cid]))
        else:
            inst["insightface_anchor_sim"] = 0.0
        inst["all_embs"] = all_embs

    print("[3/3] Computing Circularity & Cross-Scene Coherence Summary...")
    def summarize_anchor_sims(insts: List[Dict[str, Any]]) -> Dict[str, Any]:
        models = BiometricEnsemble.JUDGE_MODELS + ["Ensemble_4Judge", "InsightFace_buffalo_l"]
        out = {"n_total_slots": len(insts), "n_detected": sum(1 for i in insts if i["detected"]), "models": {}, "by_age_group": {}}
        for m in models:
            if m == "Ensemble_4Judge":
                vals = [i["ensemble_anchor_sim"] for i in insts]
            elif m == "InsightFace_buffalo_l":
                vals = [i["insightface_anchor_sim"] for i in insts]
            else:
                vals = [i["judge_anchor_sims"][m] for i in insts]
            det_vals = [v for v, i in zip(vals, insts) if i["detected"]]
            out["models"][m] = {
                "mean_all_slots": round(float(np.mean(vals)), 4) if vals else 0.0,
                "std_all_slots": round(float(np.std(vals)), 4) if vals else 0.0,
                "mean_detected_only": round(float(np.mean(det_vals)), 4) if det_vals else 0.0,
            }
        for ag in ["child", "teen", "adult"]:
            sub = [i for i in insts if i["age_group"] == ag]
            if sub:
                out["by_age_group"][ag] = {
                    "n": len(sub),
                    "Ensemble_4Judge": round(float(np.mean([i["ensemble_anchor_sim"] for i in sub])), 4),
                    "InsightFace_buffalo_l": round(float(np.mean([i["insightface_anchor_sim"] for i in sub])), 4),
                }
        return out

    v1_map = {(i["scene"], i["seed"], i["character_id"]): i for i in v1_insts}
    v2_map = {(i["scene"], i["seed"], i["character_id"]): i for i in v2_insts}
    common_keys = sorted(set(v1_map.keys()) & set(v2_map.keys()))

    paired_deltas = {}
    models_all = BiometricEnsemble.JUDGE_MODELS + ["Ensemble_4Judge", "InsightFace_buffalo_l"]
    rng = np.random.default_rng(42)
    for m in models_all:
        diffs = []
        v2_vals_c, v1_vals_c = [], []
        for ck in common_keys:
            a, b = v1_map[ck], v2_map[ck]
            if m == "Ensemble_4Judge":
                va, vb = a["ensemble_anchor_sim"], b["ensemble_anchor_sim"]
            elif m == "InsightFace_buffalo_l":
                va, vb = a["insightface_anchor_sim"], b["insightface_anchor_sim"]
            else:
                va, vb = a["judge_anchor_sims"][m], b["judge_anchor_sims"][m]
            v1_vals_c.append(va)
            v2_vals_c.append(vb)
            diffs.append(vb - va)
        diffs_arr = np.array(diffs)
        boot = rng.choice(diffs_arr, size=(10000, len(diffs_arr)), replace=True).mean(axis=1)
        paired_deltas[m] = {
            "n_pairs": len(diffs),
            "v1_mean": round(float(np.mean(v1_vals_c)), 4),
            "v2_mean": round(float(np.mean(v2_vals_c)), 4),
            "paired_delta_mean": round(float(np.mean(diffs_arr)), 4),
            "image_level_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        }

    # Correlation between InsightFace and the 4 judges / ensemble on v2 detected faces
    corr_v2 = {}
    det_v2 = [i for i in v2_insts if i["detected"]]
    if_v2_arr = np.array([i["insightface_anchor_sim"] for i in det_v2])
    for m in BiometricEnsemble.JUDGE_MODELS + ["Ensemble_4Judge"]:
        jm_arr = np.array([i["ensemble_anchor_sim"] if m == "Ensemble_4Judge" else i["judge_anchor_sims"][m] for i in det_v2])
        if len(jm_arr) > 2 and np.std(if_v2_arr) > 0 and np.std(jm_arr) > 0:
            corr_v2[m] = round(float(np.corrcoef(if_v2_arr, jm_arr)[0, 1]), 4)
        else:
            corr_v2[m] = 0.0

    all_models_coherence = BiometricEnsemble.JUDGE_MODELS + ["InsightFace_buffalo_l"]
    v1_coh = _compute_coherence(v1_insts, all_models_coherence)
    v2_coh = _compute_coherence(v2_insts, all_models_coherence)

    report = {
        "evaluation_type": "read_only_audit_existing_renders",
        "v1_dir": str(v1_dir),
        "v2_dir": str(v2_dir),
        "anchor_similarity_circularity": {
            "v1": summarize_anchor_sims(v1_insts),
            "v2": summarize_anchor_sims(v2_insts),
            "paired_v2_minus_v1": paired_deltas,
            "v2_pearson_r_insightface_vs_judges": corr_v2,
        },
        "cross_scene_coherence": {
            "v1": v1_coh,
            "v2": v2_coh,
        },
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved circularity & cross-scene coherence report to: {out_p}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
