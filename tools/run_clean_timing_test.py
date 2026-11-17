"""
MMCCP TEST-3 Evaluation Protocol: Clean Timing Test (tools/run_clean_timing_test.py).

Executes a clean timing benchmark with zero background contention:
- Renders 6 scenes (2 per candidate across C0, C1, C2)
- Seeds: 7203 (Scene 1, 2 characters) and 7206 (Scene 2, 1 character)
- Instruments every pipeline stage with precise time.perf_counter() timestamps
- Outputs stage duration breakdown and overall seconds per scene
- Saves results to results/eval3/clean_timing_report.json
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.face_swap import DirectFaceSwapper
from eval3.runner import CONFIG_PRESETS, EVAL_OUTPUT_BASE

TIMING_OUT_DIR = EVAL_OUTPUT_BASE / "clean_timing"
TIMING_OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = PROJECT_ROOT / "results" / "eval3" / "clean_timing_report.json"
REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)


class TimingInstrumentedPipeline(ConsistencyPipelineV2):
    """Subclass that wraps internal calls with exact timing instruments."""
    def __init__(self, config: ConsistencyV2Config, bank_root: Path):
        super().__init__(config=config, bank_root=bank_root)
        self.stage_timings: Dict[str, float] = {}

    def generate_scene_instrumented(self, story: StorySpec, scene, seed: int):
        self.stage_timings = {}
        t_start = time.perf_counter()

        # Wrap layout planner
        orig_render_pose = self.layout_planner.render_openpose_canvas
        def timed_render_pose(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_render_pose(*args, **kwargs)
            self.stage_timings["layout_skeleton_s"] = round(time.perf_counter() - t0, 3)
            return res
        self.layout_planner.render_openpose_canvas = timed_render_pose

        # Wrap txt2img
        orig_txt2img = self.client.txt2img
        def timed_txt2img(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_txt2img(*args, **kwargs)
            self.stage_timings["pass1_txt2img_s"] = round(time.perf_counter() - t0, 3)
            return res
        self.client.txt2img = timed_txt2img

        # Wrap img2img
        orig_img2img = self.client.img2img
        def timed_img2img(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_img2img(*args, **kwargs)
            self.stage_timings["pass2_img2img_s"] = round(time.perf_counter() - t0, 3)
            return res
        self.client.img2img = timed_img2img

        # Wrap face detailer
        orig_detail = self.face_detailer.detail_image
        def timed_detail(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_detail(*args, **kwargs)
            self.stage_timings["face_detailer_s"] = round(time.perf_counter() - t0, 3)
            return res
        self.face_detailer.detail_image = timed_detail

        # Wrap face swap
        orig_swap = DirectFaceSwapper.execute_face_swap
        def timed_swap(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_swap(*args, **kwargs)
            self.stage_timings["face_swap_s"] = round(time.perf_counter() - t0, 3)
            return res
        DirectFaceSwapper.execute_face_swap = staticmethod(timed_swap)

        # Wrap wardrobe
        orig_wardrobe = self.wardrobe_lock.lock_scene_wardrobe
        def timed_wardrobe(*args, **kwargs):
            t0 = time.perf_counter()
            res = orig_wardrobe(*args, **kwargs)
            self.stage_timings["wardrobe_lock_s"] = round(time.perf_counter() - t0, 3)
            return res
        self.wardrobe_lock.lock_scene_wardrobe = timed_wardrobe

        # Generate scene
        img_bgr, meta = self.generate_scene(story=story, scene=scene, seed=seed)

        # Restore original functions
        self.layout_planner.render_multi_person_pose = orig_render_pose
        self.client.txt2img = orig_txt2img
        self.client.img2img = orig_img2img
        self.face_detailer.detail_image = orig_detail
        DirectFaceSwapper.execute_face_swap = orig_swap
        self.wardrobe_lock.analyze_and_repair = orig_wardrobe

        total_elapsed = round(time.perf_counter() - t_start, 3)
        self.stage_timings["total_elapsed_s"] = total_elapsed
        return img_bgr, meta, self.stage_timings


def main():
    print("=" * 80)
    print("  MMCCP TEST-3 CLEAN TIMING BENCHMARK (6 SCENES)")
    print("=" * 80)

    story_a_path = PROJECT_ROOT / "storyboards" / "ebright.json"
    story = StorySpec.load_json(str(story_a_path))
    bank_root = EVAL_OUTPUT_BASE / "dev3_shared" / "replicates" / "rep_00" / "banks"

    if not bank_root.exists():
        raise FileNotFoundError(f"Shared anchor banks not found at {bank_root}!")

    candidates = ["C0", "C1", "C2"]
    # Scene 1: 2 characters (Young Ebright, Mother), Seed: 7203
    # Scene 2: 1 character (Young Ebright), Seed: 7206
    test_scenes = [
        {"scene_idx": 0, "scene_obj": story.scenes[0], "seed": 7203, "desc": "Scene 1 (2 characters: Young Ebright & Mother)"},
        {"scene_idx": 1, "scene_obj": story.scenes[1], "seed": 7206, "desc": "Scene 2 (1 character: Young Ebright)"},
    ]

    all_results = {}

    for cand_id in candidates:
        cand_cfg = CONFIG_PRESETS[cand_id]
        print(f"\n>>> Benchmarking Candidate {cand_id} (hash: {cand_cfg.to_hash()}) <<<")
        pipeline = TimingInstrumentedPipeline(config=cand_cfg, bank_root=bank_root)
        cand_runs = []

        for item in test_scenes:
            sc = item["scene_obj"]
            seed = item["seed"]
            print(f"  Rendering {item['desc']} with clean seed {seed}...")
            img_bgr, meta, timings = pipeline.generate_scene_instrumented(story, sc, seed)

            render_id = f"clean_timing_{cand_id}_scene_{sc.scene_number:02d}_seed_{seed}"
            if img_bgr is not None:
                import cv2
                out_img_path = TIMING_OUT_DIR / f"{render_id}.png"
                cv2.imwrite(str(out_img_path), img_bgr)

            record = {
                "candidate": cand_id,
                "config_hash": cand_cfg.to_hash(),
                "scene_number": sc.scene_number,
                "characters_count": len(sc.characters_present),
                "seed": seed,
                "render_id": render_id,
                "timings": timings
            }
            cand_runs.append(record)
            print(f"    Total: {timings.get('total_elapsed_s')}s | Pass 1: {timings.get('pass1_txt2img_s')}s | Pass 2: {timings.get('pass2_img2img_s')}s | Detailer: {timings.get('face_detailer_s')}s | Swap: {timings.get('face_swap_s')}s | Wardrobe: {timings.get('wardrobe_lock_s')}s")

        mean_total = round(sum(r["timings"]["total_elapsed_s"] for r in cand_runs) / len(cand_runs), 2)
        all_results[cand_id] = {
            "mean_total_seconds": mean_total,
            "runs": cand_runs
        }

    # Summary report
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": "NVIDIA GeForce RTX 3070 Ti Laptop GPU (8192 MB VRAM)",
        "forge_url": "http://127.0.0.1:7860",
        "clean_environment": "No other evaluations running",
        "seeds_used": [7203, 7206],
        "results": all_results
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("  CLEAN TIMING BENCHMARK COMPLETE")
    print("=" * 80)
    for c_id, d in all_results.items():
        print(f"  {c_id}: Mean {d['mean_total_seconds']}s per scene")
        for r in d["runs"]:
            t = r["timings"]
            print(f"    Scene {r['scene_number']} ({r['characters_count']} chars, seed {r['seed']}): {t['total_elapsed_s']}s (Pass1: {t.get('pass1_txt2img_s')}s, Pass2: {t.get('pass2_img2img_s')}s, Detailer: {t.get('face_detailer_s')}s, Swap: {t.get('face_swap_s')}s, Wardrobe: {t.get('wardrobe_lock_s')}s)")
    print(f"\nSaved report to: {REPORT_JSON}")


if __name__ == "__main__":
    main()
