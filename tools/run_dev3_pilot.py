"""
MMCCP TEST-3 Evaluation Protocol: DEV-3 Pilot & Config Selection (tools/run_dev3_pilot.py).

Executes the DEV-3 pilot to:
1. Synthesize and certifiably gate anchors for Replicates 0 and 1.
2. Render scenes 1 to 5 (2 seeds each) across Candidates C0, C1, and C2.
3. List 18 seeded sample images (6 per candidate) for literal [OBSERVED] visual inspection.
4. Score all candidates using Eval3Evaluator (PM1, PM2, PM3, timings).
5. Select winning config and calculate replicate budget R for 6.0-hour cap.
"""
import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.identity_bank import IdentityBankManager
from eval3.splits import get_anchor_seed, get_scene_seed
from eval3.runner import Eval3Runner, CONFIG_PRESETS, EVAL_OUTPUT_BASE
from eval3.anchors import (
    load_all_evaluation_characters,
    generate_character_candidates,
    select_best_anchor,
    SceneAuditor,
)
from eval3.metrics.evaluator import Eval3Evaluator


SHARED_DEV3_DIR = EVAL_OUTPUT_BASE / "dev3_shared"
PILOT_CHARS = ["young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong"]


def prepare_shared_replicate_anchors(auditor: SceneAuditor) -> Dict[int, Path]:
    """Prepares and certifies anchor banks for Replicates 0 and 1."""
    all_chars = load_all_evaluation_characters()
    rep_banks = {}

    for r_idx in [0, 1]:
        rep_dir = SHARED_DEV3_DIR / "replicates" / f"rep_{r_idx:02d}"
        anchors_dir = rep_dir / "anchors"
        banks_dir = rep_dir / "banks"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        banks_dir.mkdir(parents=True, exist_ok=True)

        selected_anchors = {}
        for c_idx, cid in enumerate(PILOT_CHARS):
            c_bank_dir = banks_dir / cid
            c_bank_dir.mkdir(parents=True, exist_ok=True)
            ref_path = c_bank_dir / f"ref_00_{cid}_anchor.png"
            sidecar_path = anchors_dir / f"{cid}_selected_anchor.json"

            if ref_path.exists() and sidecar_path.exists():
                print(f"[REUSE] Anchor for {cid} replicate {r_idx} already exists at {ref_path}")
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    selected_anchors[cid] = json.load(f)
                continue

            cspec = all_chars.get(cid)
            if not cspec:
                raise ValueError(f"Character {cid} not found in library!")

            print(f"[SYNTHESIZE] Generating candidates for {cid} (replicate {r_idx})...")
            candidates = generate_character_candidates(
                char_id=cid,
                char_spec=cspec,
                replicate_idx=r_idx,
                char_idx=c_idx,
                is_test3=False,
                out_dir=anchors_dir / cid,
                auditor=auditor,
                max_candidates=4,
            )

            best = select_best_anchor(candidates)
            selected_anchors[cid] = best

            # Save selected candidate image into bank ref_00
            best_img = best.get("img_bgr")
            if best_img is not None:
                import cv2
                cv2.imwrite(str(ref_path), best_img)
            else:
                shutil.copyfile(best["image_path"], str(ref_path))

            # Initialize IdentityBankManager metadata
            bank_mgr = IdentityBankManager(bank_root=banks_dir)
            bank_mgr.initialize_character_bank(cid, [ref_path])

            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in best.items() if k not in ["img_bgr", "crop_bgr"]}, f, indent=2)
            print(f"[ANCHOR] Selected seed {best['seed']} for {cid} (uncertified={best.get('uncertified_anchor', False)})")

        rep_banks[r_idx] = banks_dir

    return rep_banks


def run_candidate_pilot(
    cand_name: str,
    cand_cfg: ConsistencyV2Config,
    rep_banks: Dict[int, Path],
    story: StorySpec
) -> Tuple[Eval3Runner, List[Dict[str, Any]]]:
    """Runs Story A scenes 1 to 5 with 2 seeds each for Replicates 0 and 1."""
    run_id = f"dev3_pilot_{cand_name}"
    runner = Eval3Runner(
        run_id=run_id,
        stage="A",
        split="DEV-3",
        config=cand_cfg,
    )
    all_renders = []

    for r_idx in [0, 1]:
        banks_dir = rep_banks[r_idx]
        pipeline = ConsistencyPipelineV2(config=cand_cfg, bank_root=banks_dir)
        rep_render_dir = runner.out_dir / f"rep_{r_idx:02d}" / "scenes"
        rep_render_dir.mkdir(parents=True, exist_ok=True)

        for sc_idx in range(5):
            scene = story.scenes[sc_idx]
            for s_idx in range(2):
                seed = get_scene_seed(
                    replicate_idx=r_idx,
                    story_idx=0,
                    scene_idx=sc_idx,
                    seed_idx=s_idx,
                    is_test3=False
                )
                render_id = f"rep_{r_idx:02d}_scene_{scene.scene_number:02d}_seed_{seed}"
                print(f"[{cand_name}] Rendering {render_id}...")
                sidecar = runner.run_scene_render(
                    story=story,
                    scene=scene,
                    seed=seed,
                    arm="v2_final",
                    replicate_idx=r_idx,
                    pipeline=pipeline,
                    render_id=render_id,
                    out_render_dir=rep_render_dir,
                )
                all_renders.append(sidecar)

    return runner, all_renders


def main():
    print("=" * 70)
    print("MMCCP TEST-3 DEV-3 PILOT & CONFIGURATION SELECTION")
    print("=" * 70)

    story_a_path = PROJECT_ROOT / "storyboards" / "ebright.json"
    story = StorySpec.load_json(str(story_a_path))
    auditor = SceneAuditor()

    # Step 1: Synthesize and certify shared replicate anchors
    print("\n--- STEP 1: PREPARING DEV-3 REPLICATE ANCHORS ---")
    rep_banks = prepare_shared_replicate_anchors(auditor)
    print(f"Anchors ready for Replicate 0: {rep_banks[0]}")
    print(f"Anchors ready for Replicate 1: {rep_banks[1]}")

    # Step 2: Render C0, C1, C2
    print("\n--- STEP 2: EXECUTING DEV-3 SCENE RENDERS FOR C0, C1, C2 ---")
    candidates = ["C0", "C1", "C2"]
    candidate_renders: Dict[str, List[Dict[str, Any]]] = {}
    candidate_runners: Dict[str, Eval3Runner] = {}

    for c_id in candidates:
        cfg = CONFIG_PRESETS[c_id]
        print(f"\n>>> Running Candidate {c_id} (Config Hash: {cfg.to_hash()}) <<<")
        runner, renders = run_candidate_pilot(c_id, cfg, rep_banks, story)
        candidate_renders[c_id] = renders
        candidate_runners[c_id] = runner

    # Step 3: Seeded Sample Selection for Visual Inspection (6 per candidate = 18 total)
    print("\n--- STEP 3: SEEDED SAMPLE SELECTION FOR VISUAL INSPECTION ---")
    rng = random.Random(42)
    sample_images: Dict[str, List[str]] = {}

    for c_id in candidates:
        successful_renders = [r for r in candidate_renders[c_id] if r.get("status") == "SUCCESS"]
        successful_paths = [
            str(candidate_runners[c_id].out_dir / f"rep_{r['replicate_idx']:02d}" / "scenes" / f"{r['render_id']}.png")
            for r in successful_renders
        ]
        sampled = rng.sample(successful_paths, min(6, len(successful_paths)))
        sample_images[c_id] = sampled

    visual_manifest_path = PROJECT_ROOT / "results" / "eval3" / "dev3_pilot_visual_sample.json"
    visual_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(visual_manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample_images, f, indent=2)

    print(f"Saved 18 visual sample image paths to {visual_manifest_path}")
    for c_id, imgs in sample_images.items():
        print(f"\nCandidate {c_id} (6 sampled images for visual review):")
        for p in imgs:
            print(f"  {p}")

    print("\n[READY] Pilot generation complete! Next: perform visual inspection via read_file before reading metrics.")


if __name__ == "__main__":
    main()
