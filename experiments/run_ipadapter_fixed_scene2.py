"""
STAGE 14 & 15: REGENERATE EBRIGHT SCENE 2 WITH FIXED IP-ADAPTER & RUN QA
Generates Scene 2 through the canonical GenerationOrchestrator and evaluates biometric QA.
"""
import os, sys, json, time
from pathlib import Path
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from story_engine.schema import StorySpec
from generation.generation_orchestrator import GenerationOrchestrator
from qa.programmatic_auditor import ProgrammaticAuditor

def main():
    print("=" * 80)
    print("  REGENERATING EBRIGHT SCENE 2 WITH FIXED IP-ADAPTER (STAGE 14)")
    print("=" * 80)

    story = StorySpec.load_json("storyboards/ebright.json")
    scene_2 = story.scenes[1]  # Scene 2: Young Ebright in Butterfly Basement

    orchestrator = GenerationOrchestrator(identity_engine="ip_adapter")
    
    t0 = time.time()
    img_cv, meta = orchestrator.generate_scene(
        story=story,
        scene=scene_2,
        steps_pass1=25,
        steps_pass2=15,
        seed=42
    )
    elapsed = time.time() - t0

    out_img_path = Path("experiments/ipadapter_fixed_scene_2.png")
    out_json_path = Path("experiments/ipadapter_fixed_scene_2.json")

    cv2.imwrite(str(out_img_path), img_cv)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[COMPLETE] Generated {out_img_path} in {elapsed:.1f}s")
    print(f"  Resolution: {meta.get('width')}x{meta.get('height')}")
    print(f"  Identity Engine: {meta.get('identity_engine')}")
    print(f"  Denoising Strength: {meta.get('denoising_strength')}")

    # Stage 15: Run QA
    print("\n" + "=" * 80)
    print("  RUNNING PROGRAMMATIC QA ON FIXED SCENE 2 (STAGE 15)")
    print("=" * 80)
    auditor = ProgrammaticAuditor()
    auditor.audit_scene(story=story, scene=scene_2, img_path=out_img_path)
    
    passed_cnt = sum(1 for r in auditor.results if r["passed"])
    total_cnt = len(auditor.results)
    
    print(f"\n  QA Tests Passed: {passed_cnt}/{total_cnt} ({passed_cnt/total_cnt*100:.1f}%)")

    qa_report_path = Path("experiments/ipadapter_fixed_scene_2_qa.json")
    with open(qa_report_path, "w", encoding="utf-8") as f:
        json.dump(auditor.results, f, indent=2)
    print(f"  Saved QA results to {qa_report_path}")


if __name__ == "__main__":
    main()
