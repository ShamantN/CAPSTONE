"""
InstantID Runtime Test & Diagnostic Script
"""
import sys, os, time, json
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")
from story_engine.schema import StorySpec
from generation.generation_orchestrator import GenerationOrchestrator

story = StorySpec.load_json("storyboards/ebright.json")
orchestrator = GenerationOrchestrator(identity_engine="instantid")
scene = story.scenes[1]  # Scene 2 solo close up

print("Starting InstantID test generation on Scene 2...")
t0 = time.time()
img_cv, meta = orchestrator.generate_scene(story=story, scene=scene, steps_pass1=15, steps_pass2=10, seed=42)
dur = round(time.time() - t0, 2)

if img_cv is not None:
    print(f"SUCCESS! Image generated: shape={img_cv.shape}, runtime={dur}s")
    print(json.dumps(meta, indent=2))
else:
    print(f"FAILED after {dur}s! Meta: {meta}")
