"""
STAGE 22: AUTONOMOUS HEALING - SCENE 5 (HARVARD DORM DUAL-ETHNICITY ISOLATION)
Heals Scene 5 to ensure:
- Left student is strictly Caucasian Richard Ebright in Mustard Yellow Polo.
- Right student is strictly East Asian James Wong in Crimson Harvard Sweater.
- Physical plastic atom balls and molecular model on the study desk.
- Re-audits with ArcFace and CLIP.
"""
import os, sys, json, time
from pathlib import Path
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, r"D:\5TH SEM\HACKATHON")

_SD_SITE = r"C:\Stable-Diffusion\system\python\Lib\site-packages"
if os.path.exists(_SD_SITE) and _SD_SITE not in sys.path:
    sys.path.append(_SD_SITE)

from experiments.sota_generator import SOTAGenerator, generate_and_save_scene
from experiments.sota_qa_auditor import SOTAQAAuditor

OUTPUT_DIR = Path("edu_outputs/images")

def heal_scene_5():
    print("=" * 80)
    print("  STAGE 22: HEALING SCENE 5 (HARVARD BREAKTHROUGH)")
    print("=" * 80)

    # Re-render scene 5
    img_cv, meta = generate_and_save_scene(5, prefix="SOTA_scene_")
    
    # Audit healed scene 5
    auditor = SOTAQAAuditor()
    p = OUTPUT_DIR / "SOTA_scene_5.png"
    auditor.audit_scene(5, p)

if __name__ == "__main__":
    heal_scene_5()
