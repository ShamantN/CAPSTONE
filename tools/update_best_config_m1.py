"""
Script to update consistency_v2/best_config.json following Step 4 M1 re-check.
M1 nearest_pose vs single_anchor recorded delta was +0.0045 with 95% CI [0.0000, 0.01375].
Since lower bound does not rule out 0, it fails the acceptance rule.
Per protocol: default to simplest non-inferior arm ('single_anchor').
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "consistency_v2" / "best_config.json"

def update_m1_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    old_metric = cfg.get("bank_selection_metric")
    cfg["bank_selection_metric"] = "single_anchor"
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"[SUCCESS] Updated bank_selection_metric in {CONFIG_PATH}: '{old_metric}' -> 'single_anchor'")

if __name__ == "__main__":
    update_m1_config()
