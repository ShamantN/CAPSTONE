# Forensic Audit Report: Repository & Pipeline State (Phases R7–R9)

**Generated UTC:** 2026-11-03  
**Status:** FORENSIC AUDIT COMPLETE — MANDATORY ROOT CAUSE DIAGNOSIS

---

## 1. Git History & Configuration Hash Discrepancy

### Question 1: Why `git merge fix/w4-openpose-anchoring` was run in addition to PR #31, and the exact diff between `best_config.json` and the recorded hash `63937859c3fde78e`?

### Forensic Finding:
1. **Branch Lineage:**
   - PR #31 (`fix/openpose-payload-fix`) contained commit `acf1370`, which fixed OpenPose payload keys and Forge options API.
   - Branch `fix/w4-openpose-anchoring` diverged later and contained 3 additional commits:
     - `499b68c`: screening measurement framing to genuine baseline v1 with image-level bootstrap.
     - `36c6b62`: locked v2 bundle configuration (recording hash `63937859c3fde78e`), spatial verifier tools, and proposal v1.3.
     - `2adf480`: crop-upscale-swap exploration.
   - Therefore, `fix/w4-openpose-anchoring` was merged to incorporate the locked bundle (`36c6b62`) into `consistency-v2`.

2. **The Exact Config Hash Explanation:**
   - When `consistency_v2/best_config.json` is serialized directly as JSON, its SHA-256 (16-char prefix) is **`63937859c3fde78e`**!
   - In `scene_1_seed_101.json`, `"config_hash": "63937859c3fde78e"` matches `best_config.json` **bit-for-bit**.
   - However, in commit `2adf480`, a new dataclass field was introduced in `consistency_v2/config.py`:
     `enable_crop_upscale_swap: bool = False`
   - When Python instantiates `ConsistencyV2Config()` and computes `compute_hash()`, the serialized dictionary includes `"enable_crop_upscale_swap": false`, which alters the resulting hash to `f713798ac1e6c6eb`.
   - **Exact Dictionary Difference:**
     ```json
     {
  "enable_crop_upscale_swap": {
    "best_config.json": null,
    "ConsistencyV2Config_object": false
  }
}
     ```

---

## 2. Wardrobe Lock (M7) Defect Analysis

### Question 2: Why `M7 Wardrobe Lock` produced `0.0` delta in `wardrobe_pass_rate`?

### Forensic Finding:
1. **The Target Color Override Bug:**
   - In `consistency_v2/wardrobe.py` line 204:
     ```python
     def get_target_lab(self, expected_color_names, bank_wardrobe_lab=None):
         if bank_wardrobe_lab is not None:
             return np.array(bank_wardrobe_lab, dtype=np.float32)
         ...
     ```
   - When `pipeline_v2.py` initialized `char_wardrobes`, it passed `bank_wardrobe_lab` extracted from the character bank reference.
   - In `college_ebright`, the bank reference image had median LAB `[60.17, 9.86, 24.14]` (a desaturated brownish tone), whereas StorySpec specifies `"mustard yellow"` (`[75.0, 10.0, 70.0]` in Verifier v1.2's `REFERENCE_LAB_COLORS`).
   - Because `bank_wardrobe_lab` took precedence over `REFERENCE_LAB_COLORS`, `wardrobe_lock` measured distance to `[60.17, 9.86, 24.14]`.
   - In Scene 6, the desaturated render had Delta-E `22.88` relative to the bank anchor. Because `22.88 < 38.0` (`delta_e_threshold`), `wardrobe_lock` classified it as already passing and **applied 0 repairs** (`repairs_applied = 0`)!
   - However, the Frozen Verifier v1.2 strictly compares observed clothing against `"mustard yellow"` (`[75.0, 10.0, 70.0]`), resulting in Delta-E `68.64` and an immediate **FAIL**.

2. **Per-Character Verifier Breakdown on DEV-2:**
   ```json
   {
  "young_ebright": {
    "total_instances": 10,
    "passed_instances": 1,
    "pass_rate": 0.1,
    "mean_delta_e": 52.04
  },
  "mother": {
    "total_instances": 5,
    "passed_instances": 5,
    "pass_rate": 1.0,
    "mean_delta_e": 25.48
  },
  "teen_ebright": {
    "total_instances": 10,
    "passed_instances": 3,
    "pass_rate": 0.3,
    "mean_delta_e": 37.64
  },
  "college_ebright": {
    "total_instances": 10,
    "passed_instances": 0,
    "pass_rate": 0.0,
    "mean_delta_e": 62.8
  },
  "james_wong": {
    "total_instances": 5,
    "passed_instances": 3,
    "pass_rate": 0.6,
    "mean_delta_e": 37.47
  }
}
   ```
   - `mother`: 5/5 pass (100.0%)
   - `james_wong`: 3/5 pass (60.0%)
   - `teen_ebright`: 3/10 pass (30.0%)
   - `young_ebright`: 1/10 pass (10.0%)
   - `college_ebright`: 0/10 pass (0.0%, mean Delta-E = 62.80)

3. **LOO Ablation Inactivity:**
   - Because `wardrobe_lock` was inactive or targeting mismatched anchor palettes on scenes where Delta-E was already < 38 relative to desaturated anchors, disabling M7 in the Leave-One-Out ablation produced **zero change** (`0.3333` vs `0.3333`).

---

## 3. Extra Background Persons in Scenes 2, 5, and 6

### Question 3: Why extra background persons appeared in Scenes 2, 5, and 6 on DEV-2?

### Forensic Finding:
1. **Defects in Verifier v1.2 Reports:**
   - 9 out of 30 DEV-2 renders exhibited YOLO person or face count mismatches:
     - `scene_2_seed_101.png`: Expected=1, YOLO=2, Faces=1
     - `scene_2_seed_202.png`: Expected=1, YOLO=2, Faces=1
     - `scene_5_seed_101.png`: Expected=2, YOLO=4, Faces=4 (`extra_people found 4 faces`)
     - `scene_5_seed_404.png`: Expected=2, YOLO=3, Faces=3 (`extra_people found 3 faces`)
     - `scene_5_seed_505.png`: Expected=2, YOLO=4, Faces=3 (`extra_people found 3 faces`)
     - `scene_6_seed_202.png`: Expected=1, YOLO=3, Faces=6 (`extra_people found 6 faces`)
     - `scene_6_seed_303.png`: Expected=1, YOLO=5, Faces=7 (`extra_people found 7 faces`)
     - `scene_6_seed_505.png`: Expected=1, YOLO=5, Faces=5 (`extra_people found 5 faces`)

2. **Root Causes in Prompt Builder & Layout Engine:**
   - **Missing Negative Constraints in Single-Person Scenes:**
     In `story_engine/prompt_builder.py` line 124, negative guards against extra people (`"solo, 1person, single person alone, identical faces..."`) were applied **only when `num_chars >= 2`**!
     When `num_chars == 1` (Scenes 2, 3, 4, 6), NO negative prompt prevented SDXL from generating background crowds.
   - **Context-Triggered Hallucination:**
     - Scene 6 action prompt: `"Graduation day at Harvard University outdoors in the yard. Richard Ebright in black academic graduation gown holding his diploma..."`
     - Without negative crowd prompts, SDXL associates "Graduation day in the yard" with dozens of fellow graduating students and onlookers.
     - Scene 5 action prompt: `"cluttered Harvard dormitory room..."` with background prompt `"Harvard dormitory room, interior room"`. Without negative third-party prompts, SDXL places additional students/roommates in the dormitory background.
   - **Lack of Background Mask Inpainting/Suppression in M2/M3:**
     While M2 planned regional bounding boxes for the foreground characters, the non-character background regions in Pass 1 and Pass 2 had no person-suppression or negative person prompt conditioning.

---

## 4. Configuration Mode Analysis (`quality` vs `default` vs `best_config.json`)

### Question 4: What `ConsistencyV2Config(mode="quality")` actually sets compared to `mode="default"` and `best_config.json`?

### Forensic Finding:
Exact parameter diff:
```json
{
  "steps_pass1": {
    "default": 20,
    "quality": 25
  },
  "enable_best_of_n": {
    "default": false,
    "quality": true
  },
  "steps_pass2": {
    "default": 15,
    "quality": 18
  },
  "mode": {
    "default": "default",
    "quality": "quality"
  }
}
```
- **`mode`**: `"quality"` vs `"default"`
- **`steps_pass1`**: **25** (quality) vs **20** (default & best_config.json)
- **`steps_pass2`**: **18** (quality) vs **15** (default & best_config.json)
- **`enable_best_of_n`**: **True** (quality) vs **False** (default & best_config.json)

All other 48 parameters (including CFG 5.5, CodeFormer 0.85, OpenPose weight 0.7, IP-Adapter Plus 0.55, Regional Couple 0.4, Wardrobe Lock 38.0) are completely identical.
