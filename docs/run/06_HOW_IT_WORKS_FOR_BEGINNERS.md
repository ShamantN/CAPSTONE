# MMCCP v2.0 — How It Works for Beginners

- **Verification Status:** `TESTED ARCHITECTURE DOCUMENTATION` (module behavior, config flags, and TEST-2 numbers verified directly against `consistency_v2/best_config.json`, `consistency_v2/pipeline_v2.py`, and `results/numbers.json` on branch `docs/run-guides` on top of `main` `625b365`)

---

## The Core Problem

Standard text-to-image models (like SDXL) treat every image independently:
1. **Identity drift:** Changing seed or action alters bone structure, eye shape, and age.
2. **Attribute bleeding:** In two-person scenes, the model mixes prompts across the canvas (e.g., putting one character's yellow shirt onto the second character).
3. **Small-face blur:** SDXL shrinks images by 8x in latent space. In wide shots, a face spans fewer than 64 pixels in latent space, so details smear unless repaired separately.

MMCCP fixes this **without training custom LoRAs** by splitting generation into ten coordinated stages.

---

## End-to-End Pipeline Diagram

```
Story JSON (storyboards/*.json)
   │
   ▼
[1] Anchor Bootstrap ──► Generate 1 portrait per character -> hardcoded_cache.json
   │
   ▼
[2] M1 Identity Bank ──► Extract 4-judge embeddings + store ref_00_<char>_anchor.png
   │
   ▼
[3] M2 Layout Planner ──► Draw deterministic OpenPose COCO-18 skeleton + left/right boxes
   │
   ├──► [M3] Forge Couple ──► Split prompt per spatial column (stops cross-character bleed)
   └──► [M4] IP-Adapter+  ──► Inject face conditioning per character (Pass 1)
   │
   ▼
[4] Pass 1 (txt2img)  ──► 20 steps SDXL draft + FreeU + DynamicThresholding
   │
   ▼
[5] Pass 2 (img2img)  ──► Drop ControlNet/PAG to fit 8 GB VRAM; refine at adaptive denoise (15 steps)
   │
   ▼
[6] M6 Face Detailer  ──► Crop faces < 300px, re-inpaint at full resolution (20 steps, denoise 0.28)
   │
   ▼
[7] M5 InsightFace    ──► CPU inswapper_128 bound to left/right zone + Forge CodeFormer (0.85)
   │
   ▼
[8] M7 Wardrobe Lock  ──► SegFormer segments torso clothes -> CIELAB Delta-E check/repair (<= 38.0)
   │
   ▼
Final Panel PNG + JSON Sidecar ──► Evaluated offline on CPU by Frozen Verifier v1.2
```

---

## Which Modules Are ON vs OFF in Release `v2.0.0-rc3` (`consistency_v2/best_config.json`)

| Module | Flag in `best_config.json` | State | Plain-English Reason |
| :--- | :--- | :---: | :--- |
| **M1** Identity Bank | `enable_identity_bank` (`single_anchor`) | **ON** | Stores canonical anchor + centroids; `single_anchor` selected in R6 because multi-view centroid CI touched zero |
| **M2** OpenPose Layout | `enable_layout_planner` (`openpose_xl [f4251cb4]`, wt `0.70`) | **ON** | Prevents 2-person scenes from collapsing into 1 person; largest spatial helper in ablations |
| **M3** Forge Couple | `enable_regional_couple` (`regional_global_weight: 0.40`) | **ON** | Partitions cross-attention spatially so character prompts stay on their own half of the frame |
| **M4** IP-Adapter Plus-Face | `enable_ip_adapter_plus` (`wt: 0.55`) | **ON** | Guides Pass 1 facial likeness *(Note: single-seed ablations later showed `no_m4_ip_adapter` slightly beat `full_v2` on Bio; kept ON per frozen test-split rule, candidate to turn OFF in future `v2.1`)* |
| **M4b** Dual InstantID | `enable_dual_instantid` | **OFF** | Too heavy (> 7.4 GB peak risk alongside OpenPose + IP-Adapter on 8 GB VRAM) |
| **M5** Direct InsightFace Swap | `enable_reliable_face_swap` (`direct_insightface`, CodeFormer `0.85`) | **ON** | Biggest single contributor to facial Re-ID accuracy; runs `inswapper_128.onnx` on CPU to save VRAM |
| **M6** Face Detailer | `enable_face_detailer` (`threshold: 300px`, `denoise: 0.28`, `steps: 20`) | **ON** (OFF in `fast`) | Restores crisp small faces in medium/wide shots before final M5 swap |
| **M7** Wardrobe Lock | `enable_wardrobe_lock` (`Delta E <= 38.0`, `blend: 0.70`) | **ON** | Segments torso via SegFormer and pulls drifted clothing colors toward target CIELAB |
| **M8** Reference Attention | `enable_reference_attention` | **OFF** | Extra VRAM/latency without reliable DEV gain |
| **M9** Best-of-N Healing | `enable_best_of_n` | **OFF** (`ON` in `quality`) | Generates 3 candidates per scene (3x slower); left off in default 8 GB mode |
| **M10** Identity Memory | `enable_identity_memory` (`drift_guard_threshold: 0.75`) | **ON** (`0` promotions) | Guard intentionally set at `0.75` to block centroid poisoning; never triggered promotions |
| **M11** Per-Character LoRA | `use_character_lora` | **OFF** | Deliberately out of scope (training-free requirement; seeds `{606,707,808}` reserved) |
| **M12** Collage-Init | `enable_collage_init` | **OFF** | Tested in R7 (`loop-W6-iter1`) and **rejected** (`Delta = -0.0055`) due to floating-head artifacts |
| **FreeU / CFG-Fix / PAG** | `enable_freeu`, `enable_dynamic_thresholding`, `enable_pag` | **ON / ON / OFF** | FreeU + DynamicThresholding keep contrast crisp at CFG `5.5`; PAG left OFF to avoid attention memory doubling |

---

## What Works & What Does Not (Honest Summary from `TEST-2`)

- **Strong Wardrobe & Defect Gains on Ebright (`30` renders, `40` face slots):** Wardrobe Delta-E pass rose `45.0% -> 80.0%` (`+35.0%`, Wilcoxon `p=0.000183`); per-scene defects dropped `51.1%` (`3.07 -> 1.50/scene`, `p=0.001453`); Rank-1 disambiguation rose `65.0% -> 80.0%` (`+15.0%`, image CI `[+2.4%, +28.6%]`).
- **Original Pre-Registered Rules (`0/4` strict all-split MET):** Biometric Ensemble similarity on Ebright was essentially at parity with strong Baseline v1 (`0.3091 -> 0.3147`, CI crosses `0`), Holdout (`12` out-of-distribution scenes) was weaker than v1 (`Bio -0.0162`, `Spatial 87.5%`, `Wardrobe 37.5%`) mainly due to a **2-character hardcap** in M2/M3/M5 dropping the 3rd person in Holdout Scene 4, and whole-system VRAM touched `7984 MiB` (`7011 MiB` above desktop idle `973 MiB`). Full details live in [`docs_generated/v2/PREREG_RECONCILIATION.md`](../../docs_generated/v2/PREREG_RECONCILIATION.md).

---

## Beginner Glossary

- **SDXL (Stable Diffusion XL):** The `1024x1024` latent diffusion image generator (Checkpoint used here: `Juggernaut-XL v9`).
- **ControlNet:** An add-on neural network feeding spatial structure (like OpenPose stick-figure skeletons) into SDXL.
- **IP-Adapter:** Image Prompt Adapter; feeds a reference face image as conditioning tokens into cross-attention.
- **LoRA:** Low-Rank Adaptation fine-tuned weight patch; deliberately disabled (`M11`) because MMCCP must work training-free.
- **VRAM:** Dedicated GPU video memory (`8192 MiB` on an RTX 3070 Ti Laptop GPU).
- **Seed:** Integer initializing the random starting noise tensor so runs are repeatable (`11` for DEV smoke, `101..505` for DEV-2).
- **CFG Scale:** Classifier-Free Guidance (`5.5` here); how strictly diffusion obeys text vs staying natural.
- **Anchor / Identity Bank:** The clean studio reference portrait (`ref_00`) generated per character plus stored biometric embeddings.
- **Verifier (`v1.2`):** The frozen, CPU-only grader (`consistency_v2/verify/`) comparing outputs against anchors across 4 face models + YOLO + SegFormer + CLIP + DINOv2.
- **Split (`DEV` vs `TEST-2` vs `RESERVED`):** Data partitions enforcing scientific hygiene — tuning uses `DEV-*` splits only; `TEST-2` is sealed/burned; `RESERVED` (`606, 707, 808`) is permanently untouched.

