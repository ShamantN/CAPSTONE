# MMCCP TEST-3 Stage B Visual QA Report (B3)
**Run ID:** `test3_v2_final` (`v2_final` arm, completed replicates `rep_00`, `rep_01`, `rep_02`)  
**Sampling Rule:** Union of (a) seeded random sample of 10 `v2_final` renders (`seed=42`), (b) every render flagged as a failure (`PM3 == False` or failed render), and (c) the 10 weakest renders.  
**Total Unique Images Inspected:** 19 (opened directly via image viewer in shuffled blind order BEFORE reading individual scores or selection tags).

---

## Part 1: Blinded `[OBSERVED]` Notes (Recorded BEFORE Reading Scores or Selection Key)

1. **`rep_00_Story_C_sc03_s0_seed_8487`** (`Story_C` sc3: `maya_adult`, `arthur_elder`)
   - `[OBSERVED]` Two people seated behind a wooden table: dark-haired woman in dark green sweater with white collar on left; older bearded man with glasses and grey pinstriped jacket on right. Count = 2 (exact), left/right order correct, both faces sharp and clean, zero extra people. High-contrast lighting, flat backdrop, minor garbled newspaper text and slightly awkward hands on man.
   - Blind visual verdict: **PASS (minor background/hand artifacts)**.

2. **`rep_02_Story_B_sc02_s0_seed_8559`** (`Story_B` sc2: `dr_marcus_vance`, `elena_child`)
   - `[OBSERVED]` Vertical split-panel seam down center. Left panel: adult man in dark green suit standing by bookshelf. Center-right: young girl in green/red/blue dress. Far-right edge: **cropped third person (partial face/eye and braided hair of a girl intruding from right border)**. Count = 3 (1 extra partial person). Both primary faces are sharp.
   - Blind visual verdict: **FAIL (extra cropped person + vertical split-panel seam)**.

3. **`rep_00_Story_A_sc05_s0_seed_8463`** (`Story_A` sc5: `college_ebright`, `james_wong`)
   - `[OBSERVED]` Severe 3-panel collage failure: left panel shows young man in yellow/brown/navy shirt; top-right panel shows East Asian young man in red knit sweater; bottom-right panel shows an **extra third person (East Asian woman in matching red sweater)**. Count = 3 (1 extra person). Also noticeable brown color-lock blotches on left man's yellow shirt and trousers.
   - Blind visual verdict: **FAIL (3-panel collage + extra person + wardrobe color blotch)**.

4. **`rep_00_Story_A_sc04_s2_seed_8462`** (`Story_A` sc4: `teen_ebright`)
   - `[OBSERVED]` Single teenage boy in green-and-white plaid shirt leaning on wooden fence outdoors in garden. Count = 1, sharp centered face, natural proportions, zero extra people, realistic photographic look.
   - Blind visual verdict: **PASS (clean)**.

5. **`rep_02_Story_B_sc02_s1_seed_8560`** (`Story_B` sc2: `dr_marcus_vance`, `elena_child`)
   - `[OBSERVED]` Two people in a hallway (count = 2, left/right order correct, zero extra people), **BUT severe age bleeding defect**: the left character (`dr_marcus_vance`, expected adult male doctor) is rendered as a **young boy (~7–9 years old)** in a green suit and red bow tie, matching the child age of `elena_child` on the right. Noticeable cyan/green color-lock bleed along the bottom hem of the girl's dress.
   - Blind visual verdict: **FAIL (severe age-bleed identity error on left character despite correct count = 2)**.

6. **`rep_01_Story_C_sc01_s2_seed_8528`** (`Story_C` sc1: `leo_child`)
   - `[OBSERVED]` Single young boy with curly brown hair standing full-body on a wooden porch with autumn pumpkins/oranges, beige pants, dark sweater. Count = 1, sharp centered face, natural proportions, zero extra people. Minor wardrobe color-lock artifact: cobalt-blue tint along the outer edges of the dark sweater sleeves.
   - Blind visual verdict: **PASS (minor sleeve color-lock tint)**.

7. **`rep_00_Story_C_sc02_s0_seed_8484`** (`Story_C` sc2: `arthur_elder`, `leo_child`)
   - `[OBSERVED]` Two people behind a grass-covered table (count = 2, left/right order correct, zero extra people), **BUT severe composite & age-drift defects**: strong bright white cutout halo around both figures against a flat gradient backdrop; left character (`arthur_elder`) appears middle-aged (~40s) with dark hair and mustache rather than elderly white-bearded man (`arthur_elder` in `rep_00` sc3 had white hair/beard and glasses); wardrobe bleed (both wear matching dark V-neck sweaters).
   - Blind visual verdict: **FAIL (age/appearance drift on `arthur_elder` + cutout halo + wardrobe bleed)**.

8. **`rep_02_Story_B_sc03_s0_seed_8562`** (`Story_B` sc3: `clara_thorn`, `dr_marcus_vance`)
   - `[OBSERVED]` **Severe double-head / extra-face defect**: foreground left woman (`clara_thorn`) has an **extra partial head/eyes emerging directly above the top of her head**, plus a visible dark rectangular inpainting seam around her jaw/neck and surreal leaf-like garment distortion. Background right man (`dr_marcus_vance`) in bright neon-green jacket is very blurry/out of focus. Count = 3 (extra partial head).
   - Blind visual verdict: **FAIL (extra disembodied head + inpaint seam + blurry second face)**.

9. **`rep_00_Story_A_sc04_s1_seed_8461`** (`Story_A` sc4: `teen_ebright`)
   - `[OBSERVED]` Single teenage boy in green-and-white plaid shirt leaning on wooden fence outdoors, monarch butterfly on foliage at top-left. Count = 1, sharp centered face, consistent appearance with `rep_00_Story_A_sc04_s2_seed_8462`, zero extra people.
   - Blind visual verdict: **PASS (clean)**.

10. **`rep_02_Story_A_sc02_s2_seed_8546`** (`Story_A` sc2: `young_ebright`)
    - `[OBSERVED]` Single young boy in bright yellow long-sleeve shirt holding clear plastic container with monarch butterflies next to wire butterfly cage. Count = 1, sharp centered face, natural proportions, zero extra people.
    - Blind visual verdict: **PASS (clean)**.

11. **`rep_01_Story_A_sc02_s1_seed_8500`** (`Story_A` sc2: `young_ebright`)
    - `[OBSERVED]` Single young boy in bright yellow polo shirt holding small mesh container with monarch butterflies next to mesh enclosure. Count = 1, sharp centered face, natural proportions, zero extra people.
    - Blind visual verdict: **PASS (clean)**.

12. **`rep_02_Story_B_sc02_s2_seed_8561`** (`Story_B` sc2: `dr_marcus_vance`, `elena_child`)
    - `[OBSERVED]` **Severe duplicate-character defect**: adult man in dark green suit sitting on stool on left, AND **two girls stacked vertically on the right** (smaller girl with braids standing behind larger foreground portrait of a girl in cardigan). Count = 3 (1 extra duplicate child). Strong bright cutout halo around the right figures.
    - Blind visual verdict: **FAIL (duplicate extra child + cutout halo)**.

13. **`rep_00_Story_C_sc01_s1_seed_8482`** (`Story_C` sc1: `leo_child`)
    - `[OBSERVED]` Single young boy with curly brown hair standing full-body on wooden porch with autumn trees behind, beige pants, blue shoes, dark sweater. Count = 1, sharp centered face, zero extra people. Minor cobalt-blue color-lock tint on outer sweater sleeves.
    - Blind visual verdict: **PASS (minor sleeve color-lock tint)**.

14. **`rep_02_Story_A_sc04_s2_seed_8552`** (`Story_A` sc4: `teen_ebright`)
    - `[OBSERVED]` Single teenage boy in green-and-white plaid overshirt over olive t-shirt seated outdoors holding small round cup, butterfly flying at left. Count = 1, sharp centered face, zero extra people; minor painterly smudge on front of olive t-shirt.
    - Blind visual verdict: **PASS (minor shirt smudge)**.

15. **`rep_01_Story_B_sc02_s2_seed_8516`** (`Story_B` sc2: `dr_marcus_vance`, `elena_child`)
    - `[OBSERVED]` Two children standing in a room (count = 2, left/right order correct, zero extra people), **BUT severe age bleeding defect**: left character (`dr_marcus_vance`, expected adult male doctor) is rendered as a **~9-year-old boy** in a shiny copper leather jacket, age-bleeding from `elena_child` on the right.
    - Blind visual verdict: **FAIL (severe age-bleed identity error on left character despite correct count = 2)**.

16. **`rep_01_Story_C_sc03_s1_seed_8533`** (`Story_C` sc3: `maya_adult`, `arthur_elder`)
    - `[OBSERVED]` Two people seated behind large wooden table: dark-haired woman in dark green blazer and white shirt holding a bowl on left; older white-bearded man with glasses in green sweater holding a bowl of fruit with red flower on right. Count = 2, correct left/right order, both faces sharp, zero extra people. Flat wall background and minor garbled chest emblem on man's sweater.
    - Blind visual verdict: **PASS (minor sweater emblem artifact)**.

17. **`rep_02_Story_B_sc03_s2_seed_8564`** (`Story_B` sc3: `clara_thorn`, `dr_marcus_vance`)
    - `[OBSERVED]` Two people standing against a mottled wall: woman in black jacket and green skirt holding flower baskets on left; adult man with mustache in dark green blazer, yellow vest, purple tie on right. Count = 2, correct left/right order, both faces sharp, zero extra people. Noticeable bright cutout halo around figures (composite studio look).
    - Blind visual verdict: **PASS (cutout halo composite look)**.

18. **`rep_01_Story_A_sc04_s0_seed_8505`** (`Story_A` sc4: `teen_ebright`)
    - `[OBSERVED]` Single teenage boy in green-and-white plaid shirt outdoors beside porch/garden holding wooden tray with bowl of seeds. Count = 1, sharp centered face, natural proportions, zero extra people.
    - Blind visual verdict: **PASS (clean)**.

19. **`rep_00_Story_A_sc02_s1_seed_8455`** (`Story_A` sc2: `young_ebright`)
    - `[OBSERVED]` Single young boy in bright yellow long-sleeve shirt holding small clear jar with monarch butterflies between two wire butterfly racks. Count = 1, sharp centered face, natural proportions, zero extra people.
    - Blind visual verdict: **PASS (clean)**.

---

## Part 2: Unblinded Comparison Against Automated Metrics (`test3_per_scene.csv`)

### Complete 19-Image Unblinded Audit Table

| # | Render ID | Selection Tags | Blind Visual QA | Automated `PM3` | `exp / yolo / det` | `v1.2` [IN-LOOP] | Sharpness | `bon3` | Agreement |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `rep_00_Story_C_sc03_s0_seed_8487` | `random_10` | PASS | `True` | `2 / 2 / 2` | 0.7215 | 1685.9 | Yes | **AGREE (PASS)** |
| 02 | `rep_02_Story_B_sc02_s0_seed_8559` | `flagged_failure`, `weakest_10` | FAIL (extra cropped girl) | `False` | `2 / 3 / 2` | 0.4747 | 795.9 | No | **AGREE (FAIL)** |
| 03 | `rep_00_Story_A_sc05_s0_seed_8463` | `flagged_failure`, `weakest_10` | FAIL (3-panel collage, extra woman) | `False` | `2 / 3 / 3` | 0.6348 | 906.5 | No | **AGREE (FAIL)** |
| 04 | `rep_00_Story_A_sc04_s2_seed_8462` | `random_10` | PASS (clean) | `True` | `1 / 1 / 1` | 0.4611 | 148.6 | No | **AGREE (PASS)** |
| 05 | `rep_02_Story_B_sc02_s1_seed_8560` | `weakest_10` | FAIL (adult rendered as child) | `True` | `2 / 2 / 2` | 0.4519 | 895.7 | **Yes** | **DISAGREE (PM3 False Neg)** |
| 06 | `rep_01_Story_C_sc01_s2_seed_8528` | `random_10` | PASS (minor sleeve tint) | `True` | `1 / 1 / 1` | 0.8015 | 1409.4 | Yes | **AGREE (PASS)** |
| 07 | `rep_00_Story_C_sc02_s0_seed_8484` | `flagged_failure`, `weakest_10` | FAIL (age drift, halo, bleed; 2 people) | `False` | `2 / 3 / 2` | 0.6774 | 1611.6 | No | **PARTIAL DISAGREE (diff reason)** |
| 08 | `rep_02_Story_B_sc03_s0_seed_8562` | `weakest_10` | FAIL (extra partial head, seam, blur) | `True` | `2 / 2 / 3` | 0.4522 | 990.7 | No | **DISAGREE (PM3 False Neg)** |
| 09 | `rep_00_Story_A_sc04_s1_seed_8461` | `random_10`, `weakest_10` | PASS (clean) | `True` | `1 / 1 / 1` | 0.4595 | 232.0 | Yes | **AGREE (PASS; low v1.2)** |
| 10 | `rep_02_Story_A_sc02_s2_seed_8546` | `random_10` | PASS (clean) | `True` | `1 / 1 / 1` | 0.7601 | 458.8 | No | **AGREE (PASS)** |
| 11 | `rep_01_Story_A_sc02_s1_seed_8500` | `flagged_failure`, `weakest_10` | PASS (clean, 1 boy) | `False` | `1 / 2 / 1` | 0.8574 | 211.8 | No | **DISAGREE (YOLO False Pos)** |
| 12 | `rep_02_Story_B_sc02_s2_seed_8561` | `flagged_failure`, `weakest_10` | FAIL (duplicate extra girl) | `False` | `2 / 3 / 3` | 0.5046 | 855.9 | No | **AGREE (FAIL)** |
| 13 | `rep_00_Story_C_sc01_s1_seed_8482` | `flagged_failure`, `weakest_10` | PASS (clean, 1 boy) | `False` | `1 / 2 / 1` | 0.7420 | 1205.1 | No | **DISAGREE (YOLO False Pos)** |
| 14 | `rep_02_Story_A_sc04_s2_seed_8552` | `random_10` | PASS (minor shirt smudge) | `True` | `1 / 1 / 1` | 0.6744 | 240.2 | No | **AGREE (PASS)** |
| 15 | `rep_01_Story_B_sc02_s2_seed_8516` | `weakest_10` | FAIL (adult rendered as child) | `True` | `2 / 2 / 2` | 0.4212 | 1056.1 | No | **DISAGREE (PM3 False Neg)** |
| 16 | `rep_01_Story_C_sc03_s1_seed_8533` | `random_10` | PASS (minor emblem artifact) | `True` | `2 / 2 / 2` | 0.7003 | 1206.1 | No | **AGREE (PASS)** |
| 17 | `rep_02_Story_B_sc03_s2_seed_8564` | `random_10` | PASS (cutout halo) | `True` | `2 / 2 / 2` | 0.5091 | 1459.6 | No | **AGREE (PASS)** |
| 18 | `rep_01_Story_A_sc04_s0_seed_8505` | `random_10` | PASS (clean) | `True` | `1 / 1 / 1` | 0.7839 | 232.6 | No | **AGREE (PASS)** |
| 19 | `rep_00_Story_A_sc02_s1_seed_8455` | `random_10` | PASS (clean) | `True` | `1 / 1 / 1` | 0.7872 | 339.5 | No | **AGREE (PASS)** |

---

### Explicit Disagreements Between Blind Visual QA and Automated Metrics (Unsmoothed)

1. **Automated `PM3` False Negatives — Structural Metric Misses Semantic Age Drift & Extra Partial Heads (3 images):**
   - **`rep_02_Story_B_sc02_s1_seed_8560` (#5) and `rep_01_Story_B_sc02_s2_seed_8516` (#15):** Both scored `PM3=True` (`yolo=2, det=2`), yet visually exhibit severe **child-to-adult age bleeding** in `Story_B` Scene 2 (`dr_marcus_vance`, expected adult male doctor, rendered as a ~7–9-year-old boy next to `elena_child`). Crucially, **`v2_final_bon3` selected `#5`** as Tier 3 because seeds `s0` (#2) and `s2` (#12) both had `yolo=3` extra people. Thus, `v2_final_bon3`'s `PM3 = 1.0000` overstates true visual correctness on mixed-age 2-character scenes. Notably, the biometric scores *did* reflect the mismatch (`v1.2 = 0.4519` and `0.4212`, `buffalo_l = 0.1300` and `0.0851`).
   - **`rep_02_Story_B_sc03_s0_seed_8562` (#8):** Scored `PM3=True` despite a grotesque **extra partial head/eyes protruding above `clara_thorn`'s scalp**, a visible dark jaw inpaint rectangle, and a blurry background `dr_marcus_vance`. `PM3` passed it because DeepFace detected 3 faces (`det=3 >= exp=2`) while YOLO only counted 2 full bodies (`yolo=2 == exp=2`). Requiring `det == exp` (not `det >= exp`) would have caught this failure.
2. **Automated `PM3` False Positives — Spurious YOLO Person Boxes on Clean Solo Renders (2 images):**
   - **`rep_01_Story_A_sc02_s1_seed_8500` (#11) and `rep_00_Story_C_sc01_s1_seed_8482` (#13):** Both were visually clean solo-character renders (`det=1`, `v1.2 = 0.8574` and `0.7420`), yet were flagged `PM3=False` solely because YOLO fired a spurious second person box on background clutter (butterfly cage / dark porch object).
3. **Failure Flagged for Wrong Cause (1 image):**
   - **`rep_00_Story_C_sc02_s0_seed_8484` (#7):** Visually contains 2 people (not 3), but fails visually due to `arthur_elder` aging drift (middle-aged instead of elder), bright cutout halos, and matching-sweater wardrobe bleed; automated `PM3=False` flagged it for a spurious `yolo=3` count instead.
4. **Qualitative Artifacts Invisible to Biometric & Structural Metrics:**
   - **Mixed-Age Bleeding:** Pairing a child and an adult/elder in a 2-character Regional Couple scene (`Story_B` sc2, `Story_C` sc2) intermittently caused age leakage onto the adult/elder character.
   - **Cutout Halos & Flat Backdrops:** 2-character scenes (`#1`, `#7`, `#12`, `#16`, `#17`) frequently exhibit bright white foreground halos and flat studio walls rather than natural environmental integration.
   - **M7 CIELAB Wardrobe Lock Tint Bleed:** Solo `leo_child` renders (`#6`, `#13`) show cobalt-blue color-lock tinting on the outer sleeves of his charcoal sweater; `#3` shows brown blotching on `college_ebright`'s shirt/pants.
   - **Sharpness vs Visual Realism:** Clean natural-light outdoor renders (`#4`, `#9`) had low Laplacian sharpness (`148.6`, `232.0`) and moderate `v1.2` (`0.46`) compared to oversharpened cutout-halo studio renders (`#7`, `#17` with sharpness `>1450`), confirming Laplacian sharpness rewards harsh studio contrast over photographic naturalness.
