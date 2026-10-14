# Verifier Specification and Calibration Report

**Status:** FROZEN  
**Freeze SHA256:** `f7319db547dc445c2ec05722f4425d3844d6cd5a2e43a7842034b5dbabd82850`  
**Git Tag:** `verifier-frozen`  
**Runtime Isolation:** 100% CPU execution. Zero GPU VRAM consumption.  

---

## 1. Overview & Architectural Principles

To eliminate circular evaluation artifacts and prevent metric gaming, MMCCP v2 introduces an independent, multi-modal verification suite located in `consistency_v2/verify/`.

The verification suite enforces three mandatory design principles:
1. **Circularity Elimination:** While generation and face transplant use InsightFace (`buffalo_l`/`antelopev2`), the verification judges are four independent face recognizers:
   - DeepFace `Facenet512`
   - DeepFace `SFace`
   - DeepFace `GhostFaceNet`
   - DeepFace `ArcFace`
2. **Spatial & Wardrobe Binding (Fix for D5b):** The Hungarian Kuhn-Munkres matching in v1 was position-agnostic and could assign a face on the right to a character specified on the left. The v2 `BindingChecker` combines horizontal geometric zoning with SegFormer semantic wardrobe segmentation (`mattmdjaga/segformer_b2_clothes`) and CIELAB color matching ($\Delta E^* \le 38.0$).
3. **Impostor Discrimination & Drift Detection:** In addition to comparing face crops against character reference banks, the verifier computes a Rank-1 discrimination test against ALL characters in the storyboard, measures the margin to the nearest impostor, and computes pairwise cross-similarity between faces in the same scene to detect identity bleed and cloning.

---

## 2. Calibration & EER Metrics

Calibration was executed against multi-reference anchor sets across 5 characters (`young_ebright`, `teen_ebright`, `college_ebright`, `mother`, `james_wong`).

### Biometric Judge Models Calibration (`consistency_v2/verify/calibration.json`)

| Model | Embedding Dim | Genuine Mean | Impostor Mean | Cross-Age Mean | Calibrated Threshold $\tau$ | EER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Facenet512** | 512 | 0.7879 | 0.1725 | 0.4455 | **0.3200** | 0.0000 |
| **SFace** | 128 | 0.7152 | 0.2589 | 0.3695 | **0.5250** | 0.0000 |
| **GhostFaceNet** | 512 | 0.6744 | 0.2216 | 0.3340 | **0.3800** | 0.0000 |
| **ArcFace** | 512 | 0.6907 | 0.1252 | 0.3120 | **0.3050** | 0.0000 |

### Documented Model Limitations
1. **Child Facial Morphology:** As anticipated in defect diagnosis D5d, face recognition models exhibit higher intra-class variance on young children (young Ebright, 8 years old) due to softer cranial bone structures and less pronounced nasal bridges.
2. **Age Progression Gap:** Cross-age similarity between 8-year-old child Ebright and 21-year-old adult Ebright averages $0.31$ to $0.44$, reflecting genuine biological maturation rather than model failure. The pipeline correctly handles this by maintaining separate age-state reference banks.

---

## 3. Spatial and Wardrobe Binding Specification

For each character expected in the scene:
1. **Spatial Zone Verification:**
   - Left Zone: $x_c / W \le 0.48$
   - Center Zone: $0.28 \le x_c / W \le 0.72$
   - Right Zone: $x_c / W \ge 0.52$
   - Mismatch triggers immediate spatial binding failure.
2. **Wardrobe Palette Verification:**
   - SegFormer segments Upper-clothes (class 4) or Dress (class 7) directly below the detected face.
   - Converts pixels to standard CIELAB space ($L^*, a^*, b^*$).
   - Computes Euclidean distance $\Delta E^*_{ab}$ to storyboard palette.
   - Failure if $\Delta E^* > 38.0$.

---

## 4. Visual Review Protocol (VR-1 to VR-6)

- **VR-1 Contact Sheets:** Automated generation of $1600 \times 1350$ PNG contact sheets (`make_contact_sheet.py`) saved in `docs_generated/v2/visual_review/contact_sheets/`. Displays full render with colored bounding boxes (green = pass, red = fail), side-by-side bank reference vs detected face/body crops, and quantitative biometric diagnostics.
- **VR-2 Structured Rubric:** Standardized JSON rubric (`rubric.py`) requiring literal description before scoring, phenotypic feature checklist, and 14-point defect taxonomy.
- **VR-4 Blind Re-ID Test:** Anonymous numbered tiles (`make_blind_sheet.py`) with sealed answer key for human/agent re-identification benchmarking.
- **VR-5 Planted Canaries:** Verified against planted failure modes:
  - Color-inverted photographic negative: Caught with 100% confidence by CLIP solarization probe ($P = 1.0$) and CIELAB wardrobe shift ($\Delta E^* = 60.6$).

---

## 5. Frozen File Hashes

The verification suite is frozen as of commit tag `verifier-frozen`. All file hashes are recorded in [`consistency_v2/verifier_freeze.json`](file:///D:/5TH%20SEM/HACKATHON/consistency_v2/verifier_freeze.json). Under Rule 2, no verifier code or thresholds may be modified.
