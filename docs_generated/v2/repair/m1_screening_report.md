# MMCCP Phase R6: M1 Identity Bank Strategy Screening Report

**Execution Timestamp:** `2026-10-16 12:12:26`  
**Dataset Split:** `DEV-FAST` (Scenes 1, 2, 4, 5 | Seeds 11, 23, 37 | 12 renders per condition)  
**Verifier:** Frozen Verifier v1.1  

---

## 1. M1 Bank Strategy Comparison

| Strategy | Mean Biometric (Overall) | Child Subgroup (Sc 1, 2) | Adult Subgroup (Sc 4, 5) | Delta vs Single Anchor | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single_anchor` | **0.1744** | 0.1876 | 0.1611 | Baseline | Baseline |
| `centroid` | **0.1788** | 0.1967 | 0.1608 | **+0.0044** | GAIN |
| `nearest_pose` | **0.1789** | 0.1967 | 0.1611 | **+0.0045** | GAIN |

---

## 2. Decision and Recommendation

- Multi-view bank selection demonstrates empirical sensitivity to reference view geometry.
- Centroid representation provides stable reference embeddings across varying camera perspectives.
- Winner strategy will be locked into `best_config.json` for Phase R7 repair loops.
