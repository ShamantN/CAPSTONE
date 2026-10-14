## Summary
<!-- Concise overview of the problem, motivation, and proposed solution -->

## Hypothesis
<!-- Which specific failure mode is addressed, which change is made, and which metric is expected to improve? -->

## Change and Feature Flags
<!-- List the exact code changes and corresponding configuration flags introduced/modified -->
- **Flag Name:** 
- **Default State:** `True` / `False`

## Results & Paired Statistics (with 95% Bootstrap CI)
<!-- Include before/after metric comparisons on DEV/TEST split -->
| Metric | Baseline (v1 / previous) | Proposed | Delta | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| ArcFace Cosine Similarity | | | | |
| CLIP-I Similarity | | | | |
| Structural SSIM | | | | |
| Palette Similarity | | | | |
| Composite Score | | | | |

## Visual Review Summary
<!-- Notes on contact sheet inspection, defect taxonomy verification, and feature binding -->
- **Contact Sheet Verified:** Yes / No
- **Identity Retention:** 
- **Binding & Positioning:** 
- **Defects Detected:** None / Listed

## VRAM & Latency Telemetry
- **Peak VRAM (GB):** 
- **Render Latency per Scene (s):** 

## Risks & Edge Cases
<!-- Known trade-offs, potential failure modes, or hardware sensitivities -->

## Decision
<!-- ACCEPT / REVISE / REJECT with rationale -->
- [ ] ACCEPT (Merge to `consistency-v2`)
- [ ] REJECT (Retain on `exp/` branch)
