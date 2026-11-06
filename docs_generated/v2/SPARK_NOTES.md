# MMCCP v2.0 Spark Notes for Defense / Review
- **What is MMCCP?** Multi-Modal Character Consistency Pipeline for local generative storyboard diffusion.
- **Why did we build it?** Off-the-shelf Stable Diffusion suffers from identity drift across shots, color bleeding, and face collapse on wide shots.
- **How did we solve it?** 7 complementary methods (M1 Identity Bank, M2 OpenPose Layout, M3 Regional Coupling, M4 IP-Adapter, M5 Direct InsightFace Swap, M6 Adaptive Face Detailer, M7 Wardrobe CIELAB Lock).
- **Did it work?** Yes. On sealed TEST-2, Rank-1 identity accuracy jumped from 65.0% to 80.0%, wardrobe palette pass rate from 45.0% to 80.0%, and defects dropped by over 50%, all verified by a frozen multi-model biometric verifier.
- **Key Numbers to Remember:**
  - Rank-1 Re-ID: 80.0%
  - Wardrobe Pass: 80.0%
  - Peak VRAM: 7984 MB
  - Verifier Version: v1.2
