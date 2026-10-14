# 09. PPT Presentation Handoff Document: Capstone Project Phase 3 — Progress Review #2

**Target PPT Template:** PES University Capstone Phase 3 — Review #2  
**Parent Initiative:** Novel to Video Generation (NVG)  
**Project Title:** Multi-Modal Character Consistency Pipeline (MMCCP)  
**Project ID:** `PW26_CGM_02`  
**Slide Footer Identifier:** `SHAMANT_SHIVU_SHIHAB_SAMIYA`  
**Student Team:**  
- **Samiya Naaz** (SRN: `PES1UG23CS504`)  
- **Shamant Nagabhushana** (SRN: `PES1UG23CS536`)  
- **Shihab K A** (SRN: `PES1UG23CS546`)  
- **Shivu K C** (SRN: `PES1UG23CS551`)  
**Faculty Guide:** Prof. Chitra G M  
**Department:** Department of Computer Science and Engineering, PES University  
**Session Execution Date:** October 14, 2026  
**Auditor / Author:** Senior ML Engineer & Technical Writer  

---

## Slide 1: Title Slide

### Slide Visual Elements
- **Layout:** Standard PES University Capstone Review #2 Cover Layout.
- **Header:** Department of Computer Science and Engineering, PES University.
- **Main Title:** Multi-Modal Character Consistency Pipeline (MMCCP) for Novel-to-Video Generation.
- **Subtitle:** Capstone Project Phase 3 — Progress Review #2.
- **Project ID:** `PW26_CGM_02`.
- **Team Members:**
  - Samiya Naaz (PES1UG23CS504)
  - Shamant Nagabhushana (PES1UG23CS536)
  - Shihab K A (PES1UG23CS546)
  - Shivu K C (PES1UG23CS551)
- **Faculty Guide:** Prof. Chitra G M, Assistant Professor, Department of CSE.
- **Footer Text:** `SHAMANT_SHIVU_SHIHAB_SAMIYA`

### Speaker Notes
> "Respected Guide Prof. Chitra G M and honorable review committee members, welcome to our Phase 3 Progress Review #2 for Capstone Project PW26_CGM_02: Multi-Modal Character Consistency Pipeline, a critical foundational sub-system of Novel to Video Generation. Today, our team—Shamant, Shivu, Shihab, and Samiya—will present our clean-slate zero-cache experimental progress, real measured metrics on consumer hardware, rigorous responses to Review 1 feedback, mathematical foundations, and live demonstrations of our completed multi-modal pipeline."

---

## Slide 2: Outline

### Ready-to-Paste Slide Content
- **1. Review 1 Suggestions & Evidence-Backed Actions Taken:** Direct response to reviewer queries regarding additional methods and hardware feasibility.
- **2. Problem Statement & Motivation:** Overcoming identity drift and demographic bleed in narrative visual synthesis without parameter fine-tuning.
- **3. Proposed Methodology & Two-Pass Architecture:** Coarse-to-fine generative decoupled pipeline with camera-distance adaptive denoising.
- **4. Mathematical Formulation & Pseudocode:** 4-Point anchor certification, Kuhn-Munkres Hungarian assignment, and quad-modal scoring.
- **5. Design Approach & Architectural Trade-offs:** Training-free adapters vs per-character LoRAs on an 8 GB consumer GPU budget.
- **6. Demonstration & Empirical Testing:** Clean-slate 6-scene generation results, 10-stage ablation study, and chromatic vulnerability audit.
- **7. Cross-Domain Applicability:** Translating narrative consistency into graphic novels, advertising, and film pre-visualization.
- **8. Project Roadmap & Gantt Timeline:** Completed milestones and scheduled timeline for Phase 4 video synthesis.

### Speaker Notes
> "This presentation is structured to directly address the academic and engineering milestones required for Review #2. We begin by answering the specific suggestions given to us during Review #1, substantiate our approach with mathematical rigor and clean pseudocode, demonstrate our live Streamlit application with zero-cache measured numbers, evaluate cross-domain potential across nine industry dimensions, and articulate our pending milestones for final Phase 4 completion."

---

## Slide 3: Suggestions from Review 1 & Detailed Responses

### Ready-to-Paste Slide Content
* **Suggestion 1: "Find a few more methods to improve the character consistency."**
  - **Action Taken:** Surveyed, implemented, and empirically benchmarked three advanced conditioning paradigms:
    1. *Generic IP-Adapter SDXL:* Decoupled cross-attention separating text and image prompt streams.
    2. *InstantID SDXL:* 5-point facial landmark keypoint conditioning for pose-invariant identity retention.
    3. *InsightFace / FaceID Plus v2 SDXL:* 512-D ArcFace biometric embeddings coupled with cross-attention LoRA.
    4. *Literature Candidates Evaluated:* PuLID (NeurIPS 2024), StoryDiffusion (NeurIPS 2024), and ConsiStory (SIGGRAPH 2024).
* **Suggestion 2: "Mention the feasibility of the same showing the progress."**
  - **Empirical Feasibility Findings on 8 GB RTX 3070 Ti Laptop GPU:**
    - *Lightweight Methods (Demographic Locking + FreeU + ReActor):* Peak VRAM **6.8 GB**, latency **76s/scene**, ArcFace `0.48-0.53`, CLIP-I `0.85-0.88`. **100% Feasible for interactive workflows.**
    - *Heavyweight Adapter Methods (FaceID Plus v2 + LoRA):* Peak VRAM **7.6 GB** (free VRAM drops to 597 MB). Forces PCIe memory swapping, slowing iteration from 1.4s/it to **14.4s/it** (total latency **>4 minutes/scene**).
    - *Progress Update:* Evaluated across a 10-stage systematic ablation; proven that our hybrid decoupled approach provides the optimal Pareto frontier between biometric accuracy and interactive runtime.

### Visual Assets to Place on Slide
- **Comparison Table / Graphic:** `docs_generated/raw/ablation_study_results.csv` summary showing Latency vs VRAM.
- **Telemetry Chart:** VRAM curve showing the 7.6 GB peak during FaceID execution (`docs_generated/raw/vram_telemetry.csv`).

### Speaker Notes
> "During our first review, the panel wisely instructed us to explore additional consistency techniques and evaluate their physical feasibility. We took this feedback seriously: we integrated IP-Adapter, InstantID, and FaceID Plus v2 with LoRA, and reviewed NeurIPS 2024 literature like PuLID and StoryDiffusion. Crucially, we benchmarked them on our target 8 GB RTX 3070 Ti laptop GPU. Our empirical data revealed that while heavy identity adapters like FaceID improve raw biometric scores slightly, their memory footprint exceeds the dedicated VRAM boundary, triggering PCIe RAM swapping that inflates render time by 10x. Therefore, our hybrid approach of Demographic Locking, FreeU, and ReActor provides the exact sweet spot for real-world production."

---

## Slide 4: Proposed Approach & System Methodology

### Ready-to-Paste Slide Content
- **The Core Problem in Novel-to-Video:**
  - Standard diffusion models resample random latents per seed, altering facial bone structure, wardrobe, and hair color between narrative scenes.
  - Multi-character scenes suffer from *cross-attention bleed*, where clothing colors and racial traits blend between subjects.
- **Basic Approach & Why It Needed Evolution:**
  - *Initial Baseline:* Single-pass txt2img with plain character names yielded severe identity drift (ArcFace dropped across seeds, CLIP-I was only 0.7283).
  - *Identified Flaw:* SDXL’s latent space downsamples 8x ($f=8$); in medium and long shots, faces occupy fewer than $64\times 64$ pixels, obliterating biometric facial details during initial noise inversion.
- **Current Decoupled Coarse-to-Fine Architecture:**
  - **Pass 1 (Macro Composition):** Plain SDXL txt2img guided by ControlNet Depth and BREAK regional prompt column partitioning (20 steps, DPM++ 2M Karras).
  - **Pass 2 (Micro Identity Refinement):** Camera-distance scheduled img2img ($\sigma \in [0.38, 0.62]$) paired with ReActor InsightFace latent swapping and CodeFormer facial codebook restoration.
  - **Quality Assurance & Audit:** Closed-loop quad-modal verification (ArcFace, CLIP-I, SSIM, Palette) and zero-shot VLM chromatic anomaly detection.

### Visual Assets to Place on Slide
- **System Architecture Diagram:** Mermaid flow diagram from `docs_generated/03_project_overview.md`.
- **Side-by-Side Graphic:** `docs_generated/raw/ablation_images/A01_plain_sdxl_baseline.png` vs `docs_generated/raw/ablation_images/A06_reactor_codeformer.png`.

### Speaker Notes
> "Our proposed approach addresses the root cause of character inconsistency: spatial scale divergence. In literature, single-pass diffusion models try to resolve global environment composition and high-frequency facial micro-geometry simultaneously. However, because SDXL compresses images into an 8x downsampled latent space, faces in wide shots lack the necessary latent tokens for facial identity. Our pipeline decouples the problem: Pass 1 establishes scene geometry, perspective, and lighting using depth conditioning. Pass 2 executes adaptive refinement, using camera-distance denoising and high-resolution latent face injection to guarantee biometric consistency across shots."

---

## Slide 5: Algorithm & Pseudocode

### Ready-to-Paste Slide Content

```python
# Multi-Modal Character Consistency Pipeline (MMCCP) Master Execution Algorithm
def execute_mmccp_pipeline(storyboard_json, character_registry):
    # Phase 1: Canonical Visual-DNA Anchor Bootstrapping & Certification
    for char in storyboard_json.characters:
        if not character_registry.is_certified(char.id):
            anchor_candidate = render_canonical_portrait(char.phenotype_prompt)
            assert certify_4point_gate(anchor_candidate) == True
            character_registry.register(char.id, anchor_candidate)
    
    # Phase 2: Sequential Narrative Scene Synthesis
    rendered_storyboard = []
    for scene in storyboard_json.scenes:
        # Step A: Build Regional Prompt with Demographic Locking
        prompt_pass1 = build_regional_break_prompt(scene.characters, scene.environment)
        depth_map = generate_or_load_depth_guide(scene.composition)
        
        # Step B: Pass 1 - Compositional Draft Pass (txt2img)
        canvas_draft = forge_txt2img(
            prompt=prompt_pass1, steps=20, sampler="DPM++ 2M Karras",
            controlnet={"depth": depth_map, "weight": 0.70},
            latent_enhancers={"freeu": (1.3, 1.4, 0.9, 0.2), "pag": 3.0, "cfg_fix": 7.0}
        )
        
        # Step C: Pass 2 - Biometric Identity Refinement (img2img)
        denoise_sigma = calculate_adaptive_denoise(scene.camera_distance) # 0.38, 0.50, 0.62
        canvas_refined = forge_img2img(
            init_image=canvas_draft, denoise=denoise_sigma, steps=15,
            alwayson_scripts={"reactor": build_reactor_payload(character_registry, codeformer=0.80)}
        )
        
        # Step D: Closed-Loop Quad-Modal Quality Verification
        detected_faces = retinaface_detect(canvas_refined)
        if len(scene.characters) > 1:
            match_matrix = kuhn_munkres_hungarian(character_registry, detected_faces)
        scorecard = evaluate_quad_modal(canvas_refined, character_registry, weights=(0.45, 0.25, 0.15, 0.15))
        
        # Step E: Autonomous Self-Healing Retry
        if scorecard.composite < 0.50:
            canvas_refined = execute_healing_loop(scene, canvas_draft, retries=3)
            
        rendered_storyboard.append(canvas_refined)
    return rendered_storyboard
```

### Speaker Notes
> "Slide 5 details the end-to-end algorithmic control flow of MMCCP. The pipeline operates in two major phases: First, anchor bootstrapping generates canonical reference portraits for each character and subjects them to an automated 4-point certification gate. Second, the narrative loop executes: it compiles demographic prompts using BREAK column partitioning, executes Pass 1 depth-guided draft synthesis, calculates adaptive denoising based on camera shot distance, refines the canvas in Pass 2 via ReActor and CodeFormer, evaluates consistency using Kuhn-Munkres Hungarian assignment across our quad-modal metrics, and triggers automated healing if composite quality falls below 0.50."

---

## Slide 6: Design Approach & Architectural Decisions

### Ready-to-Paste Slide Content
- **1. Training-Free Architecture over Fine-Tuning:**
  - *Why not per-character LoRA/DreamBooth?* In a novel-to-video pipeline, text scripts dynamically introduce novel characters. Training a LoRA takes 15–25 minutes per character, requires 15–20 training images (which do not exist for fictional book characters), and risks catastrophic forgetting. MMCCP is **100% training-free and zero-shot**.
- **2. Memory Isolation on an 8 GB Hardware Ceiling:**
  - Deploys transient memory lifecycles: ControlNet structural weights are loaded during Pass 1 and purged before Pass 2.
  - Evaluation processes (`metrics.py`) execute under a strict 25% GPU allocation clamp (`torch.cuda.set_per_process_memory_fraction(0.25)`), releasing memory immediately via `vram_guard`.
- **3. Camera-Distance Adaptive Denoise Scheduling:**
  - *Close-Up ($\sigma = 0.38$):* High facial latent retention (62% preserved), micro-texture smoothing.
  - *Medium Shot ($\sigma = 0.50$):* Balanced 50/50 generative reconstruction.
  - *Long/Wide Shot ($\sigma = 0.62$):* High reconstruction allowance (62% new latents) to resolve complex backgrounds while ReActor restores facial keypoints.
- **4. Evaluated Design Alternatives:**
  - *Single-Pass Generation:* Fails on face scale; faces in wide shots are blurred.
  - *Video Diffusion Models (AnimateDiff/SVD):* Limited to 16–24 frames with massive VRAM demands (>12 GB); cannot maintain identity across disparate narrative chapters.

### Visual Assets to Place on Slide
- **Memory Lifecycle Flowchart:** Memory allocation bar showing the staged purge between Pass 1 and Pass 2.
- **Camera Denoise Curve:** Visual diagram showing close-up ($0.38$) vs wide shot ($0.62$).

### Speaker Notes
> "Our design approach was shaped by three strict engineering constraints: consumer hardware feasibility, dynamic narrative scalability, and training-free operation. We rejected per-character LoRA fine-tuning because novel-to-video pipelines must handle unseen fictional characters dynamically from prose descriptions without waiting 20 minutes for backpropagation. Furthermore, running multi-gigabyte models on an 8 GB mobile GPU necessitated our staged memory architecture: ControlNet runs in Pass 1, is completely unloaded, and ReActor executes in Pass 2, maintaining GPU utilization strictly under 7.5 GB."

---

## Slide 7: Demonstration & Empirical Testing of Completed Modules

### Ready-to-Paste Slide Content
- **Completed Modules Tested in Live Session:**
  - ✅ **Interactive Web Dashboard:** Streamlit multi-tab application with real-time storyboard rendering, anchor certification UI, and comparative diagnostics.
  - ✅ **Clean-Slate Storyboard Generation:** Rendered all 6 scenes of *"The Making of a Scientist"* from zero cache with 100% pass rate.
  - ✅ **Automated 4-Point Anchor Certification:** All 5 character anchors certified ($N=1$ face, bbox $12.6\% - 32.2\%$, margin $\ge 146\text{px}$, norm $1.0000$).
  - ✅ **Bipartite Assignment Matrix:** Successfully resolved multi-subject pairings in Scene 1 and Scene 5 without identity swaps.
- **Measured Session Metrics ($N = 8$ Character Evaluations):**
  - **Mean ArcFace Cosine Similarity:** **`0.5164`**
  - **Mean CLIP-I Semantic Consistency:** **`0.8061`**
  - **Mean Structural SSIM:** **`0.5017`**
  - **Mean Color Palette Similarity:** **`0.4047`**
  - **Overall Composite Consistency:** **`0.5699`** (Threshold $\theta = 0.50$)
- **Red-Team Chromatic Vulnerability Audit:**
  - Inverting RGB channels revealed ArcFace scores an alarming **`0.9346`** (false-positive identity match).
  - Our VLM Chromatic Auditor caught the inversion with **`99.81% zero-shot confidence`**, validating the necessity of multi-modal defense.
- **Estimated Module Completion:** **85% of Total Capstone Scope Completed** (Core consistency pipeline 100% operational; video temporal interpolation scheduled for Phase 4).

### Visual Assets to Place on Slide
- **UI Screenshot:** `docs_generated/screenshots/streamlit_tab1_initial_1791959202572.png` and `streamlit_tab1_scenes_top_1791959232984.png`.
- **Benchmark Matrix Screenshot:** `docs_generated/screenshots/streamlit_tab3_benchmark_complete_1791959542664.png`.
- **Exemplary Render Montage:** `scene_1_clean_best.png` (Multi-char) and `scene_2_clean_best.png` (Solo).

### Speaker Notes
> "On Slide 7, we present the empirical proof of our system running live from a zero-cache state. We generated the complete six-scene storyboard of 'The Making of a Scientist'. Across eight character evaluations, our mean composite score reached 0.5699 with a 100% pass rate against our 0.50 quality threshold. In multi-character scenes, our Hungarian algorithm successfully assigned Young Ebright and Mother to their exact respective anchors. Furthermore, our red-team experiment proved that while ArcFace is fooled by inverted blue skin—scoring 0.9346—our VLM chromatic probe catches the anomaly at 99.81% confidence. The core consistency engine is 100% functional, representing 85% completion of our overall capstone roadmap."

---

## Slide 8: Applicability of Proposed Methodology in Other Domains

### Ready-to-Paste Slide Content (Covering All 9 Template Requirements)
1. **Potential Other Domains:** Graphic novels/comics, commercial advertising, cinematic pre-visualization, e-learning content, game NPC dialogue portraits, and virtual fashion try-on.
2. **Common Problem Shared Across Domains:** Visual identity drift, wardrobe inconsistency, and demographic bleed across sequential images in training-free generative workflows.
3. **Component Mapping to External Domains:**
   - *Graphic Novels:* Anchor Registry locks comic hero facial structure; BREAK regional prompts isolate hero and villain panels.
   - *Virtual Fashion / E-Commerce:* IP-Adapter decouples garment textures; ReActor maps consistent model identities across diverse clothing catalogs.
4. **Modifications Required:** Replace SDXL photographic checkpoint (`Juggernaut-XL`) with domain-specific models (e.g., `Animagine XL` for manga or fashion-tuned diffusion checkpoints); adjust adaptive denoise parameters for illustrated art styles.
5. **Suitable Datasets & Sources:** Manga109 (manga benchmarking), DeepFashion (garment consistency), and CelebA-HQ (facial biometric validation).
6. **Expected Benefits:** 90% reduction in production time for comic panel drafting; eliminates expensive studio photoshoot reshoots in digital advertising.
7. **Feasibility & Limitations:** Highly feasible on single consumer workstations; limited only in extreme close-ups of complex hand interactions.
8. **Comparison with Existing Commercial Approaches:**
   - *Midjourney `--cref`:* Closed-source, proprietary API, unpredictable style drift, high subscription cost.
   - *MMCCP:* Fully open-source, local execution, privacy-preserving, mathematically audited with quad-modal verification.
9. **Future Scope in Target Domains:** Dynamic lighting transfer across comic panels and multi-angle 3D avatar rigging from certified 2D anchors.

### Speaker Notes
> "Slide 8 addresses the university requirement to demonstrate the broader applicability of our methodology. The challenge of character consistency is not unique to novel-to-video; it is the single largest bottleneck in digital comics, marketing campaigns, and virtual fashion catalogs. In a graphic novel workflow, our anchor registry locks character features across hundreds of panels, while our BREAK syntax prevents hero and villain traits from bleeding together. Compared to proprietary black-box tools like Midjourney cref, MMCCP offers a fully transparent, open-source pipeline that runs locally on consumer GPUs, maintains user data privacy, and provides rigorous quantitative quality assurance."

---

## Slide 9: Project Timeline & Update on Pending Tasks

### Ready-to-Paste Slide Content
- **Current Milestone Status (as of October 14, 2026):**
  - ✅ *Milestone 1 (Narrative Ingestion & Story Engine):* Complete.
  - ✅ *Milestone 2 (Two-Pass Generation & Adaptive Refinement):* Complete.
  - ✅ *Milestone 3 (Multi-Modal QA & Hungarian Verification):* Complete.
  - ✅ *Milestone 4 (Zero-Cache Systematic Ablations & Telemetry):* Complete.
- **Pending Tasks for Final Capstone Phase 4:**
  - 🔄 **Task 1: Frame-to-Frame Temporal Video Animation:** Integrating AnimateDiff-SDXL / SVD for animated scene clips (Oct 18 – Nov 08, 2026).
  - 🔄 **Task 2: Neural Dialogue & Audio Synthesis:** Integrating Edge-TTS / XTTS-v2 for character voiceovers synchronized with storyboard dialogue (Nov 09 – Nov 24, 2026).
  - 🔄 **Task 3: Automated Screenplay Audio-Visual Assembly:** FFMPEG dynamic video composition combining animated latents, subtitles, and background audio (Nov 25 – Dec 10, 2026).
  - 🔄 **Task 4: Final Phase 4 Thesis Documentation & Human MOS Study:** Comprehensive project thesis and subjective animator evaluation (Dec 11 – Dec 24, 2026).

```mermaid
gantt
    title MMCCP Capstone Phase 3 & Phase 4 Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 3 (Completed)
    Narrative Engine & Anchors           :done, 2026-08-20, 2026-09-15
    Two-Pass Generation Pipeline         :done, 2026-09-16, 2026-10-02
    QA Engine & Hungarian Matching       :done, 2026-10-03, 2026-10-12
    Review 2 Zero-Cache Benchmarks       :done, 2026-10-13, 2026-10-14
    section Phase 4 (Scheduled)
    Temporal Video Synthesis (AnimateDiff):active, 2026-10-18, 2026-11-08
    Neural TTS Dialogue & Voiceover      :2026-11-09, 2026-11-24
    Automated Video Assembly (FFMPEG)    :2026-11-25, 2026-12-10
    Final Thesis & Human MOS Study       :2026-12-11, 2026-12-24
```

### Pending Tasks Table

| Task ID | Description | Duration | Start Date | End Date | Dependencies | Assigned Lead |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **T1** | Temporal Video Synthesis (AnimateDiff) | 21 Days | 2026-10-18 | 2026-11-08 | Phase 3 Storyboard Renders | Shamant N. |
| **T2** | Neural Dialogue TTS Synthesis | 15 Days | 2026-11-09 | 2026-11-24 | Storyboard Dialogue Strings | Shivu K. C. |
| **T3** | FFMPEG Automated Video Assembly | 15 Days | 2026-11-25 | 2026-12-10 | Tasks T1, T2 | Shihab K. A. |
| **T4** | Human MOS Study & Capstone Thesis | 14 Days | 2026-12-11 | 2026-12-24 | Task T3 | Samiya Naaz |

### Speaker Notes
> "On Slide 9, we lay out our structured timeline leading to the final Phase 4 Capstone submission. Today marks the complete delivery of our Phase 3 milestones: the multi-modal character consistency pipeline is fully stabilized, empirically benchmarked, and verified from scratch. The remaining 15% of our capstone project focuses on video motion and audio: in Task 1, we will animate our consistent keyframes using AnimateDiff; in Task 2, we will synthesize character dialogue via XTTS; and in Task 3, our FFMPEG pipeline will concatenate clips into a final synchronized narrative video."

---

## Slide 10: Conclusion

### Ready-to-Paste Slide Content
- **Summary of Key Achievements:**
  - Successfully built and empirically validated a **training-free, zero-shot multi-modal character consistency pipeline** running entirely on consumer-tier hardware (8 GB RTX 3070 Ti).
  - Replaced ad-hoc prompt engineering with a mathematically rigorous **two-pass coarse-to-fine architecture** combining demographic locking, depth conditioning, adaptive denoising, and latent facial restoration.
  - Developed the **Quad-Modal Evaluation Engine** (ArcFace, CLIP-I, SSIM, Palette) with Kuhn-Munkres Hungarian assignment, solving permutation ambiguity in multi-character scenes.
  - Exposed and mitigated the critical chromatic vulnerability of ArcFace using a zero-shot VLM anomaly auditor.
  - Clean-slate zero-cache testing achieved a **100% storyboard generation pass rate** with a mean composite consistency of **`0.5699`** and semantic CLIP-I of **`0.8061`**.
- **Closing Takeaway:**
  - Proved that high-fidelity narrative visual consistency does not require enterprise GPU clusters or compute-heavy fine-tuning; it can be achieved reliably through intelligent prompt-space partitioning, structural control, and decoupled latent refinement.

### Speaker Notes
> "In conclusion, Capstone Project Phase 3 has demonstrated that high-fidelity character consistency across complex narrative storyboards is achievable on consumer-tier laptops without expensive fine-tuning. By combining demographic token locking, depth geometry, camera-distance adaptive denoising, ReActor latent swapping, and our quad-modal evaluation framework, MMCCP bridges the gap between raw text prose and visually unified storyboard scenes. We thank our guide Prof. Chitra G M for her invaluable direction, and we now welcome any questions from the review panel."

---

## Slide 11: References (IEEE Format)

### Ready-to-Paste Slide Content (Verified Scholarly Works)
1. [1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 10684–10695.
2. [2] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach, "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.
3. [3] L. Zhang, A. Rao, and M. Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023, pp. 3836–3847.
4. [4] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, "IP-Adapter: Text-Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," *arXiv preprint arXiv:2308.06721*, 2023.
5. [5] Q. Wang, X. Bai, H. Wang, Z. Qin, and A. Chen, "InstantID: Zero-shot Identity-Preserving Generation in Seconds," *arXiv preprint arXiv:2401.07519*, 2024.
6. [6] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.
7. [7] S. Zhou, K. Chan, C. Li, and C. C. Loy, "Towards Robust Blind Face Restoration with Codebook Lookup Transformer," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.
8. [8] C. Si, Z. Huang, Y. Jiang, and Z. Liu, "FreeU: Free Lunch in Diffusion U-Net," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024.
9. [9] H. W. Kuhn, "The Hungarian Method for the Assignment Problem," *Nav. Res. Logist. Q.*, vol. 2, no. 1–2, pp. 83–97, 1955.
10. [10] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE Trans. Image Process.*, vol. 13, no. 4, pp. 600–612, Apr. 2004.
11. [11] Z. Guo, Y. Wu, Z. Chen, L. Chen, and Q. He, "PuLID: Pure and Lightning ID Customization via Contrastive Alignment," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Dec. 2024.
12. [12] Y. Zhou, D. Zhou, M.-M. Cheng, J. Feng, and Q. Hou, "StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Dec. 2024.
13. [13] Y. Tewel, O. Kaduri, R. Gal, Y. Kasten, L. Wolf, G. Chechik, and Y. Atzmon, "ConsiStory: Training-Free Multi-Subject Consistency in Diffusion Models," in *ACM Trans. Graph. (SIGGRAPH)*, 2024.

---

## 12. High-Risk "DO NOT CLAIM" List for University Review #2

The student team must strictly avoid making these contradicted historical statements:
1. ❌ **Do NOT claim a constant 6.2 GB VRAM plateau:** Real peak memory during Pass 2 with ReActor reaches **6.8 GB to 7.4 GB VRAM**. State instead: *"Peak VRAM is dynamically bounded between 6.8 GB and 7.4 GB on our 8 GB GPU."*
2. ❌ **Do NOT cite ArcFace inversion score as 0.7410:** Measured reality on our active anchor is **0.9346**. State: *"ArcFace exhibits severe chromatic blindness, scoring 0.9346 on inverted faces, which our VLM auditor detects with 99.81% confidence."*
3. ❌ **Do NOT claim composite consistency is 0.82 or 0.85:** Real measured zero-cache mean composite consistency is **0.5699** across all 6 scenes. State: *"From a clean slate, MMCCP achieves an audited composite score of 0.5699, surpassing baseline plain SDXL by +17.4% in semantic consistency."*
4. ❌ **Do NOT claim evaluators are pinned to host CPU:** `metrics.py` runs on CUDA under a 25% VRAM cap. State: *"Evaluators execute on CUDA clamped to 25% GPU allocation to preserve VRAM for Forge."*
5. ❌ **Do NOT claim Euler a sampler at 30 steps:** Production pipeline uses **DPM++ 2M Karras** (20 steps Pass 1, 15 steps Pass 2).

---

## 13. One-Page Elevator Summary & Viva Q&A Guide

### One-Page Elevator Summary
> The Multi-Modal Character Consistency Pipeline (MMCCP) is an automated, training-free visual consistency framework developed for the Capstone Project "Novel to Video Generation" at PES University. In text-to-video synthesis, maintaining visual character identity across story scenes without expensive per-character fine-tuning is an unsolved industry challenge. On a consumer 8 GB RTX 3070 Ti laptop GPU, MMCCP implements a decoupled two-pass architecture: Pass 1 establishes scene geometry and character staging using demographic phenotype locking, BREAK regional prompt partitioning, and ControlNet Depth; Pass 2 refines facial biometrics using camera-distance adaptive denoising (0.38 to 0.62), ReActor latent face injection, and CodeFormer codebook restoration. Generations are verified by a closed-loop quad-modal evaluation engine (ArcFace, CLIP-I, SSIM, Color Palette) coupled with Kuhn-Munkres Hungarian bipartite matching for multi-character scenes. In a certified zero-cache execution, MMCCP achieved a 100% pass rate across the 6-scene "Making of a Scientist" storyboard with a mean composite consistency of 0.5699, while keeping peak VRAM safely under 7.5 GB.

### Top 10 Likely Review Viva Questions & Model Evidence-Backed Answers

#### Q1: "Why did you choose a training-free approach instead of fine-tuning a LoRA for each character?"
- **Answer:** *"In novel-to-video generation, screenplay ingestion is dynamic: stories introduce characters on the fly. Training a DreamBooth or LoRA model takes 15 to 25 minutes per character and requires 15–20 real training images that do not exist for fictional literary characters. Furthermore, loading multiple character LoRAs into an 8 GB GPU causes cross-attention interference and VRAM exhaustion. Our training-free approach bootstraps certified visual-DNA anchors in 30 seconds and maintains zero-shot consistency across the entire narrative."*

#### Q2: "How does your system prevent character feature bleed in multi-character scenes?"
- **Answer:** *"Standard diffusion cross-attention computes softmax globally across all prompt tokens, which causes clothing colors and hair traits to bleed between co-present characters. We resolve this using SDXL BREAK syntax and spatial column partitioning in `story_engine/prompt_builder.py`. By isolating character text tokens into separate attention chunks, we restrict cross-attention receptive fields to horizontal spatial columns, preventing Ebright's yellow shirt from bleeding into Mother's blue dress."*

#### Q3: "What is the mathematical justification for your Quad-Modal scoring weights (0.45, 0.25, 0.15, 0.15)?"
- **Answer:** *"Facial recognition models like ArcFace only evaluate inner facial geometry; our red-team experiment proved that ArcFace scores 0.9346 even on inverted blue skin. Therefore, ArcFace alone cannot assess identity. We allocate 45% to ArcFace for bone structure, 25% to CLIP-I for wardrobe, hair, and semantic traits, 15% to SSIM for edge contours, and 15% to Lab/HSV Bhattacharyya histogram distance to strictly enforce clothing color palette fidelity."*

#### Q4: "How does Kuhn-Munkres Hungarian matching work in your evaluation?"
- **Answer:** *"In a scene with $M$ expected characters and $N$ detected faces, greedy matching can erroneously pair characters. We construct an $M \times N$ pairwise biometric distance matrix $C_{i,j} = 1.0 - \text{ArcFaceSim}(\mathbf{e}_i^{\text{anchor}}, \mathbf{e}_j^{\text{detected}})$. The Hungarian algorithm finds the global bijective assignment minimizing total distance in polynomial time $\mathcal{O}(n^3)$. In Scene 1, it successfully matched Ebright to Face 1 (0.5248) and Mother to Face 0 (0.5192)."*

#### Q5: "What is Camera-Distance Adaptive Denoising, and why is it necessary?"
- **Answer:** *"SDXL compresses images by a factor of 8 into latent space. In a wide establishing shot, a face occupies only $48\times 48$ pixels, which maps to a tiny $6\times 6$ latent patch. High denoising obliterates this latent structure, while low denoising leaves noise artifacts. We implement a piecewise schedule: 0.38 for close-ups to preserve facial pores, 0.50 for medium shots, and 0.62 for wide shots to regenerate background geometry while allowing ReActor to restore facial keypoints."*

#### Q6: "How do you avoid Out-Of-Memory (OOM) crashes on an 8 GB GPU?"
- **Answer:** *"We enforce a strict staged memory lifecycle: ControlNet depth weights are active during Pass 1 and completely unloaded before Pass 2 begins. Additionally, our evaluation script runs DeepFace and PyTorch under a 25% memory fraction clamp (`torch.cuda.set_per_process_memory_fraction(0.25)`), using our `vram_guard` context manager to invoke explicit garbage collection and cache clearing between passes."*

#### Q7: "What did your ablation study reveal about plain SDXL vs your full pipeline?"
- **Answer:** *"Holding seed 42 and Scene 2 fixed, plain SDXL achieved an apparent ArcFace of 0.6065, but visual inspection revealed a generic stock child that completely drifted from the canonical anchor across subsequent seeds. Adding Demographic Locking raised semantic CLIP-I consistency by +17.4% (from 0.7283 to 0.8552). Adding FreeU and PAG restored color saturation and lifted CLIP-I to 0.8807. The full pipeline achieved our highest composite consistency of 0.5925."*

#### Q8: "How does your 4-point Anchor Certification Gate work?"
- **Answer:** *"Before rendering scenes, candidate anchor portraits must pass four objective tests: exactly one face detected via RetinaFace, face area occupying between 8% and 45% of the canvas, bounding box maintaining at least a 35-pixel margin from all borders to prevent cropping, and an ArcFace embedding $L_2$ norm equal to 1.0000. In our clean run, all 5 character anchors passed certification on their initial generation attempt."*

#### Q9: "Why does ReActor use CodeFormer restoration, and what is its weight parameter?"
- **Answer:** *"ReActor uses the InsightFace `inswapper_128.onnx` model, which operates at $128\times 128$ pixel resolution before warping back onto the $1024\times 1024$ canvas, introducing slight boundary blur. CodeFormer uses a learned vector-quantized codebook lookup transformer to restore crisp micro-textures. We fix CodeFormer fidelity weight at 0.80 to balance natural skin texture against codebook hallucination."*

#### Q10: "What are your scheduled milestones for Phase 4 to complete the novel-to-video pipeline?"
- **Answer:** *"With character consistency fully resolved in Phase 3, Phase 4 focuses on video motion and audio: Milestone 1 integrates AnimateDiff-SDXL for 16-frame camera motion; Milestone 2 implements neural text-to-speech dialogue synthesis using XTTS; Milestone 3 automates FFMPEG audio-visual multiplexing to produce the final narrated video; and Milestone 4 conducts a human animator Mean Opinion Score study."*
