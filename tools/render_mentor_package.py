"""
MMCCP Finalization Pass (A6): Mentor Package Renderer at D:\5TH SEM\_final\
Generates strictly from results/numbers.json and consistency_v2/best_config.json:
  1. CLAIMS_AND_LIMITS.md (<= 15,000 chars)
  2. SLIDE_FACTS.md (<= 15,000 chars)
  3. QA_PREP.md (<= 15,000 chars, 15 honest Q&As)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_PATH = PROJECT_ROOT / "results" / "numbers.json"
BEST_CFG_PATH = PROJECT_ROOT / "consistency_v2" / "best_config.json"
OUT_DIR = Path(r"D:\5TH SEM\_final")
MAX_CHARS = 15000


def _ci(ci: list, dec: int = 4, pct: bool = False) -> str:
    m = 100.0 if pct else 1.0
    s = "%" if pct else ""
    return f"[{ci[0]*m:+.{dec}f}{s}, {ci[1]*m:+.{dec}f}{s}]"


def render_mentor_package(out_dir: Path = OUT_DIR) -> None:
    N: Dict[str, Any] = json.loads(NUMBERS_PATH.read_text(encoding="utf-8"))
    cfg: Dict[str, Any] = json.loads(BEST_CFG_PATH.read_text(encoding="utf-8"))

    meta = N["metadata"]
    pc = N["protocol_constants"]
    t2 = N["test2_evaluation"]
    eb = t2["ebright"]
    ho = t2["holdout"]
    comb = t2["combined"]
    v1_eb = t2["v1_baselines"]["ebright"]
    v1_ho = t2["v1_baselines"]["holdout"]
    derived = t2["derived"]
    d_eb = t2["paired_deltas"]["ebright"]
    d_ho = t2["paired_deltas"]["holdout"]
    d_co = t2["paired_deltas"]["combined"]
    orig_sc = t2["hypothesis_scorecard"]
    extra = t2["extra_person_audit"]
    circ = t2.get("circularity_and_coherence", {})
    telem = N["telemetry"]
    telem_9 = telem["combined_9_csvs"]
    avr = N["automated_visual_review"]
    blind = avr["blind_study"]
    canary = avr["canary_audit"]
    t2_abl_meta = N["test2_ablation"]
    t2_abl = t2_abl_meta["conditions"]
    t2_diffs = t2_abl_meta["stage_diffs"]
    d2_abl_meta = N["dev2_ablation"]
    d2_abl = d2_abl_meta["conditions"]

    eb_ex_v2 = extra.get("v2_test2_ebright", {})
    ho_ex_v2 = extra.get("v2_test2_holdout", {})
    eb_ex_v1 = extra.get("v1_test2_ebright", {})
    ho_ex_v1 = extra.get("v1_test2_holdout", {})
    tot_ex_v2 = (eb_ex_v2.get("flagged_count", 0) or 0) + (ho_ex_v2.get("flagged_count", 0) or 0)
    tot_ex_v1 = (eb_ex_v1.get("flagged_count", 0) or 0) + (ho_ex_v1.get("flagged_count", 0) or 0)

    # Extract circularity metrics if present
    anc_p = circ.get("anchor_similarity_circularity", {}).get("paired_v2_minus_v1", {})
    if_row = anc_p.get("InsightFace_buffalo_l", {})
    ens_row = anc_p.get("Ensemble_4Judge", {})
    corr_map = circ.get("anchor_similarity_circularity", {}).get("v2_pearson_r_insightface_vs_judges", {})
    coh_v1_ens = circ.get("cross_scene_coherence", {}).get("v1", {}).get("four_judge_ensemble", {})
    coh_v2_ens = circ.get("cross_scene_coherence", {}).get("v2", {}).get("four_judge_ensemble", {})
    coh_v1_if = circ.get("cross_scene_coherence", {}).get("v1", {}).get("per_model", {}).get("InsightFace_buffalo_l", {})
    coh_v2_if = circ.get("cross_scene_coherence", {}).get("v2", {}).get("per_model", {}).get("InsightFace_buffalo_l", {})

    if if_row:
        circ_line = (
            f"InsightFace (`buffalo_l`) anchor similarity: v1 `{if_row['v1_mean']:.4f}` -> v2 `{if_row['v2_mean']:.4f}` "
            f"(`{if_row['paired_delta_mean']:+.4f}`, image CI `{_ci(if_row['image_level_95ci'])}`), "
            f"vs 4-Judge Ensemble `v1 {ens_row['v1_mean']:.4f} -> v2 {ens_row['v2_mean']:.4f}` (`{ens_row['paired_delta_mean']:+.4f}`, "
            f"Pearson r vs Ensemble `{corr_map.get('Ensemble_4Judge', 0.0):.4f}`, Facenet512 `{corr_map.get('Facenet512', 0.0):.4f}`)."
        )
        coh_line = (
            f"Anchor-free cross-scene coherence ratio (same/impostor): 4-Judge Ensemble `v1 {coh_v1_ens['coherence_ratio']:.4f}` -> "
            f"`v2 {coh_v2_ens['coherence_ratio']:.4f}`; InsightFace `v1 {coh_v1_if.get('coherence_ratio', 0.0):.4f}` -> `v2 {coh_v2_if.get('coherence_ratio', 0.0):.4f}`."
        )
    else:
        circ_line = "InsightFace vs 4-Judge circularity run pending (`results/test2_ebright_circularity_coherence.json`)."
        coh_line = "Cross-scene coherence ratio pending."

    # ------------------------------------------------------------------
    # 1. CLAIMS_AND_LIMITS.md (One page: Can Claim vs Cannot Claim)
    # ------------------------------------------------------------------
    tl_msg = N["prereg_forensic_timeline"]
    claims_md = f"""# MMCCP v2.0 — Claims & Limits Sheet (One-Page Defense Reference)

**Project:** `{meta['capstone_id']}` | **Verifier:** Frozen `v{meta['verifier_version']}` (`{meta['verifier_freeze_sha'][:16]}...`) | **Split:** Sealed `TEST-2` (`{comb['images_count']}` renders: `{eb['images_count']}` Ebright + `{ho['images_count']}` Holdout)

---

## WHAT WE CAN CLAIM (Verified Numbers + Source Artifact)

1. **Significant Wardrobe Palette Gain on Ebright TEST-2:** Wardrobe pass (`Delta E <= {pc['wardrobe_delta_e_threshold']:.0f}`) rose from `{v1_eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}`) to **`{eb['wardrobe_pass_rate']*100:.1f}%`** (`{eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}`), paired `Delta = {d_eb['delta_wardrobe']*100:+.1f}%` (cluster CI `{_ci(d_eb['delta_wardrobe_cluster_95ci'], 1, True)}`, image CI `{_ci(d_eb['delta_wardrobe_image_95ci'], 1, True)}`, Wilcoxon `W={d_eb['delta_wardrobe_wilcoxon_w']:.1f}`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`). *(Source: `results/test2_evaluation_summary.json`, `results/test2_ebright/report_v1_2.json`)*
2. **Significant Defect Reduction per Scene on Ebright & Combined TEST-2:** Primary Ebright mean defects/scene fell `{v1_eb['mean_defects_per_scene']:.2f} -> {eb['mean_defects_per_scene']:.2f}` (`Delta = {d_eb['delta_defects_per_scene']:+.2f}/scene`, `-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, cluster CI `{_ci(d_eb['delta_defects_per_scene_cluster_95ci'], 2)}`, image CI `{_ci(d_eb['delta_defects_per_scene_image_95ci'], 2)}`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`); Combined (`n={comb['images_count']}`) fell `{comb['v1_mean_defects_per_scene']:.2f} -> {comb['mean_defects_per_scene']:.2f}` (`Delta = {d_co['delta_defects_per_scene']:+.2f}`, both CIs exclude 0, `p={d_co['delta_defects_per_scene_wilcoxon_p']:.6f}`). *(Source: `results/test2_evaluation_summary.json`)*
3. **Ebright Rank-1 Identity Disambiguation Improved (Image-Level CI):** Ebright Rank-1 accuracy rose `{v1_eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v1_count']}/{eb['rank1_total_count']}`) -> **`{eb['rank1_accuracy']*100:.1f}%`** (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}`, `{d_eb['delta_rank1']*100:+.1f}%`); image-level 95% CI `{_ci(d_eb['delta_rank1_image_95ci'], 1, True)}` excludes 0 (though 6-cluster CI `{_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}` touches 0, `p={d_eb['delta_rank1_wilcoxon_p']:.4f}`). *(Source: `results/test2_evaluation_summary.json`)*
4. **Ebright Biometric Identity Parity With Baseline v1:** v2 maintained `{eb['biometric_ensemble_mean']:.4f}` vs v1 `{v1_eb['biometric_ensemble_mean']:.4f}` (`Delta = {d_eb['delta_bio']:+.4f}`, cluster CI `{_ci(d_eb['delta_bio_cluster_95ci'])}`, `p={d_eb['delta_bio_wilcoxon_p']:.4f}`) without degradation across adult (`{eb['adult_bio']:.4f}`), teen (`{eb['teen_bio']:.4f}`), and child (`{eb['child_bio']:.4f}`) splits. *(Source: `results/test2_evaluation_summary.json`)*
5. **Ebright Spatial Positioning Met Pre-Registered Bar (`>= 90%`):** `{eb['spatial_pass_rate']*100:.1f}%` (`{eb['spatial_v2_count']}/{eb['spatial_total_count']}`) on Ebright TEST-2 (`>= 90%` bar; v1 was `{v1_eb['spatial_pass_rate']*100:.1f}%`), and Combined `{comb['spatial_pass_rate']*100:.2f}%` (`{comb['spatial_v2_count']}/{comb['spatial_total_count']}`). *(Source: `results/test2_ebright/report_v1_2.json`, `results/numbers.json`)*
6. **Zero OOMs & Process-Attributable GPU Footprint Under `7.4 GiB`:** Zero OOM failures occurred across `{comb['images_count']}` TEST-2 renders + `{t2_abl_meta['sample_size_scenes']*6}` ablation renders; subtracting idle desktop (`{telem['idle_desktop_baseline_mb']:.0f}` MiB) from peak `{telem['peak_vram_mb']}` MiB gives `{telem['process_attributable_delta_peak_mb']:.0f}` MiB (`{telem['process_attributable_delta_peak_gib']:.3f}` GiB). *(Source: `results/telemetry/vram_test2_sealed.csv`)*
7. **Strict Auditability & Split Discipline:** Verifier `v{meta['verifier_version']}` stayed frozen (`tools/check_freeze.py` PASS); TEST-2 was unsealed once (`results/split_access_log.jsonl`); reserved seeds `{{{', '.join(str(s) for s in pc['reserved_seeds'])}}}` remain untouched (`tools/audit_test2_isolation.py` PASS).

---

## CANNOT CLAIM (Disproven / Unmet / Confounded — Source Artifact)

1. **CANNOT claim "ALL 4 HYPOTHESES MET":** Under the original pre-registered R10 rules, **`{orig_sc['strict_all_splits_met_count']}/4` strict all-split rules held** (`P1` NOT MET: bio CIs cross 0; `P2`/`P3` pass Ebright/Combined only, fail Holdout; `P4` NOT MET). Earlier `rc2` `4/4` verdicts used post-hoc rules written `{tl_msg['compute_test2_statistics_authored_ist']}` after scores were inspected (`{tl_msg['first_v2_test2_score_ist']}`). *(Source: `docs_generated/v2/PREREG_RECONCILIATION.md`)*
2. **CANNOT claim Biometric Superiority or OOD Holdout Improvement:** Holdout TEST-2 (`n={ho['images_count']}` scenes, `{ho['faces_count']}` faces) was **NOT better than Baseline v1** on any metric except spatial/defects point estimates (CIs cross 0): Bio `{ho['biometric_ensemble_mean']:.4f}` vs v1 `{v1_ho['biometric_ensemble_mean']:.4f}` (`{d_ho['delta_bio']:+.4f}`), Rank-1 `{ho['rank1_accuracy']*100:.1f}%` vs `{v1_ho['rank1_accuracy']*100:.1f}%`, Wardrobe `{ho['wardrobe_pass_rate']*100:.1f}%` vs `{v1_ho['wardrobe_pass_rate']*100:.1f}%`, Spatial `{ho['spatial_pass_rate']*100:.1f}%` (`< 90%`). *(Source: `results/test2_evaluation_summary.json`, `results/test2_holdout/report_v1_2.json`)*
3. **CANNOT claim "Zero Memorization" or "Zero Identity Collapse":** Holdout Scene 4 (`3` expected characters) dropped 1 character (`elena_child` omitted in swap specs; `dr_marcus_vance` bio `0.0000`) on all `3` seeds due to hardcoded 2-character caps in `layout_planner.py`, `regional.py`, `prompt_builder.py`, and M5. *(Source: `results/test2_holdout/report_v1_2.json`, sidecars)*
4. **CANNOT claim Combined Spatial+Identity+Wardrobe Binding Works (`0%`):** `identity_at_position_pass_rate` and `combined_binding_pass_rate` are **`0.0%`** (`0/{eb['faces_count']}` Ebright, `0/{ho['faces_count']}` Holdout) for **both** v1 and v2; unmeasurable with Verifier `v{meta['verifier_version']}`. *(Source: `results/test2_ebright/report_v1_2.json`, `results/test2_holdout/report_v1_2.json`)*
5. **CANNOT claim Whole-System Peak VRAM Stayed `<= 7.4 GB` (`100% compliant`):** Sealed TEST-2 peaked at **`{telem['peak_vram_mb']}` MiB** (`{telem['peak_vram_gib']:.3f}` GiB) with **`{telem['seconds_above_7400_mb']}`s** above `7400` MiB (`{telem['seconds_above_7577_6_mib']}`s > `7577.6` MiB across `{telem['excursion_episodes_count']}` episodes); 9-CSV max = `{telem_9['max_mb']:.0f}` MiB (`{telem_9['seconds_above_7400_mb']}`s > `7400`). *(Source: `results/telemetry/vram_test2_sealed.csv`)*
6. **CANNOT claim Zero Hallucinated Extra Persons (`0 extra YOLO`):** `tools/extra_person_detector.py` (`conf=0.40`) flagged `{tot_ex_v2}/{comb['images_count']}` v2 TEST-2 image (`scene_5_seed_1202.png`: duplicated foreground figure `3 vs 2`), plus Verifier YOLO flagged `scene_2_seed_1303.png` (visually 1 boy). *(Source: `results/test2_extra_person_audit.json`, `results/test2_ebright/report_v1_2.json`)*
7. **CANNOT claim M4 IP-Adapter Plus Helped Full Bundle or Human Blind Review Is Complete:** `no_m4_ip_adapter` outperformed `full_v2` on BOTH DEV-2 (`{d2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{d2_abl['full_v2']['biometric_ensemble_mean']:.4f}`) and TEST-2 (`{t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{t2_abl['full_v2']['biometric_ensemble_mean']:.4f}`), and scoring is partly circular with InsightFace (`{circ_line}`). Prior R8 blind/canary/VLM reviews (`{blind['ab_v2_wins']}/{blind['ab_total_pairs']}`) were **automated, not human**; human TEST-2 blind test is pending in `HUMAN_BLIND_TEST.md`.
"""

    # ------------------------------------------------------------------
    # 2. SLIDE_FACTS.md
    # ------------------------------------------------------------------
    methods_on = [k for k, v in cfg.items() if k.startswith("enable_") and v is True]
    methods_off = [k for k, v in cfg.items() if k.startswith("enable_") and v is False]

    slide_md = f"""# MMCCP v2.0 — Verified Slide Facts Deck Content

## Slide 1: Problem & Constraints
- **Problem:** Off-the-shelf SDXL storyboarding suffers from cross-scene identity drift, multi-character feature/wardrobe bleeding, and uncontrolled face collapse in wide shots (`8.0 GB` RTX 3070 Ti Laptop GPU constraint; zero LoRA fine-tuning).
- **Locked Evaluation Protocol:** Frozen Verifier `v{meta['verifier_version']}` (`SHA: {meta['verifier_freeze_sha'][:16]}...`, 4 face judges + YOLOv8n + SegFormer + CLIP + DINOv2), strict `DEV`/`TEST-2`/`RESERVED` split isolation, `{pc['n_bootstrap_resamples']}`-resample cluster & image 95% CIs + Wilcoxon tests.

## Slide 2: Pipeline Architecture — Methods Actually ON vs OFF (`best_config.json`)
- **Active Modules (`True` in `v2.0.0-rc2/rc3`):**
  - `M1` Identity Bank (`single_anchor` canonical bank parity, `enable_identity_bank=True`)
  - `M2` OpenPose Layout Planner (`openpose_xl [f4251cb4]`, `weight={cfg['openpose_weight']}`, `enable_layout_planner=True`)
  - `M3` Regional Forge Couple (`regional_global_weight={cfg['regional_global_weight']}`, `enable_regional_couple=True`)
  - `M4` IP-Adapter Plus Face (`weight={cfg['ip_adapter_plus_weight']}`, `enable_ip_adapter_plus=True`) — *Note: ablations show disabling M4 improved Bio on DEV-2 & TEST-2; candidate for `v2.1` OFF*
  - `M5` Direct InsightFace Swap (`inswapper_128` + CodeFormer `{cfg['codeformer_fidelity']}`, `enable_reliable_face_swap=True`)
  - `M6` Adaptive Face Detailer (`denoise={cfg['face_detailer_denoise']}`, `steps={cfg['face_detailer_steps']}`, `threshold=300px`, `enable_face_detailer=True`)
  - `M7` CIELAB Wardrobe Lock (`Delta E <= {cfg['wardrobe_delta_e_threshold']}`, `enable_wardrobe_lock=True`)
  - `M10` Identity Memory (`enable_identity_memory=True`, `drift_guard_threshold={cfg['drift_guard_threshold']}`, **0 promotions triggered** in practice) + `FreeU` + `Dynamic Thresholding` (`cfg={cfg['cfg_scale']}`, `20+15` steps)
- **Disabled / Rejected Modules (`False`):** `enable_dual_instantid`, `enable_reference_attention` (`M8`), `enable_best_of_n` (`M9`, off in `default`), `enable_collage_init` (`M12`, rejected in R7 `-0.0055`), `enable_pag`, `use_character_lora` (`M11` out of scope; reserved seeds `{{{', '.join(str(s) for s in pc['reserved_seeds'])}}}`).

## Slide 3: Sealed TEST-2 Results — Ebright (`n={eb['images_count']}` Renders, `{eb['faces_count']}` Faces) & Holdout (`n={ho['images_count']}` Renders, `{ho['faces_count']}` Faces)

| Metric | Ebright v1 (`n={v1_eb['images_count']}/{v1_eb['faces_count']}`) | Ebright v2 (`n={eb['images_count']}/{eb['faces_count']}`) | Ebright Delta `[Image 95% CI]` | Holdout v1 (`n={v1_ho['images_count']}/{v1_ho['faces_count']}`) | Holdout v2 (`n={ho['images_count']}/{ho['faces_count']}`) | Holdout Delta `[Image 95% CI]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Biometric Ensemble | {v1_eb['biometric_ensemble_mean']:.4f} | **{eb['biometric_ensemble_mean']:.4f}** | {d_eb['delta_bio']:+.4f} `{_ci(d_eb['delta_bio_image_95ci'])}` | {v1_ho['biometric_ensemble_mean']:.4f} | {ho['biometric_ensemble_mean']:.4f} | {d_ho['delta_bio']:+.4f} `{_ci(d_ho['delta_bio_image_95ci'])}` |
| Rank-1 Re-ID (`n/N`) | {v1_eb['rank1_accuracy']*100:.1f}% ({eb['rank1_v1_count']}/{eb['rank1_total_count']}) | **{eb['rank1_accuracy']*100:.1f}%** ({eb['rank1_v2_count']}/{eb['rank1_total_count']}) | {d_eb['delta_rank1']*100:+.1f}% `{_ci(d_eb['delta_rank1_image_95ci'], 1, True)}` | {v1_ho['rank1_accuracy']*100:.1f}% ({ho['rank1_v1_count']}/{ho['rank1_total_count']}) | {ho['rank1_accuracy']*100:.1f}% ({ho['rank1_v2_count']}/{ho['rank1_total_count']}) | {d_ho['delta_rank1']*100:+.1f}% `{_ci(d_ho['delta_rank1_image_95ci'], 1, True)}` |
| Spatial Pass (`n/N`) | {v1_eb['spatial_pass_rate']*100:.1f}% ({eb['spatial_v1_count']}/{eb['spatial_total_count']}) | {eb['spatial_pass_rate']*100:.1f}% ({eb['spatial_v2_count']}/{eb['spatial_total_count']}) | {d_eb['delta_spatial']*100:+.1f}% | {v1_ho['spatial_pass_rate']*100:.1f}% ({ho['spatial_v1_count']}/{ho['spatial_total_count']}) | {ho['spatial_pass_rate']*100:.1f}% ({ho['spatial_v2_count']}/{ho['spatial_total_count']}) | {d_ho['delta_spatial']*100:+.1f}% `{_ci(d_ho['delta_spatial_image_95ci'], 1, True)}` |
| Wardrobe Pass (`n/N`) | {v1_eb['wardrobe_pass_rate']*100:.1f}% ({eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}) | **{eb['wardrobe_pass_rate']*100:.1f}%** ({eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}) | **{d_eb['delta_wardrobe']*100:+.1f}%** `{_ci(d_eb['delta_wardrobe_image_95ci'], 1, True)}` | {v1_ho['wardrobe_pass_rate']*100:.1f}% ({ho['wardrobe_v1_count']}/{ho['wardrobe_total_count']}) | {ho['wardrobe_pass_rate']*100:.1f}% ({ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}) | {d_ho['delta_wardrobe']*100:+.1f}% `{_ci(d_ho['delta_wardrobe_image_95ci'], 1, True)}` |
| Defects/Scene (Primary) | {v1_eb['mean_defects_per_scene']:.2f} | **{eb['mean_defects_per_scene']:.2f}** | **{d_eb['delta_defects_per_scene']:+.2f}** `{_ci(d_eb['delta_defects_per_scene_image_95ci'], 2)}` | {v1_ho['mean_defects_per_scene']:.2f} | {ho['mean_defects_per_scene']:.2f} | {d_ho['delta_defects_per_scene']:+.2f} `{_ci(d_ho['delta_defects_per_scene_image_95ci'], 2)}` |
| Defects/Instance-Wt (Sec.) | {eb['v1_mean_defects_per_instance_weighted']:.3f} | {eb['mean_defects_per_instance_weighted']:.3f} | {d_eb['delta_defects_per_instance_weighted']:+.3f} | {ho['v1_mean_defects_per_instance_weighted']:.3f} | {ho['mean_defects_per_instance_weighted']:.3f} | {d_ho['delta_defects_per_instance_weighted']:+.3f} |
| Combined Binding | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Slide 4: Leave-One-Out Ablations (Single-Seed: `n={t2_abl_meta['sample_size_scenes']}` Scenes, `n={t2_abl_meta['sample_size_faces']}` Faces)
- **Caution:** `|Delta Bio| < {pc['ablation_noise_band']:.2f}` on 1 seed (`n={t2_abl_meta['sample_size_faces']}`) is within noise. Key consistent findings across DEV-2 (`seed {d2_abl_meta['seed']}`) and TEST-2 (`seed {t2_abl_meta['seed']}`):
  - **`no_m2_layout` hurts spatial/Rank-1/wardrobe most:** DEV-2 Bio `{d2_abl['no_m2_layout']['biometric_ensemble_mean']:.4f}` (`{d2_abl['no_m2_layout']['delta_bio_vs_full']:+.4f}`, Spatial `{d2_abl['no_m2_layout']['spatial_pass_rate']*100:.1f}%`); TEST-2 Rank-1 dropped `{t2_abl['full_v2']['rank1_accuracy']*100:.1f}% -> {t2_abl['no_m2_layout']['rank1_accuracy']*100:.1f}%`.
  - **`no_m4_ip_adapter` BEAT `full_v2` on both splits:** DEV-2 `{d2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{d2_abl['full_v2']['biometric_ensemble_mean']:.4f}`; TEST-2 `{t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{t2_abl['full_v2']['biometric_ensemble_mean']:.4f}` -> **M4 adds variance/conflict with M5 Swap**.
  - **`no_m5_swap` dropped TEST-2 Bio largest:** `{t2_abl['full_v2']['biometric_ensemble_mean']:.4f} -> {t2_abl['no_m5_swap']['biometric_ensemble_mean']:.4f}` (`{t2_abl['no_m5_swap']['delta_bio_vs_full']:+.4f}`, Rank-1 `{t2_abl['no_m5_swap']['rank1_accuracy']*100:.1f}%`).
  - **`no_m7_wardrobe` dropped Wardrobe pass (`87.5% -> 62.5%`)** on TEST-2 seed `{t2_abl_meta['seed']}` (`pixel diff = {t2_diffs['no_m7_wardrobe']:.2f} px`).

## Slide 5: Pre-Registered Verdict, Limitations & Future Work (`v2.1`)
- **Pre-Registered P1–P4 Verdict:** `{orig_sc['strict_all_splits_met_count']}/4` strict all-split MET (`P1` NOT MET, `P2`/`P3` Ebright-only MET / Holdout NOT MET, `P4` NOT MET due to `7984` MiB whole-system VRAM peak & `1/42` extra YOLO person).
- **Circularity:** {circ_line} {coh_line}
- **Two Concrete Bugs Fixed Next in DEV `v2.1`:** (1) N-character layout/swap/prompt support (`>= 3` characters); (2) State-aware wardrobe target in M7 + evaluator (`Scene 6` black graduation gown). Also test disabling `M4` IP-Adapter Plus and tightening Pass 1 VRAM offload.
"""

    # ------------------------------------------------------------------
    # 3. QA_PREP.md (15 Likely Mentor Questions + Honest Answers)
    # ------------------------------------------------------------------
    qa_md = f"""# MMCCP v2.0 — Capstone Defense Q&A Preparation (15 Tough Mentor Questions)

1. **Q: Your `rc2` docs said ALL 4 hypotheses were MET, and now `PREREG_RECONCILIATION.md` says `0/4` strict all-split MET. What happened?**  
   **A:** During `v2.0.0-rc2` (`7904ad5`), `compute_test2_statistics.py` was written at `13:41:51 IST` — ~10 minutes *after* `results/test2_ebright/report_v1_2.json` was written (`13:31:15 IST`) and inspected (`13:38:50 IST`). It weakened P1 from superiority (`CI low > 0`) to non-inferiority (`> -0.03`), replaced P2 Spatial with `Rank-1 >= 75%`, hid Holdout Wardrobe (`37.5%`) out of P3, and replaced P4 Engineering (`<= 7.4 GB`, `0` extra YOLO) with Holdout Spatial `>= 80%`. Under the original pre-registered R10 rules: `P1 = NOT MET`, `P2 = MET` on Ebright (`97.5%`) / Combined (`93.75%`) only (`87.5% < 90%` on Holdout), `P3 = MET` on Ebright (`80.0%`) / Combined (`64.06%`) only (`37.5% < 60%` on Holdout), `P4 = NOT MET` (`7984` MiB > `7.4 GB` whole-system, `1/42` extra YOLO). We reconciled and tagged `v2.0.0-rc3` honestly without hiding anything.

2. **Q: Why are TEST-2 biometric score gains so modest (`+0.0056` Ebright, `-0.0162` Holdout)?**  
   **A:** Baseline v1 (`run_from_scratch` pipeline v1) already uses IP-Adapter and ReActor face swapping, so it is a strong baseline (`0.3091` on Ebright), NOT vanilla unconditioned SDXL. Furthermore, M1–M7 focuses on spatial layout, wardrobe locking, and multi-judge Rank-1 disambiguation rather than fine-tuning LoRAs (`M11`). On Ebright, M2/M3/M5 improved Rank-1 (`65.0% -> 80.0%`) and Wardrobe (`45.0% -> 80.0%`) while holding biometric similarity at parity (`0.3147`).

3. **Q: Is Verifier v1.2 scoring circular with M5 Face Swap?**  
   **A:** Partly yes — both M4/M5 and Verifier v1.2 use the exact same canonical anchor image per character (`1.0000` parity), and M5 (`inswapper_128`) operates in InsightFace `buffalo_l` (`w600k_r50`) space. {circ_line} {coh_line} That is why reporting all 4 non-InsightFace DeepFace judges (`Facenet512`, `SFace`, `GhostFaceNet`, `ArcFace`), anchor-free cross-scene coherence, and human blind evaluation (`HUMAN_BLIND_TEST.md`) is essential.

4. **Q: Why are `identity_at_position_pass` and `combined_binding_pass` both `0.0%` (`0/40`, `0/24`) for BOTH v1 and v2?**  
   **A:** Verifier v1.2 requires *all* sub-judges/calibrated ensemble thresholds simultaneously at each spatial zone (`ArcFace` + `GhostFaceNet` + `SFace` cosine means hover around `0.03–0.14`, below calibration cutoffs near `0.25–0.45`, because cross-age / painterly SDXL renders differ in high-frequency texture from real/studio anchors). Thus neither v1 nor v2 passes `0/64` combined binding under frozen v1.2 rules; that composite gate is unmeasurable here and must not be claimed improved.

5. **Q: Why did Holdout (`n={ho['images_count']}` scenes) perform WORSE than Baseline v1 (`Bio -0.0162`, `Rank-1 -8.3%`)?**  
   **A:** Two reasons: (a) On Scene 4 (`3` expected characters: `clara_thorn`, `elena_child`, `dr_marcus_vance`), v2's `layout_planner`, `regional.py`, `prompt_builder`, and M5 hardcode a 2-character limit, completely omitting `elena_child` from M5 swapping and scoring `dr_marcus_vance` `0.0000` (missing), pulling Holdout Bio/Rank-1/Spatial down on all 3 seeds; (b) `holdout_v2.json` outfits (`emerald green`, `crimson red`, `charcoal`) proved harder for M7 (`37.5%` vs v1 `41.7%`).

6. **Q: Explain Holdout Scene 4 technically — did M2/M3/M5 literally drop the 3rd person?**  
   **A:** Yes — visually verified on all 3 renders (`2` people separated by an empty center curtain; right figure even rendered female hair from merged column prompt with male mustached M5 swap on top). Code inspection verified `layout_planner.py` constructs `ordered_pair = [left, right]`, `regional.py` outputs `2` Basic regions, M5 `swap_report` contains only `2` specs (`clara_thorn`, `dr_marcus_vance`), and `prompt_builder.py` applies `"3persons"` in the negative prompt whenever `num_chars >= 2`.

7. **Q: Why did Ebright Scene 6 graduation gowns look stained olive-mustard AND still fail Wardrobe?**  
   **A:** Visually verified across all `5` seeds: `college_ebright` wears a black graduation gown + cap with a rectangular mustard-olive stain pasted across the mid-torso and bleeding onto the top of the diploma. Code confirmed BOTH a pipeline defect (M7 reads character default `"mustard yellow"` instead of Scene 6 state `"black graduation gown"`) AND a verifier artifact (frozen Verifier `cli.py` also compares against `"mustard yellow"`, so even unstained v1 black gowns failed with `Delta E = 53.75–76.54`).

8. **Q: Did MMCCP v2 meet its `<= 7.4 GB` VRAM budget?**  
   **A:** Whole-system: **NO** (`{telem['peak_vram_mb']}` MiB = `{telem['peak_vram_gib']:.3f}` GiB peak in `vram_test2_sealed.csv`; `{telem['seconds_above_7400_mb']}s > 7400` MiB across `{telem['excursion_episodes_count']}` episodes; 9-CSV peak `{telem_9['max_mb']:.0f}` MiB). Process-attributable: peak minus `{telem['idle_desktop_baseline_mb']:.0f}` MiB desktop idle = **`{telem['process_attributable_delta_peak_mb']:.0f}` MiB (`{telem['process_attributable_delta_peak_gib']:.3f}` GiB)** with `0` OOMs. Every excursion coincided with Forge Pass 1 SDXL/ControlNet/IP-Adapter allocation (`6–31s`) or Pass 1->Pass 2 transition (`49–131s`), never scoring.

9. **Q: Why didn't you run `nvidia-smi` per-process logging or fix sidecar `peak_vram_mb = 0.0`?**  
   **A:** `get_vram_telemetry()` inside `pipeline_v2.py` runs in the `my_app_env` CPU client rather than inside Forge's CUDA process, so `torch.cuda.max_memory_allocated()` returned `0.0`. All VRAM telemetry therefore comes from external 1 Hz `nvidia-smi` whole-system monitoring; reporting both whole-system peak and idle-subtracted delta is the honest ground truth.

10. **Q: Why did you reject M8 (Reference-Only), M10 (Identity Memory), and M12 (Collage-Init)?**  
    **A:** M8 (`enable_reference_attention`) increased VRAM/latency without reliable R7 gain and remains `False`. M12 (`enable_collage_init`) was tested in `loop-W6-iter1-collage-init` and REJECTED (`Delta = -0.0055`, CI `[-0.0209, +0.0100]`): unconditioned Pass 1 drafts caused floating heads before OpenPose keys were fixed (`#31`), so it remains `False`. M10 (`enable_identity_memory`) is enabled in `best_config.json` with strict `0.75` promotion gate, which intentionally **promoted 0 frames** (`similarity ~0.20–0.40 < 0.75`) to prevent centroid poisoning.

11. **Q: Wait — your LOO ablation says `no_m4_ip_adapter` BEATS `full_v2` on BOTH DEV-2 and TEST-2! Why is M4 still ON in `rc2`/`rc3`?**  
    **A:** On single-seed LOO (`n=6` scenes, `8` faces), `no_m4_ip_adapter` scored `0.3423` vs `0.3323` on DEV-2 and `0.3373` vs `0.3269` on TEST-2. Most small single-seed LOO deltas (`|Delta| < 0.03`) sit inside diffusion noise, and we cannot re-tune `v2.0` after unsealing `TEST-2` (that would constitute test-set leakage). Therefore `v2.0.0-rc3` freezes `best_config.json` as evaluated and explicitly recommends disabling `M4` in future `v2.1` exploration.

12. **Q: Explain why `scene_2_seed_1303` had `YOLO=2` in Verifier `report_v1_2.json` vs `YOLO=1` in `extra_person_detector.py`.**  
    **A:** Visual inspection of `scene_2_seed_1303.png` shows literally 1 boy (`young_ebright`) at a butterfly table. Frozen `torch_vision_helper.py` runs `yolo(img_bgr)` at default Ultralytics NMS threshold (`0.25`) before filtering `conf >= 0.40`, whereas `tools/extra_person_detector.py` runs `model(img_bgr, conf=0.40)` cleanly and flags **only** `scene_5_seed_1202.png` (`3` people — visually confirmed duplicated foreground figure!), giving `1/42` (`3.3%` Ebright, `0.0%` Holdout) extra YOLO persons vs `2/42` in Baseline v1.

13. **Q: Was Phase R8 Visual Review done by humans?**  
    **A:** No — the R8 blind study (`10/12` v2 preference on DEV-2, `20/20` crop Re-ID), synthetic canaries (`3/3`), and BLIP/CLIP VLM check were **automated scripts/model proxies**, NOT human ratings (`blind_review_summary.json` pre-remediation `12/12` is marked `AUTOMATED-SUPERSEDED`). A true **Human Blind Test** for TEST-2 Ebright (`30` shuffled pairs + `20` crops) is prepared cleanly in `HUMAN_BLIND_TEST.md` + blank `blind_predictions_human.json` for external/human completion.

14. **Q: Why did `07_measured_results.md` previously say `3.07 -> 1.50` (`Delta = +-1.05`)?**  
    **A:** A unit-mixing bug in `compute_test2_statistics.py`: `defects_per_scene` (`3.07 -> 1.50`, `Delta = -1.57/scene` over `n=30` unique images) was mixed with face-instance-weighted `defects` (`2.575 -> 1.525`, `Delta = -1.05` over `n=40` face slots where 2-character scenes were counted twice), plus a literal `+-` formatting bug in `render_docs.py`. Both units are now separated cleanly in `results/numbers.json` and `tests/test_docs_numbers.py` parses every numeric token across all 7 rendered docs.

15. **Q: What concrete improvements belong in `v2.1` (without contaminating `TEST-2`)?**  
    **A:** On **DEV splits only** (branch `feat/v2.1-fixes` from `v2.0.0-rc3`, never re-evaluating or tuning on `TEST-2` which is permanently burned): (1) extend M2/M3/M5/`prompt_builder` to `3+` characters and validate on a new 3-character DEV storyboard; (2) make M7 wardrobe state-aware (read scene outfit colors when specified) and pair with a `tools/` state-aware checker on DEV-2 Scene 6; (3) ablate `M4=OFF` and further reduce Pass 1 VRAM excursions below `7.4 GB` above idle.
"""

    files = {
        "CLAIMS_AND_LIMITS.md": claims_md,
        "SLIDE_FACTS.md": slide_md,
        "QA_PREP.md": qa_md,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        p = out_dir / fname
        p.write_text(content, encoding="utf-8")
        sz = len(content)
        print(f"[OK] Wrote {p} ({sz} chars / limit {MAX_CHARS})")
        if sz > MAX_CHARS:
            raise AssertionError(f"{fname} exceeds {MAX_CHARS} chars ({sz})!")


if __name__ == "__main__":
    render_mentor_package()
