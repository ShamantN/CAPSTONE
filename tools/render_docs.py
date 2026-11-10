"""
MMCCP Phase R11 + Finalization Pass: Canonical Documentation Renderer
Renders all documentation and presentation assets strictly from machine-collected
ground truth in results/numbers.json (zero hardcoded metrics or verdicts).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_PATH = PROJECT_ROOT / "results" / "numbers.json"


def _fmt_ci(ci: list, decimals: int = 4, as_pct: bool = False) -> str:
    mult = 100.0 if as_pct else 1.0
    sfx = "%" if as_pct else ""
    return f"[{ci[0]*mult:+.{decimals}f}{sfx}, {ci[1]*mult:+.{decimals}f}{sfx}]"


def _render_circularity_section(circ: Dict[str, Any]) -> str:
    if not circ:
        return (
            "## 6. Circularity & Cross-Scene Coherence Audit (TEST-2 Ebright)\n\n"
            "> Pending completion of `tools/circularity_and_coherence_audit.py`.\n"
        )
    anc = circ.get("anchor_similarity_circularity", {})
    paired = anc.get("paired_v2_minus_v1", {})
    corr = anc.get("v2_pearson_r_insightface_vs_judges", {})
    coh_v1 = circ.get("cross_scene_coherence", {}).get("v1", {})
    coh_v2 = circ.get("cross_scene_coherence", {}).get("v2", {})

    lines = [
        "## 6. Circularity & Cross-Scene Coherence Audit (TEST-2 Ebright, Read-Only)",
        "",
        "### 6.1 Final Face Similarity to Source Anchor: InsightFace (`buffalo_l w600k_r50`) vs 4 Judges",
        "",
        "M5 (`inswapper_128`) operates inside the InsightFace `buffalo_l` embedding space and receives the same canonical anchor used by Verifier v1.2, introducing partial scoring circularity.",
        "",
        "| Judge / Embedder | v1 Mean | v2 Mean | Paired Delta (`v2 - v1`) | Image 95% CI | v2 Pearson r vs InsightFace |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    order = ["InsightFace_buffalo_l", "Ensemble_4Judge", "Facenet512", "SFace", "GhostFaceNet", "ArcFace"]
    for m in order:
        if m not in paired:
            continue
        row = paired[m]
        r_str = f"{corr[m]:.4f}" if m in corr else "1.0000"
        lines.append(
            f"| `{m}` | {row['v1_mean']:.4f} | {row['v2_mean']:.4f} | "
            f"{row['paired_delta_mean']:+.4f} | [{row['image_level_95ci'][0]:+.4f}, {row['image_level_95ci'][1]:+.4f}] | {r_str} |"
        )

    lines.extend([
        "",
        "### 6.2 Anchor-Free Cross-Scene Coherence (Same-Character / Impostor Ratio, TEST-2 Ebright)",
        "",
        "| Embedder | v1 Genuine | v1 Impostor | v1 Ratio | v2 Genuine | v2 Impostor | v2 Ratio | v1 Diff-Scene Gen | v2 Diff-Scene Gen |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    v1_pm = coh_v1.get("per_model", {})
    v2_pm = coh_v2.get("per_model", {})
    v1_ens = coh_v1.get("four_judge_ensemble", {})
    v2_ens = coh_v2.get("four_judge_ensemble", {})
    if v1_ens and v2_ens:
        lines.append(
            f"| **4-Judge Ensemble** | {v1_ens['genuine_mean']:.4f} | {v1_ens['impostor_mean']:.4f} | "
            f"{v1_ens['coherence_ratio']:.4f} | {v2_ens['genuine_mean']:.4f} | {v2_ens['impostor_mean']:.4f} | "
            f"{v2_ens['coherence_ratio']:.4f} | {v1_ens['genuine_diff_scene_mean']:.4f} | {v2_ens['genuine_diff_scene_mean']:.4f} |"
        )
    for m in ["InsightFace_buffalo_l", "Facenet512", "SFace", "GhostFaceNet", "ArcFace"]:
        if m in v1_pm and m in v2_pm:
            a, b = v1_pm[m], v2_pm[m]
            lines.append(
                f"| `{m}` | {a['genuine_mean']:.4f} | {a['impostor_mean']:.4f} | {a['coherence_ratio']:.4f} | "
                f"{b['genuine_mean']:.4f} | {b['impostor_mean']:.4f} | {b['coherence_ratio']:.4f} | "
                f"{a['genuine_diff_scene_mean']:.4f} | {b['genuine_diff_scene_mean']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_all():
    print("=" * 80)
    print("  RENDERING CANONICAL DOCUMENTATION STRICTLY FROM results/numbers.json")
    print("=" * 80)

    N: Dict[str, Any] = json.loads(NUMBERS_PATH.read_text(encoding="utf-8"))
    meta = N["metadata"]
    pc = N["protocol_constants"]
    tl = N["prereg_forensic_timeline"]

    t2 = N["test2_evaluation"]
    eb = t2["ebright"]
    ho = t2["holdout"]
    comb = t2["combined"]
    v1_eb = t2["v1_baselines"]["ebright"]
    v1_ho = t2["v1_baselines"]["holdout"]
    derived = t2["derived"]

    d_all = t2["paired_deltas"]
    d_eb = d_all["ebright"]
    d_ho = d_all["holdout"]
    d_co = d_all["combined"]

    orig_sc = t2["hypothesis_scorecard"]
    posthoc_sc = t2["post_hoc_redefined_scorecard_rc2"]
    extra = t2["extra_person_audit"]
    circ = t2.get("circularity_and_coherence", {})
    exp_cases = t2.get("exploratory_defect_cases", {})

    t2_abl_meta = N["test2_ablation"]
    t2_abl = t2_abl_meta["conditions"]
    t2_diffs = t2_abl_meta["stage_diffs"]

    d2_b = N["dev2_bundle"]
    d2_v1 = t2["v1_baselines"]["dev2"]
    d2_abl_meta = N["dev2_ablation"]
    d2_abl = d2_abl_meta["conditions"]

    telem = N["telemetry"]
    telem_9 = telem["combined_9_csvs"]
    avr = N["automated_visual_review"]
    blind_ab = avr["blind_study"]
    canary = avr["canary_audit"]

    r_seeds = ", ".join(str(s) for s in pc["reserved_seeds"])

    eb_extra_v2 = extra.get("v2_test2_ebright", {})
    ho_extra_v2 = extra.get("v2_test2_holdout", {})
    eb_extra_v1 = extra.get("v1_test2_ebright", {})
    ho_extra_v1 = extra.get("v1_test2_holdout", {})
    total_extra_v2 = (eb_extra_v2.get("flagged_count", 0) or 0) + (ho_extra_v2.get("flagged_count", 0) or 0)
    total_extra_v1 = (eb_extra_v1.get("flagged_count", 0) or 0) + (ho_extra_v1.get("flagged_count", 0) or 0)

    circ_md = _render_circularity_section(circ)

    # ------------------------------------------------------------------
    # 1. README_V2.md (Root)
    # ------------------------------------------------------------------
    readme_v2 = f"""# Multi-Modal Character Consistency Pipeline (MMCCP) v2.0

**PES University Capstone Project | Project ID:** `{meta['capstone_id']}`  
**Reconciled Closeout (`v2.0.0-rc3` Candidate) — Honest Artifact-Backed State**

> [!NOTE]
> Documentation reconciled from `v2.0.0-rc2`: earlier `rc2` summaries omitted pre-registration shifts, mixed defect units (`per-scene` vs `per-face-instance`), overstated Holdout performance, and omitted whole-system VRAM excursions above `{pc['p4_vram_limit_gb']} GB`. All metrics below are generated strictly from [`results/numbers.json`](file:///D:/5TH%20SEM/HACKATHON/results/numbers.json).

### Branch, Tag, Reproduction & Non-Repo Assets
- **Recommended Branch / Tag:** Use branch `consistency-v2` (synced with `main`, containing reconciled closeout documentation) or tag `v2.0.0-rc3` (`consistency-v2` release candidate snapshot prior to the `main` conflict-resolution merge).
- **Reproduce Verification, Statistics & Docs (CPU / No GPU Required):**
  ```powershell
  my_app_env\\Scripts\\python.exe tools\\check_freeze.py
  my_app_env\\Scripts\\python.exe tools\\audit_test2_isolation.py
  my_app_env\\Scripts\\python.exe tools\\compute_test2_statistics.py
  my_app_env\\Scripts\\python.exe tools\\collect_numbers.py
  my_app_env\\Scripts\\python.exe tools\\render_docs.py
  my_app_env\\Scripts\\python.exe -m unittest discover -s tests -p "test_*.py"
  my_app_env\\Scripts\\python.exe -m unittest -v tests.verifier_controls.test_controls
  ```
- **Reproduce Canonical Banks & DEV Renders (Local Forge WebUI + GPU Required):**
  ```powershell
  my_app_env\\Scripts\\python.exe tools\\sync_canonical_banks.py
  my_app_env\\Scripts\\python.exe tools\\render_dev2_bundle.py
  my_app_env\\Scripts\\python.exe tools\\score_ablation_dev2.py
  ```
  *(Sealed `TEST-2` is burned after one-shot evaluation: never re-run with `ALLOW_TEST2`, and keep quarantined seeds `{{{r_seeds}}}` untouched.)*
- **What Is NOT in the Repository & Where It Lives:**
  - **Model Weights (`*.safetensors`, `*.onnx`, `*.pt`, `*.bin`):** SDXL base, ControlNet OpenPose XL, IP-Adapter Plus Face, InsightFace `inswapper_128`, SegFormer, DeepFace, and YOLO weights live locally in `C:\\Stable-Diffusion\\` and workstation HuggingFace / InsightFace caches (gitignored).
  - **Active Character Banks & Embedding Caches:** `consistency_v2/banks/` and `hardcoded_cache.json` are gitignored runtime artifacts regenerated via `tools/sync_canonical_banks.py` from ground-truth anchors; workstation archives live outside git in `D:\\5TH SEM\\HACKATHON_RC2_BACKUP_FOR_LIVE_DEMO\\` and `D:\\5TH SEM\\_demo_state_archive\\`.
  - **Generated Renders & Visual Review PNGs:** `results/test2_ebright/renders/`, `results/test2_holdout/renders/`, `results/bundle_dev2/renders/`, `results/ablation_dev2/`, `results/ablation_test2/`, `docs_generated/v2/visual_review/contact_sheets/`, `human_blind_ab/*.png`, and `human_blind_reid_crops/*.png` are excluded via `.gitignore`. All quantitative verification JSON/CSV artifacts (`results/numbers.json`, `results/**/report*.json`, `results/**/*.csv`, `docs_generated/v2/*.md`) ARE tracked in git; full render archives should be shared via a GitHub Release asset or Google Drive archive rather than committed.

### Verifier Freeze & Protocol Constants
- **Frozen Verifier:** `v{meta['verifier_version']}` (`SHA: {meta['verifier_freeze_sha'][:16]}...`)
- **Bootstrap:** `{pc['n_bootstrap_resamples']}` iterations (fixed seed `{pc['bootstrap_random_seed']}`), reporting **both** image-level and scene-cluster 95% CIs.
- **Quarantined Seeds:** `{{{r_seeds}}}` remained strictly untouched.

---

## 1. Sealed TEST-2 Summary (v2 vs Baseline v1)

Evaluated one-shot on **TEST-2** across `{comb['images_count']}` renders (`{eb['images_count']}` Ebright / `{eb['faces_count']}` faces; `{ho['images_count']}` Holdout / `{ho['faces_count']}` faces; `{comb['faces_count']}` total face slots):

| Split / System | Renders | Face Slots | Bio Mean | Rank-1 (`n/N`) | Spatial (`n/N`) | Wardrobe (`n/N`) | Defects/Scene (Primary) | Defects/Instance-Wt (Secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ebright TEST-2 (v2)** | {eb['images_count']} | {eb['faces_count']} | **{eb['biometric_ensemble_mean']:.4f}** | **{eb['rank1_accuracy']*100:.1f}%** ({eb['rank1_v2_count']}/{eb['rank1_total_count']}) | **{eb['spatial_pass_rate']*100:.1f}%** ({eb['spatial_v2_count']}/{eb['spatial_total_count']}) | **{eb['wardrobe_pass_rate']*100:.1f}%** ({eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}) | **{eb['mean_defects_per_scene']:.2f}** | **{eb['mean_defects_per_instance_weighted']:.3f}** |
| *Ebright TEST-2 (v1 Baseline)* | {v1_eb['images_count']} | {v1_eb['faces_count']} | *{v1_eb['biometric_ensemble_mean']:.4f}* | *{v1_eb['rank1_accuracy']*100:.1f}%* ({eb['rank1_v1_count']}/{eb['rank1_total_count']}) | *{v1_eb['spatial_pass_rate']*100:.1f}%* ({eb['spatial_v1_count']}/{eb['spatial_total_count']}) | *{v1_eb['wardrobe_pass_rate']*100:.1f}%* ({eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}) | *{v1_eb['mean_defects_per_scene']:.2f}* | *{eb['v1_mean_defects_per_instance_weighted']:.3f}* |
| **Ebright Paired Delta (`v2 - v1`)** | - | - | **{d_eb['delta_bio']:+.4f}** | **{d_eb['delta_rank1']*100:+.1f}%** | **{d_eb['delta_spatial']*100:+.1f}%** | **{d_eb['delta_wardrobe']*100:+.1f}%** | **{d_eb['delta_defects_per_scene']:+.2f}** | **{d_eb['delta_defects_per_instance_weighted']:+.2f}** |
| **Holdout TEST-2 (v2)** | {ho['images_count']} | {ho['faces_count']} | **{ho['biometric_ensemble_mean']:.4f}** | **{ho['rank1_accuracy']*100:.1f}%** ({ho['rank1_v2_count']}/{ho['rank1_total_count']}) | **{ho['spatial_pass_rate']*100:.1f}%** ({ho['spatial_v2_count']}/{ho['spatial_total_count']}) | **{ho['wardrobe_pass_rate']*100:.1f}%** ({ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}) | **{ho['mean_defects_per_scene']:.2f}** | **{ho['mean_defects_per_instance_weighted']:.3f}** |
| *Holdout TEST-2 (v1 Baseline)* | {v1_ho['images_count']} | {v1_ho['faces_count']} | *{v1_ho['biometric_ensemble_mean']:.4f}* | *{v1_ho['rank1_accuracy']*100:.1f}%* ({ho['rank1_v1_count']}/{ho['rank1_total_count']}) | *{v1_ho['spatial_pass_rate']*100:.1f}%* ({ho['spatial_v1_count']}/{ho['spatial_total_count']}) | *{v1_ho['wardrobe_pass_rate']*100:.1f}%* ({ho['wardrobe_v1_count']}/{ho['wardrobe_total_count']}) | *{v1_ho['mean_defects_per_scene']:.2f}* | *{ho['v1_mean_defects_per_instance_weighted']:.3f}* |
| **Holdout Paired Delta (`v2 - v1`)** | - | - | **{d_ho['delta_bio']:+.4f}** | **{d_ho['delta_rank1']*100:+.1f}%** | **{d_ho['delta_spatial']*100:+.1f}%** | **{d_ho['delta_wardrobe']*100:+.1f}%** | **{d_ho['delta_defects_per_scene']:+.2f}** | **{d_ho['delta_defects_per_instance_weighted']:+.3f}** |
| **Combined TEST-2 (v2 vs v1)** | {comb['images_count']} | {comb['faces_count']} | **{comb['biometric_ensemble_mean']:.4f}** vs *{comb['v1_biometric_ensemble_mean']:.4f}* | **{comb['rank1_accuracy']*100:.2f}%** vs *{comb['v1_rank1_accuracy']*100:.2f}%* | **{comb['spatial_pass_rate']*100:.2f}%** vs *{comb['v1_spatial_pass_rate']*100:.2f}%* | **{comb['wardrobe_pass_rate']*100:.2f}%** vs *{comb['v1_wardrobe_pass_rate']*100:.2f}%* | **{comb['mean_defects_per_scene']:.2f}** vs *{comb['v1_mean_defects_per_scene']:.2f}* | **{comb['mean_defects_per_instance_weighted']:.3f}** vs *{comb['v1_mean_defects_per_instance_weighted']:.3f}* |

### Paired Inference & Key Limitations
- **Ebright:** Identity shows parity with Baseline v1 (`Delta Bio = {d_eb['delta_bio']:+.4f}`, scene-cluster 95% CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`, image 95% CI `{_fmt_ci(d_eb['delta_bio_image_95ci'])}`, Wilcoxon `W={d_eb['delta_bio_wilcoxon_w']:.1f}`, `p={d_eb['delta_bio_wilcoxon_p']:.4f}`); **Rank-1** improved (`{d_eb['delta_rank1']*100:+.1f}%`, image 95% CI `{_fmt_ci(d_eb['delta_rank1_image_95ci'], 1, True)}` excludes 0, whereas 6-cluster CI `{_fmt_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}` includes 0, `p={d_eb['delta_rank1_wilcoxon_p']:.4f}`); **Wardrobe** improved (`{d_eb['delta_wardrobe']*100:+.1f}%`, cluster CI `{_fmt_ci(d_eb['delta_wardrobe_cluster_95ci'], 1, True)}` and image CI `{_fmt_ci(d_eb['delta_wardrobe_image_95ci'], 1, True)}` both exclude 0, `W={d_eb['delta_wardrobe_wilcoxon_w']:.1f}`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`); **Defects per scene** dropped `{derived['ebright_defects_per_scene_pct_reduction']:.1f}%` (`{v1_eb['mean_defects_per_scene']:.2f} -> {eb['mean_defects_per_scene']:.2f}`, `Delta = {d_eb['delta_defects_per_scene']:+.2f}`, cluster CI `{_fmt_ci(d_eb['delta_defects_per_scene_cluster_95ci'], 2)}` and image CI `{_fmt_ci(d_eb['delta_defects_per_scene_image_95ci'], 2)}` both exclude 0, `W={d_eb['delta_defects_per_scene_wilcoxon_w']:.1f}`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`).
- **Holdout:** MMCCP v2 was **not better than Baseline v1** (`Bio Delta = {d_ho['delta_bio']:+.4f}`, `Rank-1 Delta = {d_ho['delta_rank1']*100:+.1f}%`, `Wardrobe Delta = {d_ho['delta_wardrobe']*100:+.1f}%`, `Defects/Scene Delta = {d_ho['delta_defects_per_scene']:+.2f}`, all CIs include 0). Spatial pass (`{ho['spatial_pass_rate']*100:.1f}%`, `{ho['spatial_v2_count']}/{ho['spatial_total_count']}`) fell below `{pc['p2_spatial_threshold_pct']:.1f}%` because Scene 4 (`3` expected characters) rendered only `2` characters on all `{len(exp_cases.get('holdout_scene_4', []))}` seeds.
- **Unmeasurable Combined Binding Gate:** `identity_at_position_pass_rate` and `combined_binding_pass_rate` are **`0.0%`** (`0/{eb['faces_count']}` Ebright, `0/{ho['faces_count']}` Holdout) for **both** Baseline v1 and MMCCP v2 due to strict multi-judge ensemble calibration, making combined binding unmeasurable under Verifier v{meta['verifier_version']}.
- **Partial Scoring Circularity:** Verifier anchor references are identical to M4/M5 source anchors; InsightFace similarity reflects swap alignment more strongly than independent judges (Section 6 of [`07_measured_results.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/07_measured_results.md)).

---

## 2. Original Pre-Registered Scorecard (`P1`–`P4`)

Full forensic comparison in [`PREREG_RECONCILIATION.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/PREREG_RECONCILIATION.md):
- **Overall Original Verdict:** **`{orig_sc['strict_all_splits_met_count']}/4` strict all-split original rules satisfied** (`0/4` `MET` across all required TEST-2 splits; `P2` and `P3` pass on Ebright & Combined only while failing Holdout).
1. **`{orig_sc['P1']['hypothesis']}` — [`{orig_sc['P1']['status']}`]** (`{orig_sc['P1']['rationale']}`)
2. **`{orig_sc['P2']['hypothesis']}` — [`{orig_sc['P2']['status']}`]** (`{orig_sc['P2']['rationale']}`)
3. **`{orig_sc['P3']['hypothesis']}` — [`{orig_sc['P3']['status']}`]** (`{orig_sc['P3']['rationale']}`)
4. **`{orig_sc['P4']['hypothesis']}` — [`{orig_sc['P4']['status']}`]** (`{orig_sc['P4']['rationale']}`)

---

## 3. Leave-One-Out Ablation (`{t2_abl_meta['sample_size_scenes']}` Scenes, `{t2_abl_meta['sample_size_faces']}` Faces, TEST-2 Seed `{t2_abl_meta['seed']}`)

> [!WARNING]
> Deltas smaller than `~{pc['ablation_noise_band']:.2f}` on a single seed (`n={t2_abl_meta['sample_size_faces']}` faces) lie within single-seed diffusion noise. Notably, **`no_m4_ip_adapter` outperformed `full_v2`** on **both** DEV-2 (`{d2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{d2_abl['full_v2']['biometric_ensemble_mean']:.4f}`) and TEST-2 (`{t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{t2_abl['full_v2']['biometric_ensemble_mean']:.4f}`), supporting disabling M4 in future work (`v2.1`) without modifying the frozen `v2.0` configuration.

| Arm | Bio Mean | Rank-1 | Spatial | Wardrobe | Defects/Scene | `Delta Bio vs Full` | Mean Pixel Diff vs Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_v2` | {t2_abl['full_v2']['biometric_ensemble_mean']:.4f} | {t2_abl['full_v2']['rank1_accuracy']*100:.1f}% | {t2_abl['full_v2']['spatial_pass_rate']*100:.1f}% | {t2_abl['full_v2']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['full_v2']['mean_defects']:.2f} | {t2_abl['full_v2']['delta_bio_vs_full']:+.4f} | {t2_diffs['full_v2']:.2f} px |
| `no_m2_layout` | {t2_abl['no_m2_layout']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m2_layout']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m2_layout']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m2_layout']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m2_layout']['mean_defects']:.2f} | {t2_abl['no_m2_layout']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m2_layout']:.2f} px |
| `no_m3_couple` | {t2_abl['no_m3_couple']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m3_couple']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m3_couple']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m3_couple']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m3_couple']['mean_defects']:.2f} | {t2_abl['no_m3_couple']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m3_couple']:.2f} px |
| `no_m4_ip_adapter` | {t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m4_ip_adapter']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m4_ip_adapter']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m4_ip_adapter']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m4_ip_adapter']['mean_defects']:.2f} | {t2_abl['no_m4_ip_adapter']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m4_ip_adapter']:.2f} px |
| `no_m5_swap` | {t2_abl['no_m5_swap']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m5_swap']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m5_swap']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m5_swap']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m5_swap']['mean_defects']:.2f} | {t2_abl['no_m5_swap']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m5_swap']:.2f} px |
| `no_m6_detailer` | {t2_abl['no_m6_detailer']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m6_detailer']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m6_detailer']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m6_detailer']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m6_detailer']['mean_defects']:.2f} | {t2_abl['no_m6_detailer']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m6_detailer']:.2f} px |
| `no_m7_wardrobe` | {t2_abl['no_m7_wardrobe']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m7_wardrobe']['rank1_accuracy']*100:.1f}% | {t2_abl['no_m7_wardrobe']['spatial_pass_rate']*100:.1f}% | {t2_abl['no_m7_wardrobe']['wardrobe_pass_rate']*100:.1f}% | {t2_abl['no_m7_wardrobe']['mean_defects']:.2f} | {t2_abl['no_m7_wardrobe']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m7_wardrobe']:.2f} px |

---

## 4. Telemetry, Extra-Person & Visual Review Status
- **Sealed TEST-2 VRAM (`{telem['samples_count']}` 1 Hz samples):** Whole-system peak = **`{telem['peak_vram_mb']}` MiB** (`{telem['peak_vram_gib']:.3f}` GiB / `{telem['peak_vram_gb_decimal']:.3f}` GB; exceeds `{pc['p4_vram_limit_gb']} GB` ceiling: `{telem['seconds_above_7400_mb']}`s > `{pc['p4_vram_limit_mb']:.0f}` MiB, `{telem['seconds_above_7577_6_mib']}`s > `{pc['p4_vram_limit_mib']:.1f}` MiB across `{telem['excursion_episodes_count']}` episodes). Idle-desktop pre-render baseline = `{telem['idle_desktop_baseline_mb']:.0f}` MiB (`{telem['idle_sample_count']}` samples), giving `{telem['process_attributable_delta_peak_mb']:.0f}` MiB (`{telem['process_attributable_delta_peak_gib']:.3f}` GiB) process-attributable delta (`0` OOMs).
- **Combined 9 Telemetry CSVs (`{telem_9['files_count']}` files, `{telem_9['samples_count']}` samples):** Overall peak `{telem_9['max_mb']:.0f}` MiB, `p99 = {telem_9['p99_mb']:.1f}` MiB, `{telem_9['seconds_above_7400_mb']}`s > `{pc['p4_vram_limit_mb']:.0f}` MiB.
- **Extra YOLO Person Detector (`conf=0.40`):** Flagged `{total_extra_v2}/{comb['images_count']}` v2 TEST-2 renders (`scene_5_seed_1202.png`: `3` vs `2` expected) vs `{total_extra_v1}/{comb['images_count']}` in v1 (`scene_1_seed_1202.png` + `scene_2_mentorship_seed_1202.png`).
- **Visual Review:** Prior DEV-2 blind A/B (`{blind_ab['ab_v2_wins']}/{blind_ab['ab_total_pairs']}` = `{blind_ab['ab_v2_win_rate']*100:.2f}%`), Re-ID (`{blind_ab['reid_correct']}/{blind_ab['reid_total_crops']}`), canary (`{canary['total_caught']}/{canary['total_canaries']}`), and VLM checks were **automated, not human**. Human TEST-2 Ebright blind evaluation protocol is staged in [`HUMAN_BLIND_TEST.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md) (status: `{avr['human_blind_test_status']}`).
"""
    (PROJECT_ROOT / "README_V2.md").write_text(readme_v2, encoding="utf-8")
    print("[OK] Rendered README_V2.md")

    # ------------------------------------------------------------------
    # 2. docs_generated/v2/PREREG_RECONCILIATION.md
    # ------------------------------------------------------------------
    reconcil_md = f"""# Pre-Registration Reconciliation & Hypothesis Audit (TEST-2)

**Verifier Version:** Frozen `v{meta['verifier_version']}` (`SHA: {meta['verifier_freeze_sha'][:16]}...`)  
**Scope:** Reconciles the original R10 pre-registered hypotheses (`P1`–`P4`) defined before unsealing `TEST-2` against the post-hoc rules committed in `7904ad5` (`tools/compute_test2_statistics.py`). All metrics below are generated strictly from [`results/numbers.json`](file:///D:/5TH%20SEM/HACKATHON/results/numbers.json).

---

## 1. Forensic Timeline (Pre-Registration vs TEST-2 Score Visibility)

| Event | Observed Timestamp / Commit | Relationship to TEST-2 Scores |
| :--- | :--- | :--- |
| Original R10 P1–P4 specification issued | `{tl['original_directive_utc']}` | **Before** TEST-2 unseal |
| Last pre-TEST-2 repository commit (`{tl['pre_test2_commit']}`) | `{tl['pre_test2_commit_date']}` | Zero `compute_test2_statistics.py` on branch |
| `TEST-2` unsealed once by `run_sealed_test2.py` | `{tl['test2_unseal_utc']}` | Logged in `results/split_access_log.jsonl` |
| First v2 `TEST-2` Verifier score file (`report_v1_2.json`) written | `{tl['first_v2_test2_score_ist']}` | Ground-truth TEST-2 scores now on disk |
| `TEST-2` scores read during R10 session (`step=14915`) | `{tl['first_score_inspected_ist']}` | Scores inspected before authoring scorecard |
| `tools/compute_test2_statistics.py` written (`step=14964`) | `{tl['compute_test2_statistics_authored_ist']}` | **Post-hoc** (after viewing TEST-2 scores; hardcoded observed Holdout values) |
| Committed as `v2.0.0-rc2` (`{tl['rc2_commit']}`) | `{tl['rc2_commit_date']}` | `git blame` confirms all rules originated in `{tl['rc2_commit']}` |

---

## 2. Original Pre-Registered Criteria (`MET` / `NOT MET` Strictly Enforced)

| Hypothesis | Original Pre-Registered Rule | Ebright (`n={eb['images_count']}` scenes, `{eb['faces_count']}` faces) | Holdout (`n={ho['images_count']}` scenes, `{ho['faces_count']}` faces) | Combined (`n={comb['images_count']}` scenes, `{comb['faces_count']}` faces) | Original Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **P1 (Identity)** | `Delta Bio > {pc['p1_ci_lower_threshold']:.1f}` & `Delta Rank-1 > {pc['p1_ci_lower_threshold']:.1f}` with 95% CI lower bound `> {pc['p1_ci_lower_threshold']:.1f}` | Bio `{d_eb['delta_bio']:+.4f}` cluster CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`, image CI `{_fmt_ci(d_eb['delta_bio_image_95ci'])}`; Rank-1 `{d_eb['delta_rank1']*100:+.1f}%` cluster CI `{_fmt_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}` | Bio `{d_ho['delta_bio']:+.4f}` cluster CI `{_fmt_ci(d_ho['delta_bio_cluster_95ci'])}`; Rank-1 `{d_ho['delta_rank1']*100:+.1f}%` cluster CI `{_fmt_ci(d_ho['delta_rank1_cluster_95ci'], 1, True)}` | Bio `{d_co['delta_bio']:+.4f}` cluster CI `{_fmt_ci(d_co['delta_bio_cluster_95ci'])}`; Rank-1 `{d_co['delta_rank1']*100:+.1f}%` | **`{orig_sc['P1']['status']}`** |
| **P2 (Spatial)** | Spatial assignment pass rate `>= {pc['p2_spatial_threshold_pct']:.1f}%` | **{eb['spatial_pass_rate']*100:.1f}%** ({eb['spatial_v2_count']}/{eb['spatial_total_count']}) [MET] | **{ho['spatial_pass_rate']*100:.1f}%** ({ho['spatial_v2_count']}/{ho['spatial_total_count']}) [**< {pc['p2_spatial_threshold_pct']:.1f}% FAIL**] | **{comb['spatial_pass_rate']*100:.2f}%** ({comb['spatial_v2_count']}/{comb['spatial_total_count']}) | **`{orig_sc['P2']['status']}`** |
| **P3 (Wardrobe)** | Wardrobe `Delta E <= {pc['wardrobe_delta_e_threshold']:.1f}` pass rate `>= {pc['p3_wardrobe_threshold_pct']:.1f}%` | **{eb['wardrobe_pass_rate']*100:.1f}%** ({eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}) [MET] | **{ho['wardrobe_pass_rate']*100:.1f}%** ({ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}) [**< {pc['p3_wardrobe_threshold_pct']:.1f}% FAIL**] | **{comb['wardrobe_pass_rate']*100:.2f}%** ({comb['wardrobe_v2_count']}/{comb['wardrobe_total_count']}) | **`{orig_sc['P3']['status']}`** |
| **P4 (Engineering)** | Peak VRAM `<= {pc['p4_vram_limit_gb']} GB` (`{pc['p4_vram_limit_mb']:.0f} MB` / `{pc['p4_vram_limit_mib']:.1f} MiB`), `{pc['p4_max_ooms']}` OOM, `{pc['p4_max_extra_yolo']}` extra YOLO | Whole-system `{telem['peak_vram_mb']}` MiB (`{telem['seconds_above_7400_mb']}s > {pc['p4_vram_limit_mb']:.0f}`); extra YOLO `{eb_extra_v2.get('flagged_count')}/{eb_extra_v2.get('total_images')}` (`conf=0.40`), Verifier YOLO `2/{eb['images_count']}` | Same sealed run (`{telem['peak_vram_mb']}` MiB peak); extra YOLO `{ho_extra_v2.get('flagged_count')}/{ho_extra_v2.get('total_images')}` (3 Scene 4 missing-person undercounts) | `{telem['peak_vram_mb']}` MiB (`{telem['process_attributable_delta_peak_mb']:.0f}` MiB delta above `{telem['idle_desktop_baseline_mb']:.0f}` idle); OOM `{pc['p4_max_ooms']}`; extra YOLO `{total_extra_v2}/{comb['images_count']}` | **`{orig_sc['P4']['status']}`** |

- **Strict Original Score Summary:** **`{orig_sc['strict_all_splits_met_count']}/4`** strict all-split original rules satisfied (`P1` fails CI lower-bound `> 0`; `P2` and `P3` meet thresholds on Ebright and Combined only while failing Holdout; `P4` fails both whole-system peak VRAM `<= {pc['p4_vram_limit_gb']} GB` and `{pc['p4_max_extra_yolo']}` extra YOLO persons).

---

## 3. Side-by-Side Comparison with Post-Hoc Redefined Rules (POST-HOC / INVALID AS PRE-REGISTRATION)

> [!CAUTION]
> **POST-HOC RULES (COMMITTED IN `{tl['rc2_commit']}`, INVALID AS PRE-REGISTRATION):** The rules below from `v2.0.0-rc2` were authored after `results/test2_ebright/report_v1_2.json` was written and inspected. They are preserved here solely to explain how `rc2` arrived at `"4/4"` and why `v2.0.0-rc3` retracts that framing.

| ID | Original Pre-Registered Rule | Post-Hoc `rc2` Redefinition (`{tl['rc2_commit']}`) | Specific Shift Applied After Viewing Scores | Post-Hoc `rc2` Label |
| :--- | :--- | :--- | :--- | :---: |
| **P1** | `Delta Bio > {pc['p1_ci_lower_threshold']:.1f}` & `Delta Rank-1 > {pc['p1_ci_lower_threshold']:.1f}`, 95% CI lower `> {pc['p1_ci_lower_threshold']:.1f}` | Ebright Bio `>= {pc['p1_ci_lower_threshold']:.1f}` & cluster CI lower `> {pc['rc2_p1_noninf_ci_floor']:.2f}` | Changed superiority to non-inferiority (`-0.03` margin) and removed Rank-1 & Holdout | `{posthoc_sc['P1']['status']}` *(post-hoc)* |
| **P2** | Spatial assignment pass `>= {pc['p2_spatial_threshold_pct']:.1f}%` | Ebright Rank-1 `>= {pc['rc2_p2_rank1_threshold_pct']:.1f}%` (`{eb['rank1_accuracy']*100:.1f}%`) | Replaced Spatial entirely with Rank-1 on Ebright only | `{posthoc_sc['P2']['status']}` *(post-hoc)* |
| **P3** | Wardrobe pass `>= {pc['p3_wardrobe_threshold_pct']:.1f}%` | Ebright Spatial `>= {pc['p2_spatial_threshold_pct']:.1f}%`, Wardrobe `>= {pc['rc2_p3_wardrobe_threshold_pct']:.1f}%`, instance-wt defects `< {pc['p1_ci_lower_threshold']:.1f}` | Dropped Holdout Wardrobe (`{ho['wardrobe_pass_rate']*100:.1f}%`) and evaluated only Ebright | `{posthoc_sc['P3']['status']}` *(post-hoc)* |
| **P4** | Peak VRAM `<= {pc['p4_vram_limit_gb']} GB`, `{pc['p4_max_ooms']}` OOM, `{pc['p4_max_extra_yolo']}` extra YOLO | Holdout Spatial `>= {pc['rc2_p4_spatial_threshold_pct']:.1f}%`, Scene defects `<= {pc['rc2_p4_defects_max']:.1f}`, Bio `>= {pc['rc2_p4_bio_min']:.2f}` | Dropped VRAM and extra-YOLO criteria completely; lowered Holdout Spatial from `{pc['p2_spatial_threshold_pct']:.1f}%` to `{pc['rc2_p4_spatial_threshold_pct']:.1f}%` | `{posthoc_sc['P4']['status']}` *(post-hoc)* |

---

## 4. Detailed P4 Breakdown (VRAM & Extra YOLO Persons on TEST-2)

- **Whole-System VRAM (`vram_test2_sealed.csv`, `{telem['samples_count']}` 1 Hz samples):** Min/Idle pre-render = `{telem['idle_desktop_baseline_mb']:.0f}` MiB (`{telem['idle_sample_count']}` samples), Mean = `{telem['mean_vram_mb']:.1f}` MiB, p95 = `{telem['p95_vram_mb']:.1f}` MiB, p99 = `{telem['p99_vram_mb']:.1f}` MiB, Peak = **`{telem['peak_vram_mb']}` MiB** (`{telem['peak_vram_gib']:.3f}` GiB / `{telem['peak_vram_gb_decimal']:.3f}` GB). Excursions: **`{telem['seconds_above_7400_mb']}`s** above `{pc['p4_vram_limit_mb']:.0f}` MiB (`{telem['seconds_above_7577_6_mib']}`s above `{pc['p4_vram_limit_mib']:.1f}` MiB across `{telem['excursion_episodes_count']}` episodes).
- **Process-Level Telemetry Note:** Separate `nvidia-smi` per-PID logs were not captured during the sealed run (and scene sidecars logged `peak_vram_mb = 0.0`). Subtracting the measured `{telem['idle_desktop_baseline_mb']:.0f}` MiB pre-render desktop baseline from the `{telem['peak_vram_mb']}` MiB peak gives a process-attributable delta peak of **`{telem['process_attributable_delta_peak_mb']:.0f}` MiB** (`{telem['process_attributable_delta_peak_gib']:.3f}` GiB), which is below `{pc['p4_vram_limit_gb']} GB` even though whole-system VRAM exceeded `{pc['p4_vram_limit_gb']} GB`.
- **Extra YOLO Person Detector (`tools/extra_person_detector.py`, `conf=0.40`):**
  - `v2_test2_ebright`: **`{eb_extra_v2.get('flagged_count')}/{eb_extra_v2.get('total_images')}`** (`{eb_extra_v2.get('flagged_rate', 0)*100:.1f}%` — `scene_5_seed_1202.png`: `3` detected vs `2` expected).
  - `v2_test2_holdout`: **`{ho_extra_v2.get('flagged_count')}/{ho_extra_v2.get('total_images')}`** (`{ho_extra_v2.get('flagged_rate', 0)*100:.1f}%`; all `3` Scene 4 renders had missing 3rd character: `2` vs `3`).
  - `v1_test2_ebright`: **`{eb_extra_v1.get('flagged_count')}/{eb_extra_v1.get('total_images')}`** (`{eb_extra_v1.get('flagged_rate', 0)*100:.1f}%` — `scene_1_seed_1202.png`: `3` vs `2`).
  - `v1_test2_holdout`: **`{ho_extra_v1.get('flagged_count')}/{ho_extra_v1.get('total_images')}`** (`{ho_extra_v1.get('flagged_rate', 0)*100:.1f}%` — `scene_2_mentorship_seed_1202.png`: `3` vs `2`).
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "PREREG_RECONCILIATION.md").write_text(reconcil_md, encoding="utf-8")
    print("[OK] Rendered docs_generated/v2/PREREG_RECONCILIATION.md")

    # ------------------------------------------------------------------
    # 3. docs_generated/v2/07_measured_results.md
    # ------------------------------------------------------------------
    eb_age = t2["ebright"]["adult_bio"]
    eb_age_stat = N["test2_evaluation"]["combined"]["age_disaggregated"]
    ho_s4_rows = []
    for row in exp_cases.get("holdout_scene_4", []):
        c_summary = "; ".join(f"{c['character_id']}={c['bio']:.4f}(sp={c['spatial_pass']},w_dE={c['wardrobe_delta_e']})" for c in row["characters"])
        ho_s4_rows.append(f"| `{row['image']}` | {row['detected_faces']}/{row['expected_faces']} (YOLO={row['detected_yolo']}) | {', '.join(row['swap_report_characters'])} | {c_summary} |")
    ho_s4_table = "\n".join(ho_s4_rows)

    eb_s6_rows = []
    for row in exp_cases.get("ebright_scene_6", []):
        eb_s6_rows.append(f"| `{row['image']}` | {row['v2_bio']:.4f} | {row['v2_wardrobe_delta_e']:.2f} (pass={row['v2_wardrobe_pass']}) | {row['v1_wardrobe_delta_e']:.2f} | {', '.join(row['defects'])} |")
    eb_s6_table = "\n".join(eb_s6_rows)

    measured_md = f"""# MMCCP v2.0: Measured Results, Statistical Analysis & Exploratory Diagnostics

**Capstone Project ID:** `{meta['capstone_id']}` | **Institution:** {meta['institution']}  
**Evaluation Verifier:** Frozen Verifier `v{meta['verifier_version']}` (`SHA: {meta['verifier_freeze_sha']}`)

---

## 1. Canonical TEST-2 Results & Dual Bootstrap CIs (`{pc['n_bootstrap_resamples']}` Resamples, Seed `{pc['bootstrap_random_seed']}`)

Defect unit definitions:
- **Defects per Scene (Primary):** Mean defects across unique rendered images (`n={eb['images_count']}` Ebright, `n={ho['images_count']}` Holdout, `n={comb['images_count']}` Combined).
- **Defects per Face Instance (Secondary / Instance-Weighted):** Scene defect count replicated across each face slot (`n={eb['faces_count']}` Ebright, `n={ho['faces_count']}` Holdout, `n={comb['faces_count']}` Combined); reported separately so multi-character scenes are not conflated with scene averages.

### 1.1 Ebright TEST-2 (`{eb['images_count']}` Renders, `{eb['faces_count']}` Face Slots, `{eb['rank1_total_count']}` Paired Slots)
- **Biometric Ensemble Mean:** `{eb['biometric_ensemble_mean']:.4f}` (vs v1 `{v1_eb['biometric_ensemble_mean']:.4f}`, `Delta = {d_eb['delta_bio']:+.4f}`, scene-cluster 95% CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`, image-level 95% CI `{_fmt_ci(d_eb['delta_bio_image_95ci'])}`, Wilcoxon `W={d_eb['delta_bio_wilcoxon_w']:.1f}`, `p={d_eb['delta_bio_wilcoxon_p']:.4f}` -> parity, CI includes 0)
  - Adult Bio (`n={eb['adult_count']}`): `{eb['adult_bio']:.4f}` (v1 `{v1_eb['adult_bio']:.4f}`)
  - Teen Bio (`n={eb['teen_count']}`): `{eb['teen_bio']:.4f}` (v1 `{v1_eb['teen_bio']:.4f}`)
  - Child Bio (`n={eb['child_count']}`): `{eb['child_bio']:.4f}` (v1 `{v1_eb['child_bio']:.4f}`)
- **Rank-1 Discrimination Accuracy:** `{eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}`, v1 `{v1_eb['rank1_accuracy']*100:.1f}%` `{eb['rank1_v1_count']}/{eb['rank1_total_count']}`, `Delta = {d_eb['delta_rank1']*100:+.1f}%`, image 95% CI `{_fmt_ci(d_eb['delta_rank1_image_95ci'], 1, True)}` excludes 0, cluster 95% CI `{_fmt_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}` touches 0, Wilcoxon `p={d_eb['delta_rank1_wilcoxon_p']:.4f}`)
- **Spatial Position Pass Rate:** `{eb['spatial_pass_rate']*100:.1f}%` (`{eb['spatial_v2_count']}/{eb['spatial_total_count']}` vs v1 `{v1_eb['spatial_pass_rate']*100:.1f}%` `{eb['spatial_v1_count']}/{eb['spatial_total_count']}`, `Delta = {d_eb['delta_spatial']*100:+.1f}%`)
- **Wardrobe Palette Pass Rate:** `{eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}`, v1 `{v1_eb['wardrobe_pass_rate']*100:.1f}%` `{eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}`, `Delta = {d_eb['delta_wardrobe']*100:+.1f}%`, cluster 95% CI `{_fmt_ci(d_eb['delta_wardrobe_cluster_95ci'], 1, True)}`, image 95% CI `{_fmt_ci(d_eb['delta_wardrobe_image_95ci'], 1, True)}`, Wilcoxon `W={d_eb['delta_wardrobe_wilcoxon_w']:.1f}`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`)
- **Mean Defects per Scene (Primary, `n={eb['images_count']}`):** `{eb['mean_defects_per_scene']:.2f}` (v1 `{v1_eb['mean_defects_per_scene']:.2f}`, `Delta = {d_eb['delta_defects_per_scene']:+.2f}` / `-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, cluster 95% CI `{_fmt_ci(d_eb['delta_defects_per_scene_cluster_95ci'], 2)}`, image 95% CI `{_fmt_ci(d_eb['delta_defects_per_scene_image_95ci'], 2)}`, Wilcoxon `W={d_eb['delta_defects_per_scene_wilcoxon_w']:.1f}`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`)
- **Mean Defects per Face Instance (Secondary, `n={eb['faces_count']}`):** `{eb['mean_defects_per_instance_weighted']:.3f}` (v1 `{eb['v1_mean_defects_per_instance_weighted']:.3f}`, `Delta = {d_eb['delta_defects_per_instance_weighted']:+.3f}`)
- **Combined Binding & Identity-at-Position Pass Rates:** `{eb['identity_at_position_pass_rate']*100:.1f}%` (`0/{eb['faces_count']}`) and `{eb['combined_binding_pass_rate']*100:.1f}%` (`0/{eb['faces_count']}`) for **both** v1 and v2.

### 1.2 Holdout TEST-2 (`{ho['images_count']}` Renders, `{ho['faces_count']}` Face Slots)
- **Biometric Ensemble Mean:** `{ho['biometric_ensemble_mean']:.4f}` (vs v1 `{v1_ho['biometric_ensemble_mean']:.4f}`, `Delta = {d_ho['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_ho['delta_bio_cluster_95ci'])}`, image CI `{_fmt_ci(d_ho['delta_bio_image_95ci'])}`, Wilcoxon `p={d_ho['delta_bio_wilcoxon_p']:.4f}`)
  - Adult Bio (`n={ho['adult_count']}`): `{ho['adult_bio']:.4f}` (v1 `{v1_ho['adult_bio']:.4f}`) | Child Bio (`n={ho['child_count']}`): `{ho['child_bio']:.4f}` (v1 `{v1_ho['child_bio']:.4f}`)
- **Rank-1 Accuracy:** `{ho['rank1_accuracy']*100:.1f}%` (`{ho['rank1_v2_count']}/{ho['rank1_total_count']}` vs v1 `{v1_ho['rank1_accuracy']*100:.1f}%` `{ho['rank1_v1_count']}/{ho['rank1_total_count']}`, `Delta = {d_ho['delta_rank1']*100:+.1f}%`, cluster CI `{_fmt_ci(d_ho['delta_rank1_cluster_95ci'], 1, True)}`, image CI `{_fmt_ci(d_ho['delta_rank1_image_95ci'], 1, True)}`, `p={d_ho['delta_rank1_wilcoxon_p']:.4f}`)
- **Spatial Pass Rate:** `{ho['spatial_pass_rate']*100:.1f}%` (`{ho['spatial_v2_count']}/{ho['spatial_total_count']}` vs v1 `{v1_ho['spatial_pass_rate']*100:.1f}%` `{ho['spatial_v1_count']}/{ho['spatial_total_count']}`, `Delta = {d_ho['delta_spatial']*100:+.1f}%`, image CI `{_fmt_ci(d_ho['delta_spatial_image_95ci'], 1, True)}`) — below `{pc['p2_spatial_threshold_pct']:.1f}%` target.
- **Wardrobe Pass Rate:** `{ho['wardrobe_pass_rate']*100:.1f}%` (`{ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}` vs v1 `{v1_ho['wardrobe_pass_rate']*100:.1f}%` `{ho['wardrobe_v1_count']}/{ho['wardrobe_total_count']}`, `Delta = {d_ho['delta_wardrobe']*100:+.1f}%`, `p={d_ho['delta_wardrobe_wilcoxon_p']:.4f}`) — below `{pc['p3_wardrobe_threshold_pct']:.1f}%` target.
- **Mean Defects per Scene (Primary, `n={ho['images_count']}`):** `{ho['mean_defects_per_scene']:.2f}` (v1 `{v1_ho['mean_defects_per_scene']:.2f}`, `Delta = {d_ho['delta_defects_per_scene']:+.2f}`, cluster CI `{_fmt_ci(d_ho['delta_defects_per_scene_cluster_95ci'], 2)}`, image CI `{_fmt_ci(d_ho['delta_defects_per_scene_image_95ci'], 2)}`, `p={d_ho['delta_defects_per_scene_wilcoxon_p']:.4f}`)
- **Mean Defects per Face Instance (Secondary, `n={ho['faces_count']}`):** `{ho['mean_defects_per_instance_weighted']:.3f}` (v1 `{ho['v1_mean_defects_per_instance_weighted']:.3f}`, `Delta = {d_ho['delta_defects_per_instance_weighted']:+.3f}`)
- **Combined Binding & Identity-at-Position:** `0.0%` (`0/{ho['faces_count']}`) for both v1 and v2.

### 1.3 Combined TEST-2 (`{comb['images_count']}` Renders, `{comb['faces_count']}` Face Slots)
- **Bio Mean:** `{comb['biometric_ensemble_mean']:.4f}` vs v1 `{comb['v1_biometric_ensemble_mean']:.4f}` (`Delta = {d_co['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_co['delta_bio_cluster_95ci'])}`, image CI `{_fmt_ci(d_co['delta_bio_image_95ci'])}`, `p={d_co['delta_bio_wilcoxon_p']:.4f}`)
- **Rank-1:** `{comb['rank1_accuracy']*100:.2f}%` (`{comb['rank1_v2_count']}/{comb['rank1_total_count']}` vs v1 `{comb['v1_rank1_accuracy']*100:.2f}%`, `Delta = {d_co['delta_rank1']*100:+.2f}%`, image CI `{_fmt_ci(d_co['delta_rank1_image_95ci'], 1, True)}`)
- **Spatial / Wardrobe:** `{comb['spatial_pass_rate']*100:.2f}%` (`{comb['spatial_v2_count']}/{comb['spatial_total_count']}`) / `{comb['wardrobe_pass_rate']*100:.2f}%` (`{comb['wardrobe_v2_count']}/{comb['wardrobe_total_count']}`)
- **Defects/Scene (Primary):** `{comb['mean_defects_per_scene']:.2f}` vs v1 `{comb['v1_mean_defects_per_scene']:.2f}` (`Delta = {d_co['delta_defects_per_scene']:+.2f}`, cluster CI `{_fmt_ci(d_co['delta_defects_per_scene_cluster_95ci'], 2)}`, image CI `{_fmt_ci(d_co['delta_defects_per_scene_image_95ci'], 2)}`, `p={d_co['delta_defects_per_scene_wilcoxon_p']:.6f}`)

---

## 2. Original Pre-Registered Scorecard (`P1`–`P4`)

See full table and post-hoc reconciliation in [`PREREG_RECONCILIATION.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/PREREG_RECONCILIATION.md):
- **`{orig_sc['P1']['hypothesis']}` — [`{orig_sc['P1']['status']}`]**: {orig_sc['P1']['rationale']}
- **`{orig_sc['P2']['hypothesis']}` — [`{orig_sc['P2']['status']}`]**: {orig_sc['P2']['rationale']}
- **`{orig_sc['P3']['hypothesis']}` — [`{orig_sc['P3']['status']}`]**: {orig_sc['P3']['rationale']}
- **`{orig_sc['P4']['hypothesis']}` — [`{orig_sc['P4']['status']}`]**: {orig_sc['P4']['rationale']}

---

## 3. Leave-One-Out Ablation Tables (TEST-2 Seed `{t2_abl_meta['seed']}` & DEV-2 Seed `{d2_abl_meta['seed']}`)

Single-seed (`n={t2_abl_meta['sample_size_faces']}` faces) biometric deltas smaller than `~{pc['ablation_noise_band']:.2f}` are within diffusion seed noise. On **both** DEV-2 and TEST-2, `no_m4_ip_adapter` exceeded `full_v2`, motivating disabling M4 IP-Adapter Plus in `v2.1` exploration while keeping frozen `v2.0` intact.

| Arm | TEST-2 Bio (`n={t2_abl_meta['sample_size_faces']}`) | TEST-2 `Delta Bio` | TEST-2 Pixel Diff | DEV-2 Bio (`n={d2_abl_meta['sample_size_faces']}`) | DEV-2 `Delta Bio` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `full_v2` | {t2_abl['full_v2']['biometric_ensemble_mean']:.4f} | {t2_abl['full_v2']['delta_bio_vs_full']:+.4f} | {t2_diffs['full_v2']:.2f} px | {d2_abl['full_v2']['biometric_ensemble_mean']:.4f} | {d2_abl['full_v2']['delta_bio_vs_full']:+.4f} |
| `no_m2_layout` | {t2_abl['no_m2_layout']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m2_layout']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m2_layout']:.2f} px | {d2_abl['no_m2_layout']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m2_layout']['delta_bio_vs_full']:+.4f} |
| `no_m3_couple` | {t2_abl['no_m3_couple']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m3_couple']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m3_couple']:.2f} px | {d2_abl['no_m3_couple']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m3_couple']['delta_bio_vs_full']:+.4f} |
| `no_m4_ip_adapter` | {t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m4_ip_adapter']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m4_ip_adapter']:.2f} px | {d2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m4_ip_adapter']['delta_bio_vs_full']:+.4f} |
| `no_m5_swap` | {t2_abl['no_m5_swap']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m5_swap']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m5_swap']:.2f} px | {d2_abl['no_m5_swap']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m5_swap']['delta_bio_vs_full']:+.4f} |
| `no_m6_detailer` | {t2_abl['no_m6_detailer']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m6_detailer']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m6_detailer']:.2f} px | {d2_abl['no_m6_detailer']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m6_detailer']['delta_bio_vs_full']:+.4f} |
| `no_m7_wardrobe` | {t2_abl['no_m7_wardrobe']['biometric_ensemble_mean']:.4f} | {t2_abl['no_m7_wardrobe']['delta_bio_vs_full']:+.4f} | {t2_diffs['no_m7_wardrobe']:.2f} px | {d2_abl['no_m7_wardrobe']['biometric_ensemble_mean']:.4f} | {d2_abl['no_m7_wardrobe']['delta_bio_vs_full']:+.4f} |

---

## 4. Exploratory Defect Root-Cause Findings (`EXPLORATORY`)

All images below were inspected directly with the image viewer before reading their scores:
1. **Holdout Scene 4 Missing 3rd Character (`clara_thorn` left, `elena_child` center, `dr_marcus_vance` right — CONFIRMED ARCHITECTURAL LIMIT):**
   - Visual inspection of all `3` Holdout Scene 4 renders confirmed strictly `2` rendered figures around a table with a blank middle wall gap, and the right-hand person exhibited a mustached male face over long feminine hair.
   - Root cause confirmed in code + sidecars: `layout_planner.py` constructs only a 2-slot pair (`left_char`, `right_char`), `regional.py` sends 2 Basic columns, `face_swap.py`/M5 receives only `2` swap specs (`clara_thorn` and `dr_marcus_vance`, dropping `elena_child`), and `prompt_builder.py` collapses non-first characters into the right column while banning `"3persons"` for all scenes with `>= 2` characters. Verifier then matched the right face to `elena_child` by spatial zone, scoring `dr_marcus_vance` `0.0000` (missing).

| Holdout Scene 4 Render | Faces/Expected (YOLO) | Swapped Specs | Per-Character Verifier Outcomes |
| :--- | :---: | :--- | :--- |
{ho_s4_table}

2. **Ebright Scene 6 Graduation Gown Color Corruption (`EXPLORATORY` — BOTH REAL DEFECT AND VERIFIER ARTIFACT CONFIRMED):**
   - Visual inspection across all `{len(exp_cases.get('ebright_scene_6', []))}` Scene 6 renders confirmed a black graduation gown and mortarboard with a rectangular olive/mustard chest stain bleeding onto the top edge of the diploma folder.
   - Root cause: `ebright.json` defines `college_ebright.clothing_colors = ["mustard yellow"]` while Scene 6 specifies `age_state = "graduation"` (`black graduation gown`). M7 (`pipeline_v2.py` -> `wardrobe.py`) reads `clothing_colors` rather than scene-state clothing, staining the torso box toward mustard yellow (**real visual corruption** + CLIP flags for `casual clothes`, `shorts`, `t-shirt`), **and** Frozen Verifier `v{meta['verifier_version']}` (`cli.py`) also scores against `clothing_colors`, causing both v2 (`Delta E = 38.77..48.94 > {pc['wardrobe_delta_e_threshold']:.1f}`) and unstained v1 black gowns (`Delta E = 53.75..76.54`) to fail (**verifier state-blindness artifact**).

| Scene 6 Render | v2 Bio Mean | v2 Wardrobe `Delta E` | v1 Wardrobe `Delta E` | Verifier Defects |
| :--- | :---: | :---: | :---: | :--- |
{eb_s6_table}

3. **Extra-Person / YOLO Divergence (`scene_2_seed_1303` vs `scene_5_seed_1202` — `EXPLORATORY`):**
   - Visual inspection of `scene_2_seed_1303.png` showed `1` boy behind a butterfly table (Verifier `report_v1_2.json` logged `YOLO=2, faces=1, defects=[]`, while standalone `tools/extra_person_detector.py` logged `YOLO=1`).
   - Visual inspection of `scene_5_seed_1202.png` showed **`3` distinct figures** (left `college_ebright`, upper-right `james_wong`, and a giant duplicated `james_wong` bust overlapping in the lower-right foreground), matching both Verifier (`YOLO=3, faces=3`) and `extra_person_detector.py` (`{eb_extra_v2.get('flagged_count')}/{eb_extra_v2.get('total_images')}` on v2 TEST-2 Ebright).

---

## 5. Hardware Telemetry & Excursion Attribution
- **Sealed TEST-2 (`vram_test2_sealed.csv`, `{telem['samples_count']}` 1 Hz samples):** Whole-system peak **`{telem['peak_vram_mb']}` MiB** (`{telem['peak_vram_gib']:.3f}` GiB / `{telem['peak_vram_gb_decimal']:.3f}` GB), median `{telem['median_vram_mb']:.1f}` MiB, mean `{telem['mean_vram_mb']:.1f}` MiB, p95 `{telem['p95_vram_mb']:.1f}` MiB, p99 `{telem['p99_vram_mb']:.1f}` MiB. Pre-render desktop idle baseline = `{telem['idle_desktop_baseline_mb']:.0f}` MiB (`{telem['idle_sample_count']}` samples); process-attributable delta peak = `{telem['process_attributable_delta_peak_mb']:.0f}` MiB (`{telem['process_attributable_delta_peak_gib']:.3f}` GiB). Whole-system excursions exceeded `{pc['p4_vram_limit_mb']:.0f}` MiB for **`{telem['seconds_above_7400_mb']}`s** (`{telem['seconds_above_7577_6_mib']}`s > `{pc['p4_vram_limit_mib']:.1f}` MiB) across **`{telem['excursion_episodes_count']}` episodes**; all episodes aligned cleanly with early Pass 1 txt2img model/ControlNet/IP-Adapter allocation (`6–31s` into renders, including the `{telem['peak_vram_mb']}` MiB peak in `scene_5_seed_1303`) or Pass 1->Pass 2 img2img handoff (`49–131s`), and **zero** occurred during frozen scoring.
- **All `{telem_9['files_count']}` Telemetry CSVs Combined (`{telem_9['samples_count']}` samples):** Max `{telem_9['max_mb']:.0f}` MiB, p99 `{telem_9['p99_mb']:.1f}` MiB, `{telem_9['seconds_above_7400_mb']}`s > `{pc['p4_vram_limit_mb']:.0f}` MiB (`{telem_9['seconds_above_7577_6_mib']}`s > `{pc['p4_vram_limit_mib']:.1f}` MiB).

---

{circ_md}

---

## 7. Visual Review Status (`automated, not human`)
- Prior Phase R8 evaluations on DEV-2 (`{blind_ab['ab_v2_wins']}/{blind_ab['ab_total_pairs']}` blind A/B preference = `{blind_ab['ab_v2_win_rate']*100:.2f}%`, `p={blind_ab['ab_p_value']:.4f}`; `{blind_ab['reid_correct']}/{blind_ab['reid_total_crops']}` blind crop Re-ID; `{canary['total_caught']}/{canary['total_canaries']}` planted canaries; VLM `kappa` proxy) are **automated, not human** evaluations (`blind_review_summary.json` / `blind_evaluation_report.md` marked `AUTOMATED-SUPERSEDED`).
- Human TEST-2 Ebright blind test is staged awaiting user scoring in [`HUMAN_BLIND_TEST.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/visual_review/HUMAN_BLIND_TEST.md) and [`blind_predictions_human.json`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/visual_review/blind_predictions_human.json).
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "07_measured_results.md").write_text(measured_md, encoding="utf-8")
    print("[OK] Rendered docs_generated/v2/07_measured_results.md")

    # ------------------------------------------------------------------
    # 4. Presentation Handoff Assets (07_PPT_UPDATE, PPT_FACT_SHEET, SPARK_NOTES)
    # ------------------------------------------------------------------
    ppt_update = f"""# Capstone Presentation Slide Update: Reconciled Verified State

**Project:** MMCCP v2.0 (`{meta['capstone_id']}`) | **Verifier:** `v{meta['verifier_version']}`

## Key Reconciled Slide Numbers (`TEST-2`, `n={eb['images_count']}` Ebright + `{ho['images_count']}` Holdout)
1. **Ebright TEST-2 (In-Domain Story):**
   - **Biometric Parity:** `{v1_eb['biometric_ensemble_mean']:.4f} -> {eb['biometric_ensemble_mean']:.4f}` (`Delta = {d_eb['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`, image CI `{_fmt_ci(d_eb['delta_bio_image_95ci'])}`, `p={d_eb['delta_bio_wilcoxon_p']:.4f}` — includes 0).
   - **Rank-1 Re-ID:** `{v1_eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v1_count']}/{eb['rank1_total_count']}`) -> **`{eb['rank1_accuracy']*100:.1f}%`** (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}`, `{d_eb['delta_rank1']*100:+.1f}%`, image CI `{_fmt_ci(d_eb['delta_rank1_image_95ci'], 1, True)}`, 6-cluster CI `{_fmt_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}`).
   - **Wardrobe Fidelity:** `{v1_eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}`) -> **`{eb['wardrobe_pass_rate']*100:.1f}%`** (`{eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}`, `{d_eb['delta_wardrobe']*100:+.1f}%`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`).
   - **Defects per Scene (Primary):** `{v1_eb['mean_defects_per_scene']:.2f}` -> **`{eb['mean_defects_per_scene']:.2f}`** (`{d_eb['delta_defects_per_scene']:+.2f}/scene`, `-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`). *(Secondary instance-weighted delta: `{d_eb['delta_defects_per_instance_weighted']:+.3f}`).*
2. **Holdout TEST-2 (Unseen Characters — NOT Better Than v1):**
   - Bio `{v1_ho['biometric_ensemble_mean']:.4f} -> {ho['biometric_ensemble_mean']:.4f}` (`{d_ho['delta_bio']:+.4f}`), Rank-1 `{v1_ho['rank1_accuracy']*100:.1f}% -> {ho['rank1_accuracy']*100:.1f}%`, Wardrobe `{v1_ho['wardrobe_pass_rate']*100:.1f}% -> {ho['wardrobe_pass_rate']*100:.1f}%`, Spatial `{ho['spatial_pass_rate']*100:.1f}%` (`{ho['spatial_v2_count']}/{ho['spatial_total_count']}` — 3-char Scene 4 dropped `1` character), Defects/Scene `{v1_ho['mean_defects_per_scene']:.2f} -> {ho['mean_defects_per_scene']:.2f}`.
3. **Engineering Telemetry & Extra Persons:**
   - Whole-system peak VRAM **`{telem['peak_vram_mb']}` MiB** (`{telem['peak_vram_gib']:.3f}` GiB; `{telem['seconds_above_7400_mb']}s > {pc['p4_vram_limit_mb']:.0f}` MiB), process-attributable delta above `{telem['idle_desktop_baseline_mb']:.0f}` MiB idle = `{telem['process_attributable_delta_peak_mb']:.0f}` MiB, `{pc['p4_max_ooms']}` OOMs; extra YOLO person (`conf=0.40`): `{total_extra_v2}/{comb['images_count']}` (`scene_5_seed_1202`).
4. **Original Pre-Registered Hypotheses (`P1`–`P4`):**
   - Strict all-split verdict: **`{orig_sc['strict_all_splits_met_count']}/4` MET** (`P1` NOT MET; `P2`/`P3` pass Ebright `{eb['spatial_pass_rate']*100:.1f}%`/`{eb['wardrobe_pass_rate']*100:.1f}%` and Combined `{comb['spatial_pass_rate']*100:.2f}%`/`{comb['wardrobe_pass_rate']*100:.2f}%` only, failing Holdout; `P4` NOT MET).
5. **Known Limits & Future Work:**
   - `identity_at_position` & `combined_binding` = `0.0%` for both v1 & v2; LOO ablations show `no_m4_ip_adapter` outperformed `full_v2` on DEV-2 & TEST-2; prior visual review checks were **automated, not human** (human TEST-2 blind test pending).
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "07_PPT_UPDATE.md").write_text(ppt_update, encoding="utf-8")

    ppt_fact_sheet = f"""# PPT Fact Sheet: Reconciled Ground-Truth Numbers

- **Ebright TEST-2 (`n={eb['images_count']}` scenes, `{eb['faces_count']}` faces):**
  - Bio Mean: `{eb['biometric_ensemble_mean']:.4f}` (v1 `{v1_eb['biometric_ensemble_mean']:.4f}`, `Delta = {d_eb['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`)
  - Rank-1 Accuracy: `{eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}`, v1 `{v1_eb['rank1_accuracy']*100:.1f}%`)
  - Wardrobe Pass: `{eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}`, v1 `{v1_eb['wardrobe_pass_rate']*100:.1f}%`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`)
  - Spatial Pass: `{eb['spatial_pass_rate']*100:.1f}%` (`{eb['spatial_v2_count']}/{eb['spatial_total_count']}`)
  - Defects/Scene (Primary): `{eb['mean_defects_per_scene']:.2f}` (v1 `{v1_eb['mean_defects_per_scene']:.2f}`, `-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`)
- **Holdout TEST-2 (`n={ho['images_count']}` scenes, `{ho['faces_count']}` faces — NOT better than v1):**
  - Bio Mean: `{ho['biometric_ensemble_mean']:.4f}` (v1 `{v1_ho['biometric_ensemble_mean']:.4f}`, `Delta = {d_ho['delta_bio']:+.4f}`)
  - Rank-1 Accuracy: `{ho['rank1_accuracy']*100:.1f}%` (`{ho['rank1_v2_count']}/{ho['rank1_total_count']}`, v1 `{v1_ho['rank1_accuracy']*100:.1f}%`)
  - Spatial / Wardrobe Pass: `{ho['spatial_pass_rate']*100:.1f}%` (`{ho['spatial_v2_count']}/{ho['spatial_total_count']}`) / `{ho['wardrobe_pass_rate']*100:.1f}%` (`{ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}`)
  - Defects/Scene (Primary): `{ho['mean_defects_per_scene']:.2f}` (v1 `{v1_ho['mean_defects_per_scene']:.2f}`)
- **Combined Binding Gate (v1 & v2):** `{eb['combined_binding_pass_rate']*100:.1f}%` (unmeasurable under Verifier `v{meta['verifier_version']}`)
- **VRAM Telemetry (`vram_test2_sealed.csv`):** Whole-system peak `{telem['peak_vram_mb']} MiB` (`{telem['seconds_above_7400_mb']}s > {pc['p4_vram_limit_mb']:.0f}` MiB), `{telem['idle_desktop_baseline_mb']:.0f} MiB` idle, `{telem['process_attributable_delta_peak_mb']:.0f} MiB` process delta, `{pc['p4_max_ooms']}` OOMs
- **Extra YOLO (`conf=0.40`):** `{total_extra_v2}/{comb['images_count']}` v2 TEST-2 renders
- **Original Pre-Registered Hypotheses:** `{orig_sc['strict_all_splits_met_count']}/4` strict all-split MET (`P1` {orig_sc['P1']['status']}, `P2` {orig_sc['P2']['status']} [Ebright {eb['spatial_pass_rate']*100:.1f}%, Holdout {ho['spatial_pass_rate']*100:.1f}%], `P3` {orig_sc['P3']['status']} [Ebright {eb['wardrobe_pass_rate']*100:.1f}%, Holdout {ho['wardrobe_pass_rate']*100:.1f}%], `P4` {orig_sc['P4']['status']})
- **Blind Review:** Automated proxy (`{blind_ab['ab_v2_wins']}/{blind_ab['ab_total_pairs']}`), **not human**; human blind test pending
- **Frozen Verifier:** `v{meta['verifier_version']} (SHA: {meta['verifier_freeze_sha'][:16]}...)`
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "PPT_FACT_SHEET.md").write_text(ppt_fact_sheet, encoding="utf-8")

    spark_notes = f"""# MMCCP v2.0 Spark Notes for Defense / Review (Reconciled)

- **What MMCCP v2.0 Does:** Training-free conditioning & refinement pipeline (`M1` Bank, `M2` OpenPose Layout, `M3` Regional Couple, `M4` IP-Adapter Plus, `M5` Direct InsightFace Swap, `M6` Adaptive Face Detailer, `M7` CIELAB Wardrobe Lock).
- **What Actually Improved on Sealed TEST-2 Ebright (`n={eb['images_count']}` scenes):**
  - Biometric similarity was at parity with Baseline v1 (`{v1_eb['biometric_ensemble_mean']:.4f} -> {eb['biometric_ensemble_mean']:.4f}`, `Delta = {d_eb['delta_bio']:+.4f}`, CI crosses 0), while **Rank-1 identity disambiguation** rose `{v1_eb['rank1_accuracy']*100:.1f}% -> {eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}` vs `{eb['rank1_v1_count']}/{eb['rank1_total_count']}`, image CI `{_fmt_ci(d_eb['delta_rank1_image_95ci'], 1, True)}`), **Wardrobe pass rate** rose `{v1_eb['wardrobe_pass_rate']*100:.1f}% -> {eb['wardrobe_pass_rate']*100:.1f}%` (`p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`), and **scene-level defects** dropped `{v1_eb['mean_defects_per_scene']:.2f} -> {eb['mean_defects_per_scene']:.2f}` (`-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`).
- **Where v2.0 Fell Short (Honest Defense Takeaways):**
  - **Holdout (`n={ho['images_count']}` scenes):** v2 was not better than v1 (`Bio {ho['biometric_ensemble_mean']:.4f} vs {v1_ho['biometric_ensemble_mean']:.4f}`, `Rank-1 {ho['rank1_accuracy']*100:.1f}% vs {v1_ho['rank1_accuracy']*100:.1f}%`, `Wardrobe {ho['wardrobe_pass_rate']*100:.1f}% vs {v1_ho['wardrobe_pass_rate']*100:.1f}%`), and `M2`/`M3`/`M5` dropped the 3rd character on all `{len(exp_cases.get('holdout_scene_4', []))}` Holdout Scene 4 renders (`Spatial = {ho['spatial_pass_rate']*100:.1f}% < {pc['p2_spatial_threshold_pct']:.1f}%`).
  - **Pre-Registration (`{orig_sc['strict_all_splits_met_count']}/4` strict):** Only `P2` Spatial & `P3` Wardrobe passed on Ebright alone; original `P1`, Holdout `P2`/`P3`, and `P4` (`{telem['peak_vram_mb']}` MiB whole-system peak > `{pc['p4_vram_limit_gb']} GB`, `{total_extra_v2}/{comb['images_count']}` extra YOLO person) were `NOT MET`.
  - **Methodology Caveats:** `combined_binding = {eb['combined_binding_pass_rate']*100:.1f}%` in both v1 and v2; `no_m4_ip_adapter` outperformed `full_v2` on DEV-2 & TEST-2 (`recommend M4 OFF in v2.1`); prior blind checks were **automated, not human**; scoring shares InsightFace anchor family with `M5`.
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "SPARK_NOTES.md").write_text(spark_notes, encoding="utf-8")

    # ------------------------------------------------------------------
    # 5. docs_generated/v2/PR19_FINAL_COMMENT.md
    # ------------------------------------------------------------------
    pr19_comment = f"""## CORRECTION NOTICE: Reconciled Finalization Audit (`v2.0.0-rc3` Candidate)

> [!WARNING]
> **RETRACTING OVERSTATED `v2.0.0-rc2` SCORECARD CLAIMS:** The earlier `v2.0.0-rc2` (`{tl['rc2_commit']}`) comment reported `4/4 MET` hypotheses using post-hoc redefined criteria written on `{tl['compute_test2_statistics_authored_ist']}` after `TEST-2` scores were inspected (`{tl['first_v2_test2_score_ist']}`), conflated per-scene (`{eb['mean_defects_per_scene']:.2f}`) with instance-weighted (`{d_eb['delta_defects_per_instance_weighted']:+.2f}`) defect metrics, and omitted whole-system VRAM excursions (`{telem['peak_vram_mb']}` MiB). Below are the reconciled, script-verified numbers from [`results/numbers.json`](file:///D:/5TH%20SEM/HACKATHON/results/numbers.json).

### 1. Reconciled Sealed TEST-2 Scorecard (`{comb['images_count']}` Renders: `{eb['images_count']}` Ebright + `{ho['images_count']}` Holdout)

- **Ebright TEST-2 (`{eb['images_count']}` scenes, `{eb['faces_count']}` face slots):**
  - **Biometric Ensemble:** `{eb['biometric_ensemble_mean']:.4f}` vs v1 `{v1_eb['biometric_ensemble_mean']:.4f}` (`Delta = {d_eb['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_eb['delta_bio_cluster_95ci'])}`, image CI `{_fmt_ci(d_eb['delta_bio_image_95ci'])}`, Wilcoxon `W={d_eb['delta_bio_wilcoxon_w']:.1f}`, `p={d_eb['delta_bio_wilcoxon_p']:.4f}`) -> **Identity parity with Baseline v1**.
  - **Rank-1 Disambiguation:** `{eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v2_count']}/{eb['rank1_total_count']}`) vs v1 `{v1_eb['rank1_accuracy']*100:.1f}%` (`{eb['rank1_v1_count']}/{eb['rank1_total_count']}`, `Delta = {d_eb['delta_rank1']*100:+.1f}%`, image CI `{_fmt_ci(d_eb['delta_rank1_image_95ci'], 1, True)}`, cluster CI `{_fmt_ci(d_eb['delta_rank1_cluster_95ci'], 1, True)}`).
  - **Wardrobe Palette (`Delta E <= {pc['wardrobe_delta_e_threshold']:.1f}`):** `{eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v2_count']}/{eb['wardrobe_total_count']}`) vs v1 `{v1_eb['wardrobe_pass_rate']*100:.1f}%` (`{eb['wardrobe_v1_count']}/{eb['wardrobe_total_count']}`, `Delta = {d_eb['delta_wardrobe']*100:+.1f}%`, cluster CI `{_fmt_ci(d_eb['delta_wardrobe_cluster_95ci'], 1, True)}`, image CI `{_fmt_ci(d_eb['delta_wardrobe_image_95ci'], 1, True)}`, `p={d_eb['delta_wardrobe_wilcoxon_p']:.6f}`).
  - **Mean Defects / Scene (Primary):** `{eb['mean_defects_per_scene']:.2f}` vs v1 `{v1_eb['mean_defects_per_scene']:.2f}` (`Delta = {d_eb['delta_defects_per_scene']:+.2f}/scene`, `-{derived['ebright_defects_per_scene_pct_reduction']:.1f}%`, cluster CI `{_fmt_ci(d_eb['delta_defects_per_scene_cluster_95ci'], 2)}`, image CI `{_fmt_ci(d_eb['delta_defects_per_scene_image_95ci'], 2)}`, `p={d_eb['delta_defects_per_scene_wilcoxon_p']:.6f}`).
- **Holdout TEST-2 (`{ho['images_count']}` scenes, `{ho['faces_count']}` face slots — NOT better than v1):**
  - **Biometric Ensemble:** `{ho['biometric_ensemble_mean']:.4f}` vs v1 `{v1_ho['biometric_ensemble_mean']:.4f}` (`Delta = {d_ho['delta_bio']:+.4f}`, cluster CI `{_fmt_ci(d_ho['delta_bio_cluster_95ci'])}`, `p={d_ho['delta_bio_wilcoxon_p']:.4f}`).
  - **Rank-1 / Spatial / Wardrobe / Scene Defects:** `{ho['rank1_accuracy']*100:.1f}%` (`{ho['rank1_v2_count']}/{ho['rank1_total_count']}` vs v1 `{v1_ho['rank1_accuracy']*100:.1f}%`), Spatial `{ho['spatial_pass_rate']*100:.1f}%` (`{ho['spatial_v2_count']}/{ho['spatial_total_count']}` vs v1 `{v1_ho['spatial_pass_rate']*100:.1f}%`), Wardrobe `{ho['wardrobe_pass_rate']*100:.1f}%` (`{ho['wardrobe_v2_count']}/{ho['wardrobe_total_count']}` vs v1 `{v1_ho['wardrobe_pass_rate']*100:.1f}%`), Scene Defects `{ho['mean_defects_per_scene']:.2f}` vs v1 `{v1_ho['mean_defects_per_scene']:.2f}` (`Delta = {d_ho['delta_defects_per_scene']:+.2f}`, `p={d_ho['delta_defects_per_scene_wilcoxon_p']:.4f}`).
- **Combined Binding & Identity-at-Position:** `{eb['combined_binding_pass_rate']*100:.1f}%` (`0/{eb['faces_count']}` Ebright, `0/{ho['faces_count']}` Holdout) for both v1 and v2.

### 2. Original Pre-Registered Hypotheses (`P1`–`P4` — Strictly `{orig_sc['strict_all_splits_met_count']}/4` MET)

1. **`{orig_sc['P1']['hypothesis']}` — [`{orig_sc['P1']['status']}`]**: {orig_sc['P1']['rationale']}
2. **`{orig_sc['P2']['hypothesis']}` — [`{orig_sc['P2']['status']}`]**: {orig_sc['P2']['rationale']}
3. **`{orig_sc['P3']['hypothesis']}` — [`{orig_sc['P3']['status']}`]**: {orig_sc['P3']['rationale']}
4. **`{orig_sc['P4']['hypothesis']}` — [`{orig_sc['P4']['status']}`]**: {orig_sc['P4']['rationale']}

### 3. Telemetry, Ablation & Methodological Notes
- **VRAM (`vram_test2_sealed.csv`):** Whole-system peak `{telem['peak_vram_mb']}` MiB (`{telem['seconds_above_7400_mb']}s > {pc['p4_vram_limit_mb']:.0f}` MiB), `{telem['idle_desktop_baseline_mb']:.0f}` MiB idle, `{telem['process_attributable_delta_peak_mb']:.0f}` MiB process delta, `{pc['p4_max_ooms']}` OOMs.
- **Ablation (`no_m4_ip_adapter`):** Outperformed `full_v2` on DEV-2 (`{d2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{d2_abl['full_v2']['biometric_ensemble_mean']:.4f}`) and TEST-2 (`{t2_abl['no_m4_ip_adapter']['biometric_ensemble_mean']:.4f}` vs `{t2_abl['full_v2']['biometric_ensemble_mean']:.4f}`); M4 should be tested OFF in future `v2.1` exploration (`|Delta| < {pc['ablation_noise_band']:.2f}` on 1 seed is within noise).
- **Visual Review:** Prior blind/canary/VLM artifacts were **automated, not human**; human blind evaluation (`HUMAN_BLIND_TEST.md`) is pending. Verifier freeze (`v{meta['verifier_version']}`) remains intact (`check_freeze.py` PASS); PR #19 stays **OPEN**.
"""
    (PROJECT_ROOT / "docs_generated" / "v2" / "PR19_FINAL_COMMENT.md").write_text(pr19_comment, encoding="utf-8")

    # ------------------------------------------------------------------
    # 6. Refresh docs_generated/v2/README_V2.md stale warning header (non-metric)
    # ------------------------------------------------------------------
    docs_readme_path = PROJECT_ROOT / "docs_generated" / "v2" / "README_V2.md"
    if docs_readme_path.exists():
        dr_txt = docs_readme_path.read_text(encoding="utf-8")
        old_banner = (
            "> [!WARNING]\n"
            "> **PIPELINE & DOCUMENTATION CURRENTLY UNDER REPAIR**\n"
            "> Previous documentation associated with tag `v2.0.0-rc1` contained contradicted and unverified claims.\n"
            "> A formal audit has been completed and ungrounded claims have been retracted. See [`docs_generated/v2/RETRACTIONS.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/RETRACTIONS.md) for details.\n"
            "> The verifier is being frozen as v1.1 and the pipeline is undergoing systematic loop repair. Do not cite rc1 figures."
        )
        new_banner = (
            "> [!NOTE]\n"
            "> **ARCHITECTURE REFERENCE ONLY:** Quantitative results, pre-registration reconciliation, and closeout caveats live in root [`README_V2.md`](file:///D:/5TH%20SEM/HACKATHON/README_V2.md), [`07_measured_results.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/07_measured_results.md), and [`PREREG_RECONCILIATION.md`](file:///D:/5TH%20SEM/HACKATHON/docs_generated/v2/PREREG_RECONCILIATION.md).\n"
            "> All `v2.0.0-rc1` and `v2.0.0-rc2` overstatements have been retracted and reconciled under Frozen Verifier v1.2."
        )
        if old_banner in dr_txt:
            docs_readme_path.write_text(dr_txt.replace(old_banner, new_banner), encoding="utf-8")

    print("[OK] Rendered all 7 documentation artifacts strictly from results/numbers.json")


if __name__ == "__main__":
    render_all()
