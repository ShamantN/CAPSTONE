"""
MMCCP TEST-3 Evaluation Protocol: Interactive Viewer & Human Rating UI (eval3/ui.py).

Strictly file-reading only (loads ZERO ML models and allocates ZERO GPU VRAM).
Provides:
1. Tab 5 — Evaluation Runs Dashboard (`render_eval_tab`):
   - Run list & manifest summary (defaults to test3_v2_final).
   - Live metrics dashboard (v2_final & v2_final_bon3 PM1/PM2/PM3, 95% CIs, chance floors, per-story table).
   - Replicate browser (certified anchors + filterable scene render gallery).
   - Cross-scene face strips per replicate.
   - Failures gallery (PM3=False renders).
   - Raw per-face & per-scene tables with CSV download.
2. Tab 6 — Blinded Human Rating Pack (`render_human_rating_tab`):
   - Task (a): 4-AFC same-person matching across scenes.
   - Task (b): 1-to-5 cross-scene sequence-consistency rating.
   - Reads ONLY rating_pack_blind.json; NEVER reads SEALED_ANSWER_KEY.json.
   - Saves ratings per rater name to results/eval3/human_ratings/ratings_<rater>.json.
"""
import os
import re
import sys
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_OUTPUT_BASE = Path(r"D:\5TH SEM\HACKATHON_eval")
HUMAN_RATINGS_DIR = PROJECT_ROOT / "results" / "eval3" / "human_ratings"


def _discover_runs_readonly(base_dir: Path = EVAL_OUTPUT_BASE) -> List[Dict[str, Any]]:
    """Pure file-based run discovery (zero ML imports)."""
    if not base_dir.exists():
        return []
    runs = []
    for run_dir in sorted(base_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    runs.append({
                        "run_id": m.get("run_id", run_dir.name),
                        "manifest_path": str(manifest_file),
                        "run_dir": str(run_dir),
                        "stage": m.get("stage", "A"),
                        "split": m.get("split", "DEV-3"),
                        "status": m.get("status", "UNKNOWN"),
                    })
                except Exception:
                    pass
    # Prioritize test3_v2_final at index 0 for immediate access
    runs.sort(key=lambda r: (0 if r["run_id"] == "test3_v2_final" else 1, r["run_id"]))
    return runs


def render_eval_tab():
    """Renders the comprehensive 'Evaluation Runs' dashboard tab (file-read only)."""
    st.header("📊 TEST-3 Pre-Registered Evaluation Dashboard")
    st.markdown(
        "Read-only interactive dashboard loading directly from disk (`D:\\5TH SEM\\HACKATHON_eval\\`). "
        "All metrics are explicitly labeled **OUT-OF-LOOP** (4-judge DeepFace ensemble, YOLO) or **IN-LOOP** (v1.2 anchor verifier, InsightFace, wardrobe CIELAB)."
    )

    runs = _discover_runs_readonly()
    if not runs:
        st.info(f"No evaluation runs detected yet in `{EVAL_OUTPUT_BASE}`.")
        return

    run_ids = [r["run_id"] for r in runs]
    selected_run_id = st.selectbox("📁 Select Evaluation Run:", run_ids, index=0, key="eval3_run_selector")
    selected_run = next(r for r in runs if r["run_id"] == selected_run_id)

    manifest_path = Path(selected_run["manifest_path"])
    run_dir = manifest_path.parent
    if not manifest_path.exists():
        st.error(f"Manifest missing at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Panel 1: Run Metadata & Execution Status
    st.markdown("---")
    st.subheader("1️⃣ Run Metadata & Execution Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Run ID", manifest.get("run_id", "N/A"))
        st.caption(f"Stage: `{manifest.get('stage', 'A')}` | Split: `{manifest.get('split', 'DEV-3')}`")
    with c2:
        st.metric("Status", manifest.get("status", "UNKNOWN"))
        st.caption(f"Config Hash: `{manifest.get('config_hash', 'N/A')[:12]}`")
    with c3:
        renders = manifest.get("renders", {})
        comp = sum(1 for r in renders.values() if r.get("status") == "SUCCESS")
        total = max(len(renders), 1)
        st.metric("Scene Renders", f"{comp} / {total}")
        st.caption(f"Complete Reps: `{len(manifest.get('completed_replicates', []))}` | Partial: `{len(manifest.get('partial_replicates', []))}`")
    with c4:
        gen_h = manifest.get("cumulative_generation_seconds", 0.0) / 3600.0
        st.metric("Generation Time", f"{gen_h:.2f} h")
        st.caption(f"Git: `{manifest.get('git_commit', 'N/A')[:8]}` | Verifier: `{manifest.get('verifier_sha256', 'N/A')[:8]}...`")

    # Load summary if available
    summary_path = run_dir / "test3_summary.json"
    summary_data: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    # Panel 2: Metrics Dashboard
    st.markdown("---")
    st.subheader("2️⃣ Primary & Secondary Metrics (OUT-OF-LOOP vs IN-LOOP)")
    if summary_data and "primary_metrics" in summary_data:
        arm_options = list(summary_data.get("arms", {}).keys()) or ["v2_final"]
        arm_choice = st.radio("Select Phase 1 Arm:", arm_options, horizontal=True, key="eval3_arm_selector")
        arm_data = summary_data.get("arms", {}).get(arm_choice, summary_data)
        pm = arm_data.get("primary_metrics", summary_data["primary_metrics"])
        sec = arm_data.get("secondary_metrics", summary_data.get("secondary_metrics", {}))

        m1, m2, m3 = st.columns(3)
        with m1:
            p1 = pm.get("pm1_cross_scene_auc", {})
            st.metric("PM1: Cross-Scene AUC [OUT-OF-LOOP]", f"{p1.get('empirical_mean', 0.5):.4f}")
            b1 = p1.get("bootstrap", {})
            if b1:
                st.caption(f"95% CI: `[{b1.get('ci_95_lower', 0.0):.4f}, {b1.get('ci_95_upper', 1.0):.4f}]` | Chance: `0.50`")
        with m2:
            p2 = pm.get("pm2_cross_scene_rank1", {})
            st.metric("PM2: Cross-Scene Rank-1 [OUT-OF-LOOP]", f"{p2.get('empirical_mean', 0.0):.4f}")
            b2 = p2.get("bootstrap", {})
            if b2:
                st.caption(
                    f"95% CI: `[{b2.get('ci_95_lower', 0.0):.4f}, {b2.get('ci_95_upper', 1.0):.4f}]` | "
                    f"Chance: `{p2.get('chance_floor', 0.0909):.4f}` | Probes: Eligible (>=2 scenes)"
                )
            if "within_story_mean_pm2" in p2:
                st.caption(f"Within-Story Mean PM2: `{p2['within_story_mean_pm2']:.4f}`")
        with m3:
            p3 = pm.get("pm3_scene_correctness_rate", {})
            st.metric("PM3: Scene Correctness [OUT-OF-LOOP]", f"{p3.get('empirical_mean', 0.0):.4f}")
            b3 = p3.get("bootstrap", {})
            if b3:
                st.caption(f"95% CI: `[{b3.get('ci_95_lower', 0.0):.4f}, {b3.get('ci_95_upper', 1.0):.4f}]` | Floor: `0.00`")

        # Per-story breakdown table
        per_story = arm_data.get("per_story", {})
        if per_story:
            st.markdown("**Per-Story Breakdown (Eligible Probes & Chance Floors):**")
            st_rows = []
            for sl, sd in per_story.items():
                st_rows.append({
                    "Story": sl,
                    "Gallery Chars": sd.get("n_gallery_characters"),
                    "n_expected/rep": sd.get("n_expected_per_rep"),
                    "n_eligible/rep": sd.get("n_eligible_per_rep"),
                    "n_ineligible/rep": sd.get("n_ineligible_per_rep"),
                    "PM2 Chance": sd.get("chance_floor_pm2"),
                    "PM1 AUC": sd.get("pm1_cross_scene_auc_mean"),
                    "PM2 Rank-1 (Elig)": sd.get("pm2_cross_scene_rank1_mean"),
                    "PM3 Correctness": sd.get("pm3_scene_correctness_mean"),
                })
            st.dataframe(pd.DataFrame(st_rows), use_container_width=True, hide_index=True)

        # Judge-specific AUCs
        j_aucs = sec.get("judge_aucs", {})
        if j_aucs:
            st.markdown("**Out-of-Loop Judge Specific AUCs:**")
            j_cols = st.columns(4)
            for i, (j_name, j_val) in enumerate(j_aucs.items()):
                with j_cols[i % 4]:
                    st.metric(f"{j_name} AUC", f"{j_val:.4f}")

        # In-loop metrics summary
        in_loop = summary_data.get("in_loop_metrics_v1_2", {})
        if in_loop:
            st.markdown("**Frozen Verifier v1.2 Metrics [IN-LOOP]:**")
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("v1.2 Ensemble [IN-LOOP]", f"{in_loop.get('mean_v1_2_ensemble_score', 0.0):.4f}")
            ic2.metric("v1.2 Rank-1 vs Anchor", f"{in_loop.get('v1_2_rank1_vs_anchor_rate', 0.0):.4f}")
            ic3.metric("Buffalo-L Anchor Sim", f"{in_loop.get('mean_buffalo_l_anchor_similarity', 0.0):.4f}")
            ic4.metric("Wardrobe CIELAB dE", f"{in_loop.get('mean_wardrobe_cielab_delta_e', 0.0):.2f}")
    else:
        st.info("Primary metric summaries will appear once scoring completes.")

    # Panel 3: Replicate Browser (Anchors & Scene Renders)
    st.markdown("---")
    st.subheader("3️⃣ Replicate Browser: Certified Anchors & Scene Renders")
    replicates = manifest.get("replicates", {})
    if replicates:
        rep_keys = sorted(list(replicates.keys()))
        rep_labels = [
            f"{rk} ({replicates[rk].get('status', 'UNKNOWN')})"
            + (" — EXCLUDED FROM PRIMARY" if replicates[rk].get("status") == "PARTIAL" else "")
            for rk in rep_keys
        ]
        sel_idx = st.selectbox("Select Replicate:", range(len(rep_keys)), format_func=lambda i: rep_labels[i], key="eval3_rep_selector")
        rep_key = rep_keys[sel_idx]
        rep_info = replicates[rep_key]

        if rep_info.get("status") == "PARTIAL":
            st.warning(f"⚠️ `{rep_key}` is a PARTIAL replicate (stopped at 8.0h generation cap) and is excluded from primary statistical analysis.")

        # Certified Anchors
        st.markdown(f"##### Certified Identity Anchors (`{rep_key}`)")
        anchors_dir = Path(rep_info.get("anchors_path", "")) if rep_info.get("anchors_path") else (Path(rep_info.get("banks_path", "")).parent / "anchors")
        if anchors_dir.exists():
            chars = rep_info.get("characters", [])
            a_cols = st.columns(5)
            for idx, cid in enumerate(chars):
                sidecar_file = anchors_dir / f"{cid}_selected_anchor.json"
                if sidecar_file.exists():
                    with open(sidecar_file, "r", encoding="utf-8") as f:
                        a_meta = json.load(f)
                    with a_cols[idx % 5]:
                        img_p = a_meta.get("image_path")
                        if img_p and Path(img_p).exists():
                            st.image(img_p, caption=f"{cid} (seed {a_meta.get('seed')})", use_container_width=True)
                            badge = "⚠️ UNCERT" if a_meta.get("uncertified_anchor", False) else "✅ CERT"
                            st.caption(f"{badge} | Score: {a_meta.get('selection_score', 0)}")

        # Scene Renders Gallery
        st.markdown(f"##### Rendered Scenes (`{rep_key}`)")
        scenes_dir = Path(rep_info.get("scenes_path", "")) if rep_info.get("scenes_path") else (run_dir / "replicates" / rep_key / "scenes")
        if scenes_dir.exists():
            story_filter = st.radio("Filter Story:", ["All", "Story_A", "Story_B", "Story_C"], horizontal=True, key="eval3_story_filter")
            scene_pngs = sorted(scenes_dir.glob("rep_*.png"))
            if story_filter != "All":
                scene_pngs = [p for p in scene_pngs if f"_{story_filter}_" in p.name]

            # Load per-scene CSV for badges if available
            per_scene_csv = run_dir / "test3_per_scene.csv"
            scene_meta_map: Dict[str, Dict[str, Any]] = {}
            if per_scene_csv.exists():
                try:
                    df_sc_all = pd.read_csv(per_scene_csv)
                    for _, row in df_sc_all.iterrows():
                        scene_meta_map[str(row["render_id"])] = row.to_dict()
                except Exception:
                    pass

            st.caption(f"Showing **{len(scene_pngs)}** scene renders.")
            g_cols = st.columns(3)
            for idx, png_p in enumerate(scene_pngs):
                rid = png_p.stem
                smeta = scene_meta_map.get(rid, {})
                bon3_tag = " ⭐ BON3" if smeta.get("bon3_selected") else ""
                pm3_tag = ""
                if "pm3_correct" in smeta:
                    pm3_tag = " | PM3 ✅" if smeta["pm3_correct"] else " | PM3 ❌"
                with g_cols[idx % 3]:
                    st.image(str(png_p), caption=f"{rid}{bon3_tag}{pm3_tag}", use_container_width=True)

        # Panel 4: Cross-Scene Face Strips
        st.markdown("---")
        st.subheader(f"4️⃣ Cross-Scene Face Strips (`{rep_key}`)")
        st.caption("Aligns certified anchor + detected face crops across distinct scenes for each character.")
        strip_rep_dir = run_dir / "face_strips" / rep_key
        if strip_rep_dir.exists():
            strips = sorted(strip_rep_dir.glob("*_strip.png"))
            for sp in strips:
                cid = sp.stem.replace("_strip", "")
                st.markdown(f"**`{cid}`**")
                st.image(str(sp))
        else:
            st.info(f"Face strips not built for `{rep_key}` (only complete primary replicates have strips).")

    # Panel 5: Failures Gallery
    st.markdown("---")
    st.subheader("5️⃣ Flagged Failures Gallery (`PM3 == False`)")
    per_scene_p = run_dir / "test3_per_scene.csv"
    if per_scene_p.exists():
        df_s = pd.read_csv(per_scene_p)
        flagged = df_s[df_s["pm3_correct"] == False]
        if len(flagged) > 0:
            st.warning(f"Found {len(flagged)} renders flagged with `PM3 == False` across primary completed replicates.")
            f_cols = st.columns(3)
            for idx, (_, row) in enumerate(flagged.iterrows()):
                rk = f"rep_{int(row['replicate_idx']):02d}"
                img_p = run_dir / "replicates" / rk / "scenes" / f"{row['render_id']}.png"
                with f_cols[idx % 3]:
                    if img_p.exists():
                        st.image(
                            str(img_p),
                            caption=f"{row['render_id']} (exp={row['expected_count']}, yolo={row['yolo_count']}, det={row['faces_detected']})",
                            use_container_width=True,
                        )
        else:
            st.success("Zero PM3 failures flagged in this run.")
    else:
        st.info("Per-scene scoring table not yet available.")

    # Panel 6: Raw Per-Face & Per-Scene Tables with CSV Download
    st.markdown("---")
    st.subheader("6️⃣ Raw Tables & CSV Export")
    per_face_p = run_dir / "test3_per_face.csv"
    t1, t2 = st.tabs(["Per-Face Records", "Per-Scene Records"])
    with t1:
        if per_face_p.exists():
            df_f = pd.read_csv(per_face_p)
            st.dataframe(df_f, use_container_width=True)
            with open(per_face_p, "rb") as f:
                st.download_button("📥 Download test3_per_face.csv", f, "test3_per_face.csv", "text/csv")
        else:
            st.info("Per-face CSV not generated yet.")
    with t2:
        if per_scene_p.exists():
            df_s2 = pd.read_csv(per_scene_p)
            st.dataframe(df_s2, use_container_width=True)
            with open(per_scene_p, "rb") as f:
                st.download_button("📥 Download test3_per_scene.csv", f, "test3_per_scene.csv", "text/csv")
        else:
            st.info("Per-scene CSV not generated yet.")


def _sanitize_rater_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return cleaned[:48]


def render_human_rating_tab():
    """
    Renders the Blinded Human Rating Tab (B4).
    STRICT BLINDING INVARIANT: Reads ONLY rating_pack_blind.json; NEVER reads SEALED_ANSWER_KEY.json.
    Saves ratings per rater name to results/eval3/human_ratings/ratings_<rater_name>.json.
    """
    st.header("📝 TEST-3 Blinded Human Rating Pack")
    st.info(
        "**Protocol Guidance:** Two independent human raters are recommended. "
        "Enter your unique rater name below; your ratings are saved automatically to "
        "`results/eval3/human_ratings/ratings_<rater>.json` so you can pause and resume at any time. "
        "The answer key is stored in a separate sealed file that this UI never reads."
    )

    blind_pack_path = EVAL_OUTPUT_BASE / "test3_v2_final" / "human_rating_pack" / "rating_pack_blind.json"
    if not blind_pack_path.exists():
        st.warning(f"Human rating pack not found at `{blind_pack_path}`. Run `tools/build_test3_human_rating_pack.py` first.")
        return

    with open(blind_pack_path, "r", encoding="utf-8") as f:
        pack = json.load(f)

    crops_dir = Path(pack["crops_dir"])
    task_a_items = pack.get("task_a_items", [])
    task_b_items = pack.get("task_b_items", [])

    # List existing rater files
    HUMAN_RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    existing_raters = sorted([p.stem.replace("ratings_", "") for p in HUMAN_RATINGS_DIR.glob("ratings_*.json")])
    if existing_raters:
        st.caption(f"Saved rater profiles on disk: `{', '.join(existing_raters)}`")

    raw_name = st.text_input("👤 Rater Name (e.g., `rater_1`, `rater_2`):", value="", key="hr_rater_name")
    rater_id = _sanitize_rater_name(raw_name)
    if not rater_id:
        st.warning("Enter your rater name above to begin or resume rating.")
        return

    rater_file = HUMAN_RATINGS_DIR / f"ratings_{rater_id}.json"
    saved: Dict[str, Any] = {"rater_name": rater_id, "pack_id": pack["pack_id"], "task_a": {}, "task_b": {}}
    if rater_file.exists():
        try:
            with open(rater_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            pass

    task_a_answers: Dict[str, Optional[str]] = dict(saved.get("task_a", {}))
    task_b_answers: Dict[str, Optional[int]] = dict(saved.get("task_b", {}))

    n_a_done = sum(1 for item in task_a_items if task_a_answers.get(item["trial_id"]) in ["A", "B", "C", "D"])
    n_b_done = sum(1 for item in task_b_items if task_b_answers.get(item["sequence_id"]) in [1, 2, 3, 4, 5])
    total_items = len(task_a_items) + len(task_b_items)
    total_done = n_a_done + n_b_done

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Task A (Same-Person 4-AFC)", f"{n_a_done} / {len(task_a_items)}")
    pcol2.metric("Task B (1–5 Consistency)", f"{n_b_done} / {len(task_b_items)}")
    pcol3.metric("Overall Progress", f"{total_done} / {total_items}")
    st.progress(min(1.0, total_done / max(total_items, 1)))

    sub_a, sub_b = st.tabs([
        f"🎯 Task A: Same-Person Matching ({n_a_done}/{len(task_a_items)})",
        f"🎞️ Task B: Sequence Consistency 1–5 ({n_b_done}/{len(task_b_items)})",
    ])

    new_task_a: Dict[str, Optional[str]] = {}
    with sub_a:
        st.markdown(pack.get("task_a_instructions", ""))
        for item in task_a_items:
            tid = item["trial_id"]
            st.markdown(f"---\n**Trial `{tid}`**")
            cols = st.columns([1.2, 0.2, 1, 1, 1, 1])
            probe_p = crops_dir / item["probe_crop"]
            with cols[0]:
                if probe_p.exists():
                    st.image(str(probe_p), caption="PROBE", use_container_width=True)
            with cols[1]:
                st.markdown("### ➡️")
            for idx_opt, opt_letter in enumerate(["A", "B", "C", "D"]):
                cand_p = crops_dir / item["candidates"][opt_letter]
                with cols[2 + idx_opt]:
                    if cand_p.exists():
                        st.image(str(cand_p), caption=f"Option {opt_letter}", use_container_width=True)

            opts = ["(Unanswered)", "A", "B", "C", "D"]
            prev_val = task_a_answers.get(tid)
            default_idx = opts.index(prev_val) if prev_val in opts else 0
            choice = st.radio(
                f"Which candidate matches the PROBE in `{tid}`?",
                opts,
                index=default_idx,
                horizontal=True,
                key=f"hr_{rater_id}_{tid}",
            )
            if choice != "(Unanswered)":
                new_task_a[tid] = choice

    new_task_b: Dict[str, Optional[int]] = {}
    with sub_b:
        st.markdown(pack.get("task_b_instructions", ""))
        scale_labels = {
            0: "(Unanswered)",
            1: "1 — Completely different people",
            2: "2 — Mostly different",
            3: "3 — Ambiguous / somewhat similar",
            4: "4 — Same person (minor drift)",
            5: "5 — Clearly exact same person",
        }
        scale_keys = [0, 1, 2, 3, 4, 5]
        for item in task_b_items:
            sid = item["sequence_id"]
            st.markdown(f"---\n**Sequence `{sid}`**")
            crops = item.get("face_crops", [])
            scols = st.columns(max(len(crops), 1))
            for i_c, cf in enumerate(crops):
                cp = crops_dir / cf
                with scols[i_c]:
                    if cp.exists():
                        st.image(str(cp), caption=f"Scene {i_c + 1}", use_container_width=True)
            prev_b = task_b_answers.get(sid)
            default_b = scale_keys.index(int(prev_b)) if prev_b in [1, 2, 3, 4, 5] else 0
            rating = st.radio(
                f"Identity consistency rating for `{sid}`:",
                scale_keys,
                index=default_b,
                format_func=lambda k: scale_labels[k],
                horizontal=True,
                key=f"hr_{rater_id}_{sid}",
            )
            if rating != 0:
                new_task_b[sid] = int(rating)

    st.markdown("---")
    if st.button("💾 Save Ratings to Disk", type="primary", use_container_width=True, key=f"hr_save_{rater_id}"):
        payload = {
            "rater_name": rater_id,
            "pack_id": pack["pack_id"],
            "saved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_a_completed": len(new_task_a),
            "task_a_total": len(task_a_items),
            "task_b_completed": len(new_task_b),
            "task_b_total": len(task_b_items),
            "all_completed": (len(new_task_a) == len(task_a_items)) and (len(new_task_b) == len(task_b_items)),
            "task_a": new_task_a,
            "task_b": new_task_b,
        }
        tmp_f = rater_file.with_suffix(".json.tmp")
        with open(tmp_f, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_f, rater_file)
        st.success(
            f"Saved ratings for `{rater_id}` to `{rater_file.relative_to(PROJECT_ROOT)}` "
            f"(Task A: {len(new_task_a)}/{len(task_a_items)}, Task B: {len(new_task_b)}/{len(task_b_items)})."
        )
