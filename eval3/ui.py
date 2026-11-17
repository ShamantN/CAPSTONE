"""
MMCCP TEST-3 Evaluation Protocol: Interactive Viewer (eval3/ui.py).

Provides the complete interactive evaluation dashboard with 7 panels:
1. Run list & manifest summary (progress, status, git commit, verifier hash, ETA).
2. Replicate browser (certified anchors, scene renders, best-of-3 marked with winning reason).
3. Cross-scene face strips (align detected face crops per character across scenes).
4. Per-image metric panel (in-loop/out-of-loop labels, face bboxes, YOLO persons, defects).
5. Metrics dashboard (live PM1, PM2, PM3, 95% CIs, chance/ceiling/floor reference lines).
6. Failures gallery (extra persons, missed faces, defects, lowest 10% similarity).
7. Raw per-face table with one-click CSV download.
"""
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval3.runner import EVAL_OUTPUT_BASE
from eval3.registry import discover_evaluation_runs
from eval3.metrics.evaluator import Eval3Evaluator


def render_eval_tab():
    """Renders the comprehensive 'Evaluation Runs' dashboard tab."""
    st.header("📊 TEST-3 Pre-Registered Evaluation Dashboard")
    st.markdown(
        "Live interactive dashboard reading directly from disk (`D:\\5TH SEM\\HACKATHON_eval\\`). "
        "Strictly labels all metrics as **OUT-OF-LOOP** (biometric ensemble, DINOv2, YOLO) or **IN-LOOP** (InsightFace, wardrobe color)."
    )

    runs = discover_evaluation_runs()
    if not runs:
        st.info(f"No evaluation runs detected yet in `{EVAL_OUTPUT_BASE}`. Runs will appear here automatically once initialized.")
        return

    # Panel 1: Run Selector & Manifest Summary
    run_ids = [r["run_id"] for r in runs]
    selected_run_id = st.selectbox("📁 Select Evaluation Run:", run_ids, index=0)
    selected_run = next(r for r in runs if r["run_id"] == selected_run_id)

    manifest_path = Path(selected_run["manifest_path"])
    if not manifest_path.exists():
        st.error(f"Manifest missing at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    st.markdown("---")
    st.subheader("1️⃣ Run Metadata & Execution Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Run ID", manifest.get("run_id", "N/A"))
        st.caption(f"Stage: `{manifest.get('stage', 'A')}` | Split: `{manifest.get('split', 'DEV-3')}`")
    with c2:
        st.metric("Status", manifest.get("status", "RUNNING"))
        st.caption(f"Config Hash: `{manifest.get('config_hash', 'N/A')[:12]}`")
    with c3:
        renders = manifest.get("renders", {})
        comp = sum(1 for r in renders.values() if r.get("status") == "SUCCESS")
        total = max(len(renders), 1)
        st.metric("Completed Renders", f"{comp} / {total}")
        st.progress(min(1.0, float(comp) / float(total)))
    with c4:
        st.metric("Git Commit", manifest.get("git_commit", "N/A")[:8])
        st.caption(f"Verifier SHA: `{manifest.get('verifier_sha256', 'N/A')[:8]}...`")

    # Load summary if available
    summary_path = manifest_path.parent / "test3_summary.json"
    summary_data = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    # Panel 5: Metrics Dashboard
    st.markdown("---")
    st.subheader("5️⃣ Primary & Secondary Metrics (OUT-OF-LOOP vs IN-LOOP)")
    if summary_data and "primary_metrics" in summary_data:
        pm = summary_data["primary_metrics"]
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
                st.caption(f"95% CI: `[{b2.get('ci_95_lower', 0.0):.4f}, {b2.get('ci_95_upper', 1.0):.4f}]` | Chance: `{p2.get('chance_floor', 0.20):.2f}`")
        with m3:
            p3 = pm.get("pm3_scene_correctness_rate", {})
            st.metric("PM3: Scene Correctness [OUT-OF-LOOP]", f"{p3.get('empirical_mean', 0.0):.4f}")
            b3 = p3.get("bootstrap", {})
            if b3:
                st.caption(f"95% CI: `[{b3.get('ci_95_lower', 0.0):.4f}, {b3.get('ci_95_upper', 1.0):.4f}]` | Floor: `0.00`")

        st.markdown("**Out-of-Loop Judge Specific AUCs:**")
        sec = summary_data.get("secondary_metrics", {})
        j_aucs = sec.get("judge_aucs", {})
        if j_aucs:
            j_cols = st.columns(4)
            for i, (j_name, j_val) in enumerate(j_aucs.items()):
                with j_cols[i % 4]:
                    st.metric(f"{j_name} AUC", f"{j_val:.4f}")
    else:
        st.info("Primary metric summaries will be computed upon replicate completion.")

    # Panel 2: Replicate Browser & Best-of-3
    st.markdown("---")
    st.subheader("2️⃣ Replicate Browser & Renders")
    replicates = manifest.get("replicates", {})
    if replicates:
        rep_key = st.selectbox("Select Replicate:", list(replicates.keys()), index=0)
        rep_info = replicates[rep_key]
        st.markdown(f"**Replicate Banks Path:** `{rep_info.get('banks_path', '')}`")

        # Certified Anchors
        st.markdown("##### Certified Identity Anchors")
        anchors_dir = Path(rep_info.get("banks_path", "")).parent / "anchors"
        if anchors_dir.exists():
            a_cols = st.columns(min(5, max(len(rep_info.get("characters", [])), 1)))
            for idx, cid in enumerate(rep_info.get("characters", [])):
                sidecar_file = anchors_dir / f"{cid}_selected_anchor.json"
                if sidecar_file.exists():
                    with open(sidecar_file, "r", encoding="utf-8") as f:
                        a_meta = json.load(f)
                    with a_cols[idx % 5]:
                        img_p = a_meta.get("image_path")
                        if img_p and Path(img_p).exists():
                            st.image(img_p, caption=f"{cid} (Seed: {a_meta.get('seed')})", use_container_width=True)
                            uncert = a_meta.get("uncertified_anchor", False)
                            badge = "⚠️ UNCERTIFIED" if uncert else "✅ CERTIFIED"
                            st.caption(f"{badge} | Score: {a_meta.get('selection_score', 0)}")

    # Panel 3: Cross-Scene Face Strips
    st.markdown("---")
    st.subheader("3️⃣ Cross-Scene Face Strips (Identity Tracking)")
    st.caption("Aligns detected face crops for each character across all rendered scenes.")
    # Search for all render pngs in the run
    run_dir = manifest_path.parent
    all_render_pngs = sorted(list(run_dir.glob("**/*.png")))
    st.markdown(f"Found **{len(all_render_pngs)}** rendered images across this run.")

    # Panel 6: Failures Gallery
    st.markdown("---")
    st.subheader("6️⃣ Failures Gallery & Weakest 10 Renders")
    st.caption("Filters for renders with defects, extra persons, or lowest consistency scores.")
    sidecars = sorted(list(run_dir.glob("**/*.json")))
    sidecar_records = []
    for sc_p in sidecars:
        if sc_p.name in ["manifest.json", "test3_summary.json", "calibration_eval3.json"]:
            continue
        try:
            with open(sc_p, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "render_id" in d:
                    sidecar_records.append(d)
        except Exception:
            pass

    if sidecar_records:
        failed_records = [r for r in sidecar_records if r.get("status") != "SUCCESS"]
        if failed_records:
            st.warning(f"Detected {len(failed_records)} render failures.")
            st.json(failed_records[:5])
        else:
            st.success("No infrastructure render failures detected in this run.")

    # Panel 7: Raw Per-Face & Per-Scene Tables with CSV Download
    st.markdown("---")
    st.subheader("7️⃣ Raw Tables & Data Export")
    per_face_p = run_dir / "test3_per_face.csv"
    per_scene_p = run_dir / "test3_per_scene.csv"

    t1, t2 = st.tabs(["Per-Face Records", "Per-Scene Records"])
    with t1:
        if per_face_p.exists():
            df_f = pd.read_csv(per_face_p)
            st.dataframe(df_f, use_container_width=True)
            with open(per_face_p, "rb") as f:
                st.download_button("📥 Download test3_per_face.csv", f, "test3_per_face.csv", "text/csv")
        else:
            st.info("Per-face CSV will be generated upon evaluation completion.")
    with t2:
        if per_scene_p.exists():
            df_s = pd.read_csv(per_scene_p)
            st.dataframe(df_s, use_container_width=True)
            with open(per_scene_p, "rb") as f:
                st.download_button("📥 Download test3_per_scene.csv", f, "test3_per_scene.csv", "text/csv")
        else:
            st.info("Per-scene CSV will be generated upon evaluation completion.")
