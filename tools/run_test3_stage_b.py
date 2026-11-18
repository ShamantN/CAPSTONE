"""
MMCCP TEST-3 Evaluation Protocol: Stage B Generation Runner (tools/run_test3_stage_b.py).

Executes Option A (R=4 replicates, 12 supported scenes x 3 seeds = 144 v2_final renders,
plus 44 character anchors with up to 8 candidates each) strictly in order:
  rep_00 -> rep_01 -> rep_02 -> rep_03
  within each replicate: anchors -> (story, scene, seed) -> optional 3-char stress scene (if cap allows)

Enforces all pre-registered invariants:
- ZERO GPU usage by this runner process (CUDA_VISIBLE_DEVICES=-1 before any ML import).
- Locked C0 config from results/eval3/selected_config.json (hash df4a0a2648ac).
- Resumable: completed renders (status==SUCCESS) are skipped on restart.
- Infrastructure failures retried ONCE (with Forge restart, same seed/config); 2nd failure logged as FAILED.
- Forge restarted every 40 renders and on any timeout/OOM/VRAM watchdog trigger (>7900 MB).
- 8.0-hour generation-only cap (ANCHOR + SCENE generation time only); stops after last complete replicate
  (min R>=3); any partial replicate archived in results/eval3/partial/.
- Saves complete Stage D archive: anchors, candidates, seed_table.json, storyboards/, config_snapshot.json,
  environment_record.json.
"""
import os
# STRICT RULE: Runner process must NEVER touch GPU VRAM (100% belongs to Forge on port 7860)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import time
import shutil
import hashlib
import datetime
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from story_engine.schema import StorySpec, SceneSpec
from consistency_v2.config import ConsistencyV2Config
from consistency_v2.pipeline_v2 import ConsistencyPipelineV2
from consistency_v2.identity_bank import IdentityBankManager
from consistency_v2.runtime_paths import get_forge_url
from eval3.splits import get_anchor_seed, get_scene_seed
from eval3.anchors import (
    load_all_evaluation_characters,
    generate_character_candidates,
    select_best_anchor,
    check_intra_replicate_separation,
    DEFAULT_SEPARATION_THRESHOLD,
    SceneAuditor,
)
from eval3.runner import (
    EVAL_OUTPUT_BASE,
    get_git_commit,
    get_verifier_freeze_sha,
    get_current_gpu_vram_mb,
    compute_file_sha256,
)

RUN_ID = "test3_v2_final"
BUDGET_HOURS = 8.0
BUDGET_SECONDS = BUDGET_HOURS * 3600.0  # 28,800s generation-only cap
TARGET_REPLICATES = 4
MIN_COMPLETE_REPLICATES = 3
FORGE_RESTART_INTERVAL = 40
VRAM_WATCHDOG_MB = 7900.0
FORGE_ROOT = Path(r"C:\Stable-Diffusion")

# Story definitions loaded by filename (keeps TEST-2 isolation audit 100% clean)
STORY_SPECS_META = [
    {"story_idx": 0, "label": "Story_A", "filename": "ebright.json", "supported_scene_indices": [0, 1, 2, 3, 4], "stress_scene_indices": []},
    {"story_idx": 1, "label": "Story_B", "filename": "holdout_v2.json", "supported_scene_indices": [0, 1, 2], "stress_scene_indices": [3]},
    {"story_idx": 2, "label": "Story_C", "filename": "eval3_story_c.json", "supported_scene_indices": [0, 1, 2, 3], "stress_scene_indices": []},
]

# Fixed deterministic character ordering across all 3 stories (11 characters total, char_idx 0..10)
ORDERED_CHAR_IDS = [
    # Story A (0..4)
    "young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong",
    # Story B (5..7)
    "elena_child", "dr_marcus_vance", "clara_thorn",
    # Story C (8..10)
    "leo_child", "maya_adult", "arthur_elder",
]

STORY_CHAR_GROUPS = {
    "Story_A": ["young_ebright", "mother", "teen_ebright", "college_ebright", "james_wong"],
    "Story_B": ["elena_child", "dr_marcus_vance", "clara_thorn"],
    "Story_C": ["leo_child", "maya_adult", "arthur_elder"],
}


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_for_forge_ready(forge_url: str, timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{forge_url}/sdapi/v1/sd-models", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def restart_forge_server(forge_url: str, log_file: Path, reason: str = "scheduled_40_renders") -> bool:
    log(f"[FORGE_RESTART] Initiating clean Forge restart (reason={reason})...")
    # 1. Kill only python processes running under C:\Stable-Diffusion
    ps_kill = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -eq 'python.exe' -or $_.Name -eq 'cmd.exe') -and "
        "($_.CommandLine -like '*Stable-Diffusion*' -or $_.ExecutablePath -like '*Stable-Diffusion*') "
        "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_kill], capture_output=True, timeout=20)
    except Exception as e:
        log(f"[FORGE_RESTART] Warning during process termination: {e}")

    time.sleep(6)

    # 2. Launch Forge detached via cmd.exe /c run.bat
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== Forge Restart ({reason}) at {datetime.datetime.now(datetime.timezone.utc).isoformat()} ===\n")
    lf_handle = open(log_file, "a", encoding="utf-8")
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008
    subprocess.Popen(
        ["cmd.exe", "/c", f"cd /d {FORGE_ROOT} && run.bat"],
        stdout=lf_handle,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )

    # 3. Wait for Forge API to respond 200
    ok = wait_for_forge_ready(forge_url, timeout_s=240)
    if ok:
        # Re-apply Forge memory parameters after restart
        try:
            requests.post(
                f"{forge_url}/sdapi/v1/options",
                json={"forge_inference_memory": 3072.0, "xl_GPU_MB": 1600.0, "disable_weights_auto_swap": False},
                timeout=15,
            )
        except Exception:
            pass
        log(f"[FORGE_RESTART] Forge is healthy and ready on {forge_url}.")
    else:
        log(f"[FORGE_RESTART] ERROR: Forge failed to come up within 240s!")
    return ok


def load_locked_config() -> Tuple[ConsistencyV2Config, Dict[str, Any]]:
    sel_path = PROJECT_ROOT / "results" / "eval3" / "selected_config.json"
    with open(sel_path, "r", encoding="utf-8") as f:
        sel_data = json.load(f)
    cfg = ConsistencyV2Config(mode="default")
    actual_hash = cfg.to_hash()
    expected_hash = sel_data["config_hash"]
    assert actual_hash == expected_hash, f"Config hash mismatch: actual={actual_hash} != locked={expected_hash}"
    assert actual_hash == "df4a0a2648ac", f"Unexpected config hash: {actual_hash}"
    return cfg, sel_data


def build_and_save_stage_d_archive(
    out_dir: Path,
    stories: Dict[int, StorySpec],
    sel_data: Dict[str, Any],
    forge_url: str,
) -> None:
    """Saves seed tables, config snapshots, storyboards, and environment record for Stage D."""
    # 1. Storyboards archive
    sb_out = out_dir / "storyboards"
    sb_out.mkdir(parents=True, exist_ok=True)
    sb_hashes = {}
    for sm in STORY_SPECS_META:
        src = PROJECT_ROOT / "storyboards" / sm["filename"]
        dst = sb_out / sm["filename"]
        if not dst.exists():
            shutil.copyfile(src, dst)
        sb_hashes[sm["filename"]] = compute_file_sha256(src)

    # 2. Config snapshot
    best_cfg_p = PROJECT_ROOT / "consistency_v2" / "best_config.json"
    sel_cfg_p = PROJECT_ROOT / "results" / "eval3" / "selected_config.json"
    config_snap = {
        "selected_config_path": str(sel_cfg_p),
        "selected_config_sha256": compute_file_sha256(sel_cfg_p),
        "best_config_sha256": compute_file_sha256(best_cfg_p) if best_cfg_p.exists() else "",
        "selected_config": sel_data,
    }
    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snap, f, indent=2)

    # 3. Complete deterministic seed table
    seed_table = {"anchors": [], "v2_final_supported_scenes": [], "v2_final_stress_scenes": [], "stage_d_reserved_preview": []}
    for r in range(TARGET_REPLICATES):
        for c_idx, cid in enumerate(ORDERED_CHAR_IDS):
            seed_table["anchors"].append({
                "replicate_idx": r,
                "char_idx": c_idx,
                "char_id": cid,
                "candidate_seeds_0_to_7": [get_anchor_seed(r, c_idx, k, is_test3=True) for k in range(8)],
            })
        for sm in STORY_SPECS_META:
            st_idx = sm["story_idx"]
            story = stories[st_idx]
            for sc_idx in sm["supported_scene_indices"]:
                sc = story.scenes[sc_idx]
                for s_idx in range(3):
                    seed_table["v2_final_supported_scenes"].append({
                        "replicate_idx": r,
                        "story_idx": st_idx,
                        "story_label": sm["label"],
                        "scene_idx": sc_idx,
                        "scene_number": sc.scene_number,
                        "scene_id": sc.scene_id,
                        "seed_idx": s_idx,
                        "seed": get_scene_seed(r, st_idx, sc_idx, s_idx, arm="v2_final", is_test3=True),
                    })
            for sc_idx in sm["stress_scene_indices"]:
                sc = story.scenes[sc_idx]
                for s_idx in range(3):
                    seed_table["v2_final_stress_scenes"].append({
                        "replicate_idx": r,
                        "story_idx": st_idx,
                        "story_label": sm["label"],
                        "scene_idx": sc_idx,
                        "scene_number": sc.scene_number,
                        "scene_id": sc.scene_id,
                        "seed_idx": s_idx,
                        "seed": get_scene_seed(r, st_idx, sc_idx, s_idx, arm="v2_final", is_test3=True),
                    })
    with open(out_dir / "seed_table.json", "w", encoding="utf-8") as f:
        json.dump(seed_table, f, indent=2)

    # 4. Environment record (Forge commit, model hashes, GPU driver, repo hashes)
    gpu_info = ""
    try:
        r_gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu_info = r_gpu.stdout.strip()
    except Exception:
        pass

    forge_git_commit = "unknown"
    try:
        r_fg = subprocess.run(
            ["git", "--no-pager", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(FORGE_ROOT / "webui"), timeout=5
        )
        if r_fg.returncode == 0:
            forge_git_commit = r_fg.stdout.strip()
    except Exception:
        pass

    forge_options = {}
    forge_models = []
    try:
        ro = requests.get(f"{forge_url}/sdapi/v1/options", timeout=10)
        if ro.status_code == 200:
            od = ro.json()
            forge_options = {
                "sd_model_checkpoint": od.get("sd_model_checkpoint"),
                "sd_checkpoint_hash": od.get("sd_checkpoint_hash"),
                "sd_vae": od.get("sd_vae"),
                "forge_inference_memory": od.get("forge_inference_memory"),
            }
        rm = requests.get(f"{forge_url}/sdapi/v1/sd-models", timeout=10)
        if rm.status_code == 200:
            forge_models = [
                {"title": m.get("title"), "model_name": m.get("model_name"), "hash": m.get("hash"), "sha256": m.get("sha256")}
                for m in rm.json()
            ]
    except Exception:
        pass

    webui_user_bat = ""
    wub = FORGE_ROOT / "webui" / "webui-user.bat"
    if wub.exists():
        webui_user_bat = wub.read_text(encoding="utf-8", errors="ignore")

    env_record = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "platform": platform.platform(),
        "python_version": sys.version,
        "gpu_info": gpu_info,
        "repo_git_commit": get_git_commit(),
        "verifier_freeze_sha256": get_verifier_freeze_sha(),
        "evaluator_sha256": compute_file_sha256(PROJECT_ROOT / "eval3" / "metrics" / "evaluator.py"),
        "calibration_sha256": compute_file_sha256(PROJECT_ROOT / "results" / "eval3" / "calibration_eval3.json"),
        "selected_config_sha256": compute_file_sha256(sel_cfg_p),
        "storyboard_sha256s": sb_hashes,
        "forge_git_commit": forge_git_commit,
        "forge_url": forge_url,
        "forge_active_options": forge_options,
        "forge_models": forge_models,
        "forge_webui_user_bat": webui_user_bat,
    }
    with open(out_dir / "environment_record.json", "w", encoding="utf-8") as f:
        json.dump(env_record, f, indent=2)
    log(f"[ARCHIVE] Saved seed_table.json, config_snapshot.json, storyboards/, and environment_record.json.")


def load_or_init_manifest(manifest_path: Path, cfg: ConsistencyV2Config) -> Dict[str, Any]:
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    m = {
        "run_id": RUN_ID,
        "stage": "B",
        "split": "TEST-3",
        "option": "A",
        "budget_hours_gen_cap": BUDGET_HOURS,
        "target_replicates": TARGET_REPLICATES,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_hash": cfg.to_hash(),
        "config_compute_hash": cfg.compute_hash(),
        "git_commit": get_git_commit(),
        "verifier_sha256": get_verifier_freeze_sha(),
        "status": "RUNNING",
        "cumulative_generation_seconds": 0.0,
        "anchor_generation_seconds": 0.0,
        "scene_generation_seconds": 0.0,
        "stress_generation_seconds": 0.0,
        "total_renders_executed": 0,
        "forge_restarts": [],
        "completed_replicates": [],
        "partial_replicates": [],
        "replicates": {},
        "renders": {},
        "stress_renders": {},
    }
    save_manifest(manifest_path, m)
    return m


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]):
    manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def archive_partial_replicate(rep_idx: int, manifest: Dict[str, Any], reason: str):
    partial_dir = PROJECT_ROOT / "results" / "eval3" / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    rep_key = f"rep_{rep_idx:02d}"
    out_p = partial_dir / f"{RUN_ID}_{rep_key}_partial.json"
    summary = {
        "run_id": RUN_ID,
        "replicate_idx": rep_idx,
        "reason": reason,
        "archived_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cumulative_generation_seconds": round(manifest["cumulative_generation_seconds"], 2),
        "replicate_info": manifest["replicates"].get(rep_key, {}),
        "renders_attempted": {
            k: v for k, v in manifest["renders"].items() if k.startswith(f"{rep_key}_")
        },
    }
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[PARTIAL] Archived partial replicate {rep_key} to {out_p}")


def main():
    log("=" * 72)
    log(f"MMCCP TEST-3 STAGE B GENERATION RUNNER — RUN ID: {RUN_ID}")
    log(f"Option A: R={TARGET_REPLICATES} replicates, 12 supported scenes x 3 seeds, cap={BUDGET_HOURS}h (gen only)")
    log("=" * 72)

    cfg, sel_data = load_locked_config()
    forge_url = get_forge_url().rstrip("/")
    out_dir = EVAL_OUTPUT_BASE / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    forge_restart_log = out_dir / "forge_restarts.log"

    # Verify Forge is reachable before starting
    if not wait_for_forge_ready(forge_url, timeout_s=30):
        log("[INIT] Forge not responding on port 7860; launching Forge...")
        ok = restart_forge_server(forge_url, forge_restart_log, reason="initial_start")
        if not ok:
            raise RuntimeError("Forge failed to start on port 7860.")

    # Load all 3 storyboards
    stories: Dict[int, StorySpec] = {}
    for sm in STORY_SPECS_META:
        sb_p = PROJECT_ROOT / "storyboards" / sm["filename"]
        stories[sm["story_idx"]] = StorySpec.load_json(str(sb_p))

    all_characters = load_all_evaluation_characters()
    auditor = SceneAuditor()

    # Save complete Stage D archive before first render
    build_and_save_stage_d_archive(out_dir, stories, sel_data, forge_url)

    manifest = load_or_init_manifest(manifest_path, cfg)
    renders_since_restart = 0

    def maybe_restart_forge(force_reason: Optional[str] = None):
        nonlocal renders_since_restart
        vram_now = get_current_gpu_vram_mb()
        reason = force_reason
        if reason is None and renders_since_restart >= FORGE_RESTART_INTERVAL:
            reason = f"scheduled_{renders_since_restart}_renders"
        elif reason is None and vram_now >= VRAM_WATCHDOG_MB:
            reason = f"vram_watchdog_{vram_now:.0f}MB"
        if reason is not None:
            ok = restart_forge_server(forge_url, forge_restart_log, reason=reason)
            manifest["forge_restarts"].append({
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": reason,
                "vram_before_mb": round(vram_now, 1),
                "success": ok,
                "total_renders_executed": manifest["total_renders_executed"],
            })
            renders_since_restart = 0
            save_manifest(manifest_path, manifest)

    def on_anchor_render(failed: bool):
        nonlocal renders_since_restart
        renders_since_restart += 1
        manifest["total_renders_executed"] += 1
        if failed:
            maybe_restart_forge(force_reason="anchor_render_failure")
        else:
            maybe_restart_forge()

    stopped_by_cap = False

    for r_idx in range(TARGET_REPLICATES):
        rep_key = f"rep_{r_idx:02d}"
        if rep_key in manifest.get("completed_replicates", []):
            log(f"[RESUME] Replicate {rep_key} already COMPLETE. Skipping.")
            continue

        # Check stopping rule BEFORE starting a new replicate
        if manifest["cumulative_generation_seconds"] >= BUDGET_SECONDS:
            log(
                f"[STOP_RULE] Cumulative generation time {manifest['cumulative_generation_seconds']/3600:.2f}h "
                f">= {BUDGET_HOURS}h cap before starting {rep_key}. Stopping after last complete replicate."
            )
            stopped_by_cap = True
            break

        rep_dir = out_dir / "replicates" / rep_key
        anchors_dir = rep_dir / "anchors"
        banks_dir = rep_dir / "banks"
        scenes_dir = rep_dir / "scenes"
        stress_dir = rep_dir / "stress_scenes"
        for d in [anchors_dir, banks_dir, scenes_dir, stress_dir]:
            d.mkdir(parents=True, exist_ok=True)

        log(f"\n{'='*72}\nSTARTING REPLICATE {rep_key} (elapsed gen: {manifest['cumulative_generation_seconds']/3600:.2f}h / {BUDGET_HOURS}h)\n{'='*72}")

        # --- STEP 1: ANCHORS FOR ALL 11 CHARACTERS ---
        selected_anchors: Dict[str, Dict[str, Any]] = {}
        all_char_candidates: Dict[str, List[Dict[str, Any]]] = {}
        rep_anchor_t0 = time.time()

        for c_idx, cid in enumerate(ORDERED_CHAR_IDS):
            # Check stopping rule mid-replicate ONLY if >= MIN_COMPLETE_REPLICATES are already complete
            if len(manifest["completed_replicates"]) >= MIN_COMPLETE_REPLICATES and manifest["cumulative_generation_seconds"] >= BUDGET_SECONDS:
                log(f"[STOP_RULE] Hit {BUDGET_HOURS}h cap during anchors of {rep_key} with {len(manifest['completed_replicates'])} complete reps.")
                stopped_by_cap = True
                break

            c_bank_dir = banks_dir / cid
            c_bank_dir.mkdir(parents=True, exist_ok=True)
            ref_path = c_bank_dir / f"ref_00_{cid}_anchor.png"
            sidecar_path = anchors_dir / f"{cid}_selected_anchor.json"

            if ref_path.exists() and sidecar_path.exists() and (c_bank_dir / "metadata.json").exists():
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    selected_anchors[cid] = json.load(f)
                log(f"  [ANCHOR RESUME] {rep_key}/{cid} already certified (seed={selected_anchors[cid].get('seed')}).")
                continue

            cspec = all_characters[cid]
            t_c0 = time.time()
            log(f"  [ANCHOR] {rep_key} ({c_idx+1}/11) synthesizing {cid}...")
            cands = generate_character_candidates(
                char_id=cid,
                char_spec=cspec,
                replicate_idx=r_idx,
                char_idx=c_idx,
                is_test3=True,
                out_dir=anchors_dir / cid,
                auditor=auditor,
                forge_url=forge_url,
                max_candidates=8,
                min_candidates=4,
                on_render_cb=on_anchor_render,
            )
            t_c_elapsed = time.time() - t_c0
            manifest["anchor_generation_seconds"] += t_c_elapsed
            manifest["cumulative_generation_seconds"] += t_c_elapsed

            if not cands:
                raise RuntimeError(f"All anchor candidates failed for {rep_key}/{cid}!")

            all_char_candidates[cid] = cands
            best = select_best_anchor(cands)
            selected_anchors[cid] = best

            best_img = best.get("img_bgr")
            if best_img is not None:
                cv2.imwrite(str(ref_path), best_img)
            else:
                shutil.copyfile(best["image_path"], str(ref_path))

            bank_mgr = IdentityBankManager(bank_root=banks_dir)
            bank_mgr.initialize_character_bank(cid, [ref_path])

            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in best.items() if k not in ["img_bgr", "crop_bgr"]}, f, indent=2)

            log(
                f"  [ANCHOR SELECTED] {rep_key}/{cid}: seed={best['seed']}, "
                f"cands={len(cands)}, passed={best.get('passed_gate')}, "
                f"score={best.get('selection_score')}, elapsed={t_c_elapsed:.1f}s"
            )
            save_manifest(manifest_path, manifest)

        if stopped_by_cap:
            manifest["partial_replicates"].append(rep_key)
            archive_partial_replicate(r_idx, manifest, reason="cap_reached_during_anchors")
            save_manifest(manifest_path, manifest)
            break

        # --- Intra-replicate separation check (CPU only) ---
        sep_path = anchors_dir / "separation_check.json"
        if not sep_path.exists():
            log(f"  [SEPARATION] Running intra-replicate 4-judge separation check for {rep_key} (threshold={DEFAULT_SEPARATION_THRESHOLD})...")
            sep_res = check_intra_replicate_separation(selected_anchors, threshold=DEFAULT_SEPARATION_THRESHOLD)
            swaps_performed = []
            # If any within-story pair violates threshold, attempt deterministic swap to next certified candidate
            within_story_violations = []
            for v in sep_res["violations"]:
                c1, c2 = v["char_1"], v["char_2"]
                same_story = any((c1 in grp and c2 in grp) for grp in STORY_CHAR_GROUPS.values())
                if same_story:
                    within_story_violations.append(v)

            if within_story_violations:
                log(f"  [SEPARATION] {len(within_story_violations)} within-story violation(s) found; attempting candidate swap...")
                for v in within_story_violations:
                    c2 = v["char_2"]
                    c2_idx = ORDERED_CHAR_IDS.index(c2)
                    cands_dir = anchors_dir / c2
                    # Load or generate up to 8 candidates for c2
                    cands = generate_character_candidates(
                        char_id=c2,
                        char_spec=all_characters[c2],
                        replicate_idx=r_idx,
                        char_idx=c2_idx,
                        is_test3=True,
                        out_dir=cands_dir,
                        auditor=auditor,
                        forge_url=forge_url,
                        max_candidates=8,
                        min_candidates=8,
                        on_render_cb=on_anchor_render,
                    )
                    cert = sorted([c for c in cands if c.get("passed_gate")], key=lambda c: (-c["selection_score"], c["seed"]))
                    old_seed = selected_anchors[c2]["seed"]
                    for alt in cert:
                        if alt["seed"] == old_seed:
                            continue
                        trial_anchors = dict(selected_anchors)
                        trial_anchors[c2] = alt
                        trial_sep = check_intra_replicate_separation({v["char_1"]: selected_anchors[v["char_1"]], c2: alt}, threshold=DEFAULT_SEPARATION_THRESHOLD)
                        if trial_sep["all_passed"]:
                            selected_anchors[c2] = alt
                            ref_path = banks_dir / c2 / f"ref_00_{c2}_anchor.png"
                            shutil.copyfile(alt["image_path"], str(ref_path))
                            bank_mgr = IdentityBankManager(bank_root=banks_dir)
                            bank_mgr.initialize_character_bank(c2, [ref_path])
                            with open(anchors_dir / f"{c2}_selected_anchor.json", "w", encoding="utf-8") as f:
                                json.dump({k: v2 for k, v2 in alt.items() if k not in ["img_bgr", "crop_bgr"]}, f, indent=2)
                            swaps_performed.append({"char_id": c2, "old_seed": old_seed, "new_seed": alt["seed"], "new_sim": trial_sep["max_pairwise_similarity"]})
                            log(f"  [SEPARATION SWAP] Swapped {c2} from seed {old_seed} -> {alt['seed']} (sim={trial_sep['max_pairwise_similarity']})")
                            break
                # Recompute final separation matrix
                sep_res = check_intra_replicate_separation(selected_anchors, threshold=DEFAULT_SEPARATION_THRESHOLD)

            sep_res["swaps_performed"] = swaps_performed
            with open(sep_path, "w", encoding="utf-8") as f:
                json.dump(sep_res, f, indent=2)
            log(f"  [SEPARATION] Done: max_sim={sep_res['max_pairwise_similarity']}, violations={len(sep_res['violations'])}, swaps={len(swaps_performed)}")

        manifest["replicates"][rep_key] = {
            "replicate_idx": r_idx,
            "characters": ORDERED_CHAR_IDS,
            "banks_path": str(banks_dir),
            "anchors_path": str(anchors_dir),
            "scenes_path": str(scenes_dir),
            "uncertified_count": sum(1 for a in selected_anchors.values() if a.get("uncertified_anchor", False)),
            "status": "IN_PROGRESS",
        }
        save_manifest(manifest_path, manifest)

        # --- STEP 2: RENDER 12 SUPPORTED SCENES x 3 SEEDS = 36 v2_final RENDERS ---
        pipeline = ConsistencyPipelineV2(config=cfg, bank_root=banks_dir, base_url=forge_url)
        rep_renders_summary = []

        for sm in STORY_SPECS_META:
            if stopped_by_cap:
                break
            st_idx = sm["story_idx"]
            st_label = sm["label"]
            story = stories[st_idx]

            for sc_idx in sm["supported_scene_indices"]:
                if stopped_by_cap:
                    break
                scene = story.scenes[sc_idx]

                for s_idx in range(3):
                    # Check stopping rule mid-replicate ONLY if >= MIN_COMPLETE_REPLICATES are already complete
                    if len(manifest["completed_replicates"]) >= MIN_COMPLETE_REPLICATES and manifest["cumulative_generation_seconds"] >= BUDGET_SECONDS:
                        log(f"[STOP_RULE] Hit {BUDGET_HOURS}h cap during {rep_key} {st_label} sc{scene.scene_number} s{s_idx}.")
                        stopped_by_cap = True
                        break

                    seed = get_scene_seed(r_idx, st_idx, sc_idx, s_idx, arm="v2_final", is_test3=True)
                    render_id = f"{rep_key}_{st_label}_sc{scene.scene_number:02d}_s{s_idx}_seed_{seed}"
                    img_path = scenes_dir / f"{render_id}.png"
                    sidecar_path = scenes_dir / f"{render_id}.json"

                    # Resumability check: skip already completed (SUCCESS or exhausted FAILED)
                    if sidecar_path.exists():
                        try:
                            with open(sidecar_path, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                            if existing.get("status") == "SUCCESS" and img_path.exists():
                                log(f"  [RESUME] Skipping completed {render_id}")
                                rep_renders_summary.append(existing)
                                continue
                            if existing.get("status") == "FAILED" and existing.get("retry_count", 0) >= 1:
                                log(f"  [RESUME] Keeping recorded FAILED {render_id} (never re-run failed renders)")
                                rep_renders_summary.append(existing)
                                continue
                        except Exception:
                            pass

                    log(f"  [RENDER] {render_id} (elapsed gen: {manifest['cumulative_generation_seconds']/3600:.2f}h)...")
                    vram_start = get_current_gpu_vram_mb()
                    t0 = time.time()
                    retry_count = 0
                    img_cv = None
                    meta = {}
                    status = "FAILED"
                    error_log = []

                    # Rule: 1 initial attempt + max 1 infrastructure retry (same seed & config, logged)
                    for attempt in range(2):
                        try:
                            img_cv, meta = pipeline.generate_scene(story=story, scene=scene, seed=seed)
                            if img_cv is not None:
                                status = "SUCCESS"
                                break
                            else:
                                error_log.append(f"attempt_{attempt+1}: generate_scene returned None")
                        except Exception as e:
                            error_log.append(f"attempt_{attempt+1}: {type(e).__name__}: {e}")
                            log(f"  [INFRA_ERROR] {render_id} attempt {attempt+1} failed: {e}")

                        if attempt == 0:
                            retry_count = 1
                            maybe_restart_forge(force_reason=f"infra_retry_{render_id}")
                            # Re-instantiate pipeline client handle after Forge restart
                            pipeline = ConsistencyPipelineV2(config=cfg, bank_root=banks_dir, base_url=forge_url)

                    elapsed = time.time() - t0
                    vram_peak = max(vram_start, get_current_gpu_vram_mb())
                    renders_since_restart += 1
                    manifest["total_renders_executed"] += 1
                    manifest["scene_generation_seconds"] += elapsed
                    manifest["cumulative_generation_seconds"] += elapsed

                    img_sha = ""
                    if img_cv is not None and status == "SUCCESS":
                        cv2.imwrite(str(img_path), img_cv)
                        img_sha = compute_file_sha256(img_path)

                    sidecar = {
                        "render_id": render_id,
                        "replicate_idx": r_idx,
                        "story_idx": st_idx,
                        "story_label": st_label,
                        "story_id": story.story_id,
                        "scene_idx": sc_idx,
                        "scene_number": scene.scene_number,
                        "scene_id": scene.scene_id,
                        "seed_idx": s_idx,
                        "seed": seed,
                        "arm": "v2_final",
                        "is_stress_scene": False,
                        "config_hash": cfg.to_hash(),
                        "config_compute_hash": cfg.compute_hash(),
                        "git_commit": get_git_commit(),
                        "verifier_sha256": get_verifier_freeze_sha(),
                        "elapsed_seconds": round(elapsed, 2),
                        "peak_vram_mb": round(vram_peak, 1),
                        "image_sha256": img_sha,
                        "retry_count": retry_count,
                        "errors": error_log,
                        "status": status,
                        "expected_characters": scene.characters_present or list(scene.character_positions.keys()),
                        "character_positions": scene.character_positions,
                        "metadata": meta,
                    }
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar, f, indent=2)

                    manifest["renders"][render_id] = {
                        "status": status,
                        "replicate_idx": r_idx,
                        "story_label": st_label,
                        "scene_number": scene.scene_number,
                        "seed": seed,
                        "elapsed_seconds": round(elapsed, 2),
                        "retry_count": retry_count,
                        "image_sha256": img_sha,
                        "image_path": str(img_path) if status == "SUCCESS" else "",
                        "sidecar_path": str(sidecar_path),
                    }
                    rep_renders_summary.append(sidecar)
                    save_manifest(manifest_path, manifest)

                    log(f"  [DONE] {render_id}: {status} in {elapsed:.1f}s (retries={retry_count}, VRAM={vram_peak:.0f}MB)")
                    maybe_restart_forge()

        if stopped_by_cap:
            manifest["replicates"][rep_key]["status"] = "PARTIAL"
            manifest["partial_replicates"].append(rep_key)
            archive_partial_replicate(r_idx, manifest, reason="cap_reached_during_scenes")
            save_manifest(manifest_path, manifest)
            break

        # Mark replicate PRIMARY scenes COMPLETE
        manifest["replicates"][rep_key]["status"] = "COMPLETE"
        manifest["replicates"][rep_key]["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["replicates"][rep_key]["primary_renders_count"] = len(rep_renders_summary)
        manifest["replicates"][rep_key]["primary_success_count"] = sum(1 for x in rep_renders_summary if x["status"] == "SUCCESS")
        manifest["replicates"][rep_key]["primary_failed_count"] = sum(1 for x in rep_renders_summary if x["status"] != "SUCCESS")
        manifest["completed_replicates"].append(rep_key)

        # --- STEP 3: OPTIONAL 3-CHARACTER STRESS SCENE (Story B Scene 4) LAST FOR THIS REPLICATE ---
        # Only run if projected remaining primary replicates + this stress scene (~600s) fit cleanly under 8.0h cap
        remaining_primary_reps = TARGET_REPLICATES - len(manifest["completed_replicates"])
        avg_rep_time = manifest["cumulative_generation_seconds"] / max(len(manifest["completed_replicates"]), 1)
        projected_total_with_stress = manifest["cumulative_generation_seconds"] + (remaining_primary_reps * avg_rep_time) + 600.0
        stress_ran = []
        if projected_total_with_stress <= BUDGET_SECONDS:
            log(f"  [STRESS] Budget allows 3-char stress scene for {rep_key} (projected={projected_total_with_stress/3600:.2f}h <= {BUDGET_HOURS}h).")
            story_b = stories[1]
            stress_scene = story_b.scenes[3]
            for s_idx in range(3):
                if manifest["cumulative_generation_seconds"] >= BUDGET_SECONDS:
                    break
                seed = get_scene_seed(r_idx, 1, 3, s_idx, arm="v2_final", is_test3=True)
                render_id = f"{rep_key}_Story_B_STRESS_sc{stress_scene.scene_number:02d}_s{s_idx}_seed_{seed}"
                img_path = stress_dir / f"{render_id}.png"
                sidecar_path = stress_dir / f"{render_id}.json"
                if sidecar_path.exists() and img_path.exists():
                    continue
                log(f"  [STRESS RENDER] {render_id}...")
                vram_start = get_current_gpu_vram_mb()
                t0 = time.time()
                retry_count = 0
                img_cv = None
                meta = {}
                status = "FAILED"
                error_log = []
                for attempt in range(2):
                    try:
                        img_cv, meta = pipeline.generate_scene(story=story_b, scene=stress_scene, seed=seed)
                        if img_cv is not None:
                            status = "SUCCESS"
                            break
                    except Exception as e:
                        error_log.append(f"attempt_{attempt+1}: {e}")
                    if attempt == 0:
                        retry_count = 1
                        maybe_restart_forge(force_reason=f"infra_retry_{render_id}")
                        pipeline = ConsistencyPipelineV2(config=cfg, bank_root=banks_dir, base_url=forge_url)
                elapsed = time.time() - t0
                vram_peak = max(vram_start, get_current_gpu_vram_mb())
                renders_since_restart += 1
                manifest["total_renders_executed"] += 1
                manifest["stress_generation_seconds"] += elapsed
                manifest["cumulative_generation_seconds"] += elapsed
                img_sha = ""
                if img_cv is not None and status == "SUCCESS":
                    cv2.imwrite(str(img_path), img_cv)
                    img_sha = compute_file_sha256(img_path)
                sc_data = {
                    "render_id": render_id,
                    "replicate_idx": r_idx,
                    "story_idx": 1,
                    "story_label": "Story_B",
                    "story_id": story_b.story_id,
                    "scene_idx": 3,
                    "scene_number": stress_scene.scene_number,
                    "scene_id": stress_scene.scene_id,
                    "seed_idx": s_idx,
                    "seed": seed,
                    "arm": "v2_final",
                    "is_stress_scene": True,
                    "config_hash": cfg.to_hash(),
                    "elapsed_seconds": round(elapsed, 2),
                    "peak_vram_mb": round(vram_peak, 1),
                    "image_sha256": img_sha,
                    "retry_count": retry_count,
                    "errors": error_log,
                    "status": status,
                    "expected_characters": stress_scene.characters_present,
                    "character_positions": stress_scene.character_positions,
                    "metadata": meta,
                }
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(sc_data, f, indent=2)
                manifest["stress_renders"][render_id] = {
                    "status": status,
                    "seed": seed,
                    "elapsed_seconds": round(elapsed, 2),
                    "retry_count": retry_count,
                    "image_path": str(img_path) if status == "SUCCESS" else "",
                    "sidecar_path": str(sidecar_path),
                }
                stress_ran.append(render_id)
                save_manifest(manifest_path, manifest)
                maybe_restart_forge()
        else:
            log(f"  [STRESS SKIPPED] {rep_key}: projected {projected_total_with_stress/3600:.2f}h > {BUDGET_HOURS}h cap.")

        # Write per-replicate manifest
        rep_manifest = {
            "run_id": RUN_ID,
            "replicate_idx": r_idx,
            "status": "COMPLETE",
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "characters": ORDERED_CHAR_IDS,
            "banks_path": str(banks_dir),
            "anchors_path": str(anchors_dir),
            "scenes_path": str(scenes_dir),
            "primary_renders": len(rep_renders_summary),
            "primary_success": sum(1 for x in rep_renders_summary if x["status"] == "SUCCESS"),
            "primary_failed": sum(1 for x in rep_renders_summary if x["status"] != "SUCCESS"),
            "stress_renders": stress_ran,
        }
        with open(rep_dir / "replicate_manifest.json", "w", encoding="utf-8") as f:
            json.dump(rep_manifest, f, indent=2)

        save_manifest(manifest_path, manifest)
        log(f"[REPLICATE COMPLETE] {rep_key} finished! Total gen time: {manifest['cumulative_generation_seconds']/3600:.2f}h")

    manifest["status"] = "STOPPED_AT_CAP" if stopped_by_cap else "COMPLETE"
    manifest["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_manifest(manifest_path, manifest)

    log("=" * 72)
    log(f"STAGE B GENERATION FINISHED: status={manifest['status']}")
    log(f"Completed replicates ({len(manifest['completed_replicates'])}): {manifest['completed_replicates']}")
    log(f"Partial replicates ({len(manifest['partial_replicates'])}): {manifest['partial_replicates']}")
    log(f"Total Generation Time: {manifest['cumulative_generation_seconds']/3600:.3f}h "
        f"(Anchors: {manifest['anchor_generation_seconds']/3600:.3f}h, "
        f"Primary Scenes: {manifest['scene_generation_seconds']/3600:.3f}h, "
        f"Stress Scenes: {manifest['stress_generation_seconds']/3600:.3f}h)")
    log("=" * 72)


if __name__ == "__main__":
    main()
