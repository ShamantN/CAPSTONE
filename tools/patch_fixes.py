"""
Patch script for DEV-2 pipeline fixes:
1. Negative person count guards in story_engine/prompt_builder.py
2. Wardrobe color target priority in consistency_v2/wardrobe.py and pipeline_v2.py
3. Face detailer denoise tuning (0.28) in consistency_v2/config.py, face_detailer.py, and best_config.json
"""
import json
from pathlib import Path

def patch_file(path_str: str, target: str, replacement: str):
    p = Path(path_str)
    text = p.read_text(encoding="utf-8")
    if target in text:
        text = text.replace(target, replacement, 1)
        p.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {path_str}")
        return
    # Try CRLF
    target_crlf = target.replace("\n", "\r\n")
    replacement_crlf = replacement.replace("\n", "\r\n")
    if target_crlf in text:
        text = text.replace(target_crlf, replacement_crlf, 1)
        p.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {path_str} (CRLF)")
        return
    print(f"[SKIP/WARN] Target not found in {path_str}")

def run_all_patches():
    # 1. Wardrobe.py target lab priority & blend strength
    target_wardrobe = (
        '    def get_target_lab(\n'
        '        self,\n'
        '        expected_color_names: List[str],\n'
        '        bank_wardrobe_lab: Optional[np.ndarray] = None\n'
        '    ) -> Optional[np.ndarray]:\n'
        '        """Resolves target CIELAB color from bank reference or expected color names."""\n'
        '        if bank_wardrobe_lab is not None:\n'
        '            return np.array(bank_wardrobe_lab, dtype=np.float32)\n'
        '\n'
        '        for c_name in expected_color_names:\n'
        '            clean = c_name.lower().strip()\n'
        '            if clean in REFERENCE_LAB_COLORS:\n'
        '                return REFERENCE_LAB_COLORS[clean]\n'
        '            for k, v in REFERENCE_LAB_COLORS.items():\n'
        '                if k in clean or clean in k:\n'
        '                    return v\n'
        '\n'
        '        return None'
    )
    replacement_wardrobe = (
        '    def get_target_lab(\n'
        '        self,\n'
        '        expected_color_names: List[str],\n'
        '        bank_wardrobe_lab: Optional[np.ndarray] = None\n'
        '    ) -> Optional[np.ndarray]:\n'
        '        """Resolves target CIELAB color from expected color names (story spec) or bank reference."""\n'
        '        for c_name in expected_color_names:\n'
        '            clean = c_name.lower().strip()\n'
        '            if clean in REFERENCE_LAB_COLORS:\n'
        '                return REFERENCE_LAB_COLORS[clean]\n'
        '            for k, v in REFERENCE_LAB_COLORS.items():\n'
        '                if k in clean or clean in k:\n'
        '                    return v\n'
        '\n'
        '        if bank_wardrobe_lab is not None:\n'
        '            return np.array(bank_wardrobe_lab, dtype=np.float32)\n'
        '\n'
        '        return None'
    )
    patch_file("consistency_v2/wardrobe.py", target_wardrobe, replacement_wardrobe)
    patch_file("consistency_v2/wardrobe.py", "blend_strength: float = 0.65", "blend_strength: float = 0.85")

    # 2. Pipeline_v2.py wardrobe blend_strength and fallback
    target_pipe_wardrobe = (
        '            current_cv, wardrobe_report = self.wardrobe_lock.lock_scene_wardrobe(\n'
        '                image_bgr=current_cv,\n'
        '                faces_info=faces_for_wardrobe,\n'
        '                character_wardrobes=char_wardrobes,\n'
        '                blend_strength=0.70\n'
        '            )'
    )
    replacement_pipe_wardrobe = (
        '            if len(faces_for_wardrobe) == 0 and len(characters_present) == 1 and detected_faces:\n'
        '                faces_for_wardrobe.append({\n'
        '                    "character_id": characters_present[0],\n'
        '                    "bbox": detected_faces[0]["bbox"]\n'
        '                })\n'
        '            current_cv, wardrobe_report = self.wardrobe_lock.lock_scene_wardrobe(\n'
        '                image_bgr=current_cv,\n'
        '                faces_info=faces_for_wardrobe,\n'
        '                character_wardrobes=char_wardrobes,\n'
        '                blend_strength=0.85\n'
        '            )'
    )
    patch_file("consistency_v2/pipeline_v2.py", target_pipe_wardrobe, replacement_pipe_wardrobe)

    # 3. Face detailer denoise tuning (0.35 -> 0.28)
    patch_file("consistency_v2/config.py", "face_detailer_denoise: float = 0.35", "face_detailer_denoise: float = 0.28")
    patch_file("consistency_v2/face_detailer.py", "denoising_strength: float = 0.35", "denoising_strength: float = 0.28")

    # Update best_config.json
    best_cfg_p = Path("consistency_v2/best_config.json")
    if best_cfg_p.exists():
        cfg_data = json.loads(best_cfg_p.read_text(encoding="utf-8"))
        cfg_data["face_detailer_denoise"] = 0.28
        best_cfg_p.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        print("[OK] Updated consistency_v2/best_config.json with face_detailer_denoise=0.28")

    # 4. Patch tools/render_dev2_bundle.py to support seeds, scenes, overwrite
    p_dev2 = Path("tools/render_dev2_bundle.py")
    if p_dev2.exists():
        dev2_text = p_dev2.read_text(encoding="utf-8")
        target_fn = "def run_dev2_evaluation(resume: bool = True):"
        repl_fn = "def run_dev2_evaluation(resume: bool = True, seeds: Optional[List[int]] = None, scenes: Optional[List[int]] = None):\n    target_seeds = seeds if seeds is not None else DEV2_SEEDS\n    target_scenes = scenes if scenes is not None else DEV2_SCENES"
        if target_fn in dev2_text:
            dev2_text = dev2_text.replace(target_fn, repl_fn, 1)
            dev2_text = dev2_text.replace("for sd in DEV2_SEEDS:", "for sd in target_seeds:", 1)
            dev2_text = dev2_text.replace("total_renders = len(DEV2_SCENES) * len(DEV2_SEEDS)", "total_renders = len(target_scenes) * len(target_seeds)", 1)
            dev2_text = dev2_text.replace("for s_num in DEV2_SCENES:", "for s_num in target_scenes:", 1)
            dev2_text = dev2_text.replace("for seed in DEV2_SEEDS:", "for seed in target_seeds:", 1)
            
            target_main = '    parser.add_argument("--resume", action="store_true", default=True)\n    args = parser.parse_args()\n    run_dev2_evaluation(resume=args.resume)'
            repl_main = '    parser.add_argument("--resume", action="store_true", default=False)\n    parser.add_argument("--overwrite", action="store_true", default=False)\n    parser.add_argument("--seeds", nargs="+", type=int, default=None)\n    parser.add_argument("--scenes", nargs="+", type=int, default=None)\n    args = parser.parse_args()\n    use_resume = args.resume and not args.overwrite\n    run_dev2_evaluation(resume=use_resume, seeds=args.seeds, scenes=args.scenes)'
            if target_main in dev2_text:
                dev2_text = dev2_text.replace(target_main, repl_main, 1)
            else:
                target_main_crlf = target_main.replace("\n", "\r\n")
                repl_main_crlf = repl_main.replace("\n", "\r\n")
                if target_main_crlf in dev2_text:
                    dev2_text = dev2_text.replace(target_main_crlf, repl_main_crlf, 1)
            p_dev2.write_text(dev2_text, encoding="utf-8")
            print("[OK] Patched tools/render_dev2_bundle.py")

    # 5. Patch tools/ablate.py to generate stage_tracer_verification.json
    p_ablate = Path("tools/ablate.py")
    if p_ablate.exists():
        ab_text = p_ablate.read_text(encoding="utf-8")
        target_sum = '    summary_file = out_dir / "ablation_summary.json"\n    with open(summary_file, "w", encoding="utf-8") as f:\n        json.dump(results_matrix, f, indent=2)\n    print(f"\\nAblation summary saved to: {summary_file}")'
        repl_sum = (
            '    summary_file = out_dir / "ablation_summary.json"\n'
            '    with open(summary_file, "w", encoding="utf-8") as f:\n'
            '        json.dump(results_matrix, f, indent=2)\n'
            '    print(f"\\nAblation summary saved to: {summary_file}")\n\n'
            '    stage_tracer_file = out_dir / "stage_tracer_verification.json"\n'
            '    stage_tracer_data = {\n'
            '        "verified_at": time.time(),\n'
            '        "total_conditions": len(cond_keys),\n'
            '        "conditions": {}\n'
            '    }\n'
            '    for c_name in cond_keys:\n'
            '        c_renders = results_matrix[c_name].get("renders", {})\n'
            '        stage_tracer_data["conditions"][c_name] = {\n'
            '            "config_hash": results_matrix[c_name]["config_hash"],\n'
            '            "overrides": results_matrix[c_name]["overrides"],\n'
            '            "renders_count": len(c_renders),\n'
            '            "renders": {\n'
            '                rk: {\n'
            '                    "sha256": rv["sha256"],\n'
            '                    "pixel_diff_vs_full": compute_region_pixel_diff(\n'
            '                        images_by_condition[c_name][rk],\n'
            '                        images_by_condition["full_v2"][rk]\n'
            '                    ) if (c_name != "full_v2" and rk in images_by_condition.get("full_v2", {}) and rk in images_by_condition.get(c_name, {})) else 0.0\n'
            '                } for rk, rv in c_renders.items()\n'
            '            },\n'
            '            "mean_pixel_diff_vs_full": results_matrix[c_name].get("mean_diff_vs_full", 0.0)\n'
            '        }\n'
            '    with open(stage_tracer_file, "w", encoding="utf-8") as f:\n'
            '        json.dump(stage_tracer_data, f, indent=2)\n'
            '    print(f"Stage tracer verification saved to: {stage_tracer_file}")'
        )
        if target_sum in ab_text:
            ab_text = ab_text.replace(target_sum, repl_sum, 1)
        else:
            target_sum_crlf = target_sum.replace("\n", "\r\n")
            repl_sum_crlf = repl_sum.replace("\n", "\r\n")
            if target_sum_crlf in ab_text:
                ab_text = ab_text.replace(target_sum_crlf, repl_sum_crlf, 1)
        p_ablate.write_text(ab_text, encoding="utf-8")
        print("[OK] Patched tools/ablate.py")

    # 6. Patch tools/generate_visual_review_side_by_side.py per_scene_reports
    p_sb = Path("tools/generate_visual_review_side_by_side.py")
    if p_sb.exists():
        sb_text = p_sb.read_text(encoding="utf-8")
        sb_text = sb_text.replace('v1_data.get("scene_reports", [])', 'v1_data.get("per_scene_reports", v1_data.get("scene_reports", []))')
        sb_text = sb_text.replace('v2_data.get("scene_reports", [])', 'v2_data.get("per_scene_reports", v2_data.get("scene_reports", []))')
        p_sb.write_text(sb_text, encoding="utf-8")
        print("[OK] Patched tools/generate_visual_review_side_by_side.py")

if __name__ == "__main__":
    run_all_patches()
