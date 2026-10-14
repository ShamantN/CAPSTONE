import os, json, glob

def main():
    print("=== SCENE METADATA AUDIT ===")
    for p in sorted(glob.glob("edu_outputs/images/*.json")):
        if "manifest" in p or "report" in p:
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"File: {p}")
        print(f"  Story: {data.get('story_id')}")
        print(f"  Scene: {data.get('scene_number')} - {data.get('title')}")
        print(f"  Identity Engine: {data.get('identity_engine')}")
        print(f"  Resolution: {data.get('width')}x{data.get('height')}")
        print(f"  Steps: Pass1={data.get('steps_pass1')}, Pass2={data.get('steps_pass2')}, Denoise={data.get('denoising_strength')}")
        print(f"  Runtime: {data.get('runtime_s')}s, Peak VRAM: {data.get('peak_vram_mb')} MB")
        cond = data.get("native_conditioning", {})
        print(f"  Conditioning: OpenPose={cond.get('openpose_active')}, Depth={cond.get('depth_active')}, FreeU={cond.get('freeu_active')}, PAG={cond.get('pag_active')}, CFG-Fix={cond.get('cfg_fix_active')}, Regional={cond.get('regional_isolation', {}).get('regional_active')}")
        print()

    print("=== QA REPORTS SUMMARY ===")
    for r_path in ["experiments/benchmark_a_qa_report.json", "experiments/benchmark_b_qa_report.json"]:
        if os.path.exists(r_path):
            with open(r_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            print(f"Report: {r_path}")
            print(f"  Story: {rep.get('title')}")
            print(f"  Total Tests: {rep.get('total_tests')}, Passed: {rep.get('passed')}, Failed: {rep.get('failed')}, Pass Rate: {rep.get('pass_rate_pct')}%")
            fails = [t for t in rep.get("results", []) if not t["passed"]]
            print(f"  Failed Tests ({len(fails)}):")
            for ft in fails:
                print(f"    - [{ft['test_id']}] Scene {ft['scene']} ({ft['category']}): {ft['description']} | Val: {ft.get('measured_value')} vs Thresh: {ft.get('threshold')}")
            print()

if __name__ == "__main__":
    main()
