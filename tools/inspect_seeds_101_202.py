import json

with open('results/bundle_dev2/report_v1_2.json', 'r', encoding='utf-8') as f:
    rep = json.load(f)

scenes = rep.get('per_scene_reports', [])
target_scenes = [s for s in scenes if any(f'_seed_{sd}.png' in s.get('image_filename', '') for sd in [101, 202])]

print(f"Total target scene evaluations: {len(target_scenes)}")

wardrobe_passes = []
spatial_passes = []
biometrics = []
extra_yolo_persons = 0

for s in target_scenes:
    name = s.get('image_filename', '')
    aud = s.get('scene_audit', {})
    yolo_p = aud.get('detected_persons_yolo_count', 0)
    retina_f = aud.get('detected_faces_count', 0)
    chars = s.get('character_evaluations', [])
    exp_p = 2 if ('scene_1' in name or 'scene_5' in name) else 1
    extra = max(0, yolo_p - exp_p)
    extra_yolo_persons += extra
    print(f"{name:25s} | Expected: {exp_p} | YOLO: {yolo_p} | Faces: {retina_f} | Extra: {extra}")
    for c in chars:
        cid = c.get('character_id', '')
        sp = c.get('spatial_pass', False)
        wp = c.get('wardrobe_pass', False)
        ens = c.get('biometric_ensemble_score', 0.0)
        wardrobe_passes.append(bool(wp))
        spatial_passes.append(bool(sp))
        biometrics.append(ens)
        de = c.get("wardrobe_delta_e")
        de_str = f"{de:.2f}" if de is not None else "None"
        print(f"   -> {cid:15s} | Spatial: {str(sp):5s} | Wardrobe: {str(wp):5s} (DE={de_str}) | Bio: {ens:.4f}")

print("\n--- SUMMARY ACROSS SEEDS 101 & 202 (12 scenes, 16 character instances) ---")
if spatial_passes:
    print(f"Spatial Pass Rate:   {sum(spatial_passes)/len(spatial_passes)*100:.1f}% ({sum(spatial_passes)}/{len(spatial_passes)})")
if wardrobe_passes:
    print(f"Wardrobe Pass Rate:  {sum(wardrobe_passes)/len(wardrobe_passes)*100:.1f}% ({sum(wardrobe_passes)}/{len(wardrobe_passes)})")
if biometrics:
    print(f"Mean Biometric:      {sum(biometrics)/len(biometrics):.4f}")
print(f"Total Extra YOLO Persons: {extra_yolo_persons}")
