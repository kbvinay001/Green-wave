import os, glob, json, time, cv2
from infer import AmbulanceDetector

MODEL = r"models\yolov11s-ambulance.pt"
CFG   = r"..\common\config.yaml"
IMDIR = r"data\test\images"
OUTDIR= r"..\..\outputs\vision_preds"  # saves under project-level outputs

os.makedirs(OUTDIR, exist_ok=True)

det = AmbulanceDetector(MODEL, config_path=CFG, device='cpu')  # change to 'cuda' if you have GPU

results = []
paths = sorted(glob.glob(os.path.join(IMDIR, "*.*")))
print(f"Found {len(paths)} test images")

for p in paths:
    img = cv2.imread(p)
    if img is None:
        print(f"Skipping unreadable: {p}")
        continue
    t = time.time()
    dets = det.detect(img, timestamp=t, track=False)
    vis  = det.visualize(img, dets, show_tracks=False, show_velocity=False)
    outp = os.path.join(OUTDIR, os.path.basename(p))
    cv2.imwrite(outp, vis)
    results.append({"image": p, "detections":[d.to_dict() for d in dets]})
    print(f"{os.path.basename(p)} -> {len(dets)} detections")

with open(os.path.join(OUTDIR, "detections.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved annotated images to:", OUTDIR)
print("Saved JSON to:", os.path.join(OUTDIR, "detections.json"))
