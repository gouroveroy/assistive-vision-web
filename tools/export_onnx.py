"""Export a trained checkpoint to the ONNX file the web app loads.

    python webapp/tools/export_onnx.py                      # results/deliverables/best.pt
    python webapp/tools/export_onnx.py path/to/best.pt

Needs `pip install ultralytics onnx onnxslim` (CPU is enough). Writes:

    webapp/model/hazard.onnx   float32, input [1, 3, H, W] with H and W free,
                               so one file serves 320, 480 and 640 px
    webapp/model/model.json    class names in output order + a version
                               string that changes whenever the weights do,
                               so phones fetch the new model instead of the
                               cached one

Run it again after every training run whose model should reach the app.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)
PROJECT = os.path.dirname(WEBAPP)

weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    PROJECT, "results", "deliverables", "best.pt")
if not os.path.isfile(weights):
    sys.exit(f"no checkpoint at {weights}")

from ultralytics import YOLO  # noqa: E402

model = YOLO(weights)
names = [model.names[i] for i in sorted(model.names)]
exported = model.export(format="onnx", imgsz=640, dynamic=True, simplify=True,
                        nms=False, device="cpu")

out_dir = os.path.join(WEBAPP, "model")
os.makedirs(out_dir, exist_ok=True)
dst = os.path.join(out_dir, "hazard.onnx")
shutil.move(exported, dst)

with open(dst, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()[:12]

manifest = {}
run_manifest = os.path.join(os.path.dirname(weights), "run_manifest.json")
if os.path.isfile(run_manifest):
    with open(run_manifest, encoding="utf-8") as f:
        manifest = json.load(f)

info = {
    "file": "hazard.onnx",
    "version": digest,
    "name": manifest.get("run_name") or os.path.basename(weights),
    "pipeline_version": manifest.get("pipeline_version"),
    "base_model": manifest.get("base_model"),
    "labels": names,
}
with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")

print(f"\nwrote {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB, version {digest})")
print(f"wrote {os.path.join(out_dir, 'model.json')}  ({len(names)} classes)")
