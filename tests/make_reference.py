"""Reference detections from the original PyTorch checkpoint.

    python webapp/tests/make_reference.py OUT.json IMAGE [IMAGE ...]
        [--weights results/deliverables/best.pt]

tests/e2e.mjs runs the same images through the web app and compares. This
is what catches a wrong channel order, a wrong letterbox or a shifted
class list - mistakes that still produce plausible-looking boxes.

Needs `pip install ultralytics`.
"""
import json
import os
import sys

args = sys.argv[1:]
weights = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "deliverables", "best.pt")
if "--weights" in args:
    i = args.index("--weights")
    weights = args[i + 1]
    del args[i:i + 2]
if len(args) < 2:
    sys.exit(__doc__)
out_path, images = args[0], args[1:]

from ultralytics import YOLO  # noqa: E402

model = YOLO(weights)
ref = {"weights": os.path.basename(weights), "imgsz": 640, "images": []}
for path in images:
    # rect=False: square letterbox, the same geometry the web app uses.
    r = model.predict(path, imgsz=640, conf=0.25, iou=0.5, rect=False,
                      device="cpu", verbose=False)[0]
    h, w = r.orig_shape
    dets = []
    for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                              r.boxes.conf.tolist()):
        dets.append({"name": r.names[int(cls)], "score": round(conf, 4),
                     "box": [round(box[0] / w, 5), round(box[1] / h, 5),
                             round(box[2] / w, 5), round(box[3] / h, 5)]})
    ref["images"].append({"path": os.path.abspath(path), "width": w,
                          "height": h, "detections": dets})
    print(f"{os.path.basename(path):<28} "
          + ", ".join(f"{d['name']} {d['score']:.2f}" for d in dets))

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ref, f, indent=1)
print(f"wrote {out_path}")
