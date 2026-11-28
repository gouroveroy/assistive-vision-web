# Assistive Vision — web app

Opens the phone's rear camera in the browser, runs the hazard detector on
the device, and speaks short warnings: "Pothole ahead", "Vehicle on your
left, close". Works in Chrome on Android and Safari on iPhone from one
URL. No install, no build step, no server-side code; no image leaves the
phone.

This is the stand-in for a native app until an iOS build is possible (that
needs a Mac). The detection and alert logic are plain JavaScript modules
that a Flutter or native app can mirror later.

## Run it on this computer

```
python webapp/serve.py
```

Open http://localhost:8000 in Chrome or Edge and press **Start**. The
laptop's webcam stands in for the phone camera.

## Run it on a phone

Browsers only allow camera access over **HTTPS** (or on `localhost`), so
the folder has to be published. Any static host works. With GitHub Pages:

1. Make this folder its own repository and push it (public, on a free
   account):
   ```
   cd webapp
   git init -b main
   git add .
   git commit -m "Assistive Vision web app"
   gh repo create assistive-vision-web --public --source=. --push
   ```
2. On GitHub: **Settings → Pages → Deploy from a branch → `main` / root**.
3. Open `https://<your-user>.github.io/assistive-vision-web/` on the phone,
   allow the camera, press **Start**.
4. Optional: "Add to Home Screen" gives it an icon and a full-screen window.

The first visit downloads about 67 MB on a phone with WebGPU and about
53 MB without it or on iPhone (38 MB of that is the model). After that it
runs from the cache, offline.

### iPhone and iPad

Safari on iOS closes the whole page, with no error, when a tab uses too
much memory. ONNX Runtime Web's WebGPU build did exactly that on iPhone
while Android ran smoothly: it runs on a 29 MB "asyncify" WebAssembly
file, and Safari's optimising compiler can use gigabytes compiling such
files ([WebKit bug 304810](https://bugs.webkit.org/show_bug.cgi?id=304810),
still open on iOS 26.6). Each WebAssembly thread is a Web Worker with its
own instance, which adds to it.

So on iPhone and iPad (any browser there is Safari underneath) the app
uses the plain WebAssembly build on **one thread**, at 320 px in
Automatic, with a 960×540 camera stream. All the rules are in
[`js/device.js`](js/device.js).

For testing on the phone, the address can ask for more: `?gpu` tries
WebGPU, `?threads=2` two threads, both together work too. If a test
crashes, the next visit says so (below) and steps back.

**Crash notes.** A page killed by iOS cannot report anything, so the app
writes down what it is about to do (load the model, watch the camera)
and wipes the note when that ends normally or the page goes to the
background. If the note is still there on the next visit, the page died
there: the status line says where and with which set-up, and the app
steps down one level (no GPU, then one thread, then 320 px). Settings
shows how often it happened and has **Forget the crash limits**. This
works on every phone, not only iPhone.

## What it does with each frame

```
camera frame → letterbox to a square → model (ONNX Runtime Web)
            → boxes + NMS → alert engine → speech / vibration
```

- **Runtime.** WebGPU when the browser has it, WebAssembly otherwise (up
  to 4 threads). iPhone and iPad: WebAssembly on one thread (see above).
- **Detail.** "Automatic" uses 480 px on WebGPU and 320 px on WebAssembly.
  320 px is fast but loses small hazards: on held-out test frames the model
  missed a pothole and a pole at 320 px that it found at 480 px.
- **When it speaks.** A thing must be seen in 3 consecutive frames (2 when
  slow). The same alert is not repeated for 5 s unless the object has come
  close or changed side. Vehicles outrank ground hazards, which outrank
  obstacles. Only a close vehicle interrupts a sentence in progress.
- **What it says.** Car, bus and truck are all "vehicle". Benches, chairs,
  bins and similar are silent unless switched on in Settings. `left turn`
  and `right turn` (3.3 models only) are never spoken: 3.3 confused them.
- **Which model.** `model/` holds pipeline **3.5** (2026-11-28; version
  `02bf2a6e6d6e`), trained with SideGuide's walker-height frames. At the
  app's thresholds it found 4,024 more objects than 3.4 on the held-out
  frames for 22 more false alarms. The 3.3 and 3.5 models are kept in
  `archive/webapp_model_v3.3/` and `archive/webapp_model_v3.5/`.
- **"Close"** is a guess from the box position and size, assuming the phone
  is held upright facing forward. It is not a distance measurement.

All of this is tuned in [`js/config.js`](js/config.js) (per-class words,
priority, confidence threshold) and [`js/alerts.js`](js/alerts.js).

## Files

| | |
|---|---|
| `index.html`, `css/style.css` | the one screen |
| `js/app.js` | camera, main loop, settings, drawing |
| `js/detector.js` | loads the model, runs it on a frame |
| `js/yolo.js` | letterbox, output decoding, NMS (pure functions) |
| `js/alerts.js` | what to say and when (pure logic) |
| `js/config.js` | per-class behaviour and defaults |
| `js/device.js` | iPhone rules, crash notes and the limits they teach (pure logic) |
| `js/speech.js` | speech synthesis wrapper |
| `sw.js` | offline cache; adds the headers WebAssembly threads need |
| `model/hazard.onnx`, `model/model.json` | the model and its class list |
| `vendor/ort/` | ONNX Runtime Web 1.31.1, copied unmodified |
| `tools/export_onnx.py` | checkpoint → `model/` |
| `tests/` | unit tests and the browser test |

## Putting a newly trained model in

From pipeline 3.4 on, every Kaggle run writes the two files the app needs.
Copy them over the old ones:

```
deliverables/webapp_model/hazard.onnx   →  webapp/model/hazard.onnx
deliverables/webapp_model/model.json    →  webapp/model/model.json
```

For an older run, or any other checkpoint, make them locally instead:

```
pip install ultralytics onnx onnxslim
python webapp/tools/export_onnx.py results/deliverables/best.pt
```

Either way the class list comes from the model and is matched **by name**
in `js/config.js`, so a changed class set needs no code change; a class the
config does not know is announced by its own name at normal priority.
Publish the folder again and phones pick up the new model on their next
online visit. Before replacing a model, read Cell 7's verdict: if the new
run is not better than the one it started from, keep the old files.

## Tests

```
node --test webapp/tests/unit.test.mjs
```

29 checks on the decoding, alert and device logic; needs only Node.

The browser test drives installed Chrome with a video file as the camera.
It compares the app's detections with the original PyTorch checkpoint,
runs the camera-to-speech loop, and checks that the app still works on a
host without isolation headers and with the network switched off:

```
cd webapp && npm install --no-save playwright-core
python tests/make_reference.py ref.json photo1.jpg photo2.jpg
ffmpeg -loop 1 -i photo1.jpg -t 2 -r 15 -pix_fmt yuv420p camera.y4m
node tests/e2e.mjs --ref ref.json --video camera.y4m          # WebGPU if available
node tests/e2e.mjs --ref ref.json --video camera.y4m --cpu    # WebAssembly only
node tests/e2e.mjs --ref ref.json --video camera.y4m --ios    # the iPhone path
```

`--ios` cannot reproduce Safari's memory limit on a computer; it checks
that the iPhone set-up works and that a crash note leads to a step down.

## Measured so far

On the development laptop (Intel i5-1135G7, Chrome), model `yolo11s-hazard-v33`:

| Input | WebGPU | WebAssembly, 4 threads |
|---|---|---|
| 320 px | ~50 ms | ~180 ms |
| 480 px | ~85 ms | ~410 ms |
| 640 px | ~150 ms | ~880 ms |

On seven test images the app's boxes matched the PyTorch model's at
IoU ≥ 0.99.

The iPhone set-up (WebAssembly, one thread) on the same laptop with
`yolo11s-hazard-v35`: 222 ms at 320 px, 494 ms at 480 px, 867 ms at
640 px, against 140 / 291 / 489 ms on 4 threads. Live at 320 px: 3.8
frames a second. An iPhone 13's single core is roughly as fast as this
laptop's; Safari's WebAssembly speed on it is not measured yet.

**Not measured:** anything on a phone. Speed, heat, battery and how well
the camera view suits the model are all unknown until it runs on the
Android 16 and iPhone 13 targets.

## Known limits

- The page must stay open and in the foreground with the screen on. No
  browser lets a web page use the camera in the background.
- iPhone speaks only after a tap, which is why Start is a button.
- iPhone runs without the graphics processor and on one thread (see
  "iPhone and iPad"), so it is slower than a recent Android phone.
- WebAssembly is slow for this model size. A phone without WebGPU will
  manage a few frames per second at 320 px at best.
- It is a research prototype. It misses hazards and makes mistakes, and it
  does not replace a cane, a guide dog or the user's own judgement.
