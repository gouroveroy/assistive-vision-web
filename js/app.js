// Page controller: camera -> detector -> alert engine -> speech.

import { Detector } from './detector.js';
import { AlertEngine } from './alerts.js';
import { Speaker } from './speech.js';
import { classTable, DEFAULT_SETTINGS } from './config.js';

const $ = (id) => document.getElementById(id);
const el = {
  video: $('cam'), overlay: $('overlay'), stage: $('stage'),
  status: $('status'), said: $('said'), toggle: $('toggle'), stats: $('stats'),
  size: $('set-size'), sens: $('set-sens'), sensOut: $('set-sens-out'),
  rate: $('set-rate'), rateOut: $('set-rate-out'), boxes: $('set-boxes'),
  furniture: $('set-furniture'), vibrate: $('set-vibrate'), gpu: $('set-gpu'),
  modelInfo: $('model-info'),
};
const params = new URLSearchParams(location.search);

// ── settings ────────────────────────────────────────────────────────────
const SETTINGS_KEY = 'assistive-vision-settings-v1';
function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') || {};
  } catch { /* private mode or blocked storage: defaults are fine */ }
  const s = { ...DEFAULT_SETTINGS, ...saved };
  if (![0, 320, 480, 640].includes(s.inputSize)) s.inputSize = DEFAULT_SETTINGS.inputSize;
  return s;
}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch { /* nothing to do */ }
}
const settings = loadSettings();

// ── state ───────────────────────────────────────────────────────────────
const speaker = new Speaker();
const detector = new Detector();
let model = null;        // model/model.json
let classes = [];        // classTable(labels)
let thresholds = [];
let engine = null;
let running = false;
let starting = false;
let stream = null;
let wakeLock = null;
let lastDetections = [];
let lastSeenAnything = 0;
let fps = 0;
let lastTickEnd = 0;
let frames = 0;
const stat = { backend: '-', threads: 1, inferMs: 0, totalMs: 0, fps: 0, size: 0 };

// For debugging in the console and for the automated browser test.
window.__av = {
  get running() { return running; },
  get detections() { return lastDetections; },
  get spoken() { return speaker.log.slice(); },
  get stats() { return { ...stat, frames }; },
  get classes() { return classes; },
  get isolated() { return !!window.crossOriginIsolated; },
  settings,
  /** Run the model once on a still image: __av.detectImage('photo.jpg', 640) */
  async detectImage(url, size) {
    await ensureDetector();
    size = size || inputSize();
    const img = new Image();
    img.src = url;
    await img.decode();
    const r = await detector.detect(img, img.naturalWidth, img.naturalHeight, size, thresholds);
    stat.backend = detector.backend;
    return {
      backend: detector.backend,
      threads: detector.threads,
      inferMs: r.inferMs,
      detections: r.detections.map((d) => ({ ...d, name: classes[d.cls].name })),
    };
  },
};

/** The input size in use: the chosen one, or the automatic pick. */
function inputSize() {
  return settings.inputSize || (detector.backend === 'webgpu' ? 480 : 320);
}

function setStatus(text) {
  el.status.textContent = text;
}

function recomputeThresholds() {
  thresholds = classes.map((c) => Math.min(0.9, Math.max(0.15, c.conf + settings.sensitivity)));
}

// ── service worker: offline cache + cross-origin isolation ──────────────
async function setupServiceWorker() {
  if (!('serviceWorker' in navigator) || params.has('nosw')) return;
  try {
    await navigator.serviceWorker.register('sw.js');
    // The very first page load arrives before the worker can add the
    // isolation headers that WebAssembly threads need. Reload once, as
    // soon as the worker controls the page, and never loop.
    let reloaded = false;
    try {
      reloaded = !!sessionStorage.getItem('av-isolation-reload');
    } catch { reloaded = true; }
    if (!window.crossOriginIsolated && !reloaded) {
      await navigator.serviceWorker.ready;
      if (!navigator.serviceWorker.controller) {
        await new Promise((resolve) => {
          navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true });
          setTimeout(resolve, 3000);
        });
      }
      if (navigator.serviceWorker.controller && !running && !starting) {
        try {
          sessionStorage.setItem('av-isolation-reload', '1');
        } catch { return; }
        location.reload();
      }
    }
  } catch (err) {
    console.warn('service worker unavailable', err);
  }
}

// ── model ───────────────────────────────────────────────────────────────
async function loadModelInfo() {
  const res = await fetch('model/model.json', { cache: 'no-cache' });
  if (!res.ok) throw new Error('model/model.json is missing');
  model = await res.json();
  classes = classTable(model.labels);
  recomputeThresholds();
  engine = new AlertEngine(classes);
  el.modelInfo.textContent =
    `Model ${model.name || model.file} · ${model.labels.length} classes · version ${model.version}`;
}

async function ensureDetector() {
  if (detector.session) return;
  setStatus('Downloading the detection model…');
  const url = `model/${model.file}?v=${encodeURIComponent(model.version)}`;
  await detector.load(url, {
    preferGpu: settings.preferGpu && !params.has('cpu'),
    onProgress: (f) => setStatus(`Downloading the detection model… ${Math.round(f * 100)}%`),
  });
  stat.backend = detector.backend;
  stat.threads = detector.threads;
}

// ── camera ──────────────────────────────────────────────────────────────
async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('This browser cannot open the camera. The page must be served over HTTPS.');
  }
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  el.video.srcObject = stream;
  await el.video.play();
  if (!el.video.videoWidth) {
    await new Promise((resolve) => el.video.addEventListener('loadedmetadata', resolve, { once: true }));
  }
  el.stage.style.setProperty('--ar', String(el.video.videoWidth / el.video.videoHeight));
}

function closeCamera() {
  if (stream) for (const t of stream.getTracks()) t.stop();
  stream = null;
  el.video.srcObject = null;
}

async function holdWakeLock() {
  try {
    if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
  } catch { /* not allowed right now; the screen may dim */ }
}

// ── drawing ─────────────────────────────────────────────────────────────
const COLORS = { 1: '#ff5252', 2: '#ffb300', 3: '#40c4ff', 4: '#b0bec5' };

function draw(dets) {
  const c = el.overlay;
  const w = c.clientWidth;
  const h = c.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!settings.showBoxes) return;
  ctx.lineWidth = 3;
  ctx.font = '600 14px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  for (const d of dets) {
    const info = classes[d.cls];
    const color = COLORS[info ? info.priority : 3];
    const x = d.x1 * w;
    const y = d.y1 * h;
    const bw = (d.x2 - d.x1) * w;
    const bh = (d.y2 - d.y1) * h;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, bw, bh);
    const label = `${info ? info.name : d.cls} ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(label).width + 8;
    const ty = y > 20 ? y - 20 : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 4, ty + 3);
  }
}

// ── main loop ───────────────────────────────────────────────────────────
function nextFrame(delayMs) {
  return new Promise((resolve) => {
    const go = () => {
      if (typeof el.video.requestVideoFrameCallback === 'function' && !document.hidden) {
        let done = false;
        const finish = () => {
          if (!done) {
            done = true;
            resolve();
          }
        };
        el.video.requestVideoFrameCallback(finish);
        setTimeout(finish, 250); // never wait on a stalled video
      } else {
        setTimeout(resolve, 30);
      }
    };
    if (delayMs > 0) setTimeout(go, delayMs);
    else go();
  });
}

async function loop() {
  let gpuFailures = 0;
  while (running) {
    const v = el.video;
    if (v.readyState < 2 || !v.videoWidth) {
      await nextFrame(50);
      continue;
    }
    const t0 = performance.now();
    let result;
    try {
      result = await detector.detect(v, v.videoWidth, v.videoHeight, inputSize(), thresholds);
    } catch (err) {
      console.error('inference failed', err);
      if (detector.backend === 'webgpu' && gpuFailures++ < 1 && (await detector.fallbackToWasm())) {
        stat.backend = detector.backend;
        continue;
      }
      stop(`Detection stopped: ${err.message || err}`);
      return;
    }
    if (!running) return;
    const now = performance.now();
    lastDetections = result.detections;
    frames++;
    const dt = now - (lastTickEnd || t0);
    lastTickEnd = now;
    if (dt > 0) fps = fps ? fps * 0.8 + (1000 / dt) * 0.2 : 1000 / dt;
    stat.inferMs = result.inferMs;
    stat.totalMs = now - t0;
    stat.fps = fps;
    stat.size = inputSize();

    const { speak } = engine.update(result.detections, now, {
      fps,
      speechBusy: speaker.busy,
      announceFurniture: settings.announceFurniture,
    });
    if (speak) {
      speaker.speak(speak.text, { interrupt: speak.interrupt });
      el.said.textContent = speak.text;
      if (settings.vibrate && speak.priority <= 2 && navigator.vibrate) {
        try {
          navigator.vibrate(speak.close ? [120, 60, 120] : 120);
        } catch { /* no haptics here */ }
      }
    }
    draw(result.detections);
    if (frames % 5 === 0) {
      el.stats.textContent =
        `${stat.backend}${stat.backend === 'wasm' ? ` ×${stat.threads}` : ''} · ` +
        `${stat.size}px · ${stat.inferMs.toFixed(0)} ms · ${fps.toFixed(1)} fps`;
    }

    // Save battery: with nothing in view for a while, look less often.
    if (result.detections.length) lastSeenAnything = now;
    await nextFrame(now - lastSeenAnything > 3000 ? 150 : 0);
  }
}

async function start() {
  if (running || starting) return;
  starting = true;
  el.toggle.disabled = true;
  // Must happen inside the tap, or iOS will never speak.
  speaker.rate = settings.speechRate;
  speaker.unlock('Starting');
  try {
    setStatus('Opening the camera…');
    await openCamera();
    await ensureDetector();
    await holdWakeLock();
    running = true;
    engine.reset();
    lastSeenAnything = performance.now();
    lastTickEnd = 0;
    fps = 0;
    el.toggle.textContent = 'Stop';
    el.toggle.setAttribute('aria-pressed', 'true');
    document.body.classList.add('running');
    setStatus('Watching. Hold the phone upright, camera facing forward.');
    speaker.speak('Assistive vision is on');
    loop();
  } catch (err) {
    console.error(err);
    closeCamera();
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    const msg = denied
      ? 'Camera permission was refused. Allow camera access for this site and try again.'
      : `Could not start: ${err.message || err}`;
    setStatus(msg);
    speaker.speak(denied ? 'Camera permission is needed' : 'Could not start');
  } finally {
    starting = false;
    el.toggle.disabled = false;
  }
}

function stop(message) {
  const was = running;
  running = false;
  closeCamera();
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
  lastDetections = [];
  draw([]);
  el.toggle.textContent = 'Start';
  el.toggle.setAttribute('aria-pressed', 'false');
  document.body.classList.remove('running');
  setStatus(message || 'Stopped. Tap Start to begin.');
  if (was) {
    speaker.stop();
    speaker.speak(message ? 'Detection stopped' : 'Assistive vision is off');
  }
}

// ── wiring ──────────────────────────────────────────────────────────────
function bindSettings() {
  el.size.value = String(settings.inputSize);
  el.sens.value = String(-settings.sensitivity);
  el.rate.value = String(settings.speechRate);
  el.boxes.checked = settings.showBoxes;
  el.furniture.checked = settings.announceFurniture;
  el.vibrate.checked = settings.vibrate;
  el.gpu.checked = settings.preferGpu;
  const show = () => {
    const s = -settings.sensitivity;
    el.sensOut.textContent = s === 0 ? 'normal' : s > 0 ? `more alerts (+${s.toFixed(2)})` : `fewer alerts (${s.toFixed(2)})`;
    el.rateOut.textContent = `${settings.speechRate.toFixed(1)}×`;
  };
  show();
  const on = (node, fn) =>
    node.addEventListener('input', () => {
      fn();
      show();
      saveSettings();
    });
  on(el.size, () => { settings.inputSize = Number(el.size.value); });
  on(el.sens, () => {
    settings.sensitivity = -Number(el.sens.value);
    recomputeThresholds();
  });
  on(el.rate, () => {
    settings.speechRate = Number(el.rate.value);
    speaker.rate = settings.speechRate;
  });
  on(el.boxes, () => { settings.showBoxes = el.boxes.checked; });
  on(el.furniture, () => { settings.announceFurniture = el.furniture.checked; });
  on(el.vibrate, () => { settings.vibrate = el.vibrate.checked; });
  on(el.gpu, () => { settings.preferGpu = el.gpu.checked; });
}

el.toggle.addEventListener('click', () => (running ? stop() : start()));

document.addEventListener('visibilitychange', () => {
  // The camera is taken away when the page is hidden; pick the lock back up.
  if (!document.hidden && running) holdWakeLock();
});

(async function init() {
  bindSettings();
  setupServiceWorker();
  try {
    await loadModelInfo();
    setStatus('Ready. Tap Start, then hold the phone upright with the camera facing forward.');
    el.toggle.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`Cannot load the model description: ${err.message || err}`);
  }
})();
