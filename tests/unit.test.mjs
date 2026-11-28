// Offline checks for the parts that decide what gets drawn and said, and
// how the detector is set up on each device.
//     node --test webapp/tests/unit.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { letterboxParams, rgbaToCHW, decode, nms, iou, toSourceCoords, postprocess } from '../js/yolo.js';
import { AlertEngine, zoneOf, isClose, phrase } from '../js/alerts.js';
import { classTable, CLASS_INFO } from '../js/config.js';
import { isAppleMobile, runtimePlan, CrashGuard, describeCrash } from '../js/device.js';

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

// ── yolo.js ─────────────────────────────────────────────────────────────
test('letterbox: landscape frame is padded top and bottom', () => {
  const lb = letterboxParams(1280, 720, 320);
  assert.equal(lb.newW, 320);
  assert.equal(lb.newH, 180);
  assert.equal(lb.left, 0);
  assert.equal(lb.top, 70);
});

test('letterbox: portrait frame is padded left and right', () => {
  const lb = letterboxParams(720, 1280, 640);
  assert.equal(lb.newW, 360);
  assert.equal(lb.newH, 640);
  assert.equal(lb.left, 140);
  assert.equal(lb.top, 0);
});

test('rgbaToCHW: planar RGB in 0..1, alpha dropped', () => {
  const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 51, 102, 204, 255]);
  const chw = rgbaToCHW(rgba, 2);
  assert.deepEqual(Array.from(chw.slice(0, 4)), [1, 0, 0, 0.2].map(Math.fround));
  assert.deepEqual(Array.from(chw.slice(4, 8)), [0, 1, 0, 0.4].map(Math.fround));
  assert.deepEqual(Array.from(chw.slice(8, 12)), [0, 0, 1, 0.8].map(Math.fround));
});

/** Build a [1, 4+nc, n] output from a list of {cx, cy, w, h, scores[]}. */
function fakeOutput(rows, nc) {
  const n = rows.length;
  const data = new Float32Array((4 + nc) * n);
  rows.forEach((r, i) => {
    data[i] = r.cx;
    data[n + i] = r.cy;
    data[2 * n + i] = r.w;
    data[3 * n + i] = r.h;
    r.scores.forEach((s, c) => { data[(4 + c) * n + i] = s; });
  });
  return { data, dims: [1, 4 + nc, n] };
}

test('decode: channel-major layout, best class per anchor, per-class thresholds', () => {
  const { data, dims } = fakeOutput([
    { cx: 100, cy: 100, w: 40, h: 20, scores: [0.1, 0.9, 0.2] },
    { cx: 200, cy: 150, w: 10, h: 10, scores: [0.45, 0.1, 0.0] },
    { cx: 50, cy: 50, w: 10, h: 10, scores: [0.0, 0.0, 0.3] },
  ], 3);
  const got = decode(data, dims, [0.4, 0.5, 0.5]);
  assert.equal(got.length, 2);
  assert.equal(got[0].cls, 1);
  assert.ok(near(got[0].score, 0.9, 1e-6));
  assert.deepEqual([got[0].x1, got[0].y1, got[0].x2, got[0].y2], [80, 90, 120, 110]);
  assert.equal(got[1].cls, 0);
  // A single number applies to every class.
  assert.equal(decode(data, dims, 0.25).length, 3);
});

test('decode: rejects an output that is not [1, 4+nc, n]', () => {
  assert.throws(() => decode(new Float32Array(10), [1, 4, 2], 0.5));
});

test('nms: overlapping boxes of one class collapse, other classes survive', () => {
  const boxes = [
    { cls: 0, score: 0.9, x1: 0, y1: 0, x2: 100, y2: 100 },
    { cls: 0, score: 0.8, x1: 5, y1: 5, x2: 105, y2: 105 },
    { cls: 1, score: 0.7, x1: 5, y1: 5, x2: 105, y2: 105 },
    { cls: 0, score: 0.6, x1: 300, y1: 300, x2: 340, y2: 340 },
  ];
  const kept = nms(boxes, 0.5);
  assert.deepEqual(kept.map((b) => b.score), [0.9, 0.7, 0.6]);
  assert.ok(near(iou(boxes[0], boxes[3]), 0));
  assert.ok(near(iou(boxes[0], boxes[0]), 1));
});

test('toSourceCoords: undoes the letterbox and clamps to the frame', () => {
  const lb = letterboxParams(1280, 720, 320); // top pad 70, content 320x180
  const d = toSourceCoords({ cls: 2, score: 0.5, x1: 80, y1: 70 + 45, x2: 240, y2: 70 + 135 }, lb);
  assert.ok(near(d.x1, 0.25) && near(d.x2, 0.75) && near(d.y1, 0.25) && near(d.y2, 0.75));
  const edge = toSourceCoords({ cls: 0, score: 0.5, x1: -10, y1: 0, x2: 400, y2: 400 }, lb);
  assert.deepEqual([edge.x1, edge.y1, edge.x2, edge.y2], [0, 0, 1, 1]);
});

test('postprocess: end to end on a synthetic output', () => {
  const lb = letterboxParams(640, 640, 320);
  const { data, dims } = fakeOutput([
    { cx: 160, cy: 160, w: 160, h: 160, scores: [0.9, 0.0] },
    { cx: 162, cy: 160, w: 160, h: 160, scores: [0.7, 0.0] },
    { cx: 40, cy: 40, w: 20, h: 20, scores: [0.0, 0.2] },
  ], 2);
  const dets = postprocess(data, dims, lb, [0.4, 0.4]);
  assert.equal(dets.length, 1);
  assert.ok(near(dets[0].x1, 0.25) && near(dets[0].y2, 0.75));
});

// ── config.js ───────────────────────────────────────────────────────────
const LABELS = ['person', 'car', 'bus', 'pothole', 'bench', 'left turn', 'something new'];
const classes = classTable(LABELS);
const id = (name) => LABELS.indexOf(name);

test('classTable: by name, with a safe default for unknown classes', () => {
  assert.equal(classes[id('car')].say, 'vehicle');
  assert.equal(classes[id('bus')].say, 'vehicle');
  assert.equal(classes[id('pothole')].kind, 'ground');
  assert.equal(classes[id('left turn')].announce, false);
  assert.equal(classes[id('something new')].announce, true);
  assert.equal(classes[id('something new')].priority, 3);
  for (const [name, info] of Object.entries(CLASS_INFO)) {
    assert.ok(info.priority >= 1 && info.priority <= 4, name);
    assert.ok(info.conf > 0.2 && info.conf < 0.9, name);
  }
});

// ── alerts.js ───────────────────────────────────────────────────────────
const det = (name, x1, y1, x2, y2, score = 0.8) => ({ cls: id(name), score, x1, y1, x2, y2 });

test('zones and closeness', () => {
  assert.equal(zoneOf({ x1: 0.0, x2: 0.3 }), 'left');
  assert.equal(zoneOf({ x1: 0.4, x2: 0.6 }), 'ahead');
  assert.equal(zoneOf({ x1: 0.7, x2: 1.0 }), 'right');
  assert.equal(isClose({ y1: 0.6, y2: 0.9 }, 'ground'), true);
  assert.equal(isClose({ y1: 0.4, y2: 0.6 }, 'ground'), false);
  assert.equal(isClose({ y1: 0.2, y2: 0.85 }, 'upright'), true);
  assert.equal(isClose({ y1: 0.4, y2: 0.6 }, 'upright'), false);
  assert.equal(phrase('pothole', 'ahead', true), 'Pothole ahead, close');
  assert.equal(phrase('vehicle', 'left', false), 'Vehicle on your left');
});

/** Feed the same detections for several frames at 10 fps; collect speech. */
function run(engine, frames, t0 = 1000, opts = {}) {
  const said = [];
  let t = t0;
  for (const dets of frames) {
    const { speak } = engine.update(dets, t, { fps: 10, ...opts });
    if (speak) said.push({ t, ...speak });
    t += 100;
  }
  return said;
}

test('nothing is said until a thing is seen in three consecutive frames', () => {
  const e = new AlertEngine(classes);
  const p = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  const said = run(e, [p, p, p, p]);
  assert.equal(said.length, 1);
  assert.equal(said[0].t, 1200);
  assert.equal(said[0].text, 'Pothole ahead');
});

test('a single-frame false positive is never spoken', () => {
  const e = new AlertEngine(classes);
  const p = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  const said = run(e, [p, [], [], [], [], [], [], [], [], [], [], p, [], [], []]);
  assert.equal(said.length, 0);
});

test('at a low frame rate two frames are enough', () => {
  const e = new AlertEngine(classes);
  const p = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  assert.equal(e.update(p, 0, { fps: 3 }).speak, null);
  assert.equal(e.update(p, 330, { fps: 3 }).speak.text, 'Pothole ahead');
});

test('the same alert is not repeated inside the cooldown', () => {
  const e = new AlertEngine(classes);
  const p = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  const said = run(e, Array(60).fill(p)); // 6 seconds
  assert.deepEqual(said.map((s) => s.t), [1200, 6200]);
});

test('coming close is announced again without waiting for the cooldown', () => {
  const e = new AlertEngine(classes);
  const far = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  const close = [det('pothole', 0.4, 0.7, 0.6, 0.95)];
  const said = run(e, [...Array(10).fill(far), ...Array(20).fill(close)]);
  assert.deepEqual(said.map((s) => s.text), ['Pothole ahead', 'Pothole ahead, close']);
  assert.ok(said[1].t - said[0].t >= 1500);
});

test('moving to another direction is announced after a short wait', () => {
  const e = new AlertEngine(classes);
  const mid = [det('person', 0.4, 0.3, 0.6, 0.7)];
  const left = [det('person', 0.0, 0.3, 0.2, 0.7)];
  const said = run(e, [...Array(5).fill(mid), ...Array(40).fill(left)]);
  assert.deepEqual(said.map((s) => s.text), ['Person ahead', 'Person on your left']);
  assert.ok(said[1].t - said[0].t >= 2500);
});

test('car, bus and truck are one spoken thing; several become plural', () => {
  const e = new AlertEngine(classes);
  const f = [det('car', 0.0, 0.4, 0.2, 0.6), det('bus', 0.05, 0.3, 0.3, 0.7)];
  const said = run(e, [f, f, f]);
  assert.equal(said.length, 1);
  assert.equal(said[0].text, 'Vehicles on your left');
});

test('the most urgent thing is spoken first; vehicles beat potholes beat people', () => {
  const e = new AlertEngine(classes);
  const f = [det('person', 0.4, 0.3, 0.6, 0.7), det('pothole', 0.7, 0.5, 0.9, 0.7), det('car', 0.0, 0.4, 0.2, 0.6)];
  const said = run(e, Array(5).fill(f));
  assert.deepEqual(said.map((s) => s.word), ['vehicle', 'pothole', 'person']);
});

test('while speaking, only a close vehicle interrupts', () => {
  const e = new AlertEngine(classes);
  const pothole = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  assert.equal(run(e, Array(5).fill(pothole), 1000, { speechBusy: true }).length, 0);
  const farCar = [det('car', 0.4, 0.4, 0.6, 0.6)];
  assert.equal(run(e, Array(5).fill(farCar), 2000, { speechBusy: true }).length, 0);
  const closeCar = [det('car', 0.2, 0.2, 0.8, 0.9)];
  const said = run(e, Array(5).fill(closeCar), 3000, { speechBusy: true });
  assert.equal(said.length, 1);
  assert.equal(said[0].interrupt, true);
  assert.equal(said[0].text, 'Vehicle ahead, close');
});

test('street furniture is silent unless switched on; turn classes are always silent', () => {
  const f = [det('bench', 0.4, 0.4, 0.6, 0.6), det('left turn', 0.1, 0.1, 0.9, 0.9)];
  assert.equal(run(new AlertEngine(classes), Array(5).fill(f)).length, 0);
  const said = run(new AlertEngine(classes), Array(5).fill(f), 1000, { announceFurniture: true });
  assert.deepEqual(said.map((s) => s.text), ['Bench ahead']);
});

test('a detection that flickers off and on does not re-announce itself', () => {
  const e = new AlertEngine(classes);
  const p = [det('pothole', 0.4, 0.5, 0.6, 0.7)];
  const frames = [];
  for (let i = 0; i < 45; i++) frames.push(i % 4 === 3 ? [] : p); // one dropped frame in four
  const said = run(e, frames);
  assert.equal(said.length, 1);
});

// ── device.js ───────────────────────────────────────────────────────────
const IPHONE = { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1', maxTouchPoints: 5 };
const IPAD = { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15', maxTouchPoints: 5 };
const MAC = { userAgent: IPAD.userAgent, maxTouchPoints: 0 };
const ANDROID = { userAgent: 'Mozilla/5.0 (Linux; Android 16; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36', maxTouchPoints: 5 };

/** localStorage stand-in; `broken` throws like Safari's private mode once did. */
function memoryStorage(broken = false) {
  const m = new Map();
  const guard = () => { if (broken) throw new Error('QuotaExceededError'); };
  return {
    getItem: (k) => { guard(); return m.has(k) ? m.get(k) : null; },
    setItem: (k, v) => { guard(); m.set(k, String(v)); },
    removeItem: (k) => { guard(); m.delete(k); },
  };
}

test('iPhone and iPad (which claims to be a Mac) are recognised; Mac and Android are not', () => {
  assert.equal(isAppleMobile(IPHONE), true);
  assert.equal(isAppleMobile(IPAD), true);
  assert.equal(isAppleMobile(MAC), false);
  assert.equal(isAppleMobile(ANDROID), false);
  assert.equal(isAppleMobile(undefined), false);
});

test('runtime plan: iPhone never gets WebGPU or extra threads unless the address asks', () => {
  const base = { preferGpu: true, isolated: true, cores: 6 };
  assert.deepEqual(runtimePlan({ ...base, apple: false }), { gpu: true, threads: 4, maxSize: 0 });
  assert.deepEqual(runtimePlan({ ...base, apple: true }), { gpu: false, threads: 1, maxSize: 0 });
  const addr = new URLSearchParams('gpu&threads=2');
  assert.deepEqual(runtimePlan({ ...base, apple: true, params: addr }), { gpu: true, threads: 2, maxSize: 0 });
  // ?cpu beats ?gpu; threads need isolation whatever the address says.
  assert.equal(runtimePlan({ ...base, params: new URLSearchParams('cpu&gpu') }).gpu, false);
  assert.equal(runtimePlan({ ...base, isolated: false, params: new URLSearchParams('threads=4') }).threads, 1);
  assert.equal(runtimePlan({ ...base, preferGpu: false }).gpu, false);
});

test('runtime plan: limits learned from a crash apply on every device', () => {
  const limits = { noGpu: true, maxThreads: 1, maxSize: 320 };
  assert.deepEqual(runtimePlan({ preferGpu: true, isolated: true, cores: 8, limits }),
    { gpu: false, threads: 1, maxSize: 320 });
});

test('crash guard: a stage that ends normally leaves nothing behind', () => {
  const store = memoryStorage();
  const g = new CrashGuard(store);
  g.enter('loading', { gpu: true, threads: 4 });
  g.leave();
  assert.equal(new CrashGuard(store).recover(), null);
});

test('crash guard: each crash steps down one level (GPU, then threads, then size), then stops', () => {
  const store = memoryStorage();
  const visit = (stage, setup) => {
    const g = new CrashGuard(store);
    const c = g.recover();
    g.enter(stage, setup); // ... and the page dies here
    return c;
  };
  assert.equal(visit('loading', { gpu: true, threads: 4 }), null);
  let c = visit('running', { gpu: false, threads: 4, size: 480 });
  assert.equal(c.step, 'gpu');
  assert.equal(c.stage, 'loading');
  c = visit('running', { gpu: false, threads: 1, size: 480 });
  assert.equal(c.step, 'threads');
  c = visit('running', { gpu: false, threads: 1, size: 320 });
  assert.equal(c.step, 'size');
  c = new CrashGuard(store).recover();
  assert.equal(c.step, null);
  assert.match(describeCrash(c), /no lighter setting/);
  const limits = new CrashGuard(store).limits;
  assert.deepEqual([limits.noGpu, limits.maxThreads, limits.maxSize], [true, 1, 320]);
  assert.equal(limits.crashes.length, 4);
  new CrashGuard(store).resetLimits();
  assert.deepEqual(new CrashGuard(store).limits, {});
});

test('crash guard: the background is not a crash, coming back is', () => {
  const store = memoryStorage();
  const g = new CrashGuard(store);
  g.enter('running', { gpu: false, threads: 1, size: 480 });
  g.suspend(); // iOS ends the page in the background
  assert.equal(new CrashGuard(store).recover(), null);
  g.resume(); // back in front, then the page dies
  assert.equal(new CrashGuard(store).recover().step, 'size');
});

test('crash guard: blocked storage remembers nothing and never throws', () => {
  for (const store of [memoryStorage(true), null]) {
    const g = new CrashGuard(store);
    g.enter('loading', { gpu: true });
    assert.equal(g.recover(), null);
    assert.deepEqual(g.limits, {});
    g.leave();
    g.resetLimits();
  }
});

test('crash message names the stage, the set-up and the change', () => {
  const t = describeCrash({ stage: 'loading', gpu: true, step: 'gpu' });
  assert.match(t, /loading the detection model \(graphics processor\)/);
  assert.match(t, /use the processor instead/);
  assert.match(describeCrash({ stage: 'running', gpu: false, threads: 1, size: 480, step: 'size' }),
    /watching the camera \(processor, 1 thread, 480 px\).*320 px/);
});
