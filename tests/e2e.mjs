// End-to-end check in a real browser (installed Chrome, headless).
//
//   node webapp/tests/e2e.mjs --ref REF.json [--video CAMERA.y4m] [--cpu] [--ios]
//
//   --ref    reference detections from tests/make_reference.py
//   --video  a .y4m or .mjpeg file Chrome plays as the camera; enables the
//            live part: tap Start, wait, check what was spoken
//   --cpu    force WebAssembly even if the browser has WebGPU
//   --ios    take the iPhone path (?ios): WebAssembly on one thread, no GPU
//
// Needs playwright-core (`npm install --no-save playwright-core` in
// webapp/, or point E2E_NODE_MODULES at a folder that has it) and Python
// for serve.py.
//
// Part 1 - still images: every confident reference detection must be found
// by the web app with the same class name and an overlapping box.
// Part 2 - live: the fake camera shows one picture; the app must process
// frames, confirm what it sees and speak about it, without page errors.
// Part 3 - crash recovery: a crash note left by a "killed" page makes the
// next visit say so and step down to one thread.
// Part 4 - static host without isolation headers, then offline.

import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const webapp = path.dirname(here);
const require = createRequire(
  process.env.E2E_NODE_MODULES ? path.join(process.env.E2E_NODE_MODULES, 'x.js') : import.meta.url,
);
const { chromium } = require('playwright-core');

const argv = process.argv.slice(2);
const arg = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : null);
const refPath = arg('--ref');
const video = arg('--video');
const forceCpu = argv.includes('--cpu');
const ios = argv.includes('--ios');
const modeQuery = [forceCpu && 'cpu', ios && 'ios'].filter(Boolean).join('&');
if (!refPath) {
  console.error('usage: node e2e.mjs --ref REF.json [--video CAMERA.y4m] [--cpu] [--ios]');
  process.exit(2);
}
const ref = JSON.parse(readFileSync(refPath, 'utf8'));

const PORT = 8137;
const PLAIN_PORT = 8138; // no isolation headers, like GitHub Pages
const servers = [
  spawn('python', [path.join(webapp, 'serve.py'), String(PORT), '--quiet'], { stdio: 'ignore' }),
  spawn('python', [path.join(webapp, 'serve.py'), String(PLAIN_PORT), '--quiet', '--plain'], { stdio: 'ignore' }),
];
const cleanup = () => { for (const s of servers) { try { s.kill(); } catch { /* gone */ } } };
process.on('exit', cleanup);

let failures = 0;
const check = (ok, msg) => {
  console.log(`  ${ok ? 'OK  ' : 'FAIL'} ${msg}`);
  if (!ok) failures++;
};
const iou = (a, b) => {
  const ix = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const iy = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (ix <= 0 || iy <= 0) return 0;
  const inter = ix * iy;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
};

async function waitForServer(port) {
  for (let i = 0; i < 50; i++) {
    try {
      const r = await fetch(`http://localhost:${port}/model/model.json`);
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error('serve.py did not start');
}

const args = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  '--autoplay-policy=no-user-gesture-required', '--enable-unsafe-webgpu'];
if (video) args.push(`--use-file-for-fake-video-capture=${path.resolve(video)}`);

await waitForServer(PORT);
await waitForServer(PLAIN_PORT);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args });
const context = await browser.newContext({ viewport: { width: 412, height: 915 }, permissions: ['camera'] });
const page = await context.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(String(e)));
page.on('console', (m) => {
  if (m.type() === 'error') pageErrors.push(m.text());
});

try {
  const url = `http://localhost:${PORT}/index.html?nosw${modeQuery ? `&${modeQuery}` : ''}`;
  await page.goto(url);
  await page.waitForFunction(() => window.__av && window.__av.classes.length > 0, null, { timeout: 20000 });
  const labels = await page.evaluate(() => window.__av.classes.map((c) => c.name));
  console.log(`\nmodel: ${labels.length} classes; page isolated: ${await page.evaluate(() => window.__av.isolated)}`);

  // ── Part 1: still images against the PyTorch reference ────────────────
  console.log('\nstill images, 640 px, web app vs PyTorch reference');
  for (const im of ref.images) {
    const b64 = readFileSync(im.path).toString('base64');
    const got = await page.evaluate(
      ({ data }) => window.__av.detectImage(`data:image/jpeg;base64,${data}`, 640),
      { data: b64 },
    );
    const name = path.basename(im.path);
    const mine = got.detections.map((d) => ({ name: d.name, score: d.score, box: [d.x1, d.y1, d.x2, d.y2] }));
    // The app applies its own, stricter thresholds (0.40-0.50) than the
    // reference's 0.25, so only confident reference boxes must be matched.
    const must = im.detections.filter((d) => d.score >= 0.6 && labels.includes(d.name));
    let matched = 0;
    let worst = 1;
    for (const r of must) {
      const best = mine.filter((m) => m.name === r.name).map((m) => iou(m.box, r.box)).sort((a, b) => b - a)[0] || 0;
      if (best >= 0.75) matched++;
      worst = Math.min(worst, best);
    }
    const extra = mine.filter((m) => m.score >= 0.6 &&
      !im.detections.some((r) => r.name === m.name && iou(m.box, r.box) >= 0.5));
    check(matched === must.length && extra.length === 0,
      `${name}: ${matched}/${must.length} confident reference boxes matched` +
      `${must.length ? ` (worst IoU ${worst.toFixed(2)})` : ''}, ${extra.length} unexplained` +
      ` · ${got.backend} ${got.inferMs.toFixed(0)} ms · ` +
      (mine.map((m) => `${m.name} ${m.score.toFixed(2)}`).join(', ') || 'nothing'));
  }

  if (ios) {
    const s = await page.evaluate(() => window.__av.stats);
    check(s.backend === 'wasm' && s.threads === 1,
      `iPhone path: plain WebAssembly on one thread (${s.backend}, ${s.threads} thread(s))`);
    const gpuDisabled = await page.evaluate(() => document.getElementById('set-gpu').disabled);
    check(gpuDisabled, 'iPhone path: the graphics setting is shown as unavailable');
  }

  // Speed at each input size, on this computer (not a phone).
  const first = readFileSync(ref.images[0].path).toString('base64');
  for (const size of [320, 480, 640]) {
    const times = await page.evaluate(async ({ data, size }) => {
      const out = [];
      for (let i = 0; i < 6; i++) out.push((await window.__av.detectImage(`data:image/jpeg;base64,${data}`, size)).inferMs);
      return out.slice(1);
    }, { data: first, size });
    const s = await page.evaluate(() => window.__av.stats);
    console.log(`  speed ${size} px: median ${times.sort((a, b) => a - b)[2].toFixed(0)} ms ` +
      `(${s.backend}${s.backend === 'wasm' ? `, ${s.threads} thread(s)` : ''})`);
  }

  // ── Part 2: live camera -> speech ─────────────────────────────────────
  if (video) {
    console.log('\nlive loop with a simulated camera');
    await page.click('#toggle');
    await page.waitForFunction(() => window.__av.running, null, { timeout: 60000 });
    await page.waitForFunction(() => window.__av.stats.frames >= 12, null, { timeout: 120000 });
    // Give a second, lower-priority sentence the chance to follow the first.
    await page.waitForFunction(
      () => window.__av.spoken.filter((s) => !/^(Starting|Assistive vision)/.test(s.text)).length >= 2,
      null, { timeout: 12000 }).catch(() => {});
    const live = await page.evaluate(() => ({
      stats: window.__av.stats,
      spoken: window.__av.spoken.map((s) => s.text),
      spokenAt: window.__av.spoken.map((s) => s.at),
      dets: window.__av.detections.map((d) => window.__av.classes[d.cls].name),
      status: document.getElementById('status').textContent,
      button: document.getElementById('toggle').textContent,
      video: [document.getElementById('cam').videoWidth, document.getElementById('cam').videoHeight],
    }));
    console.log(`  camera ${live.video.join('x')} · ${live.stats.backend} · ${live.stats.size} px · ` +
      `${live.stats.inferMs.toFixed(0)} ms · ${live.stats.fps.toFixed(1)} fps · ${live.stats.frames} frames`);
    console.log(`  in view: ${[...new Set(live.dets)].join(', ') || 'nothing'}`);
    console.log(`  spoken:  ${live.spoken.map((s) => `"${s}"`).join(' · ')}`);
    check(live.button === 'Stop', 'button reads Stop while running');
    check(live.dets.length > 0, 'detections on the live camera frame');
    const alerts = live.spoken.filter((s) => !/^(Starting|Assistive vision)/.test(s));
    check(alerts.length >= 1, `at least one hazard sentence spoken (${alerts.length})`);
    let tooSoon = 0;
    live.spoken.forEach((text, i) => {
      const prev = live.spoken.lastIndexOf(text, i - 1);
      if (i > 0 && prev >= 0 && live.spokenAt[i] - live.spokenAt[prev] < 4900) tooSoon++;
    });
    check(tooSoon === 0, 'no sentence repeated inside the 5 s cooldown');
    await page.screenshot({ path: path.join(here, 'e2e-live.png') });

    await page.click('#toggle');
    await page.waitForFunction(() => !window.__av.running, null, { timeout: 5000 });
    const after = await page.evaluate(() => ({
      button: document.getElementById('toggle').textContent,
      tracks: document.getElementById('cam').srcObject,
    }));
    check(after.button === 'Start' && after.tracks === null, 'Stop releases the camera');
  }

  // ── Part 3: crash recovery ────────────────────────────────────────────
  console.log('\ncrash recovery');
  const page3 = await context.newPage();
  page3.on('pageerror', (e) => pageErrors.push(String(e)));
  // What a page killed while watching on 4 threads at 480 px leaves behind.
  // Planted before the app's script runs: a reload would not do, because
  // a page that unloads normally (pagehide) wipes the note, as it should.
  await page3.addInitScript(() => {
    if (sessionStorage.getItem('planted')) return;
    sessionStorage.setItem('planted', '1');
    localStorage.removeItem('assistive-vision-limits-v1');
    localStorage.setItem('assistive-vision-guard-v1',
      JSON.stringify({ stage: 'running', gpu: false, threads: 4, size: 480, at: Date.now() }));
  });
  await page3.goto(`http://localhost:${PORT}/index.html?nosw`);
  await page3.waitForFunction(() => window.__av && window.__av.classes.length > 0, null, { timeout: 20000 });
  const rec = await page3.evaluate(() => ({
    status: document.getElementById('status').textContent,
    limits: window.__av.limits,
    reset: !document.getElementById('reset-limits').hidden,
  }));
  console.log(`  status: ${rec.status}`);
  check(/closed unexpectedly/.test(rec.status) && rec.limits.maxThreads === 1 && rec.reset,
    'a crash note is reported and teaches a one-thread limit');
  const after1 = await page3.evaluate(
    ({ data }) => window.__av.detectImage(`data:image/jpeg;base64,${data}`, 320),
    { data: readFileSync(ref.images[0].path).toString('base64') });
  check(after1.threads === 1 && after1.detections.length > 0,
    `the next model load uses one thread (${after1.backend}, ${after1.threads} thread(s))`);
  await page3.evaluate(() => document.getElementById('reset-limits').click());
  const cleared = await page3.evaluate(() => ({
    stored: localStorage.getItem('assistive-vision-limits-v1'),
    reset: !document.getElementById('reset-limits').hidden,
  }));
  check(cleared.stored === null && !cleared.reset, '"Forget the crash limits" clears them');
  await page3.close();

  // ── Part 4: static host without isolation headers, then offline ───────
  console.log('\nservice worker on a plain static host (as on GitHub Pages)');
  const page2 = await context.newPage();
  page2.on('pageerror', (e) => pageErrors.push(String(e)));
  await page2.goto(`http://localhost:${PLAIN_PORT}/index.html${modeQuery ? `?${modeQuery}` : ''}`);
  // First load is not isolated; the page reloads itself once the worker controls it.
  await page2.waitForFunction(
    () => window.__av && window.__av.isolated && navigator.serviceWorker.controller && window.__av.classes.length > 0,
    null, { timeout: 30000 }).catch(() => {});
  const sw = await page2.evaluate(() => ({
    isolated: window.crossOriginIsolated,
    controlled: !!navigator.serviceWorker.controller,
  }));
  check(sw.controlled, 'service worker controls the page');
  check(sw.isolated, 'page is cross-origin isolated through the worker (threads available)');
  const b64 = readFileSync(ref.images[0].path).toString('base64');
  const warm = await page2.evaluate(
    ({ data }) => window.__av.detectImage(`data:image/jpeg;base64,${data}`, 320), { data: b64 });
  check(warm.detections.length > 0, `model runs under the worker (${warm.backend}, ${warm.threads} thread(s))`);
  await page2.waitForTimeout(1500); // let the cache writes finish
  await context.setOffline(true);
  await page2.reload();
  await page2.waitForFunction(() => window.__av && window.__av.classes.length > 0, null, { timeout: 20000 }).catch(() => {});
  const off = await page2.evaluate(
    async ({ data }) => {
      try {
        const r = await window.__av.detectImage(`data:image/jpeg;base64,${data}`, 320);
        return { ok: r.detections.length > 0, backend: r.backend };
      } catch (e) {
        return { ok: false, error: String(e) };
      }
    }, { data: b64 }).catch((e) => ({ ok: false, error: String(e) }));
  check(off.ok, `works with the network switched off${off.error ? ` (${off.error})` : ''}`);
  await context.setOffline(false);
  await page2.close();

  check(pageErrors.length === 0, `no page errors${pageErrors.length ? `: ${pageErrors.slice(0, 3).join(' | ')}` : ''}`);
} finally {
  await browser.close();
  cleanup();
}

console.log(failures ? `\n${failures} CHECK(S) FAILED` : '\nALL E2E CHECKS PASSED');
process.exit(failures ? 1 : 0);
