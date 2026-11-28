// Loads the ONNX model and runs it on video frames.
//
// Two ONNX Runtime Web builds are vendored in vendor/ort/:
//   ort.webgpu.min.mjs  WebGPU, with a WebAssembly fallback inside it
//   ort.wasm.min.mjs    WebAssembly only - half the download
// The WebGPU build is fetched only when the caller asks for it and the
// browser has a GPU adapter. app.js never asks on iPhone (see device.js).

import { letterboxParams, rgbaToCHW, postprocess, PAD_VALUE } from './yolo.js';

const ORT_DIR = new URL('../vendor/ort/', import.meta.url).href;

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download the model (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > buf.length) {
      // content-length lied (compressed transfer): fall back to chunks
      const rest = [buf.subarray(0, got), value];
      let size = got + value.length;
      for (;;) {
        const r = await reader.read();
        if (r.done) break;
        rest.push(r.value);
        size += r.value.length;
      }
      const all = new Uint8Array(size);
      let o = 0;
      for (const part of rest) {
        all.set(part, o);
        o += part.length;
      }
      return all;
    }
    buf.set(value, got);
    got += value.length;
    if (onProgress) onProgress(got / total);
  }
  return got === total ? buf : buf.subarray(0, got);
}

async function hasWebGPU() {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

export class Detector {
  constructor() {
    this.session = null;
    this.ort = null;
    this.backend = null; // 'webgpu' | 'wasm'
    this.threads = 1;
    this.inputName = null;
    this.outputName = null;
    this.canvas = null;
    this.ctx = null;
    this.chw = null;
    this.size = 0;
    this.modelUrl = null;
  }

  /**
   * @param modelUrl   .onnx file
   * @param opts.gpu      try WebGPU first (when the browser has an adapter)
   * @param opts.threads  WebAssembly threads; above 1 needs a cross-origin
   *                      isolated page (see sw.js), else one is used
   * @param opts.onProgress  (fraction 0..1) while the model downloads
   */
  async load(modelUrl, { gpu = true, threads = 1, onProgress } = {}) {
    const useGpu = gpu && (await hasWebGPU());
    const ort = await import(`${ORT_DIR}${useGpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
    ort.env.wasm.wasmPaths = ORT_DIR;
    // Warnings such as "some nodes run on the CPU" are normal under WebGPU.
    ort.env.logLevel = 'error';
    // Each thread is a Web Worker with its own copy of the runtime. Without
    // isolation ONNX Runtime would silently run on one thread anyway.
    this.threads = globalThis.crossOriginIsolated ? Math.max(1, threads) : 1;
    ort.env.wasm.numThreads = this.threads;
    this.ort = ort;
    this.modelUrl = modelUrl;

    let bytes = await fetchWithProgress(modelUrl, onProgress);
    let session = null;
    let backend = 'wasm';
    if (useGpu) {
      try {
        session = await this._create(bytes, 'webgpu');
        backend = 'webgpu';
      } catch (err) {
        console.warn('WebGPU session failed, using WebAssembly instead', err);
      }
    }
    if (!session) session = await this._create(bytes, 'wasm');
    // The runtime has its own copy now. Not keeping this one saves 38 MB,
    // which matters on iPhone; a fallback fetches it again from the cache.
    bytes = null;

    this.session = session;
    this.backend = backend;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    return this;
  }

  _create(bytes, ep) {
    return this.ort.InferenceSession.create(bytes, {
      executionProviders: [ep],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3, // errors only
    });
  }

  /** Drop to WebAssembly after a WebGPU failure at run time. */
  async fallbackToWasm() {
    if (this.backend === 'wasm' || !this.modelUrl) return false;
    try {
      await this.session.release();
    } catch { /* already gone */ }
    this.session = null;
    // Served from the service worker's cache, so no new download.
    this.session = await this._create(await fetchWithProgress(this.modelUrl), 'wasm');
    this.backend = 'wasm';
    return true;
  }

  _ensureCanvas(size) {
    if (this.size === size && this.canvas) return;
    // Safari keeps a dropped canvas's pixels until garbage collection and
    // caps the total canvas memory; shrinking it frees them at once.
    if (this.canvas) {
      this.canvas.width = 0;
      this.canvas.height = 0;
    }
    this.size = size;
    const opts = { willReadFrequently: true, alpha: false };
    this.canvas = null;
    this.ctx = null;
    if (typeof OffscreenCanvas !== 'undefined') {
      this.canvas = new OffscreenCanvas(size, size);
      this.ctx = this.canvas.getContext('2d', opts); // null on some older Safari
    }
    if (!this.ctx) {
      this.canvas = Object.assign(document.createElement('canvas'), { width: size, height: size });
      this.ctx = this.canvas.getContext('2d', opts);
    }
    this.chw = new Float32Array(3 * size * size);
  }

  /**
   * @param source  anything drawImage accepts (video, canvas, image)
   * @param srcW, srcH  its pixel size
   * @param size    model input side (320, 480, 640)
   * @param thresholds  one confidence threshold per class
   * @returns {detections, inferMs} with boxes normalised to the source frame
   */
  async detect(source, srcW, srcH, size, thresholds) {
    this._ensureCanvas(size);
    const lb = letterboxParams(srcW, srcH, size);
    const ctx = this.ctx;
    ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(source, 0, 0, srcW, srcH, lb.left, lb.top, lb.newW, lb.newH);
    const rgba = ctx.getImageData(0, 0, size, size).data;
    rgbaToCHW(rgba, size, this.chw);

    const t0 = performance.now();
    const input = new this.ort.Tensor('float32', this.chw, [1, 3, size, size]);
    const result = await this.session.run({ [this.inputName]: input });
    const out = result[this.outputName];
    const data = out.data || (await out.getData());
    const inferMs = performance.now() - t0;

    const detections = postprocess(data, out.dims, lb, thresholds);
    if (typeof out.dispose === 'function') out.dispose();
    return { detections, inferMs };
  }
}
