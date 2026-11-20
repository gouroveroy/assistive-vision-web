// Loads the ONNX model and runs it on video frames.
//
// Two ONNX Runtime Web builds are vendored in vendor/ort/:
//   ort.webgpu.min.mjs  WebGPU, with a WebAssembly fallback inside it
//   ort.wasm.min.mjs    WebAssembly only - half the download
// The WebGPU build is fetched only when the browser has a GPU adapter.

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
  }

  /**
   * @param modelUrl   .onnx file
   * @param opts.preferGpu   try WebGPU first
   * @param opts.onProgress  (fraction 0..1) while the model downloads
   */
  async load(modelUrl, { preferGpu = true, onProgress } = {}) {
    const gpu = preferGpu && (await hasWebGPU());
    const ort = await import(`${ORT_DIR}${gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
    ort.env.wasm.wasmPaths = ORT_DIR;
    // Warnings such as "some nodes run on the CPU" are normal under WebGPU.
    ort.env.logLevel = 'error';
    // Threads need a cross-origin isolated page (see sw.js). Without it
    // ONNX Runtime silently runs on one thread.
    this.threads = globalThis.crossOriginIsolated
      ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 2))
      : 1;
    ort.env.wasm.numThreads = this.threads;

    const bytes = await fetchWithProgress(modelUrl, onProgress);
    const create = (ep) =>
      ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3, // errors only
      });

    let session = null;
    let backend = 'wasm';
    if (gpu) {
      try {
        session = await create('webgpu');
        backend = 'webgpu';
      } catch (err) {
        console.warn('WebGPU session failed, using WebAssembly instead', err);
      }
    }
    if (!session) session = await create('wasm');

    this.ort = ort;
    this.session = session;
    this.backend = backend;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    this._bytes = bytes;
    return this;
  }

  /** Drop to WebAssembly after a WebGPU failure at run time. */
  async fallbackToWasm() {
    if (this.backend === 'wasm' || !this._bytes) return false;
    try {
      await this.session.release();
    } catch { /* already gone */ }
    this.session = await this.ort.InferenceSession.create(this._bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    this.backend = 'wasm';
    return true;
  }

  _ensureCanvas(size) {
    if (this.size === size && this.canvas) return;
    this.size = size;
    this.canvas =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(size, size)
        : Object.assign(document.createElement('canvas'), { width: size, height: size });
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
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
