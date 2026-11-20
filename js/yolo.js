// Pre- and post-processing for the exported YOLO model.
// Pure functions only (no DOM, no ONNX Runtime) so Node can test them.
//
// Model contract (Ultralytics ONNX export without NMS):
//   input   float32 [1, 3, S, S], RGB, values 0..1, letterboxed with grey 114
//   output  float32 [1, 4 + nc, N]: rows 0-3 are cx, cy, w, h in input
//           pixels, rows 4.. are per-class scores already passed through
//           sigmoid. N is 2100 at S=320 and 8400 at S=640.

export const PAD_VALUE = 114;

/** Where a srcW x srcH frame lands inside the S x S model input. */
export function letterboxParams(srcW, srcH, size) {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  // Same rounding as Ultralytics' LetterBox so boxes line up with training.
  const left = Math.max(0, Math.round((size - newW) / 2 - 0.1));
  const top = Math.max(0, Math.round((size - newH) / 2 - 0.1));
  return { size, scale, newW, newH, left, top, srcW, srcH };
}

/** RGBA bytes (canvas ImageData) -> planar RGB floats in 0..1. */
export function rgbaToCHW(rgba, size, out) {
  const area = size * size;
  const chw = out && out.length === 3 * area ? out : new Float32Array(3 * area);
  const k = 1 / 255;
  for (let i = 0, p = 0; i < area; i++, p += 4) {
    chw[i] = rgba[p] * k;
    chw[area + i] = rgba[p + 1] * k;
    chw[2 * area + i] = rgba[p + 2] * k;
  }
  return chw;
}

/**
 * Raw output -> candidate boxes in model-input pixels.
 * `thresholds` is one number or an array with one threshold per class.
 */
export function decode(data, dims, thresholds, maxCandidates = 300) {
  const channels = dims[dims.length - 2];
  const n = dims[dims.length - 1];
  const nc = channels - 4;
  if (nc <= 0 || data.length < channels * n) {
    throw new Error(`unexpected model output shape [${dims.join(', ')}]`);
  }
  const perClass = typeof thresholds !== 'number';

  const best = new Float32Array(n);
  const bestCls = new Uint16Array(n);
  for (let c = 0; c < nc; c++) {
    const off = (4 + c) * n;
    for (let i = 0; i < n; i++) {
      const s = data[off + i];
      if (s > best[i]) {
        best[i] = s;
        bestCls[i] = c;
      }
    }
  }

  const out = [];
  for (let i = 0; i < n; i++) {
    const score = best[i];
    const cls = bestCls[i];
    const thr = perClass ? thresholds[cls] : thresholds;
    if (!(score >= thr)) continue;
    const cx = data[i];
    const cy = data[n + i];
    const w = data[2 * n + i];
    const h = data[3 * n + i];
    out.push({
      cls, score,
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
    });
  }
  if (out.length > maxCandidates) {
    out.sort((a, b) => b.score - a.score);
    out.length = maxCandidates;
  }
  return out;
}

export function iou(a, b) {
  const ix = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const iy = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (ix <= 0 || iy <= 0) return 0;
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

/** Class-aware non-maximum suppression. Returns the kept boxes, best first. */
export function nms(boxes, iouThreshold = 0.5, maxDetections = 50) {
  const sorted = boxes.slice().sort((a, b) => b.score - a.score);
  const kept = [];
  for (const cand of sorted) {
    let keep = true;
    for (const k of kept) {
      if (k.cls === cand.cls && iou(k, cand) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      kept.push(cand);
      if (kept.length >= maxDetections) break;
    }
  }
  return kept;
}

/** Model-input pixels -> coordinates normalised to the source frame (0..1). */
export function toSourceCoords(box, lb) {
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  return {
    cls: box.cls,
    score: box.score,
    x1: clamp01((box.x1 - lb.left) / lb.newW),
    y1: clamp01((box.y1 - lb.top) / lb.newH),
    x2: clamp01((box.x2 - lb.left) / lb.newW),
    y2: clamp01((box.y2 - lb.top) / lb.newH),
  };
}

/** Full post-processing: raw output -> detections in source coordinates. */
export function postprocess(data, dims, lb, thresholds, iouThreshold = 0.5, maxDetections = 50) {
  const kept = nms(decode(data, dims, thresholds), iouThreshold, maxDetections);
  const out = [];
  for (const b of kept) {
    const d = toSourceCoords(b, lb);
    if (d.x2 - d.x1 > 0.002 && d.y2 - d.y1 > 0.002) out.push(d);
  }
  return out;
}
