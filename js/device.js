// Per-device runtime choices: iPhone / iPad rules, and what a crash taught us.
// Pure logic (storage and navigator are passed in) so Node can test it.
//
// Why iPhone is different. Safari on iOS closes the whole page, with no
// error the page could catch, as soon as the tab uses too much memory
// ("A problem repeatedly occurred"). Two things in ONNX Runtime Web push
// it over on iOS 26, while Chrome on Android copes:
//   - The WebGPU build runs on a 29 MB "asyncify" WebAssembly file, and
//     Safari's optimising compiler can use gigabytes compiling such files
//     (WebKit bug 304810, still reproducing on iOS 26.6).
//   - Every WebAssembly thread is a Web Worker with its own instance of the
//     module, and that compiler cost grows with the number of instances.
// So on iPhone and iPad the app uses the plain WebAssembly build on one
// thread. For testing, the address can ask otherwise: ?gpu, ?threads=2.
//
// A crash leaves nothing to catch, so the app writes down what it is about
// to do (load the model, watch the camera) and wipes the note when that
// ends normally. A note still there on the next visit means the page died
// there; the app then steps down one level (no GPU, then one thread, then
// 320 px) and says so.

/** iPhone, iPod or iPad (iPadOS reports itself as a Mac with touch). */
export function isAppleMobile(nav = globalThis.navigator) {
  if (!nav) return false;
  const ua = nav.userAgent || '';
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints || 0) > 1);
}

/**
 * Which ONNX Runtime set-up to try.
 * @param o.apple      isAppleMobile()
 * @param o.preferGpu  the user's setting
 * @param o.isolated   crossOriginIsolated (threads need it)
 * @param o.cores      navigator.hardwareConcurrency
 * @param o.limits     CrashGuard.limits
 * @param o.params     URLSearchParams of the page address
 * @returns {{gpu: boolean, threads: number, maxSize: number}} maxSize 0 = no cap
 */
export function runtimePlan({ apple = false, preferGpu = true, isolated = false, cores = 2, limits = {}, params } = {}) {
  const p = params || new URLSearchParams();
  let gpu = preferGpu && !apple && !limits.noGpu;
  let threads = isolated ? Math.min(4, Math.max(1, cores || 2)) : 1;
  if (apple) threads = 1;
  if (limits.maxThreads) threads = Math.min(threads, limits.maxThreads);
  // Test switches typed into the address win over everything above.
  if (p.has('cpu')) gpu = false;
  else if (p.has('gpu')) gpu = true;
  const forced = Number(p.get('threads'));
  if (Number.isInteger(forced) && forced >= 1) threads = isolated ? Math.min(forced, 8) : 1;
  return { gpu, threads, maxSize: limits.maxSize || 0 };
}

const GUARD_KEY = 'assistive-vision-guard-v1';
const LIMITS_KEY = 'assistive-vision-limits-v1';

/**
 * Remembers the stage the page is in, so the next visit can tell that the
 * page was killed, and what limits that taught.
 * `storage` is localStorage or a stand-in; every access may throw (private
 * mode), and then the guard simply remembers nothing.
 */
export class CrashGuard {
  constructor(storage) {
    this.storage = storage;
    this.active = null; // {stage, setup} while a stage is open
  }

  _read(key) {
    try {
      return JSON.parse(this.storage.getItem(key) || 'null');
    } catch {
      return null;
    }
  }

  _write(key, value) {
    try {
      if (value == null) this.storage.removeItem(key);
      else this.storage.setItem(key, JSON.stringify(value));
    } catch { /* storage blocked: nothing is remembered */ }
  }

  /** A stage that could take the page down begins ('loading' | 'running'). */
  enter(stage, setup = {}) {
    this.active = { stage, setup };
    this._write(GUARD_KEY, { stage, ...setup, at: Date.now() });
  }

  /** The stage ended normally. */
  leave() {
    this.active = null;
    this._write(GUARD_KEY, null);
  }

  /** The page went to the background: iOS may end it there, which is not a crash. */
  suspend() {
    this._write(GUARD_KEY, null);
  }

  /** Back in the foreground: the open stage is at risk again. */
  resume() {
    if (this.active) this.enter(this.active.stage, this.active.setup);
  }

  get limits() {
    return this._read(LIMITS_KEY) || {};
  }

  resetLimits() {
    this._write(LIMITS_KEY, null);
  }

  /**
   * Call once at start-up. Returns null if the last visit ended normally,
   * else what was going on when it died and the step taken because of it:
   * {stage, gpu, threads, size, at, step: 'gpu'|'threads'|'size'|null}.
   */
  recover() {
    const died = this._read(GUARD_KEY);
    if (!died || typeof died !== 'object') return null;
    this._write(GUARD_KEY, null);
    const limits = this.limits;
    let step = null;
    if (died.gpu && !limits.noGpu) {
      limits.noGpu = true;
      step = 'gpu';
    } else if ((died.threads || 1) > 1 && limits.maxThreads !== 1) {
      limits.maxThreads = 1;
      step = 'threads';
    } else if (died.stage === 'running' && (died.size || 0) > 320 && limits.maxSize !== 320) {
      limits.maxSize = 320;
      step = 'size';
    }
    limits.crashes = [...(limits.crashes || []), { ...died, step }].slice(-5);
    this._write(LIMITS_KEY, limits);
    return { ...died, step };
  }
}

/** One line for people: what died and what changed. */
export function describeCrash(c) {
  const where = c.stage === 'loading' ? 'loading the detection model' : 'watching the camera';
  const setup = `${c.gpu ? 'graphics processor' : `processor, ${c.threads || 1} thread${(c.threads || 1) > 1 ? 's' : ''}`}` +
    `${c.size ? `, ${c.size} px` : ''}`;
  const change = {
    gpu: 'It will now use the processor instead of the graphics processor.',
    threads: 'It will now use one processor thread.',
    size: 'It will now look at 320 px detail.',
  }[c.step] || 'There is no lighter setting left to try.';
  return `The page closed unexpectedly last time, while ${where} (${setup}). ${change}`;
}
