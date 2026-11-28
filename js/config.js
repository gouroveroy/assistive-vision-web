// What the app says about each class, and how urgently.
//
// Keyed by the class NAME in model/labels.json, never by index: the class
// list changes between training runs (3.3 dropped four classes and added
// four), and names are the only thing that stays put.
//
//   say       spoken word (defaults to the class name)
//   plural    spoken when two or more are in view
//   priority  1 moving vehicles · 2 things to trip on or step into
//             3 things to walk into · 4 street furniture (off by default)
//   kind      'ground' = lies on the walking surface, 'upright' = stands
//             in the way. Decides how "close" is estimated.
//   conf      minimum confidence before the class counts at all
//   announce  false = detected and drawn, never spoken

const VEHICLE = { say: 'vehicle', plural: 'vehicles', priority: 1, kind: 'upright', conf: 0.4 };

export const CLASS_INFO = {
  car: VEHICLE,
  bus: VEHICLE,
  truck: VEHICLE,
  motorcycle: { plural: 'motorcycles', priority: 1, kind: 'upright', conf: 0.4 },
  bicycle: { plural: 'bicycles', priority: 1, kind: 'upright', conf: 0.4 },
  // Dhaka classes, from pipeline 3.6 on (a 3.5 model has no rows for them).
  rickshaw: { plural: 'rickshaws', priority: 1, kind: 'upright', conf: 0.45 },
  'auto-rickshaw': { say: 'auto rickshaw', plural: 'auto rickshaws', priority: 1, kind: 'upright', conf: 0.45 },

  pothole: { plural: 'potholes', priority: 2, kind: 'ground', conf: 0.4 },
  puddle: { plural: 'puddles', priority: 2, kind: 'ground', conf: 0.45 },
  drain: { say: 'drain cover', plural: 'drain covers', priority: 2, kind: 'ground', conf: 0.45 },
  stair: { say: 'stairs', plural: 'stairs', priority: 2, kind: 'ground', conf: 0.45 },
  'bad road': { say: 'rough ground', plural: 'rough ground', priority: 2, kind: 'ground', conf: 0.5 },

  pole: { plural: 'poles', priority: 3, kind: 'upright', conf: 0.45 },
  barrier: { plural: 'barriers', priority: 3, kind: 'upright', conf: 0.45 },
  person: { plural: 'people', priority: 3, kind: 'upright', conf: 0.5 },
  dog: { plural: 'dogs', priority: 3, kind: 'upright', conf: 0.5 },
  'zebra cross': { say: 'crosswalk', plural: 'crosswalk', priority: 3, kind: 'ground', conf: 0.5 },
  'traffic light': { plural: 'traffic lights', priority: 3, kind: 'upright', conf: 0.5 },
  tree: { plural: 'trees', priority: 3, kind: 'upright', conf: 0.5 },
  'vendor stall': { say: 'stall', plural: 'stalls', priority: 3, kind: 'upright', conf: 0.5 },
  cow: { plural: 'cows', priority: 3, kind: 'upright', conf: 0.5 },

  'garbage bin': { say: 'bin', plural: 'bins', priority: 4, kind: 'upright', conf: 0.5 },
  bench: { plural: 'benches', priority: 4, kind: 'upright', conf: 0.5 },
  chair: { plural: 'chairs', priority: 4, kind: 'upright', conf: 0.5 },
  'potted plant': { say: 'plant pot', plural: 'plant pots', priority: 4, kind: 'upright', conf: 0.5 },
  'fire hydrant': { say: 'hydrant', plural: 'hydrants', priority: 4, kind: 'upright', conf: 0.5 },
  'parking meter': { plural: 'parking meters', priority: 4, kind: 'upright', conf: 0.5 },
  'stop sign': { plural: 'stop signs', priority: 4, kind: 'upright', conf: 0.5 },

  // Never spoken. The 3.3 model scored them AP50 0.36 and 0.16 and mistook
  // one for the other: 58% of real right turns were called left. Wrong
  // directions are worse than none. Models from pipeline 3.4 on (the app
  // ships 3.5) do not have these classes at all.
  'right turn': { announce: false, priority: 4, kind: 'ground', conf: 0.5 },
  'left turn': { announce: false, priority: 4, kind: 'ground', conf: 0.5 },
};

const FALLBACK = { priority: 3, kind: 'upright', conf: 0.5 };

/** Resolve the settings for every class index of a loaded model. */
export function classTable(labels) {
  return labels.map((name) => {
    const info = CLASS_INFO[name] || FALLBACK;
    return {
      name,
      say: info.say || name,
      plural: info.plural || info.say || name,
      priority: info.priority,
      kind: info.kind,
      conf: info.conf,
      announce: info.announce !== false,
    };
  });
}

export const DEFAULT_SETTINGS = {
  // Model input side in pixels: 320, 480, 640, or 0 = choose for me.
  // Automatic picks 480 on WebGPU and 320 on WebAssembly. 320 is the only
  // size the CPU path runs at a usable rate, but it loses small things:
  // on held-out test frames the 3.3 model missed a pothole and a pole at
  // 320 px that it found at 480 px.
  inputSize: 0,
  sensitivity: 0,        // added to every class threshold; negative = more alerts
  speechRate: 1.1,
  showBoxes: true,
  announceFurniture: false, // priority-4 classes
  vibrate: true,
  preferGpu: true,
};

export const ALERT_TIMING = {
  cooldownMs: 5000,       // never repeat the same alert sooner than this
  escalateMs: 1500,       // ...unless it has become "close" since
  zoneChangeMs: 2500,     // ...or has moved to another direction
  forgetMs: 2000,         // unseen this long = gone
};
