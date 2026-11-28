// Service worker. Two jobs:
//
// 1. Offline use. After one visit the page, the runtime and the model are
//    served from the cache, so the app works with no connection.
//
// 2. Cross-origin isolation. WebAssembly threads need the page to be sent
//    with COOP/COEP headers, and static hosts such as GitHub Pages cannot
//    set them. Every response that passes through here gets them added.
//    Without isolation the app still works, on one thread.
//
// Bump CACHE when the files in SHELL change in a way old copies would break.

const CACHE = 'assistive-vision-v1';

const SHELL = [
  './',
  'index.html',
  'css/style.css',
  'js/app.js',
  'js/detector.js',
  'js/yolo.js',
  'js/alerts.js',
  'js/speech.js',
  'js/config.js',
  'js/device.js',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'model/model.json',
];

// Large files that never change under the same URL: the runtime is pinned
// to a version and the model URL carries ?v=<version>.
const IMMUTABLE = /\.(wasm|onnx)(\?|$)|\/vendor\//;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .catch(() => { /* offline install: fill the cache on demand instead */ })
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key !== CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

function isolated(response) {
  if (!response || response.status === 0 || response.type === 'opaque') return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

async function store(cache, request, response) {
  if (!response || response.status !== 200) return;
  try {
    await cache.put(request, response);
  } catch { /* over quota: keep working from the network */ }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request);
  if (hit) return hit;
  const fresh = await fetch(request);
  store(cache, request, fresh.clone());
  return fresh;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const fresh = await fetch(request, { cache: 'no-cache' });
    store(cache, request, fresh.clone());
    return fresh;
  } catch (err) {
    const hit =
      (await cache.match(request, { ignoreSearch: request.mode === 'navigate' })) ||
      (request.mode === 'navigate' ? await cache.match('index.html') : null);
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith(
    (IMMUTABLE.test(url.pathname + url.search) ? cacheFirst(request) : networkFirst(request))
      .then(isolated),
  );
});
