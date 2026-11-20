"""Local test server for the web app.

    python webapp/serve.py            # http://localhost:8000
    python webapp/serve.py 9000
    python webapp/serve.py 9000 --plain   # no isolation headers, like GitHub Pages

Plain `python -m http.server` is not enough: on Windows it can serve .mjs
and .wasm with the wrong content type, and it cannot send the two headers
that let WebAssembly use several threads.

The camera only works on http://localhost or over HTTPS. To try the app on
a phone, publish the folder to an HTTPS host (see README.md).
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
        ".webmanifest": "application/manifest+json",
        ".onnx": "application/octet-stream",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # --plain behaves like GitHub Pages: no isolation headers, so the
        # service worker has to supply them.
        if "--plain" not in sys.argv:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "--quiet" not in sys.argv:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Assistive Vision web app: http://localhost:{PORT}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
