"""A minimal HTTP server exposing an SLM over a /generate endpoint."""

from __future__ import annotations

import json
from typing import Optional

from .slm import SLM


class Server:
    """Wrap an SLM in a tiny HTTP server (stdlib only, no extra deps required)."""

    def __init__(self, model: SLM, host: str = "0.0.0.0", port: int = 8000):
        self.model = model
        self.host = host
        self.port = port

    def start(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        slm = self.model

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path != "/generate":
                    self._json(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    prompt = data.get("prompt", "")
                    max_tokens = int(data.get("max_tokens", 256))
                    temperature = float(data.get("temperature", 0.7))
                    reply = slm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                    self._json(200, {"reply": reply})
                except Exception as exc:  # pragma: no cover
                    self._json(500, {"error": str(exc)})

            def log_message(self, *args) -> None:
                pass  # silence default logging

        httpd = HTTPServer((self.host, self.port), Handler)
        print(f"> Listening on :{self.port}  ·  ready")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
