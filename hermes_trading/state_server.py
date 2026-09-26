"""Serves the worker's state over HTTP so the dashboard and reflection task can sync in one request.

  POST /state   Authorization: Bearer $HERMES_STATE_TOKEN
                body: {"have": {"<path relative to state/>": {"size": n, "sha256": hex}}}

  POST /sell    {"asset": "BTC/USDT"}                  close that asset's position now
  POST /add     {"asset": "ADA/USDT", "buy": true}     start trading an asset (optionally buy now)
                Both reply {"message": "..."} or {"error": "..."}; same bearer token.

/state lists every state file, sending only what the caller is missing:
  {"same": true}                    the caller's copy is identical
  {"append": "<text>"}              the caller's copy is a prefix (trades/hypotheses logs only grow)
  {"data": "<text>"}                anything else: the whole file
State is ~100 KB, so hashing it per request is cheap. Runs in a daemon thread next to the worker;
started only when HERMES_STATE_TOKEN is set. Railway routes the service's domain to $PORT.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config

MAX_BODY = 1_000_000
ACTION_TIMEOUT_S = 90  # Alpaca fills and price fetches can take a while
_worker = None  # set by attach(); actions run on the worker's event loop


def attach(worker) -> None:
    global _worker
    _worker = worker


def _run_action(path: str, body: dict) -> str:
    worker = _worker
    if worker is None or worker.loop is None:
        raise RuntimeError("worker is still starting")
    asset = str(body.get("asset") or "")
    if path == "/sell":
        coro = worker.book(asset).manual_sell()
    else:
        coro = worker.add_asset(asset, bool(body.get("buy")))
    return asyncio.run_coroutine_threadsafe(coro, worker.loop).result(timeout=ACTION_TIMEOUT_S)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_files() -> dict[str, bytes]:
    """Every state file, keyed by its posix path under STATE. Skips dotfiles (atomic_write temps)."""
    files = {}
    for path in config.STATE.rglob("*"):
        rel = path.relative_to(config.STATE)
        if not path.is_file() or any(p.startswith(".") or p == "lost+found" for p in rel.parts):
            continue
        data = path.read_bytes()
        if path.suffix == ".jsonl":
            data = data[: data.rfind(b"\n") + 1]  # never hand out a half-written last line
        files[rel.as_posix()] = data
    return files


def diff(have: dict) -> dict:
    out = {}
    for rel, data in state_files().items():
        known = have.get(rel)
        if known and known.get("sha256") == _sha(data):
            out[rel] = {"same": True}
        elif known and rel.endswith(".jsonl") and 0 < known.get("size", 0) <= len(data) \
                and _sha(data[: known["size"]]) == known.get("sha256"):
            out[rel] = {"append": data[known["size"]:].decode("utf-8")}
        else:
            out[rel] = {"data": data.decode("utf-8")}
    return out


def _handler(token: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/state", "/sell", "/add"):
                return self._send(404, {"error": "not found"})
            sent = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(sent.encode(), token.encode()):
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/state":
                    return self._send(200, {"files": diff(body.get("have") or {})})
                self._send(200, {"message": _run_action(self.path, body)})
            except ValueError as e:  # refused: unknown asset, market closed, no position…
                self._send(400, {"error": str(e)[:300]})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

        def log_message(self, *args) -> None:  # keep the worker log for trading decisions
            pass

    return Handler


def start() -> None:
    token = config.env("HERMES_STATE_TOKEN")
    if not token:
        return
    port = int(config.env("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"State server listening on :{port}", flush=True)
