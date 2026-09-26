"""Serves the worker's state over HTTP so the dashboard and reflection task can sync in one request.

  POST /state   Authorization: Bearer $HERMES_STATE_TOKEN
                body: {"have": {"<path relative to state/>": {"size": n, "sha256": hex}}}

  POST /sell    {"asset": "BTC/USDT"}                  close that asset's position now
  POST /add     {"asset": "ADA/USDT", "buy": true}     start trading an asset (optionally buy now)
  POST /strategy {"asset": "BTC/USDT", "changes": {"stop_loss_pct": 1.5}}   edit its strategy
                Both reply {"message": "..."} or {"error": "..."}; same bearer token.

/state lists every state file, sending only what the caller is missing:
  {"same": true}                    the caller's copy is identical
  {"append": "<text>"}              the caller's copy is a prefix (trades/hypotheses logs only grow)
  {"data": "<text>"}                anything else: the whole file
State is ~100 KB, so hashing it per request is cheap. Runs in a daemon thread next to the worker;
started only when HERMES_STATE_TOKEN is set. Railway routes the service's domain to $PORT.

With HERMES_DASHBOARD_PASSWORD set it also serves the dashboard (dashboard.html) at / for browsers:
  GET /             the dashboard, or a login form without a session
  POST /login       form field `password` → a signed session cookie (7 days)
  POST /logout      drops the session
  GET /api/state, GET /api/candles, POST /api/{sell,add,strategy,sync}   same API as the local
                    dashboard, on the worker's live state; they need the session cookie, and the
                    POSTs a same-origin Origin header
Changing the password logs every session out. Repeated wrong passwords from one IP are refused.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config
from .storage import load_yaml

MAX_BODY = 1_000_000
SESSION_COOKIE = "hermes_dash"
SESSION_S = 7 * 24 * 3600
LOGIN_WINDOW_S = 900
LOGIN_MAX_FAILURES = 5  # per IP per window
ACTION_TIMEOUT_S = 90  # Alpaca fills and price fetches can take a while
_worker = None  # set by attach(); actions run on the worker's event loop


def attach(worker) -> None:
    global _worker
    _worker = worker


def _run_action(path: str, body: dict) -> str:
    worker = _worker
    if worker is None or worker.loop is None:
        raise RuntimeError("el worker aún está arrancando")
    asset = str(body.get("asset") or "")
    if path == "/sell":
        coro = worker.book(asset).manual_sell()
    elif path == "/strategy":
        changes = body.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("no hay cambios que guardar")
        coro = worker.book(asset).set_strategy(changes)
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


LOGIN_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>hermes-trading</title>
<style>
  :root { color-scheme: light dark; --bg: #f6f7f9; --card: #fff; --fg: #1d2330; --muted: #667085; --line: #d0d5dd; --accent: #2e6be6; --bad: #c0392b; }
  @media (prefers-color-scheme: dark) { :root { --bg: #111418; --card: #1a1f26; --fg: #e6e9ee; --muted: #98a2b3; --line: #344054; --accent: #5b8def; --bad: #f07167; } }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
  form { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 28px; width: min(320px, calc(100vw - 32px)); box-sizing: border-box; }
  h1 { font-size: 18px; margin: 0 0 4px; } p { color: var(--muted); margin: 0 0 18px; font-size: 13px; }
  input, button { width: 100%; box-sizing: border-box; font: inherit; padding: 9px 11px; border-radius: 8px; }
  input { border: 1px solid var(--line); background: transparent; color: inherit; margin-bottom: 12px; }
  button { border: 0; background: var(--accent); color: #fff; cursor: pointer; }
  .err { color: var(--bad); font-size: 13px; margin: -4px 0 12px; }
</style></head><body>
<form method="post" action="/login">
  <h1>hermes-trading</h1><p>Dashboard de trading simulado</p>
  __ERROR__
  <input type="password" name="password" placeholder="Contraseña" autocomplete="current-password" autofocus required>
  <button type="submit">Entrar</button>
</form></body></html>"""


class _LoginGuard:
    """Counts failed logins per IP; refuses an IP after LOGIN_MAX_FAILURES in LOGIN_WINDOW_S."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def blocked(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(ip, []) if now - t < LOGIN_WINDOW_S]
            self._failures[ip] = recent
            return len(recent) >= LOGIN_MAX_FAILURES

    def fail(self, ip: str) -> None:
        with self._lock:
            self._failures.setdefault(ip, []).append(time.time())


def _hosted_action(path: str, body: dict) -> str:
    """The dashboard's /api/* actions, run on this worker."""
    if path == "/api/add":
        kind = "stock" if body.get("kind") == "stock" else "crypto"
        asset = config.normalize_asset(str(body.get("symbol") or ""), kind)
        return _run_action("/add", {"asset": asset, "buy": bool(body.get("buy"))})
    return _run_action(path.removeprefix("/api"), body)


def _handler(token: str, password: str = ""):
    # Session cookies are signed with a key tied to the password, so changing it ends them all.
    session_key = hmac.new(token.encode(), b"dashboard-session:" + password.encode(), hashlib.sha256).digest()
    guard = _LoginGuard()

    def sign(expires: int) -> str:
        return hmac.new(session_key, str(expires).encode(), hashlib.sha256).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            self._raw(code, json.dumps(payload, default=str).encode(), "application/json")

        def _raw(self, code: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        # --- dashboard session ----------------------------------------------------

        def _ip(self) -> str:
            # Railway's proxy puts the client first in X-Forwarded-For
            return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

        def _session_ok(self) -> bool:
            morsel = SimpleCookie(self.headers.get("Cookie") or "").get(SESSION_COOKIE)
            if morsel is None or "." not in morsel.value:
                return False
            expires, sig = morsel.value.split(".", 1)
            return expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(sig, sign(int(expires)))

        def _same_origin(self) -> bool:
            origin, host = self.headers.get("Origin"), self.headers.get("Host")
            return bool(origin and host) and origin in (f"https://{host}", f"http://{host}")

        def _login_page(self, code: int = 200, error: str = "") -> None:
            html = LOGIN_PAGE.replace("__ERROR__", f'<div class="err">{error}</div>' if error else "")
            self._raw(code, html.encode(), "text/html; charset=utf-8")

        def _login(self) -> None:
            ip = self._ip()
            if guard.blocked(ip):
                return self._login_page(429, "Demasiados intentos. Espera unos minutos.")
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            sent = (parse_qs(self.rfile.read(length).decode("utf-8", "replace")).get("password") or [""])[0]
            if not hmac.compare_digest(sent.encode(), password.encode()):
                guard.fail(ip)
                time.sleep(1)  # slow down guessing
                return self._login_page(401, "Contraseña incorrecta.")
            expires = int(time.time()) + SESSION_S
            cookie = (f"{SESSION_COOKIE}={expires}.{sign(expires)}; Max-Age={SESSION_S}; Path=/; "
                      "HttpOnly; Secure; SameSite=Strict")
            self._raw(303, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})

        def _dashboard_get(self, url) -> None:
            from . import dashboard  # heavy (ccxt, pandas); only loaded once someone opens the page

            if url.path in ("/", "/index.html"):
                if not self._session_ok():
                    return self._login_page()
                return self._raw(200, dashboard.PAGE.read_bytes(), "text/html; charset=utf-8")
            if not self._session_ok():
                return self._send(401, {"error": "sesión caducada; vuelve a entrar"})
            try:
                if url.path == "/api/state":
                    return self._send(200, dashboard.build_state(config.STATE, hosted=True))
                query = parse_qs(url.query)
                asset, tf = (query.get("asset") or [""])[0], (query.get("tf") or ["1m"])[0]
                if asset not in config.goal_assets(load_yaml(config.GOAL_FILE)) or tf not in dashboard.TIMEFRAMES:
                    return self._send(404, {"error": "activo o periodo desconocido"})
                self._send(200, dashboard.candles(asset, tf))
            except Exception as e:  # e.g. an exchange outage; the page keeps its last render
                self._send(503, {"error": f"{type(e).__name__}: {e}"[:300]})

        def _dashboard_post(self, path: str) -> None:
            if path == "/login":
                return self._login()
            if path == "/logout":
                expired = f"{SESSION_COOKIE}=; Max-Age=0; Path=/; HttpOnly; Secure; SameSite=Strict"
                return self._raw(303, b"", "text/plain", {"Location": "/", "Set-Cookie": expired})
            if not self._session_ok():
                return self._send(401, {"error": "sesión caducada; vuelve a entrar"})
            if not self._same_origin():
                return self._send(403, {"error": "origen no permitido"})
            if path == "/api/sync":  # the page reads live state; nothing to sync
                return self._send(200, {"running": False, "error": None})
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, {"message": _hosted_action(path, body)})
            except ValueError as e:  # refused: unknown asset, market closed, no position…
                self._send(400, {"error": str(e)[:300]})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

        # --- routing --------------------------------------------------------------

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/health":
                self._send(200, {"ok": True})
            elif password and url.path in ("/", "/index.html", "/api/state", "/api/candles"):
                self._dashboard_get(url)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if password and self.path in ("/login", "/logout", "/api/sync", "/api/sell", "/api/add", "/api/strategy"):
                return self._dashboard_post(self.path)
            if self.path not in ("/state", "/sell", "/add", "/strategy"):
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
    password = config.env("HERMES_DASHBOARD_PASSWORD")
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(token, password))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    web = "dashboard on /" if password else "no dashboard (HERMES_DASHBOARD_PASSWORD not set)"
    print(f"State server listening on :{port} — {web}", flush=True)
