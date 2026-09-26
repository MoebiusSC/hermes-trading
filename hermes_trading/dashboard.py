"""Local dashboard: `python -m hermes_trading.dashboard [--port 8765] [--no-sync] [--no-browser]`.

Serves dashboard.html on http://127.0.0.1:<port> (localhost only). Data sources:
  • dashboard_state/ — the dashboard's own mirror of the Railway volume, refreshed on start,
    every minute (every 5 without the worker's state server) and from the "Sync" button.
    Separate from remote_state/, which the scheduled reflection task owns.
  • live 1-minute candles from the exchange, for the price and RSI charts.

The same page is also served on the internet by the Railway worker itself (state_server.py,
behind a password), reading its state directly instead of a mirror.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ccxt
import httpx
import pandas as pd

from . import config, remote
from .loop import RSI_PERIOD
from .score import metrics, score
from .storage import load_yaml, read_jsonl

DASH_DIR = config.ROOT / "dashboard_state"
PAGE = Path(__file__).with_name("dashboard.html")
SYNC_EVERY_FAST_S = 60   # with the state server: one small request
SYNC_EVERY_CLI_S = 300   # Railway CLI fallback: ~40s per pull
CANDLES = 180  # 3 hours of 1-minute candles
CANDLE_TTL_S = 30
# price chart timeframes: ccxt timeframe, bars, Alpaca timeframe, days of history to ask Alpaca for
TIMEFRAMES = {
    "1m": ("1m", CANDLES, "1Min", 5),
    "5m": ("5m", 288, "5Min", 10),
    "1h": ("1h", 168, "1Hour", 40),
    "1d": ("1d", 180, "1Day", 400),
}
EQUITY_POINTS = 1500  # the equity curve is downsampled to about this many snapshots
EVENTS_SHOWN = 300
FALLBACK_EXCHANGES = ("binance", "kraken", "okx")


# --- Railway sync --------------------------------------------------------------


class Sync:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = False
        self.error: str | None = None
        self.finished_at: float | None = None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            remote.pull(DASH_DIR)
            self.error = None
        except BaseException as e:  # remote raises SystemExit with the CLI's message
            self.error = str(e)[-400:] or type(e).__name__
        finally:
            self.finished_at = time.time()
            self.running = False

    def status(self) -> dict:
        return {"running": self.running, "error": self.error}


SYNC = Sync()


def _auto_sync() -> None:
    while True:
        fast = config.env("HERMES_STATE_URL") and config.env("HERMES_STATE_TOKEN")
        time.sleep(SYNC_EVERY_FAST_S if fast else SYNC_EVERY_CLI_S)
        SYNC.start()


# --- state from the mirror ------------------------------------------------------


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evaluate_changes(trades: list[dict], hypotheses: list[dict], goal: dict) -> list[dict]:
    """Did each strategy change help? Compares the score of the `reflection_every` trades closed
    before it with the ones opened under it (until the next change). Small samples: a hint only."""
    n = int(goal["reflection_every"])
    out = []
    for h in hypotheses:
        end = next((x["ts"] for x in hypotheses if x["ts"] > h["ts"]), None)
        before = [t for t in trades if t["closed_at"] <= h["ts"]][-n:]
        after = [t for t in trades if t["opened_at"] >= h["ts"] and (end is None or t["opened_at"] < end)][:n]
        ev: dict = {"n_before": len(before), "n_after": len(after), "needed": n}
        if before and (len(after) >= n or (end is not None and len(after) >= 2)):
            sb, sa = score(before, goal), score(after, goal)
            ev.update(status="done", score_before=sb, score_after=sa, improved=sa > sb,
                      avg_before=sum(float(t["pnl_pct"]) for t in before) / len(before),
                      avg_after=sum(float(t["pnl_pct"]) for t in after) / len(after))
            if h.get("predicted_direction") in ("up", "down"):
                ev["matched"] = (sa > sb) == (h["predicted_direction"] == "up")
        else:
            ev["status"] = "pending" if end is None else "insufficient"
        out.append({**h, "evaluation": ev})
    return out


def _equity_curve(path: Path) -> list[dict]:
    points = read_jsonl(path)
    if len(points) > EQUITY_POINTS:
        step = len(points) / EQUITY_POINTS
        points = [points[int(i * step)] for i in range(EQUITY_POINTS)] + [points[-1]]
    return points


def build_state(state_dir: Path = DASH_DIR, hosted: bool = False) -> dict:
    """The page's data. `hosted`: read the worker's live state (on Railway), so there's no sync."""
    sync = {"running": False, "error": None} if hosted else SYNC.status()
    if not (state_dir / "goal.yaml").exists():
        return {"ready": False, "sync": sync}
    goal = load_yaml(state_dir / "goal.yaml")
    heartbeat = _read_json(state_dir / "heartbeat.json") or {}
    pulled = state_dir / ".pulled"
    if hosted:
        pulled_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    else:
        pulled_at = pulled.read_text().strip() if pulled.exists() else None
    assets = []
    for asset in config.goal_assets(goal):
        root = state_dir / "assets" / config.asset_slug(asset)
        trades = read_jsonl(root / "trades.jsonl")
        start = config.start_equity(asset, goal)
        assets.append(
            {
                "asset": asset,
                "kind": "stock" if config.is_stock(asset) else "crypto",
                "start_equity": start,
                "strategy": load_yaml(root / "strategy.yaml") if (root / "strategy.yaml").exists() else None,
                "paper": _read_json(root / "paper_account.json") or {"equity": start, "position": None},
                "trades": trades,
                "hypotheses": evaluate_changes(trades, read_jsonl(root / "hypotheses.jsonl"), goal),
                "metrics": metrics(trades),
                "score": score(trades, goal),
                "tick": (heartbeat.get("assets") or {}).get(asset),
            }
        )
    return {
        "ready": True,
        "goal": goal,
        "worker": {k: v for k, v in heartbeat.items() if k != "assets"},
        "pulled_at": pulled_at,
        "assets": assets,
        "equity": _equity_curve(state_dir / "equity.jsonl"),
        "events": read_jsonl(state_dir / "events.jsonl")[-EVENTS_SHOWN:],
        "sync": sync,
        "hosted": hosted,
    }


# --- manual actions (forwarded to the worker's state server) -------------------------


def worker_action(path: str, body: dict) -> tuple[int, dict]:
    url, token = config.env("HERMES_STATE_URL"), config.env("HERMES_STATE_TOKEN")
    if not url or not token:
        return 503, {"error": "faltan HERMES_STATE_URL / HERMES_STATE_TOKEN en .env"}
    try:
        r = httpx.post(url.rstrip("/") + path, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=100)
        payload = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return 502, {"error": f"no se pudo contactar con el worker: {type(e).__name__}: {e}"[:300]}
    if r.status_code == 200:
        SYNC.start()  # show the result without waiting for the next auto-sync
    return r.status_code, payload


def add_asset(body: dict) -> tuple[int, dict]:
    try:
        asset = config.normalize_asset(str(body.get("symbol") or ""), "stock" if body.get("kind") == "stock" else "crypto")
    except ValueError as e:
        return 400, {"error": str(e)}
    code, payload = worker_action("/add", {"asset": asset, "buy": bool(body.get("buy"))})
    local_goal = config.ROOT / "state" / "goal.yaml"
    if code == 200 and asset not in config.goal_assets(load_yaml(local_goal)):
        # keep the repo's goal.yaml in step, so a later `push-goal` doesn't drop the asset
        config.add_goal_asset(local_goal, asset)
    return code, payload


# --- live candles ------------------------------------------------------------------

_clients: dict = {}
_preferred: dict = {}
_candle_cache: dict = {}
_candle_lock = threading.Lock()


def _rsi_series(closes: list[float], period: int = RSI_PERIOD) -> list[float | None]:
    delta = pd.Series(closes, dtype=float).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    out: list[float | None] = []
    for i, (g, l) in enumerate(zip(gain, loss)):
        if i < period or math.isnan(g) or math.isnan(l):
            out.append(None)  # not enough history for a stable reading
        elif l == 0:
            out.append(50.0 if g == 0 else 100.0)
        else:
            out.append(round(100 - 100 / (1 + g / l), 2))
    return out


def _stock_candles(symbol: str, tf: str = "1m") -> dict:
    """Alpaca IEX bars (read-only data API; the same keys the worker uses)."""
    _, bars_wanted, alpaca_tf, days = TIMEFRAMES[tf]
    key, secret = config.env("ALPACA_API_KEY"), config.env("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in .env")
    feed = config.env("ALPACA_DATA_FEED", "iex")
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    r = httpx.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        params={"timeframe": alpaca_tf, "limit": bars_wanted, "feed": feed, "sort": "desc", "start": start},
        timeout=15,
    )
    r.raise_for_status()
    bars = (r.json().get("bars") or [])[::-1]
    closes = [float(b["c"]) for b in bars]
    return {
        "asset": symbol,
        "tf": tf,
        "source": f"Alpaca ({feed.upper()})",
        "t": [int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000) for b in bars],
        "open": [float(b["o"]) for b in bars],
        "high": [float(b["h"]) for b in bars],
        "low": [float(b["l"]) for b in bars],
        "close": closes,
        "rsi": _rsi_series(closes),
    }


def candles(asset: str, tf: str = "1m") -> dict:
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}")
    key = (asset, tf)
    with _candle_lock:
        hit = _candle_cache.get(key)
        if hit and time.time() - hit[0] < CANDLE_TTL_S:
            return hit[1]
        if config.is_stock(asset):
            data = _stock_candles(asset, tf)
            _candle_cache[key] = (time.time(), data)
            return data
        ccxt_tf, bars_wanted = TIMEFRAMES[tf][:2]
        primary = config.env("EXCHANGE_ID", "binance")
        order = [primary] + [e for e in FALLBACK_EXCHANGES if e != primary]
        if asset in _preferred:
            order.remove(_preferred[asset])
            order.insert(0, _preferred[asset])
        errors = []
        for exchange_id in order:
            try:
                client = _clients.setdefault(exchange_id, getattr(ccxt, exchange_id)({"enableRateLimit": True}))
                rows = client.fetch_ohlcv(asset, timeframe=ccxt_tf, limit=bars_wanted)
            except Exception as e:
                errors.append(f"{exchange_id}: {type(e).__name__}")
                continue
            if not rows:
                continue
            _preferred[asset] = exchange_id
            closes = [float(r[4]) for r in rows]
            data = {
                "asset": asset,
                "tf": tf,
                "source": exchange_id,
                "t": [int(r[0]) for r in rows],
                "open": [float(r[1]) for r in rows],
                "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows],
                "close": closes,
                "rsi": _rsi_series(closes),
            }
            _candle_cache[key] = (time.time(), data)
            return data
        raise RuntimeError("no exchange returned candles — " + "; ".join(errors))


# --- HTTP ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "hermes-dashboard"

    def _allowed(self) -> bool:
        # localhost only, and refuse other sites' pages (DNS rebinding / cross-site POSTs)
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in hosts}

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def do_GET(self) -> None:
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(build_state())
            elif url.path == "/api/candles":
                query = parse_qs(url.query)
                self._json(candles((query.get("asset") or [""])[0], (query.get("tf") or ["1m"])[0]))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # e.g. reading the mirror mid-swap; the page keeps its last render
            self._json({"error": f"{type(e).__name__}: {e}"}, 503)

    def do_POST(self) -> None:
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")
        path = urlparse(self.path).path
        if path == "/api/sync":
            SYNC.start()
            return self._json(SYNC.status())
        if path in ("/api/sell", "/api/add", "/api/strategy"):
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "JSON no válido"}, 400)
            if path == "/api/sell":
                code, payload = worker_action("/sell", {"asset": str(body.get("asset") or "")})
            elif path == "/api/strategy":
                code, payload = worker_action("/strategy", {"asset": str(body.get("asset") or ""), "changes": body.get("changes")})
            else:
                code, payload = add_asset(body)
            return self._json(payload, code)
        self._send(404, b"not found", "text/plain")

    def log_message(self, format, *args) -> None:  # keep the terminal quiet
        pass


def main(argv: list[str] | None = None) -> None:
    config.load_env()
    parser = argparse.ArgumentParser(description="Local trading dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-sync", action="store_true", help="don't pull from Railway")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_sync:
        SYNC.start()
        threading.Thread(target=_auto_sync, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
