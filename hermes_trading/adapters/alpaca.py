"""Alpaca paper broker + IEX market data, for stocks and ETFs.

Only the paper endpoint is accepted: live trading is not implemented, and a mistyped
ALPACA_BASE_URL must not be able to route real orders.
Keys come from ALPACA_API_KEY / ALPACA_API_SECRET.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
import time

import httpx

from ..config import env

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
CLOCK_TTL_S = 30
FILL_TIMEOUT_S = 20


class AlpacaError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Alpaca {status}: {message}")
        self.status = status
        self.message = message


def _parse_ts(ts: str) -> dt.datetime:
    # Alpaca sends nanoseconds ("...36.863725981-04:00"); fromisoformat wants at most microseconds.
    return dt.datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", ts.replace("Z", "+00:00")))


class Alpaca:
    def __init__(self) -> None:
        key, secret = env("ALPACA_API_KEY"), env("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")
        base = env("ALPACA_BASE_URL", PAPER_URL).rstrip("/")
        if base != PAPER_URL:
            raise RuntimeError(f"Only Alpaca paper trading is supported; ALPACA_BASE_URL must be {PAPER_URL}")
        self.feed = env("ALPACA_DATA_FEED", "iex")
        self._http = httpx.AsyncClient(
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, timeout=15
        )
        self._clock: tuple[float, dict] | None = None
        self._calendar: dict[str, dict] = {}

    async def _req(self, method: str, url: str, **kwargs):
        r = await self._http.request(method, url, **kwargs)
        if r.status_code >= 400:
            try:
                message = r.json().get("message", r.text)
            except ValueError:
                message = r.text
            raise AlpacaError(r.status_code, str(message)[:300])
        return r.json() if r.content else None

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- market calendar ---------------------------------------------------------

    async def clock(self) -> dict:
        if self._clock and time.monotonic() - self._clock[0] < CLOCK_TTL_S:
            return self._clock[1]
        clock = await self._req("GET", f"{PAPER_URL}/v2/clock")
        self._clock = (time.monotonic(), clock)
        return clock

    async def session(self) -> dict:
        """{is_open, date (ET), minutes_since_open, next_open}."""
        clock = await self.clock()
        now = _parse_ts(clock["timestamp"])
        date = now.date().isoformat()
        info = {"is_open": clock["is_open"], "date": date, "next_open": clock["next_open"], "minutes_since_open": None}
        if clock["is_open"]:
            if date not in self._calendar:
                days = await self._req("GET", f"{PAPER_URL}/v2/calendar", params={"start": date, "end": date})
                self._calendar = {date: days[0]} if days else {}
            day = self._calendar.get(date)
            if day:
                hh, mm = map(int, day["open"].split(":"))
                opened = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                info["minutes_since_open"] = (now - opened).total_seconds() / 60
        return info

    # --- market data -------------------------------------------------------------

    async def bars(self, symbol: str, limit: int = 100) -> list[tuple[int, float]]:
        """Most recent 1-minute bars as (epoch ms, close), oldest first. Spans the overnight gap."""
        start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).isoformat()
        data = await self._req(
            "GET",
            f"{DATA_URL}/v2/stocks/{symbol}/bars",
            params={"timeframe": "1Min", "limit": limit, "feed": self.feed, "sort": "desc", "start": start},
        )
        rows = [(int(_parse_ts(b["t"]).timestamp() * 1000), float(b["c"])) for b in data.get("bars") or []]
        return rows[::-1]

    _TF = {"1m": ("1Min", 5), "5m": ("5Min", 10), "15m": ("15Min", 20), "1h": ("1Hour", 60), "4h": ("4Hour", 200), "1d": ("1Day", 400)}

    async def ohlcv(self, symbol: str, tf: str, limit: int) -> dict:
        """Most recent bars as {t, open, high, low, close}, oldest first; cached a fraction of a bar."""
        from ..strategy import TF_SECONDS

        key = (symbol, tf)
        cache = self.__dict__.setdefault("_ohlcv_cache", {})
        hit = cache.get(key)
        if hit and time.monotonic() - hit[0] < min(TF_SECONDS[tf] / 5, 300):
            return hit[1]
        timeframe, days = self._TF[tf]
        start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        data = await self._req("GET", f"{DATA_URL}/v2/stocks/{symbol}/bars",
                               params={"timeframe": timeframe, "limit": limit, "feed": self.feed, "sort": "desc", "start": start})
        bars = (data.get("bars") or [])[::-1]
        candles = {"t": [int(_parse_ts(b["t"]).timestamp() * 1000) for b in bars], "open": [float(b["o"]) for b in bars],
                   "high": [float(b["h"]) for b in bars], "low": [float(b["l"]) for b in bars], "close": [float(b["c"]) for b in bars]}
        cache[key] = (time.monotonic(), candles)
        return candles

    # --- trading -----------------------------------------------------------------

    async def account(self) -> dict:
        return await self._req("GET", f"{PAPER_URL}/v2/account")

    async def position(self, symbol: str) -> dict | None:
        try:
            return await self._req("GET", f"{PAPER_URL}/v2/positions/{symbol}")
        except AlpacaError as e:
            if e.status == 404:
                return None
            raise

    async def market_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> dict:
        return await self._req(
            "POST",
            f"{PAPER_URL}/v2/orders",
            json={
                "symbol": symbol,
                "qty": f"{qty:.6f}",
                "side": side,
                "type": "market",
                "time_in_force": "day",  # required for fractional quantities
                "client_order_id": client_order_id,
            },
        )

    async def close_position(self, symbol: str) -> dict:
        return await self._req("DELETE", f"{PAPER_URL}/v2/positions/{symbol}")

    async def wait_filled(self, order_id: str) -> dict:
        deadline = time.monotonic() + FILL_TIMEOUT_S
        while True:
            order = await self._req("GET", f"{PAPER_URL}/v2/orders/{order_id}")
            if order["status"] == "filled":
                return order
            if order["status"] in ("canceled", "rejected", "expired"):
                raise AlpacaError(422, f"order {order_id} {order['status']}")
            if time.monotonic() > deadline:
                try:
                    await self._req("DELETE", f"{PAPER_URL}/v2/orders/{order_id}")
                except AlpacaError:
                    pass  # it may have filled meanwhile
                order = await self._req("GET", f"{PAPER_URL}/v2/orders/{order_id}")
                if float(order.get("filled_qty") or 0) > 0:
                    return order
                raise AlpacaError(408, f"order {order_id} not filled within {FILL_TIMEOUT_S}s; canceled")
            await asyncio.sleep(1)

    async def last_sell_fill(self, symbol: str, after_iso: str) -> dict | None:
        orders = await self._req(
            "GET",
            f"{PAPER_URL}/v2/orders",
            params={"status": "closed", "symbols": symbol, "after": after_iso, "direction": "desc", "limit": 20},
        )
        for order in orders or []:
            if order["side"] == "sell" and order["status"] == "filled":
                return order
        return None


_client: Alpaca | None = None


def client() -> Alpaca:
    global _client
    if _client is None:
        _client = Alpaca()
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
