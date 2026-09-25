"""1-minute OHLCV via ccxt public endpoints. EXCHANGE_ID / EXCHANGE_API_KEY override the default.

Exchange clients are reused across ticks, and each asset remembers the exchange that last
served it, so a geo-blocked primary isn't retried for every asset every minute.
"""
from __future__ import annotations

import ccxt.async_support as ccxt_async

from ..config import env
from . import SCHEMA_VERSION

FALLBACK_EXCHANGES = ("binance", "kraken", "okx")
MIN_CANDLES = 20

_clients: dict[str, object] = {}
_preferred: dict[str, str] = {}  # asset -> exchange id that last worked


def _client(exchange_id: str, is_primary: bool):
    client = _clients.get(exchange_id)
    if client is None:
        options = {"enableRateLimit": True}
        key, secret = env("EXCHANGE_API_KEY"), env("EXCHANGE_API_SECRET")
        if is_primary and key and secret:
            options.update(apiKey=key, secret=secret)
        client = getattr(ccxt_async, exchange_id)(options)
        _clients[exchange_id] = client
    return client


async def close() -> None:
    for client in _clients.values():
        await client.close()
    _clients.clear()


async def fetch(asset: str) -> dict:
    primary = env("EXCHANGE_ID", "binance")
    order = [primary] + [e for e in FALLBACK_EXCHANGES if e != primary]
    preferred = _preferred.get(asset)
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)

    errors = []
    for exchange_id in order:
        try:
            client = _client(exchange_id, exchange_id == primary)
            candles = await client.fetch_ohlcv(asset, timeframe="1m", limit=100)
        except Exception as e:  # geo-blocks, symbol not listed, network
            errors.append(f"{exchange_id}: {type(e).__name__}: {e}"[:160])
            continue
        if len(candles) < MIN_CANDLES:
            errors.append(f"{exchange_id}: only {len(candles)} candles")
            continue
        _preferred[asset] = exchange_id
        return {
            "schema_version": SCHEMA_VERSION,
            "source": exchange_id,
            "asset": asset,
            "timeframe": "1m",
            "timestamps": [int(c[0]) for c in candles],
            "closes": [float(c[4]) for c in candles],
            "last": float(candles[-1][4]),
        }
    raise RuntimeError("no exchange returned prices — " + "; ".join(errors))
