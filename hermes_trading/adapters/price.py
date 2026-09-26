"""1-minute OHLCV via ccxt public endpoints. EXCHANGE_ID / EXCHANGE_API_KEY override the default.

Exchange clients are reused across ticks, and each asset remembers the exchange that last
served it, so a geo-blocked primary isn't retried for every asset every minute.
"""
from __future__ import annotations

import ccxt.async_support as ccxt_async

from ..config import env
from . import SCHEMA_VERSION

# OKX before Kraken: some Kraken USDT pairs barely trade (79% of BNB/USDT's 15m candles were flat),
# which shrinks ATR and distorts RSI.
FALLBACK_EXCHANGES = ("binance", "okx", "kraken")
MIN_CANDLES = 20
MAX_FLAT_SHARE = 0.5  # candles with high == low; above this the market is too thin to trade on


def _thin(rows: list) -> bool:
    return bool(rows) and sum(1 for r in rows if r[2] == r[3]) / len(rows) > MAX_FLAT_SHARE

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
        if _thin(candles):
            errors.append(f"{exchange_id}: too many flat candles (illiquid)")
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


_ohlcv_cache: dict[tuple[str, str], tuple[float, dict]] = {}


async def ohlcv(asset: str, tf: str, limit: int) -> dict:
    """OHLCV candles {t, open, high, low, close} (t = bar start ms), from the exchange that last
    served this asset. Cached for a fraction of the bar length: indicators only use closed bars."""
    import time

    from ..strategy import TF_SECONDS

    key = (asset, tf)
    hit = _ohlcv_cache.get(key)
    if hit and time.monotonic() - hit[0] < min(TF_SECONDS[tf] / 5, 300):
        return hit[1]
    primary = env("EXCHANGE_ID", "binance")
    order = [primary] + [e for e in FALLBACK_EXCHANGES if e != primary]
    preferred = _preferred.get(asset)
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    errors = []
    for exchange_id in order:
        try:
            rows = await _client(exchange_id, exchange_id == primary).fetch_ohlcv(asset, timeframe=tf, limit=limit)
        except Exception as e:
            errors.append(f"{exchange_id}: {type(e).__name__}: {e}"[:160])
            continue
        if not rows:
            errors.append(f"{exchange_id}: no {tf} candles")
            continue
        if _thin(rows):
            errors.append(f"{exchange_id}: too many flat {tf} candles (illiquid)")
            continue
        candles = {"t": [int(r[0]) for r in rows], "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                   "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows]}
        _ohlcv_cache[key] = (time.monotonic(), candles)
        return candles
    raise RuntimeError(f"no exchange returned {tf} candles — " + "; ".join(errors))
