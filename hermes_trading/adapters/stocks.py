"""1-minute stock/ETF bars from Alpaca market data (free IEX feed by default)."""
from __future__ import annotations

from . import SCHEMA_VERSION, alpaca

MIN_CANDLES = 20


async def fetch(symbol: str) -> dict:
    client = alpaca.client()
    rows = await client.bars(symbol, limit=100)
    if len(rows) < MIN_CANDLES:
        raise RuntimeError(f"only {len(rows)} bars for {symbol}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": f"alpaca-{client.feed}",
        "asset": symbol,
        "timeframe": "1m",
        "timestamps": [t for t, _ in rows],
        "closes": [c for _, c in rows],
        "last": rows[-1][1],
    }


async def ohlcv(symbol: str, tf: str, limit: int) -> dict:
    """OHLCV bars {t, open, high, low, close} from Alpaca, regular and extended hours as Alpaca sends them."""
    return await alpaca.client().ohlcv(symbol, tf, limit)
